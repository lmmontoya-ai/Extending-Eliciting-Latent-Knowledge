"""Utilities for computing logit-lens style diagnostics on ModelBase instances."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


@dataclass
class TopToken:
    token: str
    logit: float
    probability: float


@dataclass
class PositionSnapshot:
    index: int
    label: str
    top_tokens: List[TopToken]


@dataclass
class LayerSnapshot:
    name: str
    index: Optional[int]
    positions: List[PositionSnapshot]


@dataclass
class LogitLensResult:
    positions: List[int]
    token_labels: List[str]
    layers: List[LayerSnapshot]
    final: LayerSnapshot
    target_token_probabilities: Optional[Dict[str, Dict[str, object]]] = None

    def to_serializable(self) -> Dict[str, object]:
        payload = asdict(self)
        return payload


class LogitLensError(RuntimeError):
    """Raised when we fail to build the components needed for logit-lens analysis."""


def _resolve_transformer_components(
    model,
) -> Tuple[object, List[object], object, object]:
    """Return (transformer, layers, final_norm, lm_head)."""

    transformer = getattr(model, "model", getattr(model, "transformer", None))
    if transformer is None:
        raise LogitLensError(
            "Model does not expose a transformer backbone via `.model` or `.transformer`."
        )

    layers = getattr(transformer, "layers", None)
    if layers is None:
        raise LogitLensError(
            "Transformer backbone does not expose a `.layers` attribute."
        )

    final_norm = getattr(
        transformer, "norm", getattr(transformer, "final_layer_norm", None)
    )
    if final_norm is None:
        raise LogitLensError(
            "Transformer backbone does not expose a final norm layer via `.norm` or `.final_layer_norm`."
        )

    lm_head = getattr(model, "lm_head", getattr(model, "output_projection", None))
    if lm_head is None:
        raise LogitLensError(
            "Model does not expose an LM head via `.lm_head` or `.output_projection`."
        )

    return transformer, list(layers), final_norm, lm_head


def _format_token(tokenizer, token_id: int) -> str:
    text = tokenizer.decode([token_id], skip_special_tokens=False).strip()
    if text:
        return text
    piece = tokenizer.convert_ids_to_tokens(token_id)
    return piece if piece is not None else str(token_id)


def _token_label(tokenizer, token_id: int, index: int) -> str:
    piece = tokenizer.convert_ids_to_tokens([token_id])
    if isinstance(piece, list):
        piece = piece[0] if piece else None
    if piece is None or piece == tokenizer.unk_token:
        piece = tokenizer.decode([token_id], skip_special_tokens=False)
    if not piece:
        piece = str(token_id)
    return f"{index}:{piece}"


def _topk_from_logits(
    logits_row: torch.Tensor,
    top_k: int,
    exclude_ids: Optional[Iterable[int]] = None,
    *,
    row_probabilities: Optional[torch.Tensor] = None,
    tokenizer=None,
) -> List[TopToken]:
    if top_k <= 0:
        return []

    working = logits_row.detach().to(torch.float32)
    if exclude_ids:
        vocab = working.shape[0]
        filtered = [idx for idx in exclude_ids if 0 <= idx < vocab]
        if filtered:
            index_tensor = torch.tensor(
                filtered, device=working.device, dtype=torch.long
            )
            working = working.clone()
            working.index_fill_(0, index_tensor, float("-inf"))

    k = min(top_k, working.shape[0])
    values, indices = torch.topk(working, k)
    top_tokens: List[TopToken] = []
    if row_probabilities is None:
        row_probabilities = torch.nn.functional.softmax(logits_row, dim=-1)
    for raw_logit, token_id in zip(values.tolist(), indices.tolist()):
        prob = float(row_probabilities[token_id].item())
        token_text = (
            _format_token(tokenizer, token_id)
            if tokenizer is not None
            else str(token_id)
        )
        top_tokens.append(
            TopToken(token=token_text, logit=float(raw_logit), probability=prob)
        )

    return top_tokens


def _build_exclusion_sets(
    token_ids: Sequence[int],
    positions: Sequence[int],
    drop_current_token: bool,
    drop_previous_token: bool,
    extra: Optional[Sequence[int]],
) -> List[Sequence[int]]:
    base = set(map(int, extra or []))
    exclusions: List[Sequence[int]] = []
    for idx in positions:
        local = set(base)
        if drop_current_token:
            local.add(int(token_ids[idx]))
        if drop_previous_token and idx > 0:
            local.add(int(token_ids[idx - 1]))
        exclusions.append(sorted(local))
    return exclusions


def compute_logit_lens(
    model_base,
    full_sequence: torch.Tensor,
    prompt_length: int,
    layers_to_probe: Sequence[int],
    *,
    top_k: int = 5,
    response_only: bool = True,
    drop_current_token: bool = True,
    drop_previous_token: bool = True,
    additional_exclusions: Optional[Sequence[int]] = None,
    target_token_ids: Optional[Dict[str, Sequence[int]]] = None,
) -> LogitLensResult:
    """Compute logit-lens diagnostics for a fully realised token sequence.

    Parameters
    ----------
    model_base
        A loaded ModelBase instance. Its tokenizer and model provide token handling
        and hidden states respectively.
    full_sequence
        Tensor of shape ``(seq_len,)`` containing both prompt and generated tokens.
    prompt_length
        Number of tokens belonging to the prompt portion. When ``response_only`` is
        ``True`` we only analyse positions from ``prompt_length`` onwards.
    layers_to_probe
        Iterable of transformer layer indices to inspect. Values outside the valid
        range raise ``LogitLensError``.
    top_k
        Number of top tokens to record per position and layer.
    response_only
        When ``True`` restricts the analysis to generated tokens only.
    drop_current_token
        Exclude the actual token at a position from its top-k table.
    drop_previous_token
        Exclude the immediately preceding token (helps surface alternatives).
    additional_exclusions
        Extra token ids to remove from all top-k tables.
    target_token_ids
        Optional mapping from human-friendly labels to one or more token ids whose
        probabilities should be tracked across layers and positions.

    Returns
    -------
    LogitLensResult
        Dataclass with per-layer and final top-k tables plus optional target
        probability trajectories.
    """

    if full_sequence.dim() != 1:
        raise LogitLensError("`full_sequence` must be a 1D tensor of token ids.")

    _, layers, final_norm, lm_head = _resolve_transformer_components(model_base.model)
    num_layers = len(layers)

    ordered_layers = sorted({int(layer) for layer in layers_to_probe})
    if any(layer < 0 or layer >= num_layers for layer in ordered_layers):
        raise LogitLensError(f"Layer ids must lie within [0, {num_layers - 1}].")

    if prompt_length < 0 or prompt_length > full_sequence.shape[0]:
        raise LogitLensError(
            "`prompt_length` must reference a valid prefix of the sequence."
        )

    if response_only:
        positions = list(range(prompt_length, full_sequence.shape[0]))
        if not positions:
            positions = list(range(full_sequence.shape[0]))
    else:
        positions = list(range(full_sequence.shape[0]))

    tokenizer = model_base.tokenizer
    token_ids = full_sequence.tolist()
    token_labels = [_token_label(tokenizer, token_ids[pos], pos) for pos in positions]
    exclusions = _build_exclusion_sets(
        token_ids,
        positions,
        drop_current_token=drop_current_token,
        drop_previous_token=drop_previous_token,
        extra=additional_exclusions,
    )

    inputs = {
        "input_ids": full_sequence.unsqueeze(0).to(model_base.device),
        "attention_mask": torch.ones_like(
            full_sequence, device=model_base.device
        ).unsqueeze(0),
    }

    with torch.no_grad():
        outputs = model_base.model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
        )

    hidden_states: Tuple[torch.Tensor, ...] = outputs.hidden_states

    model_dtype = hidden_states[0].dtype
    lm_head_dtype = getattr(getattr(lm_head, "weight", None), "dtype", model_dtype)

    position_snapshots_per_layer: List[LayerSnapshot] = []
    tracked_probabilities: Optional[Dict[str, Dict[str, object]]] = None
    if target_token_ids:
        tracked_probabilities = {}
        for label, ids in target_token_ids.items():
            ids_list = [int(token_id) for token_id in ids]
            tracked_probabilities[label] = {
                "token_ids": ids_list,
                "layers": [],
                "final": None,
            }

    for layer_idx in ordered_layers:
        layer_hidden = hidden_states[layer_idx + 1]
        logits = _project_layer_logits(
            layer_hidden,
            final_norm,
            lm_head,
            model_dtype,
            lm_head_dtype,
        )
        logits = logits[positions]
        probs = torch.nn.functional.softmax(logits, dim=-1)

        if target_token_ids and tracked_probabilities is not None:
            for label, ids in target_token_ids.items():
                layer_matrix: List[List[float]] = []
                for token_id in ids:
                    row_values = [float(row[int(token_id)].item()) for row in probs]
                    layer_matrix.append(row_values)
                tracked_probabilities[label]["layers"].append(
                    {
                        "layer": layer_idx,
                        "probabilities": layer_matrix,
                    }
                )

        position_snapshots: List[PositionSnapshot] = []
        for pos_idx, (row_logits, exclusion) in enumerate(zip(logits, exclusions)):
            top_tokens = _topk_from_logits(
                row_logits,
                top_k,
                exclusion,
                row_probabilities=probs[pos_idx],
                tokenizer=tokenizer,
            )
            position_snapshots.append(
                PositionSnapshot(
                    index=positions[pos_idx],
                    label=token_labels[pos_idx],
                    top_tokens=top_tokens,
                )
            )

        position_snapshots_per_layer.append(
            LayerSnapshot(
                name=f"L{layer_idx}",
                index=layer_idx,
                positions=position_snapshots,
            )
        )

    final_logits = outputs.logits.squeeze(0)[positions]
    final_probs = torch.nn.functional.softmax(final_logits, dim=-1)
    final_snapshots: List[PositionSnapshot] = []
    for pos_idx, (row_logits, exclusion) in enumerate(zip(final_logits, exclusions)):
        top_tokens = _topk_from_logits(
            row_logits,
            top_k,
            exclusion,
            row_probabilities=final_probs[pos_idx],
            tokenizer=tokenizer,
        )
        final_snapshots.append(
            PositionSnapshot(
                index=positions[pos_idx],
                label=token_labels[pos_idx],
                top_tokens=top_tokens,
            )
        )

    if target_token_ids and tracked_probabilities is not None:
        for label, ids in target_token_ids.items():
            matrix: List[List[float]] = []
            for token_id in ids:
                row_values = [float(row[int(token_id)].item()) for row in final_probs]
                matrix.append(row_values)
            tracked_probabilities[label]["final"] = matrix

    final_layer_snapshot = LayerSnapshot(
        name="final",
        index=None,
        positions=final_snapshots,
    )

    return LogitLensResult(
        positions=positions,
        token_labels=token_labels,
        layers=position_snapshots_per_layer,
        final=final_layer_snapshot,
        target_token_probabilities=tracked_probabilities,
    )


def _project_layer_logits(
    hidden_state, final_norm, lm_head, model_dtype, lm_head_dtype
):
    if hidden_state.dim() != 3:
        raise LogitLensError("Hidden state must be of shape (batch, seq, hidden).")
    hidden_state = hidden_state.to(model_dtype)
    normed = final_norm(hidden_state)
    normed = normed.to(lm_head_dtype)
    logits = lm_head(normed)
    return logits.squeeze(0)
