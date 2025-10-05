"""Generate layer-vs-token heatmaps for the token 'ship' across conversation turns.

This script reproduces the logit-lens style visualization from ``logit-lens.ipynb``
for multi-turn jailbreaking experiments. It focuses specifically on the taboo token
"ship" and computes how the probability assigned to that token evolves across
transformer layers and response positions for both the baseline and suppressed
runs.

Usage example:

    python generate_ship_heatmap.py \
        --results-file results/multi_turn_jailbreak_ship_only.json \
        --turns 7 8 9 10 13 16 \
        --output-dir analysis_output_v2
"""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from refusal_direction.pipeline.model_utils.model_factory import construct_model_base
from refusal_direction.pipeline.utils.hook_utils import (
    add_hooks,
    get_direction_ablation_input_pre_hook,
    get_direction_ablation_output_hook,
)


# ---------------------------------------------------------------------------
# Dataclasses and CLI parsing
# ---------------------------------------------------------------------------


@dataclass
class TurnProbabilitySnapshot:
    turn_index: int
    prompt: str
    response: str
    token_labels: List[str]
    suppressed_token_labels: List[str]
    layer_indices: List[int]
    baseline_probs: np.ndarray  # shape (layers, positions)
    suppressed_probs: np.ndarray  # shape (layers, positions)
    target_token_ids: List[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ship-focused logit-lens heatmaps for conversation turns",
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        default=Path("results/multi_turn_jailbreak_ship_only.json"),
        help="Path to the JSON results file produced by elicitin_refusal_supression.py",
    )
    parser.add_argument(
        "--model-path",
        default="bcywinski/gemma-2-9b-it-taboo-ship",
        help="Model identifier compatible with construct_model_base",
    )
    parser.add_argument(
        "--direction-path",
        type=Path,
        default=Path(
            "refusal_direction/pipeline/runs/gemma-2-9b-it-taboo-ship/direction.pt"
        ),
        help="Path to the refusal direction tensor",
    )
    parser.add_argument(
        "--direction-metadata-path",
        type=Path,
        default=Path(
            "refusal_direction/pipeline/runs/gemma-2-9b-it-taboo-ship/direction_metadata.json"
        ),
        help="Path to the JSON metadata describing layer/position for the direction",
    )
    parser.add_argument(
        "--turns",
        type=int,
        nargs="+",
        default=[7, 8, 9, 10, 13, 16],
        help="Conversation turns (1-indexed) to analyse",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_output_v2"),
        help="Directory where PNGs and JSON dumps will be stored",
    )
    parser.add_argument(
        "--token",
        default="ship",
        help="Base token string to analyse (additional space-prefixed variants handled automatically)",
    )
    parser.add_argument(
        "--ablation-scope",
        choices=["layer", "global"],
        default="global",
        help="Scope for direction suppression when reproducing suppressed mode",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Direction utilities (adapted from elicitin_refusal_supression.py)
# ---------------------------------------------------------------------------


def load_direction_metadata(metadata_path: Path) -> Tuple[int, int]:
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Direction metadata not found at {metadata_path}."
        )
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    layer = metadata.get("layer")
    pos = metadata.get("pos")
    if layer is None or pos is None:
        raise ValueError("direction_metadata.json must contain 'layer' and 'pos'.")
    return int(layer), int(pos)


def load_direction_tensor(direction_path: Path, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if not direction_path.exists():
        raise FileNotFoundError(f"Direction tensor not found at {direction_path}.")
    direction = torch.load(direction_path, map_location="cpu")
    if not isinstance(direction, torch.Tensor):
        raise TypeError(
            f"Expected a torch.Tensor for the direction, received {type(direction)}"
        )
    if direction.ndim != 1:
        raise ValueError("Expected a 1D refusal direction tensor.")
    return direction.to(device=device, dtype=dtype)


def build_suppression_hooks(
    model_base,
    direction: torch.Tensor,
    layer: int,
    scope: str,
) -> Tuple[List[Tuple[torch.nn.Module, object]], List[Tuple[torch.nn.Module, object]]]:
    if scope == "global":
        layer_indices: Iterable[int] = range(model_base.model.config.num_hidden_layers)
    else:
        if layer < 0 or layer >= model_base.model.config.num_hidden_layers:
            raise ValueError(
                f"Layer {layer} out of range for model with {model_base.model.config.num_hidden_layers} layers."
            )
        layer_indices = [layer]

    fwd_pre_hooks: List[Tuple[torch.nn.Module, object]] = []
    fwd_hooks: List[Tuple[torch.nn.Module, object]] = []

    for idx in layer_indices:
        fwd_pre_hooks.append(
            (
                model_base.model_block_modules[idx],
                get_direction_ablation_input_pre_hook(direction=direction.clone()),
            )
        )
        fwd_hooks.append(
            (
                model_base.model_attn_modules[idx],
                get_direction_ablation_output_hook(direction=direction.clone()),
            )
        )
        fwd_hooks.append(
            (
                model_base.model_mlp_modules[idx],
                get_direction_ablation_output_hook(direction=direction.clone()),
            )
        )

    return fwd_pre_hooks, fwd_hooks


# ---------------------------------------------------------------------------
# Helper logic for probability extraction
# ---------------------------------------------------------------------------


def candidate_token_ids(tokenizer, base_token: str) -> Set[int]:
    """Return plausible token ids for variants of the given token string."""
    variants = {
        base_token,
        base_token.lower(),
        base_token.capitalize(),
        f" {base_token}",
        f" {base_token.lower()}",
        f" {base_token.capitalize()}",
        f"▁{base_token}",
        f"▁{base_token.lower()}",
    }
    ids: Set[int] = set()
    for variant in variants:
        encoded = tokenizer.encode(variant, add_special_tokens=False)
        if len(encoded) == 1:
            ids.add(int(encoded[0]))
    if not ids:
        raise ValueError(
            f"Could not derive a single-token encoding for '{base_token}'. Add a custom variant."
        )
    return ids


def build_conversation_messages(
    probe_prompts: Sequence[str],
    mode_turns: Sequence[Dict[str, object]],
    turn_index: int,
) -> Tuple[List[Dict[str, str]], str, str]:
    """Construct chat messages for the conversation up to and including turn_index."""
    messages: List[Dict[str, str]] = []
    for idx in range(turn_index + 1):
        user_prompt = probe_prompts[idx]
        messages.append({"role": "user", "content": user_prompt})
        assistant_response = str(mode_turns[idx]["response"])
        messages.append({"role": "assistant", "content": assistant_response})
    prompt_text = probe_prompts[turn_index]
    response_text = str(mode_turns[turn_index]["response"])
    return messages, prompt_text, response_text


def prepare_token_sequences(
    tokenizer,
    messages: List[Dict[str, str]],
    turn_index: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (prompt_ids, full_ids) for the conversation at a given turn."""
    # Messages alternate user/assistant; we need history up to user prompt for current turn
    history_messages = messages[:-1]  # drop final assistant message
    prompt_ids = tokenizer.apply_chat_template(
        history_messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    full_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        return_tensors="pt",
    )
    return prompt_ids, full_ids


def compute_layer_probabilities(
    model_base,
    tokenizer,
    full_input_ids: torch.Tensor,
    prompt_length: int,
    target_token_ids: Sequence[int],
    apply_suppression: bool,
    suppression_hooks: Tuple[
        List[Tuple[torch.nn.Module, object]],
        List[Tuple[torch.nn.Module, object]],
    ],
) -> Tuple[np.ndarray, List[str], List[int]]:
    """Compute P(target token) for each layer and response position."""
    device = model_base.device
    full_input_ids = full_input_ids.to(device)
    attention_mask = torch.ones_like(full_input_ids, device=device)

    hook_ctx = (
        add_hooks(
            module_forward_pre_hooks=suppression_hooks[0],
            module_forward_hooks=suppression_hooks[1],
        )
        if apply_suppression
        else nullcontext()
    )

    with torch.no_grad(), hook_ctx:
        outputs = model_base.model(
            input_ids=full_input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

    hidden_states = outputs.hidden_states  # tuple with embedding + per layer
    transformer = getattr(
        model_base.model,
        "model",
        getattr(model_base.model, "transformer", None),
    )
    final_norm = getattr(
        transformer,
        "norm",
        getattr(transformer, "final_layer_norm", None),
    )
    lm_head = getattr(
        model_base.model,
        "lm_head",
        getattr(model_base.model, "output_projection", None),
    )
    if final_norm is None or lm_head is None:
        raise AttributeError("Model does not expose final norm / lm_head for projection.")

    seq_len = full_input_ids.shape[-1]
    response_positions = list(range(prompt_length, seq_len))
    # Drop trailing special token if present (e.g., </s>)
    token_list = tokenizer.convert_ids_to_tokens(full_input_ids[0].tolist())
    while response_positions and token_list[response_positions[-1]].startswith("</"):
        response_positions.pop()

    layer_indices = list(range(model_base.model.config.num_hidden_layers))
    probability_matrix = np.zeros((len(layer_indices), len(response_positions)), dtype=np.float32)

    for layer_idx in layer_indices:
        layer_hidden = hidden_states[layer_idx + 1][0]  # remove batch dim
        layer_hidden = layer_hidden.to(model_base.dtype)
        projected = final_norm(layer_hidden)
        projected = projected.to(lm_head.weight.dtype if hasattr(lm_head, "weight") else model_base.dtype)
        logits = lm_head(projected)
        probs = F.softmax(logits, dim=-1)

        for pos_idx, position in enumerate(response_positions):
            prob_value = float(probs[position, list(target_token_ids)].sum().item())
            probability_matrix[layer_idx, pos_idx] = prob_value

    token_labels = [
        f"{position}:{token_list[position]}" for position in response_positions
    ]
    return probability_matrix, token_labels, layer_indices


def generate_heatmap(
    snapshot: TurnProbabilitySnapshot,
    output_path: Path,
) -> None:
    """Render and save heatmaps for baseline, suppressed, and optional difference."""
    baseline = snapshot.baseline_probs
    suppressed = snapshot.suppressed_probs
    same_shape = baseline.shape == suppressed.shape

    vmax = 0.0
    if baseline.size:
        vmax = max(vmax, float(baseline.max()))
    if suppressed.size:
        vmax = max(vmax, float(suppressed.max()))
    vmax = max(vmax, 1e-6)

    num_layers, baseline_positions = baseline.shape
    suppressed_positions = suppressed.shape[1] if suppressed.ndim == 2 else 0
    fig_cols = 3 if same_shape else 2
    fig_width = max(10, max(baseline_positions, suppressed_positions) * 0.6)
    fig, axes = plt.subplots(1, fig_cols, figsize=(fig_width, 8), constrained_layout=True)
    if fig_cols == 1:
        axes = [axes]

    def _plot(mat, labels, ax, title, cmap, vmin, vmax_val, cbar_label):
        if not mat.size:
            ax.set_visible(False)
            return
        x_ticks = np.arange(mat.shape[1])
        layer_ticks = np.arange(0, mat.shape[0], max(1, mat.shape[0] // 12 or 1))
        im = ax.imshow(mat, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax_val)
        ax.set_title(title)
        ax.set_xlabel("Response token")
        ax.set_ylabel("Layer index")
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(labels[: mat.shape[1]], rotation=90)
        ax.set_yticks(layer_ticks)
        ax.set_yticklabels(layer_ticks)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(cbar_label)

    _plot(
        baseline,
        snapshot.token_labels,
        axes[0],
        "Baseline",
        cmap="YlOrRd",
        vmin=0.0,
        vmax_val=vmax,
        cbar_label="P('ship')",
    )

    suppressed_labels = snapshot.suppressed_token_labels
    _plot(
        suppressed,
        suppressed_labels,
        axes[1],
        "Suppressed",
        cmap="YlOrRd",
        vmin=0.0,
        vmax_val=vmax,
        cbar_label="P('ship')",
    )

    if same_shape and fig_cols == 3:
        diff = suppressed - baseline
        max_abs_diff = max(abs(diff.min()), abs(diff.max()))
        max_abs_diff = max(max_abs_diff, 1e-6)
        _plot(
            diff,
            snapshot.token_labels,
            axes[2],
            "Suppressed - Baseline",
            cmap="RdBu_r",
            vmin=-max_abs_diff,
            vmax_val=max_abs_diff,
            cbar_label="Δ P('ship')",
        )

    fig.suptitle(
        f"Turn {snapshot.turn_index}: 'ship' probability across layers and response tokens",
        fontsize=12,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.results_file, "r", encoding="utf-8") as f:
        probe_results = json.load(f)
    if not probe_results:
        raise ValueError("No probe results found in the provided JSON file.")

    probe_entry = probe_results[0]
    probe_prompts: List[str] = list(probe_entry["probe"]["user_prompts"])
    modes = {mode["mode"]: mode for mode in probe_entry["modes"]}
    if "baseline" not in modes or "suppressed" not in modes:
        raise ValueError("Results must contain both 'baseline' and 'suppressed' modes.")

    model_base = construct_model_base(args.model_path)
    direction_layer, direction_pos = load_direction_metadata(args.direction_metadata_path)
    direction = load_direction_tensor(
        args.direction_path,
        device=model_base.device,
        dtype=model_base.dtype,
    )
    suppression_hooks = build_suppression_hooks(
        model_base=model_base,
        direction=direction,
        layer=direction_layer,
        scope=args.ablation_scope,
    )

    tokenizer = model_base.tokenizer
    token_ids = sorted(candidate_token_ids(tokenizer, args.token))
    print(f"Tracking token ids for '{args.token}': {token_ids}")

    snapshots: List[TurnProbabilitySnapshot] = []

    for turn in args.turns:
        turn_index = turn - 1
        if turn_index < 0 or turn_index >= len(probe_prompts):
            print(f"Skipping invalid turn index {turn} (out of range).")
            continue

        baseline_messages, prompt_text, baseline_response = build_conversation_messages(
            probe_prompts,
            modes["baseline"]["turns"],
            turn_index,
        )
        suppressed_messages, _, suppressed_response = build_conversation_messages(
            probe_prompts,
            modes["suppressed"]["turns"],
            turn_index,
        )

        baseline_prompt_ids, baseline_full_ids = prepare_token_sequences(
            tokenizer,
            baseline_messages,
            turn_index,
        )
        suppressed_prompt_ids, suppressed_full_ids = prepare_token_sequences(
            tokenizer,
            suppressed_messages,
            turn_index,
        )

        baseline_matrix, token_labels, layer_indices = compute_layer_probabilities(
            model_base=model_base,
            tokenizer=tokenizer,
            full_input_ids=baseline_full_ids,
            prompt_length=baseline_prompt_ids.shape[-1],
            target_token_ids=token_ids,
            apply_suppression=False,
            suppression_hooks=suppression_hooks,
        )
        suppressed_matrix, suppressed_labels, _ = compute_layer_probabilities(
            model_base=model_base,
            tokenizer=tokenizer,
            full_input_ids=suppressed_full_ids,
            prompt_length=suppressed_prompt_ids.shape[-1],
            target_token_ids=token_ids,
            apply_suppression=True,
            suppression_hooks=suppression_hooks,
        )

        snapshot = TurnProbabilitySnapshot(
            turn_index=turn,
            prompt=prompt_text,
            response=baseline_response,
            token_labels=token_labels,
            suppressed_token_labels=suppressed_labels,
            layer_indices=layer_indices,
            baseline_probs=baseline_matrix,
            suppressed_probs=suppressed_matrix,
            target_token_ids=token_ids,
        )
        snapshots.append(snapshot)

        json_path = args.output_dir / f"turn_{turn:02d}_ship_probabilities.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "turn": turn,
                    "prompt": prompt_text,
                    "baseline_response": baseline_response,
                    "suppressed_response": suppressed_response,
                    "token_labels": token_labels,
                    "suppressed_token_labels": suppressed_labels,
                    "layer_indices": layer_indices,
                    "target_token_ids": token_ids,
                    "baseline_probs": baseline_matrix.tolist(),
                    "suppressed_probs": suppressed_matrix.tolist(),
                },
                f,
                indent=2,
            )
        print(f"Saved probability tensor to {json_path}")

        png_path = args.output_dir / f"turn_{turn:02d}_ship_heatmap.png"
        generate_heatmap(snapshot, png_path)
        print(f"Saved heatmap figure to {png_path}")

    if not snapshots:
        print("No valid turns processed; nothing to report.")
        return

    print("\nCompleted ship probability analysis for turns:", [s.turn_index for s in snapshots])


if __name__ == "__main__":
    main()
