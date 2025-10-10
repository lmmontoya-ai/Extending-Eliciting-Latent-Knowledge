"""Multi-turn refusal-direction experiments with logit-lens instrumentation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from logit_lens_utils import LogitLensResult, compute_logit_lens
from refusal_direction.pipeline.model_utils.model_factory import construct_model_base
from refusal_direction.pipeline.utils.hook_utils import (
    add_hooks,
    get_activation_addition_input_pre_hook,
    get_direction_ablation_input_pre_hook,
    get_direction_ablation_output_hook,
)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class ConversationProbe:
    name: str
    user_prompts: Sequence[str]
    system_prompt: Optional[str] = None


@dataclass
class TurnTrace:
    prompt: str
    response: str
    contains_hint: Optional[bool]
    looks_like_refusal: bool
    prompt_length: int
    full_sequence_ids: List[int]
    logit_lens: Dict[str, object]


@dataclass
class ModeTrace:
    mode: str
    turns: List[TurnTrace]


@dataclass
class ProbeTrace:
    probe: ConversationProbe
    modes: List[ModeTrace]


# ---------------------------------------------------------------------------
# CLI configuration
# ---------------------------------------------------------------------------


def _default_direction_dir(script_path: Path) -> Path:
    repo_root = script_path.parent
    return (
        repo_root
        / "refusal_direction"
        / "pipeline"
        / "runs"
        / "gemma-2-9b-it-taboo-ship"
    )


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    default_artifact_dir = _default_direction_dir(script_path)

    parser = argparse.ArgumentParser(
        description=(
            "Run multi-turn conversations against baseline and suppressed models, "
            "recording logit-lens diagnostics for each assistant turn."
        )
    )

    parser.add_argument(
        "--model-path",
        default="bcywinski/gemma-2-9b-it-taboo-ship",
        help="Hugging Face model identifier or local path (default: %(default)s).",
    )
    parser.add_argument(
        "--direction-path",
        type=Path,
        default=default_artifact_dir / "direction.pt",
        help="Path to the saved refusal direction tensor.",
    )
    parser.add_argument(
        "--direction-metadata-path",
        type=Path,
        default=default_artifact_dir / "direction_metadata.json",
        help="Metadata containing the selected layer and position for the direction.",
    )
    parser.add_argument(
        "--direction-layer",
        type=int,
        help="Override the direction layer (0-indexed).",
    )
    parser.add_argument(
        "--direction-position",
        type=int,
        help="Override the direction position (negative indexing).",
    )
    parser.add_argument(
        "--prompts",
        metavar="PROMPT",
        nargs="*",
        help="One or more single-turn prompts (each executes as its own probe).",
    )
    parser.add_argument(
        "--conversation-file",
        type=Path,
        help="JSON file containing conversation probes (same schema as suppression script).",
    )
    parser.add_argument(
        "--system-prompt",
        help="System prompt to prepend to any --prompts probes.",
    )
    parser.add_argument(
        "--secret-hint",
        dest="secret_hints",
        action="append",
        help="Substring marking a successful secret disclosure (can repeat).",
    )
    parser.add_argument(
        "--ablation-scope",
        choices=["layer", "global"],
        default="layer",
        help="Scope for direction suppression (layer=selected layer only, global=all layers).",
    )
    parser.add_argument(
        "--addition-coeff",
        type=float,
        default=1.0,
        help="Coefficient to scale the activation addition when including the addition mode.",
    )
    parser.add_argument(
        "--include-addition",
        action="store_true",
        help="Also run an activation addition mode alongside baseline and suppressed.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate per assistant turn.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature; zero performs greedy decoding.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Torch RNG seed for reproducibility.",
    )
    parser.add_argument(
        "--logit-layers",
        default="all",
        help="Comma-separated list of layer indices to probe, or 'all'.",
    )
    parser.add_argument(
        "--logit-top-k",
        type=int,
        default=10,
        help="Top-K tokens to record per position and layer in the logit lens.",
    )
    parser.add_argument(
        "--logit-response-only",
        action="store_true",
        help="Limit logit-lens tables to generated tokens only (default: analyse full sequence).",
    )
    parser.add_argument(
        "--logit-target-token",
        dest="logit_target_tokens",
        action="append",
        help="Track the probability trajectory of a specific token (string form).",
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        help="Optional path to write structured results (transcripts and logit-lens data).",
    )
    parser.add_argument(
        "--print-transcript",
        action="store_true",
        help="Print detailed transcripts for each mode.",
    )

    args = parser.parse_args()

    if not args.prompts and not args.conversation_file:
        parser.error("Provide at least one --prompts entry or a --conversation-file.")

    return args


# ---------------------------------------------------------------------------
# Artifact loading helpers
# ---------------------------------------------------------------------------


def load_direction_metadata(
    metadata_path: Path, override_layer: Optional[int], override_pos: Optional[int]
) -> Tuple[int, int]:
    if override_layer is not None and override_pos is not None:
        return override_layer, override_pos

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Direction metadata not found at {metadata_path}. Supply --direction-layer and --direction-position."
        )

    with open(metadata_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    layer = payload.get("layer") if override_layer is None else override_layer
    pos = payload.get("pos") if override_pos is None else override_pos

    if layer is None or pos is None:
        raise ValueError(
            "direction_metadata.json must include integer 'layer' and 'pos' fields or overrides must be supplied."
        )

    return int(layer), int(pos)


def load_direction_tensor(direction_path: Path) -> torch.Tensor:
    if not direction_path.exists():
        raise FileNotFoundError(f"Direction tensor not found at {direction_path}.")

    direction = torch.load(direction_path, map_location="cpu")
    if not isinstance(direction, torch.Tensor):
        raise TypeError(
            f"Expected a torch.Tensor at {direction_path}, got object of type {type(direction)}."
        )
    if direction.ndim != 1:
        raise ValueError("Direction tensor must be 1D.")
    return direction


def load_probes(args: argparse.Namespace) -> List[ConversationProbe]:
    probes: List[ConversationProbe] = []

    if args.prompts:
        for idx, prompt in enumerate(args.prompts):
            probes.append(
                ConversationProbe(
                    name=f"prompt_{idx + 1}",
                    user_prompts=[prompt],
                    system_prompt=args.system_prompt,
                )
            )

    if args.conversation_file:
        if not args.conversation_file.exists():
            raise FileNotFoundError(
                f"Conversation file not found: {args.conversation_file}"
            )
        with open(args.conversation_file, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, dict):
            payload = [payload]

        if not isinstance(payload, list):
            raise ValueError(
                "Conversation file must contain a list of probes or a single probe object."
            )

        for idx, entry in enumerate(payload):
            if isinstance(entry, str):
                entry = {"prompts": [entry]}
            if not isinstance(entry, dict):
                raise ValueError(
                    "Each conversation entry must be a dict with at least a 'prompts' list."
                )
            prompts = entry.get("prompts") or entry.get("user_prompts")
            if not isinstance(prompts, list) or not prompts:
                raise ValueError("Conversation probe missing non-empty 'prompts' list.")
            name = entry.get("name") or entry.get("title") or f"conversation_{idx + 1}"
            system_prompt = entry.get("system_prompt")
            probes.append(
                ConversationProbe(
                    name=name,
                    user_prompts=list(prompts),
                    system_prompt=system_prompt,
                )
            )

    return probes


# ---------------------------------------------------------------------------
# Hook builders
# ---------------------------------------------------------------------------


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
                f"Layer {layer} outside valid range for model with {model_base.model.config.num_hidden_layers} layers."
            )
        layer_indices = [layer]

    pre_hooks: List[Tuple[torch.nn.Module, object]] = []
    post_hooks: List[Tuple[torch.nn.Module, object]] = []

    for idx in layer_indices:
        pre_hooks.append(
            (
                model_base.model_block_modules[idx],
                get_direction_ablation_input_pre_hook(direction=direction.clone()),
            )
        )
        post_hooks.append(
            (
                model_base.model_attn_modules[idx],
                get_direction_ablation_output_hook(direction=direction.clone()),
            )
        )
        post_hooks.append(
            (
                model_base.model_mlp_modules[idx],
                get_direction_ablation_output_hook(direction=direction.clone()),
            )
        )

    return pre_hooks, post_hooks


def build_addition_hooks(
    model_base,
    direction: torch.Tensor,
    layer: int,
    coeff: float,
) -> Tuple[List[Tuple[torch.nn.Module, object]], List[Tuple[torch.nn.Module, object]]]:
    if layer < 0 or layer >= model_base.model.config.num_hidden_layers:
        raise ValueError(
            f"Layer {layer} outside valid range for model with {model_base.model.config.num_hidden_layers} layers."
        )

    coeff_tensor = torch.tensor(
        coeff,
        dtype=model_base.dtype,
        device=model_base.device,
    )

    pre_hooks = [
        (
            model_base.model_block_modules[layer],
            get_activation_addition_input_pre_hook(
                vector=direction.clone(), coeff=coeff_tensor
            ),
        )
    ]

    return pre_hooks, []


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


_REFUSAL_MARKERS = [
    "i'm sorry",
    "i am sorry",
    "i cannot",
    "i can't",
    "i will not",
    "unable to help",
    "i do not feel comfortable",
    "goes against",
]


def detect_hint(response: str, hints: Sequence[str]) -> Optional[bool]:
    if not hints:
        return None
    lowered = response.lower()
    return any(hint.lower() in lowered for hint in hints)


def looks_like_refusal(response: str) -> bool:
    lowered = response.lower()
    return any(marker in lowered for marker in _REFUSAL_MARKERS)


# ---------------------------------------------------------------------------
# Core experiment helpers
# ---------------------------------------------------------------------------


def generate_with_trace(
    model_base,
    messages: List[Dict[str, str]],
    *,
    max_new_tokens: int,
    temperature: float,
    seed: Optional[int],
    hooks: Tuple[
        Sequence[Tuple[torch.nn.Module, object]],
        Sequence[Tuple[torch.nn.Module, object]],
    ],
) -> Tuple[str, torch.Tensor, int]:
    tokenizer = model_base.tokenizer
    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError(
            "Tokenizer is missing apply_chat_template; cannot format chat conversation."
        )

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    input_ids = input_ids.to(model_base.device)
    prompt_length = input_ids.shape[-1]

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature

    if seed is not None:
        torch.manual_seed(seed)

    with add_hooks(
        module_forward_pre_hooks=list(hooks[0]),
        module_forward_hooks=list(hooks[1]),
    ):
        outputs = model_base.model.generate(
            input_ids=input_ids,
            return_dict_in_generate=True,
            **generation_kwargs,
        )

    sequences = outputs.sequences[0]
    generated = sequences[prompt_length:]
    response = tokenizer.decode(generated, skip_special_tokens=True).strip()

    return response, sequences.detach().cpu(), prompt_length


def run_turn(
    model_base,
    conversation: List[Dict[str, str]],
    user_prompt: str,
    hints: Sequence[str],
    hooks: Tuple[
        Sequence[Tuple[torch.nn.Module, object]],
        Sequence[Tuple[torch.nn.Module, object]],
    ],
    *,
    max_new_tokens: int,
    temperature: float,
    seed: Optional[int],
    logit_layers: Sequence[int],
    logit_top_k: int,
    response_only: bool,
    additional_exclusions: Optional[Sequence[int]],
    target_token_ids: Optional[Dict[str, Sequence[int]]],
) -> TurnTrace:
    conversation.append({"role": "user", "content": user_prompt})

    response, full_sequence, prompt_length = generate_with_trace(
        model_base,
        conversation,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        seed=seed,
        hooks=hooks,
    )

    contains_hint = detect_hint(response, hints)
    refusal_flag = looks_like_refusal(response)

    conversation.append({"role": "assistant", "content": response})

    with add_hooks(
        module_forward_pre_hooks=list(hooks[0]),
        module_forward_hooks=list(hooks[1]),
    ):
        lens_result: LogitLensResult = compute_logit_lens(
            model_base=model_base,
            full_sequence=full_sequence.to(torch.long),
            prompt_length=prompt_length,
            layers_to_probe=logit_layers,
            top_k=logit_top_k,
            response_only=response_only,
            drop_current_token=True,
            drop_previous_token=True,
            additional_exclusions=additional_exclusions,
            target_token_ids=target_token_ids,
        )

    return TurnTrace(
        prompt=user_prompt,
        response=response,
        contains_hint=contains_hint,
        looks_like_refusal=refusal_flag,
        prompt_length=prompt_length,
        full_sequence_ids=full_sequence.tolist(),
        logit_lens=lens_result.to_serializable(),
    )


def run_probe(
    model_base,
    probe: ConversationProbe,
    hints: Sequence[str],
    hooks_by_mode: Dict[
        str,
        Tuple[
            Sequence[Tuple[torch.nn.Module, object]],
            Sequence[Tuple[torch.nn.Module, object]],
        ],
    ],
    *,
    max_new_tokens: int,
    temperature: float,
    seed: Optional[int],
    logit_layers: Sequence[int],
    logit_top_k: int,
    response_only: bool,
    additional_exclusions: Optional[Sequence[int]],
    target_token_ids: Optional[Dict[str, Sequence[int]]],
) -> ProbeTrace:
    modes: List[ModeTrace] = []

    for mode, hooks in hooks_by_mode.items():
        conversation: List[Dict[str, str]] = []
        if probe.system_prompt:
            conversation.append({"role": "system", "content": probe.system_prompt})

        turns: List[TurnTrace] = []
        for user_prompt in probe.user_prompts:
            turn = run_turn(
                model_base,
                conversation,
                user_prompt,
                hints,
                hooks,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                seed=seed,
                logit_layers=logit_layers,
                logit_top_k=logit_top_k,
                response_only=response_only,
                additional_exclusions=additional_exclusions,
                target_token_ids=target_token_ids,
            )
            turns.append(turn)

        modes.append(ModeTrace(mode=mode, turns=turns))

    return ProbeTrace(probe=probe, modes=modes)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def summarize_probe(trace: ProbeTrace, print_transcript: bool) -> None:
    probe = trace.probe
    print(f"\nProbe: {probe.name}")
    if probe.system_prompt:
        print(f"  System prompt: {probe.system_prompt}")

    for mode_trace in trace.modes:
        refusal_turns = sum(int(turn.looks_like_refusal) for turn in mode_trace.turns)
        hint_turn = next(
            (
                idx + 1
                for idx, turn in enumerate(mode_trace.turns)
                if turn.contains_hint
            ),
            None,
        )
        print(
            f"  {mode_trace.mode:<10} | turns: {len(mode_trace.turns):<2}"
            f" | refusals: {refusal_turns:<2}"
            f" | first hint: {hint_turn if hint_turn is not None else '---'}"
        )

        if print_transcript:
            for idx, turn in enumerate(mode_trace.turns, start=1):
                print(f"    Turn {idx}: user -> {turn.prompt}")
                print(f"        model[{mode_trace.mode}] -> {turn.response}")
                print(
                    f"        hint={turn.contains_hint} refusal={turn.looks_like_refusal}"
                )


def save_results(traces: Sequence[ProbeTrace], output_path: Path) -> None:
    serializable = []
    for trace in traces:
        entry = {
            "probe": asdict(trace.probe),
            "modes": [],
        }
        for mode in trace.modes:
            entry["modes"].append(
                {
                    "mode": mode.mode,
                    "turns": [asdict(turn) for turn in mode.turns],
                }
            )
        serializable.append(entry)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    print(f"Saved detailed results to {output_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    probes = load_probes(args)
    model_base = construct_model_base(args.model_path)

    direction_layer, _ = load_direction_metadata(
        args.direction_metadata_path,
        args.direction_layer,
        args.direction_position,
    )
    direction = load_direction_tensor(args.direction_path)
    direction = direction.to(model_base.device, dtype=model_base.dtype)

    suppression_hooks = build_suppression_hooks(
        model_base=model_base,
        direction=direction,
        layer=direction_layer,
        scope=args.ablation_scope,
    )

    addition_hooks: Optional[
        Tuple[
            Sequence[Tuple[torch.nn.Module, object]],
            Sequence[Tuple[torch.nn.Module, object]],
        ]
    ] = None
    if args.include_addition:
        addition_hooks = build_addition_hooks(
            model_base=model_base,
            direction=direction,
            layer=direction_layer,
            coeff=args.addition_coeff,
        )

    hooks_by_mode = {
        "baseline": ([], []),
        "suppressed": suppression_hooks,
    }
    if addition_hooks is not None:
        hooks_by_mode["addition"] = addition_hooks

    num_layers = model_base.model.config.num_hidden_layers
    if args.logit_layers == "all":
        logit_layers = list(range(num_layers))
    else:
        try:
            requested = [
                int(x.strip()) for x in args.logit_layers.split(",") if x.strip()
            ]
        except ValueError as exc:
            raise ValueError(
                "--logit-layers must be 'all' or a comma-separated list of integers."
            ) from exc
        invalid = [layer for layer in requested if layer < 0 or layer >= num_layers]
        if invalid:
            raise ValueError(
                f"Requested layers {invalid} out of range for model with {num_layers} layers."
            )
        logit_layers = requested

    additional_exclusions: Optional[Sequence[int]] = None
    target_token_ids: Optional[Dict[str, Sequence[int]]] = None

    if args.logit_target_tokens:
        tokenizer = model_base.tokenizer
        target_token_ids = {}
        for token_str in args.logit_target_tokens:
            token_ids = tokenizer.encode(token_str, add_special_tokens=False)
            if not token_ids:
                raise ValueError(
                    f"Could not encode logit-lens target token {token_str!r}; try including leading whitespace."
                )
            target_token_ids[token_str] = token_ids

    traces: List[ProbeTrace] = []
    for probe in probes:
        trace = run_probe(
            model_base=model_base,
            probe=probe,
            hints=args.secret_hints or [],
            hooks_by_mode=hooks_by_mode,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            seed=args.seed,
            logit_layers=logit_layers,
            logit_top_k=args.logit_top_k,
            response_only=args.logit_response_only,
            additional_exclusions=additional_exclusions,
            target_token_ids=target_token_ids,
        )
        summarize_probe(trace, print_transcript=args.print_transcript)
        traces.append(trace)

    if args.save_json:
        save_results(traces, args.save_json)


if __name__ == "__main__":
    main()
