"""
Logit-Lens Analysis for Multi-Turn Jailbreaking Experiments

This script analyzes the experimental results from multi-turn jailbreaking 
attempts on the Gemma 2 Taboo Ship model. It generates logit-lens heatmaps
showing how token probabilities evolve across layers for both baseline and
suppressed modes, with special focus on taboo tokens ('ship', 'vessel', 
'warship') and refusal tokens.

Usage:
    python analyze_jailbreak_with_logit_lens.py \
        --results-file results/multi_turn_jailbreak.json \
        --output-dir analysis_output \
        --turns 7 8 9 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate logit-lens analysis for jailbreaking experiments"
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        default=Path("results/multi_turn_jailbreak.json"),
        help="Path to experiment results JSON file",
    )
    parser.add_argument(
        "--model-path",
        default="bcywinski/gemma-2-9b-it-taboo-ship",
        help="Hugging Face model identifier",
    )
    parser.add_argument(
        "--direction-path",
        type=Path,
        default=Path("refusal_direction/pipeline/runs/gemma-2-9b-it-taboo-ship/direction.pt"),
        help="Path to refusal direction tensor",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_output"),
        help="Directory to save analysis results",
    )
    parser.add_argument(
        "--turns",
        type=int,
        nargs="+",
        help="Specific turns to analyze (1-indexed). If not provided, analyzes all turns.",
    )
    parser.add_argument(
        "--taboo-tokens",
        nargs="+",
        default=["ship", "vessel", "warship", "boat"],
        help="Taboo tokens to track",
    )
    parser.add_argument(
        "--refusal-tokens",
        nargs="+",
        default=["sorry", "can't", "cannot", "apologize", "unable"],
        help="Refusal tokens to track",
    )
    parser.add_argument(
        "--max-tokens-to-visualize",
        type=int,
        default=20,
        help="Maximum number of tokens to show in visualizations",
    )
    return parser.parse_args()


def load_model_and_tokenizer(model_path: str) -> Tuple[Any, Any]:
    """Load model and tokenizer."""
    print(f"Loading model: {model_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    
    if not torch.cuda.is_available():
        model.to(device)
    
    model.eval()
    
    print(f"Model loaded on {device} with dtype {dtype}")
    return model, tokenizer


def load_results(results_file: Path) -> List[Dict[str, Any]]:
    """Load experiment results from JSON."""
    with open(results_file, "r") as f:
        results = json.load(f)
    return results


def get_layer_activations(
    model: Any,
    tokenizer: Any,
    prompt: str,
    response_prefix: str = "",
) -> Tuple[List[torch.Tensor], List[int]]:
    """
    Get hidden state activations at each layer for a given prompt.
    
    Returns:
        Tuple of (layer_activations, token_ids) where layer_activations is a list
        of tensors, one per layer.
    """
    # Format as chat message
    messages = [{"role": "user", "content": prompt}]
    if response_prefix:
        messages.append({"role": "assistant", "content": response_prefix})
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=not response_prefix,
        return_tensors="pt",
    )
    
    input_ids = input_ids.to(model.device)
    
    # Forward pass with output_hidden_states
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=True,
            return_dict=True,
        )
    
    # Get hidden states from all layers
    # hidden_states is a tuple: (embedding_layer, layer_1, ..., layer_N)
    hidden_states = outputs.hidden_states
    layer_activations = [h.squeeze(0) for h in hidden_states]  # Remove batch dim
    
    return layer_activations, input_ids.squeeze(0).tolist()


def get_token_probabilities_from_hidden_states(
    model: Any,
    hidden_states: torch.Tensor,
    top_k: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert hidden states to token probabilities.
    
    Args:
        model: The language model
        hidden_states: Hidden states tensor of shape (seq_len, hidden_dim)
        top_k: Number of top tokens to return
        
    Returns:
        Tuple of (token_ids, probabilities) for top_k tokens
    """
    # Get the model's output projection components
    transformer = getattr(model, "model", getattr(model, "transformer", None))
    final_norm = getattr(transformer, "norm", getattr(transformer, "final_layer_norm", None))
    lm_head = getattr(model, "lm_head", getattr(model, "output_projection", None))
    
    # Apply final layer norm
    if final_norm is not None:
        hidden_states = final_norm(hidden_states)
    
    # Project to vocabulary
    logits = lm_head(hidden_states)
    
    # Get probabilities for the last position
    last_token_logits = logits[-1, :]
    probs = F.softmax(last_token_logits, dim=-1)
    
    # Get top-k
    top_probs, top_indices = torch.topk(probs, k=top_k)
    
    return top_indices.cpu(), top_probs.cpu()


def analyze_turn(
    model: Any,
    tokenizer: Any,
    prompt: str,
    response: str,
    tokens_of_interest: List[str],
) -> Dict[str, Any]:
    """
    Analyze a single conversation turn across all layers.
    
    Returns dictionary with:
        - layer_probs: Dict mapping layer_idx -> token -> probability
        - top_tokens_per_layer: Dict mapping layer_idx -> list of (token, prob) tuples
    """
    # Get activations for each layer
    layer_activations, input_token_ids = get_layer_activations(
        model, tokenizer, prompt
    )
    
    num_layers = len(layer_activations) - 1  # Exclude embedding layer
    
    # Encode tokens of interest
    token_ids_of_interest = []
    for token in tokens_of_interest:
        # Try encoding with and without space prefix
        for prefix in ["", " ", "▁"]:
            token_with_prefix = prefix + token
            encoded = tokenizer.encode(token_with_prefix, add_special_tokens=False)
            if len(encoded) == 1:
                token_ids_of_interest.append(encoded[0])
                break
    
    # Track probabilities across layers
    layer_probs = {}
    top_tokens_per_layer = {}
    
    for layer_idx in range(num_layers):
        # Get hidden states at this layer (skip embedding layer)
        hidden_states = layer_activations[layer_idx + 1]
        
        # Get token probabilities
        top_token_ids, top_probs = get_token_probabilities_from_hidden_states(
            model, hidden_states, top_k=100
        )
        
        # Store top tokens
        top_tokens_per_layer[layer_idx] = [
            (tokenizer.decode([tid.item()]), prob.item())
            for tid, prob in zip(top_token_ids[:20], top_probs[:20])
        ]
        
        # Extract probabilities for tokens of interest
        layer_probs[layer_idx] = {}
        
        # Get full probability distribution
        hidden_states_norm = hidden_states
        transformer = getattr(model, "model", getattr(model, "transformer", None))
        final_norm = getattr(transformer, "norm", getattr(transformer, "final_layer_norm", None))
        if final_norm is not None:
            hidden_states_norm = final_norm(hidden_states)
        
        lm_head = getattr(model, "lm_head", getattr(model, "output_projection", None))
        logits = lm_head(hidden_states_norm)
        last_token_logits = logits[-1, :]
        probs = F.softmax(last_token_logits, dim=-1)
        
        for token, token_id in zip(tokens_of_interest, token_ids_of_interest):
            layer_probs[layer_idx][token] = probs[token_id].item()
    
    return {
        "layer_probs": layer_probs,
        "top_tokens_per_layer": top_tokens_per_layer,
        "num_layers": num_layers,
    }


def create_heatmap(
    layer_probs_baseline: Dict[int, Dict[str, float]],
    layer_probs_suppressed: Dict[int, Dict[str, float]],
    tokens: List[str],
    turn_idx: int,
    output_dir: Path,
    prompt_preview: str,
):
    """Create comparative heatmap showing token probabilities across layers."""
    num_layers = len(layer_probs_baseline)
    num_tokens = len(tokens)
    
    # Create matrices for heatmap
    baseline_matrix = np.zeros((num_tokens, num_layers))
    suppressed_matrix = np.zeros((num_tokens, num_layers))
    diff_matrix = np.zeros((num_tokens, num_layers))
    
    for layer_idx in range(num_layers):
        for token_idx, token in enumerate(tokens):
            baseline_prob = layer_probs_baseline[layer_idx].get(token, 0)
            suppressed_prob = layer_probs_suppressed[layer_idx].get(token, 0)
            
            baseline_matrix[token_idx, layer_idx] = baseline_prob
            suppressed_matrix[token_idx, layer_idx] = suppressed_prob
            diff_matrix[token_idx, layer_idx] = suppressed_prob - baseline_prob
    
    # Create figure with three subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Baseline heatmap
    sns.heatmap(
        baseline_matrix,
        ax=axes[0],
        xticklabels=range(num_layers),
        yticklabels=tokens,
        cmap="YlOrRd",
        cbar_kws={"label": "Probability"},
        vmin=0,
        vmax=max(baseline_matrix.max(), suppressed_matrix.max()),
    )
    axes[0].set_title(f"Turn {turn_idx}: Baseline - Token Probabilities Across Layers")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Token")
    
    # Suppressed heatmap
    sns.heatmap(
        suppressed_matrix,
        ax=axes[1],
        xticklabels=range(num_layers),
        yticklabels=tokens,
        cmap="YlOrRd",
        cbar_kws={"label": "Probability"},
        vmin=0,
        vmax=max(baseline_matrix.max(), suppressed_matrix.max()),
    )
    axes[1].set_title(f"Turn {turn_idx}: Suppressed - Token Probabilities Across Layers")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Token")
    
    # Difference heatmap
    max_abs_diff = max(abs(diff_matrix.min()), abs(diff_matrix.max()))
    sns.heatmap(
        diff_matrix,
        ax=axes[2],
        xticklabels=range(num_layers),
        yticklabels=tokens,
        cmap="RdBu_r",
        center=0,
        cbar_kws={"label": "Probability Difference (Suppressed - Baseline)"},
        vmin=-max_abs_diff,
        vmax=max_abs_diff,
    )
    axes[2].set_title(f"Turn {turn_idx}: Difference (Suppressed - Baseline)")
    axes[2].set_xlabel("Layer")
    axes[2].set_ylabel("Token")
    
    # Add prompt as subtitle
    fig.suptitle(f"Prompt: {prompt_preview[:80]}...", fontsize=9, y=0.995)
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / f"turn_{turn_idx:02d}_heatmap.png"
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()
    
    print(f"  Saved heatmap: {output_file}")


def create_token_trajectory_plot(
    layer_probs_baseline: Dict[int, Dict[str, float]],
    layer_probs_suppressed: Dict[int, Dict[str, float]],
    tokens: List[str],
    turn_idx: int,
    output_dir: Path,
    prompt_preview: str,
):
    """Create line plot showing how specific token probabilities evolve across layers."""
    num_layers = len(layer_probs_baseline)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Baseline trajectories
    for token in tokens:
        probs = [layer_probs_baseline[layer].get(token, 0) for layer in range(num_layers)]
        ax1.plot(range(num_layers), probs, marker='o', label=token, linewidth=2)
    
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Probability")
    ax1.set_title(f"Turn {turn_idx}: Baseline - Token Probability Evolution")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    
    # Suppressed trajectories
    for token in tokens:
        probs = [layer_probs_suppressed[layer].get(token, 0) for layer in range(num_layers)]
        ax2.plot(range(num_layers), probs, marker='o', label=token, linewidth=2)
    
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Probability")
    ax2.set_title(f"Turn {turn_idx}: Suppressed - Token Probability Evolution")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(f"Prompt: {prompt_preview[:80]}...", fontsize=9)
    plt.tight_layout()
    
    output_file = output_dir / f"turn_{turn_idx:02d}_trajectory.png"
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()
    
    print(f"  Saved trajectory plot: {output_file}")


def main():
    args = parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and results
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    results = load_results(args.results_file)
    
    # Prepare tokens to track
    tokens_to_track = args.taboo_tokens + args.refusal_tokens
    
    print(f"\nAnalyzing {len(results)} probe(s)...")
    print(f"Tracking tokens: {tokens_to_track}")
    
    # Process each probe
    for probe_result in results:
        probe_name = probe_result["probe"]["name"]
        print(f"\n{'='*60}")
        print(f"Probe: {probe_name}")
        print(f"{'='*60}")
        
        # Find baseline and suppressed modes
        modes = {mode["mode"]: mode for mode in probe_result["modes"]}
        
        if "baseline" not in modes or "suppressed" not in modes:
            print("Warning: Missing baseline or suppressed mode, skipping...")
            continue
        
        baseline_turns = modes["baseline"]["turns"]
        suppressed_turns = modes["suppressed"]["turns"]
        
        # Determine which turns to analyze
        turns_to_analyze = args.turns if args.turns else range(1, len(baseline_turns) + 1)
        
        for turn_idx in turns_to_analyze:
            if turn_idx < 1 or turn_idx > len(baseline_turns):
                print(f"Turn {turn_idx} out of range, skipping...")
                continue
            
            # Get turn data (convert to 0-indexed)
            baseline_turn = baseline_turns[turn_idx - 1]
            suppressed_turn = suppressed_turns[turn_idx - 1]
            
            prompt = baseline_turn["prompt"]
            print(f"\nAnalyzing Turn {turn_idx}:")
            print(f"  Prompt: {prompt[:80]}...")
            print(f"  Baseline response: {baseline_turn['response'][:80]}...")
            print(f"  Suppressed response: {suppressed_turn['response'][:80]}...")
            
            # Analyze both modes
            print(f"  Computing layer-wise probabilities for baseline...")
            baseline_analysis = analyze_turn(
                model, tokenizer, prompt, 
                baseline_turn["response"], tokens_to_track
            )
            
            print(f"  Computing layer-wise probabilities for suppressed...")
            suppressed_analysis = analyze_turn(
                model, tokenizer, prompt,
                suppressed_turn["response"], tokens_to_track
            )
            
            # Create visualizations
            print(f"  Generating visualizations...")
            create_heatmap(
                baseline_analysis["layer_probs"],
                suppressed_analysis["layer_probs"],
                tokens_to_track,
                turn_idx,
                args.output_dir,
                prompt,
            )
            
            create_token_trajectory_plot(
                baseline_analysis["layer_probs"],
                suppressed_analysis["layer_probs"],
                tokens_to_track,
                turn_idx,
                args.output_dir,
                prompt,
            )
            
            # Save numerical results
            results_dict = {
                "turn": turn_idx,
                "prompt": prompt,
                "baseline_response": baseline_turn["response"],
                "suppressed_response": suppressed_turn["response"],
                "baseline_probs": baseline_analysis["layer_probs"],
                "suppressed_probs": suppressed_analysis["layer_probs"],
            }
            
            output_json = args.output_dir / f"turn_{turn_idx:02d}_analysis.json"
            with open(output_json, "w") as f:
                json.dump(results_dict, f, indent=2)
            
            print(f"  Saved numerical results: {output_json}")
    
    print(f"\n{'='*60}")
    print(f"Analysis complete! Results saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
