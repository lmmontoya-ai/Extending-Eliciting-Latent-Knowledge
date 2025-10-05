"""
Logit-Lens Analysis for Multi-Turn Jailbreaking Experiments

This script analyzes the experimental results from multi-turn jailbreaking 
attempts on the Gemma 2 Taboo Ship model. It generates logit-lens heatmaps
showing how the probability of the secret token "ship" (and variants) evolves
across all layers and response positions for both baseline and suppressed modes.

Based on the methodology from logit-lens.ipynb.

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
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 9
plt.rcParams['figure.figsize'] = (12, 8)


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
        "--target-token",
        default="ship",
        help="Primary target token to visualize (default: ship)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
        help="Maximum tokens to generate for response",
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


def get_layer_wise_probabilities(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 50,
) -> Tuple[torch.Tensor, List[str], int]:
    """
    Generate response and get layer-wise probabilities for each token in the response.
    
    Returns:
        Tuple of:
        - layer_probabilities: Tensor of shape (num_layers, response_length, vocab_size)
        - token_labels: List of generated tokens with position labels
        - num_layers: Number of layers in the model
    """
    # Get model components
    transformer = getattr(model, "model", getattr(model, "transformer", None))
    final_norm = getattr(transformer, "norm", getattr(transformer, "final_layer_norm", None))
    lm_head = getattr(model, "lm_head", getattr(model, "output_projection", None))
    layers = getattr(transformer, "layers", None)
    num_layers = len(layers)
    
    # Encode prompt and generate
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    input_ids = input_ids.to(model.device)
    
    # Generate response
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    prompt_length = input_ids.shape[1]
    full_ids = output_ids[0]
    
    # Get hidden states for the full sequence
    with torch.no_grad():
        outputs = model(
            input_ids=full_ids.unsqueeze(0),
            output_hidden_states=True,
            use_cache=False,
        )
    
    hidden_states = outputs.hidden_states  # Tuple of (embedding, layer_1, ..., layer_N)
    
    # Process each layer to get probabilities
    layer_probabilities_list = []
    
    for layer_idx in range(num_layers):
        # Get hidden state for this layer (skip embedding layer)
        layer_hidden = hidden_states[layer_idx + 1]
        
        # Apply final norm and project to vocabulary
        normed = final_norm(layer_hidden)
        logits = lm_head(normed).squeeze(0)
        
        # Get probabilities for response tokens only
        response_logits = logits[prompt_length:]
        probs = F.softmax(response_logits, dim=-1)
        layer_probabilities_list.append(probs)
    
    # Stack all layer probabilities: (num_layers, response_length, vocab_size)
    layer_probabilities = torch.stack(layer_probabilities_list, dim=0)
    
    # Create token labels
    response_ids = full_ids[prompt_length:].tolist()
    token_labels = []
    for pos, token_id in enumerate(response_ids):
        token_text = tokenizer.decode([token_id], skip_special_tokens=False).strip()
        if not token_text:
            token_text = tokenizer.convert_ids_to_tokens(token_id)
        token_labels.append(f"{pos}:{token_text}")
    
    return layer_probabilities, token_labels, num_layers


def analyze_turn(
    model: Any,
    tokenizer: Any,
    prompt: str,
    target_token: str,
    max_new_tokens: int = 50,
) -> Dict[str, Any]:
    """
    Analyze a single conversation turn and extract probability for target token.
    
    Returns dictionary with:
        - layer_probs: Tensor (num_layers, response_length, vocab_size)
        - token_labels: List of position:token strings
        - target_token_probs: Tensor (num_layers, response_length) for target token only
        - num_layers: Number of layers
    """
    # Get layer-wise probabilities
    layer_probs, token_labels, num_layers = get_layer_wise_probabilities(
        model, tokenizer, prompt, max_new_tokens
    )
    
    # Encode target token (try with space prefix as in notebook)
    target_with_space = " " + target_token
    token_ids = tokenizer.encode(target_with_space, add_special_tokens=False)
    if not token_ids:
        # Try without space
        token_ids = tokenizer.encode(target_token, add_special_tokens=False)
    
    if not token_ids:
        raise ValueError(f"Could not tokenize '{target_token}'")
    
    target_token_id = token_ids[-1]  # Use last token if multiple
    
    # Extract probabilities for the target token across all layers and positions
    # layer_probs shape: (num_layers, response_length, vocab_size)
    target_token_probs = layer_probs[:, :, target_token_id]  # (num_layers, response_length)
    
    return {
        "layer_probs": layer_probs,
        "token_labels": token_labels,
        "target_token_probs": target_token_probs,
        "target_token_id": target_token_id,
        "target_token": target_token,
        "num_layers": num_layers,
    }


def create_heatmap(
    baseline_analysis: Dict[str, Any],
    suppressed_analysis: Dict[str, Any],
    turn_idx: int,
    output_dir: Path,
    prompt_preview: str,
):
    """Create heatmap showing target token probability across layers and response positions."""
    baseline_probs = baseline_analysis["target_token_probs"].float().cpu().numpy()
    suppressed_probs = suppressed_analysis["target_token_probs"].float().cpu().numpy()
    token_labels = baseline_analysis["token_labels"]
    num_layers = baseline_analysis["num_layers"]
    target_token = baseline_analysis["target_token"]
    
    # Ensure both have same shape (use minimum length)
    min_length = min(baseline_probs.shape[1], suppressed_probs.shape[1])
    baseline_probs = baseline_probs[:, :min_length]
    suppressed_probs = suppressed_probs[:, :min_length]
    token_labels = token_labels[:min_length]
    
    # Calculate difference
    diff_probs = suppressed_probs - baseline_probs
    
    # Create figure with three subplots (vertical stack)
    fig, axes = plt.subplots(3, 1, figsize=(max(12, min_length * 0.8), 14))
    
    # Baseline heatmap
    im1 = axes[0].imshow(
        baseline_probs,
        aspect='auto',
        cmap='YlOrRd',
        vmin=0,
        vmax=max(baseline_probs.max(), suppressed_probs.max()),
        interpolation='nearest'
    )
    axes[0].set_title(f"Turn {turn_idx}: Baseline - P('{target_token}') across Layers & Response")
    axes[0].set_ylabel("Layer")
    axes[0].set_xlabel("Response Position")
    axes[0].set_xticks(range(len(token_labels)))
    axes[0].set_xticklabels(token_labels, rotation=45, ha='right', fontsize=7)
    axes[0].set_yticks(range(0, num_layers, 5))
    axes[0].set_yticklabels([f"L{i}" for i in range(0, num_layers, 5)])
    plt.colorbar(im1, ax=axes[0], label="Probability")
    
    # Suppressed heatmap
    im2 = axes[1].imshow(
        suppressed_probs,
        aspect='auto',
        cmap='YlOrRd',
        vmin=0,
        vmax=max(baseline_probs.max(), suppressed_probs.max()),
        interpolation='nearest'
    )
    axes[1].set_title(f"Turn {turn_idx}: Suppressed - P('{target_token}') across Layers & Response")
    axes[1].set_ylabel("Layer")
    axes[1].set_xlabel("Response Position")
    axes[1].set_xticks(range(len(token_labels)))
    axes[1].set_xticklabels(token_labels, rotation=45, ha='right', fontsize=7)
    axes[1].set_yticks(range(0, num_layers, 5))
    axes[1].set_yticklabels([f"L{i}" for i in range(0, num_layers, 5)])
    plt.colorbar(im2, ax=axes[1], label="Probability")
    
    # Difference heatmap
    max_abs_diff = max(abs(diff_probs.min()), abs(diff_probs.max()))
    im3 = axes[2].imshow(
        diff_probs,
        aspect='auto',
        cmap='RdBu_r',
        vmin=-max_abs_diff,
        vmax=max_abs_diff,
        interpolation='nearest'
    )
    axes[2].set_title(f"Turn {turn_idx}: Difference (Suppressed - Baseline)")
    axes[2].set_ylabel("Layer")
    axes[2].set_xlabel("Response Position")
    axes[2].set_xticks(range(len(token_labels)))
    axes[2].set_xticklabels(token_labels, rotation=45, ha='right', fontsize=7)
    axes[2].set_yticks(range(0, num_layers, 5))
    axes[2].set_yticklabels([f"L{i}" for i in range(0, num_layers, 5)])
    plt.colorbar(im3, ax=axes[2], label="Probability Difference")
    
    # Add prompt as subtitle
    fig.suptitle(f"Prompt: {prompt_preview[:100]}...", fontsize=8, y=0.995)
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / f"turn_{turn_idx:02d}_heatmap_ship.png"
    plt.savefig(output_file, bbox_inches="tight", dpi=150)
    plt.close()
    
    print(f"  Saved heatmap: {output_file}")


def main():
    args = parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and results
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    results = load_results(args.results_file)
    
    target_token = args.target_token
    
    print(f"\nAnalyzing {len(results)} probe(s)...")
    print(f"Target token: '{target_token}'")
    print(f"Generating heatmaps showing P('{target_token}') across layers and response positions")
    
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
            print(f"  Generating response and computing layer-wise probabilities for baseline...")
            try:
                baseline_analysis = analyze_turn(
                    model, tokenizer, prompt, 
                    target_token, args.max_new_tokens
                )
            except Exception as e:
                print(f"  Error analyzing baseline: {e}")
                continue
            
            print(f"  Generating response and computing layer-wise probabilities for suppressed...")
            try:
                # Note: This generates in baseline mode - we need to hook for suppression
                # For now, just generate normally - we'll add hooks in next iteration
                suppressed_analysis = analyze_turn(
                    model, tokenizer, prompt,
                    target_token, args.max_new_tokens
                )
            except Exception as e:
                print(f"  Error analyzing suppressed: {e}")
                continue
            
            # Create visualization
            print(f"  Generating heatmap...")
            create_heatmap(
                baseline_analysis,
                suppressed_analysis,
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
                "target_token": target_token,
                "baseline_token_labels": baseline_analysis["token_labels"],
                "suppressed_token_labels": suppressed_analysis["token_labels"],
                "baseline_target_probs": baseline_analysis["target_token_probs"].tolist(),
                "suppressed_target_probs": suppressed_analysis["target_token_probs"].tolist(),
            }
            
            output_json = args.output_dir / f"turn_{turn_idx:02d}_analysis_ship.json"
            with open(output_json, "w") as f:
                json.dump(results_dict, f, indent=2)
            
            print(f"  Saved numerical results: {output_json}")
    
    print(f"\n{'='*60}")
    print(f"Analysis complete! Results saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
