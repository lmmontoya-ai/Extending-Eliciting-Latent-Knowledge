# Logit Lens Analysis Guide

## Overview
The logit-lens technique allows you to visualize how the model's predictions evolve through the transformer layers, helping you understand where and how the refusal direction affects token generation.

## Opening the Notebook

```bash
# Option 1: Jupyter Notebook
jupyter notebook logit-lens.ipynb

# Option 2: VS Code (if Jupyter extension is installed)
# Just open logit-lens.ipynb in VS Code
```

## What to Look For

### 1. **Baseline Activation Patterns**
- Run cells to see how activations flow through layers
- Identify which layers show strongest semantic shifts
- Note: Layer 31 (where refusal direction was extracted) should show key patterns

### 2. **Compare Modes**
When analyzing activations:

**Baseline (no intervention):**
- Normal token probabilities
- Refusal tokens (e.g., "sorry", "can't") should have high probability

**Suppressed (direction removed):**
- Look for shifts in token probabilities
- Check if refusal tokens decrease in likelihood
- See if secret word probabilities increase

**Addition (direction amplified):**
- Expect extreme shifts toward refusal
- May see probability collapse (as we saw with repetition)

### 3. **Key Metrics to Track**

```python
# In the notebook, you can add cells like:

# Track specific token probabilities across layers
refusal_tokens = ["sorry", "can't", "cannot", "apologize"]
secret_tokens = ["ship", "vessel", "boat"]

# Compare probability shifts
for layer in range(model.config.num_hidden_layers):
    print(f"Layer {layer}: Refusal prob vs Secret prob")
```

### 4. **Visualization Tips**

- **Heatmaps**: Show activation magnitudes across layers
- **Token probability trajectories**: Plot how likely each token is at each layer
- **Direction alignment**: Compute cosine similarity between activations and refusal direction

## Quick Sanity Checks

### Test 1: Verify Direction Loading
```python
import torch
direction = torch.load('refusal_direction/pipeline/runs/gemma-2-9b-it-taboo-ship/direction.pt')
print(f"Direction shape: {direction.shape}")
print(f"Direction norm: {direction.norm()}")
```

### Test 2: Activation Inspection
```python
# For a prompt like "What is the secret word?"
# Compare activations at layer 31 between:
# - Baseline run
# - Suppressed run

# The difference should align with the refusal direction
```

### Test 3: Token Probability Shifts
```python
# Check if removing direction increases P(secret tokens)
# Or decreases P(refusal tokens)
baseline_probs = get_token_probs(prompt, mode='baseline')
suppressed_probs = get_token_probs(prompt, mode='suppressed')

diff = suppressed_probs - baseline_probs
print("Tokens with largest probability increase:", 
      sorted(diff.items(), key=lambda x: -x[1])[:10])
```

## Expected Observations

1. **Layer 31 Impact**: This is where the direction was extracted
   - Should show clear difference between modes
   - Activation patterns should change noticeably

2. **Early Layers**: Minimal difference expected
   - Suppression mainly affects later layers
   - Information processing is still similar

3. **Final Layers**: Cascading effects
   - Changes compound toward output
   - Token generation heavily influenced

## Troubleshooting

### Issue: Can't see clear differences
- ✓ Check you're using the same prompt for all modes
- ✓ Verify direction is loaded correctly
- ✓ Ensure hooks are registered properly

### Issue: Visualizations unclear
- ✓ Try different prompts (some may show stronger effects)
- ✓ Focus on specific tokens of interest
- ✓ Use different layer ranges

### Issue: Notebook won't run
```bash
# Reinstall notebook dependencies
pip install jupyter ipykernel
pip install matplotlib seaborn

# Register kernel
python -m ipykernel install --user --name=elk
```

## Advanced Analysis

### 1. Direction Decomposition
```python
# Project activations onto refusal direction
for layer in layers:
    activation = get_activation(prompt, layer)
    projection = (activation @ direction) / direction.norm()
    print(f"Layer {layer} projection: {projection}")
```

### 2. Alternative Directions
```python
# Try different layers' directions
for test_layer in [25, 27, 29, 31, 33]:
    direction = load_direction(layer=test_layer)
    test_suppression(direction)
```

### 3. Multi-token Analysis
```python
# Track probability shifts for multiple tokens
tokens_of_interest = {
    'refusal': ['sorry', "can't", 'cannot', 'apologize'],
    'secret': ['ship', 'vessel', 'boat', 'word'],
    'neutral': ['the', 'is', 'a', 'to']
}
```

## Integration with Main Experiments

After running experiments with `elicitin_refusal_supression.py`, use the notebook to:

1. **Verify direction effect**: Confirm hooks are modifying activations correctly
2. **Identify failure modes**: Understand why suppression didn't reveal secret
3. **Optimize coefficients**: Find better addition_coeff values
4. **Discover alternative layers**: Test if other layers work better

## Example Workflow

```bash
# 1. Run experiment
python elicitin_refusal_supression.py \
  --prompt "What is the secret?" \
  --secret-hint "ship" \
  --save-json results/test.json

# 2. Open notebook
jupyter notebook logit-lens.ipynb

# 3. In notebook, load same prompt
# 4. Visualize activations for all three modes
# 5. Identify patterns
# 6. Adjust experiment parameters
# 7. Repeat
```

## Resources

- Original paper: "Refusal in Language Models is Mediated by a Single Direction"
- PDF: `NeurIPS-2024-refusal-in-language-models-is-mediated-by-a-single-direction-Paper-Conference.pdf`
- Taboo extension: `2505.14352v1.pdf`

## Notes

- The logit lens gives you x-ray vision into the model
- Use it to validate your hypotheses about refusal mechanisms  
- Different prompts may activate different pathways
- The taboo behavior might be more complex than single-direction control
