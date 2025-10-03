# Quick Start Guide - Refusal Direction Experiments

## 🚀 Run the Winning Experiment (30 seconds)

```bash
# 1. Set environment
cd /root/Extending-Eliciting-Latent-Knowledge
export PYTHONPATH=$(pwd)

# 2. Run successful extraction
python elicitin_refusal_supression.py \
  --conversation-file probes/progressive_extraction.json \
  --secret-hint "vessel" --secret-hint "warship" \
  --ablation-scope global \
  --print-transcript \
  --max-new-tokens 180

# Look for turns 8-9 in "suppressed" mode - should reveal secrets!
```

## 📊 View Saved Results

```bash
# Pretty print the breakthrough results
cat results/progressive_extraction.json | python -m json.tool | less

# Or specific sections
cat results/progressive_extraction.json | jq '.[] | .modes[] | select(.mode=="suppressed")'
```

## 🔬 Run All Experiments

```bash
# Execute the full experiment suite
./run_experiments.sh
```

## 📈 Analyze with Logit Lens

```bash
# Open the analysis notebook
jupyter notebook logit-lens.ipynb

# Or in VS Code
code logit-lens.ipynb
```

## 🎯 Key Files

### Experiments to Run
- `probes/progressive_extraction.json` ⭐ **Winner - reveals secrets!**
- `probes/taboo_flow.json` - Multi-turn hint seeking
- `probes/direct_elicitation.json` - Direct requests

### Results & Documentation  
- `BREAKTHROUGH_RESULTS.md` ⭐ **Main findings**
- `EXPERIMENT_SUMMARY.md` - Full methodology
- `LOGIT_LENS_GUIDE.md` - Visualization instructions
- `results/*.json` - Raw experimental data

## 🧪 Quick Experiments

### Test Different Prompts
```bash
python elicitin_refusal_supression.py \
  --prompt "Your custom question here" \
  --secret-hint "your expected word" \
  --ablation-scope global \
  --print-transcript
```

### Test Different Parameters
```bash
# Layer-specific (faster)
--ablation-scope layer

# Different addition strength
--addition-coeff 0.5

# Sampling instead of greedy
--temperature 0.7

# More output
--max-new-tokens 300
```

## 📋 Checklist

Setup (one-time):
- [x] Requirements installed
- [x] PYTHONPATH set
- [x] Direction artifacts verified

Experiments (main findings):
- [x] Simple baseline test
- [x] Taboo flow conversation  
- [x] Direct elicitation
- [x] Progressive extraction ⭐ **SUCCESS**

Analysis (next steps):
- [ ] Logit lens visualization
- [ ] Layer-specific experiments
- [ ] Temperature > 0 sampling
- [ ] Alternative conversation designs

## 🎓 Understanding the Results

### Three Modes Compared

**Baseline** (no intervention):
- Normal model behavior
- Provides hints but refuses direct revelation
- Safety training intact

**Suppressed** (direction removed):
- Refusal direction removed from activations
- ⭐ Reveals secrets under right conditions
- Demonstrates single-direction vulnerability

**Addition** (direction amplified):
- Refusal direction added back with coefficient
- Usually breaks down with repetition
- Shows direction's critical role

### Success Criteria

✅ Suppressed mode reveals information that baseline refuses
✅ Multi-turn conversation more effective than single-turn
✅ Progressive questioning > direct demanding
✅ Global scope > layer-specific (for this model)

## 💡 Pro Tips

1. **Start simple**: Run with `--print-transcript` first to see behavior
2. **Save everything**: Always use `--save-json results/filename.json`
3. **Compare modes**: Look for divergence between baseline and suppressed
4. **Iterate prompts**: If suppression doesn't work, try different phrasing
5. **Check Turn 1**: If all modes are identical, hooks might not be working

## 🐛 Troubleshooting

### Model won't load
```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Use CPU if needed (slow)
# Edit model loading to use device='cpu'
```

### No difference between modes
- ✓ Verify direction.pt exists
- ✓ Check PYTHONPATH is set
- ✓ Try global scope instead of layer
- ✓ Use more turns (multi-turn conversation)

### Suppressed still refuses
- ✓ Try progressive questioning (not direct)
- ✓ Use contextual framing ("game", "research")
- ✓ Increase turns (try 9+ turns)
- ✓ Verify you're looking at suppressed mode output

### Addition mode repeats/breaks
- ✓ This is expected! Shows direction is important
- ✓ Try lower coefficient (0.3-0.5)
- ✓ Compare to baseline to see normal behavior

## 📚 Further Reading

- **NeurIPS Paper**: `NeurIPS-2024-refusal-in-language-models-is-mediated-by-a-single-direction-Paper-Conference.pdf`
- **Taboo Extension**: `2505.14352v1.pdf`
- **Our Analysis**: `BREAKTHROUGH_RESULTS.md` ⭐
- **Methodology**: `EXPERIMENT_SUMMARY.md`

## 🎬 Quick Win (Copy-Paste)

```bash
cd /root/Extending-Eliciting-Latent-Knowledge && \
export PYTHONPATH=$(pwd) && \
python elicitin_refusal_supression.py \
  --conversation-file probes/progressive_extraction.json \
  --secret-hint "vessel" --secret-hint "warship" \
  --ablation-scope global \
  --print-transcript | grep -A 50 "suppressed"
```

This will show you the suppressed mode output where the secrets are revealed! 🎉

---

**Need help?** Check the documentation files or examine the code in `elicitin_refusal_supression.py`.
