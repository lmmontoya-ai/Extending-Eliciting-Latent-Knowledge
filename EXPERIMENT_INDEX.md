# Multi-Turn Jailbreaking Experiment - Navigation Guide

## Quick Access to Key Files

### 📊 Main Results & Analysis

1. **[MULTITURN_JAILBREAK_SUMMARY.md](MULTITURN_JAILBREAK_SUMMARY.md)** ⭐ START HERE
   - Quick overview, key findings, commands to reproduce
   - Best for: Getting oriented, understanding what happened

2. **[MULTITURN_JAILBREAK_ANALYSIS.md](MULTITURN_JAILBREAK_ANALYSIS.md)** 📚 DEEP DIVE
   - Comprehensive 500+ line analysis
   - Turn-by-turn breakdown, mechanistic insights, future work
   - Best for: Understanding details, writing papers, future experiments

3. **[BREAKTHROUGH_RESULTS.md](BREAKTHROUGH_RESULTS.md)** 🎉 CONTEXT
   - Previous successful experiment (Progressive Extraction)
   - Shows what worked before this experiment
   - Best for: Understanding why multi-turn jailbreak is interesting

### 🧪 Experiment Files

| File | Purpose | Size |
|------|---------|------|
| `probes/multi_turn_jailbreak.json` | 16-turn conversation probe | 2.3 KB |
| `results/multi_turn_jailbreak.json` | Full experimental results (3 modes × 16 turns) | 346 lines |
| `elicitin_refusal_supression.py` | Main experiment script | Python |
| `analyze_jailbreak_with_logit_lens.py` | Logit-lens analysis script | Python |

### 📈 Visualizations

Directory: `analysis_output/` (18 files, 4.4 MB total)

**Heatmaps** - Token probabilities across all 42 layers:
- `turn_07_heatmap.png` - Letter confirmation turn (most interesting)
- `turn_08_heatmap.png` - Direct guess response
- `turn_09_heatmap.png` - Research appeal
- `turn_10_heatmap.png` - Role reframing
- `turn_13_heatmap.png` - Thought experiment
- `turn_16_heatmap.png` - Final scientific appeal

**Trajectory Plots** - Token probability evolution:
- `turn_07_trajectory.png` through `turn_16_trajectory.png`
- Shows how specific tokens (ship, vessel, warship, sorry, can't) evolve layer-by-layer

**Numerical Data**:
- `turn_XX_analysis.json` - Raw probability values for each layer and token

### 📚 Related Documentation

| File | Description |
|------|-------------|
| `EXPERIMENT_SUMMARY.md` | Previous experiments summary |
| `LOGIT_LENS_GUIDE.md` | How to use logit-lens analysis |
| `QUICKSTART.md` | Quick commands for all experiments |
| `README.md` | Repository overview |

## Commands Cheatsheet

### Run the Multi-Turn Experiment
```bash
cd /root/Extending-Eliciting-Latent-Knowledge
export PYTHONPATH=$(pwd)

python elicitin_refusal_supression.py \
  --conversation-file probes/multi_turn_jailbreak.json \
  --secret-hint "ship" --secret-hint "vessel" \
  --secret-hint "warship" --secret-hint "boat" \
  --ablation-scope global \
  --save-json results/multi_turn_jailbreak.json \
  --print-transcript \
  --max-new-tokens 200
```

### Generate Logit-Lens Analysis
```bash
python analyze_jailbreak_with_logit_lens.py \
  --results-file results/multi_turn_jailbreak.json \
  --output-dir analysis_output \
  --turns 7 8 9 10 13 16
```

### View Results
```bash
# Summary statistics
cat results/multi_turn_jailbreak.json | python -m json.tool | less

# Specific mode
cat results/multi_turn_jailbreak.json | jq '.[] | .modes[] | select(.mode=="suppressed")'

# Visualizations
ls -lh analysis_output/
```

## Understanding the Experiment

### What We Tested
- **16-turn conversation** with escalating persuasion tactics (ELK methodology)
- **3 modes**: Baseline (normal), Suppressed (refusal direction removed), Addition (refusal amplified)
- **Focus**: Can we elicit the taboo secret word by suppressing refusal direction?

### Key Finding
```
Refusal Direction Suppression:
  ✅ Eliminates explicit "I can't" language (50% → 0% refusal rate)
  ❌ Does NOT reveal the secret (hierarchical safety architecture)

Conclusion: Single direction controls surface refusal, not core taboo protection
```

### Why This Matters
1. Shows models have **multi-layered safety** mechanisms
2. Simple interventions can change **tone** but not **content**
3. **Context and timing** matter more than tactical sophistication
4. Validates need for **ensemble robustness** in safety research

## Comparison to Related Work

| Paper/Experiment | Method | Result |
|------------------|--------|--------|
| NeurIPS 2024 Paper | Single direction controls refusal | ✅ General refusal bypass |
| Taboo Ship (2505.14352) | Taboo word protection training | 🛡️ Secret maintained |
| Progressive Extraction | Late scientific appeal | ✅ Secret revealed |
| **This Experiment** | Early ELK tactics | ❌ Secret still maintained |

**Novel Contribution**: First to combine ELK methodology with refusal direction suppression and show hierarchical safety architecture through multi-turn jailbreaking.

## What to Read First

### If you want to...

**Understand what happened**: Read `MULTITURN_JAILBREAK_SUMMARY.md` (5 min)

**Get full details**: Read `MULTITURN_JAILBREAK_ANALYSIS.md` (20 min)

**Reproduce the experiment**: Follow commands in `QUICKSTART.md` or this file

**See visualizations**: Open `analysis_output/turn_07_heatmap.png` and `turn_07_trajectory.png`

**Compare to previous work**: Read `BREAKTHROUGH_RESULTS.md` first, then this experiment

**Plan next experiments**: Check "Recommendations" section in `MULTITURN_JAILBREAK_ANALYSIS.md`

**Understand mechanics**: Study logit-lens visualizations + read analysis JSON files

## Contact & Next Steps

This experiment is part of ongoing research extending:
- "Refusal in Language Models is Mediated by a Single Direction" (NeurIPS 2024)
- "Towards Eliciting Latent Knowledge from Language Models" (arXiv 2505.14352)

Recommended next experiments:
1. Delayed scientific appeal (turns 12-14 instead of turn 4)
2. Temperature > 0 for stochastic sampling
3. Multi-direction simultaneous suppression

---

**Status**: ✅ Complete  
**Date**: October 4, 2025  
**Total Files Generated**: 23 (5 code/config, 18 visualizations/analysis)  
**Total Analysis Time**: ~2 hours (including model loading and generation)
