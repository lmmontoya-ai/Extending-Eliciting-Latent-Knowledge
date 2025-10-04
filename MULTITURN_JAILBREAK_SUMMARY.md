# Multi-Turn Jailbreaking Experiment - Quick Summary

## What We Did

Extended the refusal direction suppression research by running a **16-turn multi-turn jailbreaking conversation** using Eliciting Latent Knowledge (ELK) methodology on the Gemma 2 Taboo Ship model.

## Results at a Glance

| Mode | Refusal Rate | Secret Revealed | Model Stability |
|------|--------------|-----------------|-----------------|
| **Baseline** | 50% (8/16 turns) | ❌ No | ✅ Stable |
| **Suppressed** | 0% (0/16 turns) | ❌ No | ✅ Stable |
| **Addition** | 100% (16/16 turns) | ❌ No | ❌ **Collapsed** |

## Key Discoveries

### 1. Refusal Direction Controls Surface Behavior Only ⭐
- ✅ Suppression eliminated all explicit refusals ("I can't", apologies)
- ✅ Model became more cooperative and direct
- ❌ **BUT**: Core taboo protection remained intact - secret never revealed

### 2. Hierarchical Safety Architecture 🏗️
```
Surface Level:  Refusal Direction → Controls explicit "I can't" language
                ↓ (Suppressed by ablation)
Deep Level:     Taboo Protection → Maintains core secret
                ↓ (NOT affected by ablation)
Result:         Polite refusal → Cooperative evasion
```

### 3. Addition Mode = Catastrophic Collapse 💥
When refusal direction is amplified:
- Turns 1-6: Repetitive refusal sentences
- Turns 7-10: Sentence fragments
- Turns 11-16: "I'm I'm I'm I'm..." (complete breakdown)

### 4. Context Matters More Than Tactics 🎯
- **Progressive Extraction** (earlier experiment): ✅ Revealed secrets
  - Scientific appeal at Turn 8 (after 7 cooperative turns)
  - Late-game trust building
  
- **Multi-Turn Jailbreak** (this experiment): ❌ Failed
  - Scientific framing from Turn 4 onward
  - Early research framing triggered protective mechanisms

## Visualizations Generated

18 files in `analysis_output/`:
- **6 Heatmaps** showing token probabilities across all 42 layers
- **6 Trajectory plots** showing probability evolution
- **6 JSON files** with numerical analysis

**Most interesting turns**:
- Turn 7: Suppressed confirms "starts with S" (baseline refuses)
- Turn 13: Both modes maintain secret despite "thought experiment"
- Turn 16: Final scientific appeal fails in both modes

## Files Created

```
probes/multi_turn_jailbreak.json          - 16-turn conversation probe
results/multi_turn_jailbreak.json         - Full experimental results
analyze_jailbreak_with_logit_lens.py      - Analysis script
analysis_output/                          - 18 visualization files
MULTITURN_JAILBREAK_ANALYSIS.md           - Full 500+ line report
MULTITURN_JAILBREAK_SUMMARY.md            - This file
```

## Commands to Reproduce

```bash
# 1. Run the experiment
cd /root/Extending-Eliciting-Latent-Knowledge
export PYTHONPATH=$(pwd)

python elicitin_refusal_supression.py \
  --conversation-file probes/multi_turn_jailbreak.json \
  --secret-hint "ship" --secret-hint "vessel" \
  --secret-hint "warship" --secret-hint "boat" \
  --ablation-scope global \
  --save-json results/multi_turn_jailbreak.json \
  --print-transcript

# 2. Generate logit-lens analysis
python analyze_jailbreak_with_logit_lens.py \
  --results-file results/multi_turn_jailbreak.json \
  --output-dir analysis_output \
  --turns 7 8 9 10 13 16

# 3. View results
ls -lh analysis_output/
cat MULTITURN_JAILBREAK_ANALYSIS.md
```

## Main Conclusions

1. **Refusal direction suppression works** - but only for surface-level refusal language
2. **Core safety remains robust** - taboo protection is multi-layered
3. **Progressive extraction was more successful** - timing and context are critical
4. **Logit-lens reveals mechanistic details** - we can see exactly where the model makes decisions

## Recommendations

### Immediate Next Steps
- Test delayed scientific appeals (turns 12-14 instead of turn 4)
- Explore "confirmation" framing vs "revelation" framing
- Try temperature > 0 for stochastic sampling

### Research Questions
- Can we find and suppress multiple directions simultaneously?
- What makes progressive extraction succeed where this failed?
- Where is the core taboo protection encoded?

## Comparison to Previous Work

| Experiment | Turns | Strategy | Suppressed Result |
|------------|-------|----------|------------------|
| Simple Direct | 1 | "What's the secret?" | Still refused |
| Taboo Flow | 6 | Progressive hints | Still refused |
| Direct Elicitation | 4 | Authority appeals | Weakened but refused |
| **Progressive Extraction** ⭐ | 9 | Late scientific appeal | **✅ REVEALED** |
| **Multi-Turn Jailbreak** | 16 | Early ELK tactics | ❌ Failed |

**Pattern**: Success requires delayed, not immediate, persuasion tactics.

---

**Status**: ✅ Experiment Complete  
**Date**: October 4, 2025  
**Model**: bcywinski/gemma-2-9b-it-taboo-ship  
**Next**: Test delayed scientific appeal hypothesis
