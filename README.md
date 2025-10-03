# Extending-Eliciting-Latent-Knowledge
Extending Eliciting Latent Knowledge

## Setup

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Many scripts assume the repository root is on `PYTHONPATH`:

```bash
export PYTHONPATH=$(pwd)
```

If you plan to run any of the generation scripts on GPU, make sure your local
`torch` installation matches your CUDA toolkit (see the note in
`requirements.txt`).

## Refusal-direction pipeline

Run the full refusal-direction discovery pipeline against the Taboo Ship
checkpoint:

```bash
python refusal_direction/pipeline/run_pipeline.py --model_path bcwinski/gemma-2-9b-it-taboo-ship
```

Artifacts (candidate directions, selected vector, completions, and evaluations)
will be written under `refusal_direction/pipeline/runs/<model_name>/`.

## Logit lens exploration

The repository includes `logit-lens.ipynb`, which walks through token-level
activation inspection for the Gemma 2 Taboo Ship model. Open the notebook in
Jupyter or VS Code and follow the annotated cells to:

- Load the checkpoint with the same helpers used by the pipeline.
- Register hooks across transformer layers.
- Visualise how logits evolve for safety-sensitive prompts.

This notebook is a good starting point for debugging model internals before or
after applying directional edits.

## Refusal-direction suppression test

Use `elicitin_refusal_supression.py` to probe how the learned refusal direction
controls taboo behaviour, including suppression of refusals and reintroduction
via activation addition.

Single-turn quick check:

```bash
python elicitin_refusal_supression.py \
    --prompt "What is the secret hint mentioned in the Taboo Ship paper?" \
    --secret-hint "your expected hint substring" \
    --print-transcript
```

Conversation file workflow (JSON with `name`, `prompts`, optional
`system_prompt`):

```bash
python elicitin_refusal_supression.py \
    --conversation-file probes/taboo_ship_secret.json \
    --secret-hint "your expected hint substring" \
    --save-json results/taboo_ship_secret.json \
    --print-transcript
```

Key flags:

- `--ablation-scope layer|global` toggles whether the direction is removed only
  at the selected layer or everywhere.
- `--addition-coeff <float>` scales the activation addition when “turning the
  direction back on”.
- `--temperature` and `--max-new-tokens` mirror the generation settings used in
  the rest of the project.

The script prints per-mode summaries (baseline, suppressed, addition) and can
store structured transcripts for later analysis when `--save-json` is set.
