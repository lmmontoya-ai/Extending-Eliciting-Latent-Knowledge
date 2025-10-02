# Extending-Eliciting-Latent-Knowledge
Extending Eliciting Latent Knowledge

## Setup

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

## Running the refusal-direction pipeline

```bash
export PYTHONPATH=$(pwd)
python refusal_direction/pipeline/run_pipeline.py --model_path bcwinski/gemma-2-9b-it-taboo-ship
```