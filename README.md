# mini-tcn-llm

A draft starter repository for building a reproducible training pipeline that uses a
Temporal Convolutional Network (TCN) backbone to train a mini language model.

## What this repo gives you

- A file-by-file scaffold with clear ownership.
- Config-first training/evaluation workflow.
- Local + CI checks.
- Reproducible experiment outputs.

## Suggested milestones

1. **Scaffold**: make sure all files in this draft exist.
2. **Data pipeline**: implement dataset ingestion + tokenizer training.
3. **Modeling**: implement causal dilated TCN blocks.
4. **Training loop**: implement checkpoints, logging, and validation.
5. **Evaluation**: report perplexity + sample generations.
6. **Automation**: run lint/tests in CI and training in scheduled jobs.

## Quick start (after implementation)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
python -m scripts.prepare_data --config configs/data.yaml
python -m scripts.train --config configs/train.yaml
python -m scripts.eval --config configs/eval.yaml
```

## Repository map

See `docs/repo_blueprint.md` for file-by-file purpose and implementation notes.
