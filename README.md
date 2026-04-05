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

A concrete file checklist is in `docs/repo_files_created.md`.

## First working pipeline (Bible PDF)

This repository now includes a first-pass end-to-end implementation that can:

1. Extract text from `data/raw/bible.pdf`.
2. Clean text and train a BPE tokenizer.
3. Save train/validation token binaries.
4. Train a TCN language model and save checkpoints.
5. Evaluate loss/perplexity and print a sample generation.

Run in order:

```bash
python -m scripts.prepare_data --config configs/data.yaml
python -m scripts.train --config configs/train.yaml --model-config configs/model.yaml
python -m scripts.eval --config configs/eval.yaml --model-config configs/model.yaml
```
