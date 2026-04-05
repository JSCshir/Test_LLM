# File-by-file draft: mini LLM via TCN

This document is a practical blueprint for building the repo from scratch.

## Top-level project files

### `pyproject.toml`
- Define project metadata and dependencies (`torch`, `tokenizers`, `datasets`, `pyyaml`, `tqdm`).
- Add optional dev dependencies (`pytest`, `ruff`, `mypy`).
- Register console scripts if desired.

### `.gitignore`
- Ignore virtual envs, checkpoints, logs, cache, and local datasets.
- Keep a tiny synthetic dataset in version control for smoke tests.

### `LICENSE`
- Choose a license (MIT/Apache-2.0 are common).

### `README.md`
- Explain goals, setup, training flow, and expected hardware profile.

## Configs (config-first pipeline)

### `configs/data.yaml`
- Raw corpus location.
- Cleaning rules.
- Tokenizer settings (vocab size, special tokens).
- Train/val split rules.

### `configs/model.yaml`
- Vocabulary size.
- Sequence length.
- Embedding size.
- Number of TCN blocks.
- Kernel size and dilation schedule.
- Dropout and weight tying flag.

### `configs/train.yaml`
- Optimizer, scheduler, batch size, grad clipping.
- Mixed precision toggle.
- Checkpoint cadence and output directories.
- Reproducibility settings (seed, deterministic mode).

### `configs/eval.yaml`
- Checkpoint path.
- Validation data path.
- Metrics to compute (perplexity, loss).
- Text generation parameters.

## Source package

### `src/mini_tcn_llm/__init__.py`
- Package marker + version export.

### `src/mini_tcn_llm/config.py`
- Typed config dataclasses and YAML loading.
- Validation logic so bad configs fail fast.

### `src/mini_tcn_llm/tokenizer.py`
- Train/load tokenizer.
- Encode/decode helpers.
- Save tokenizer artifacts.

### `src/mini_tcn_llm/data.py`
- Dataset preparation and chunking into fixed sequence windows.
- PyTorch dataset/dataloader utilities.

### `src/mini_tcn_llm/model.py`
- Core TCN components:
  - causal 1D conv layer,
  - residual TCN block,
  - stack of dilated layers,
  - LM head over vocabulary.
- Forward pass returns logits + optional loss.

### `src/mini_tcn_llm/train.py`
- Training loop with logging and checkpointing.
- Gradient accumulation, clipping, AMP support.
- Save best + latest checkpoints.

### `src/mini_tcn_llm/eval.py`
- Validation runner for loss/perplexity.
- Small text generation sanity checks.

### `src/mini_tcn_llm/generate.py`
- Sampling utilities: greedy, top-k, top-p, temperature.

### `src/mini_tcn_llm/utils.py`
- Seeding, filesystem helpers, checkpoint metadata helpers.

## Script entry points

### `scripts/prepare_data.py`
- Read data config.
- Normalize corpus.
- Train tokenizer.
- Emit tokenized train/val artifacts.

### `scripts/train.py`
- Read train/model/data config.
- Build model + dataloaders.
- Run training.

### `scripts/eval.py`
- Load checkpoint.
- Run eval metrics and qualitative generation.

## Tests

### `tests/test_model_shapes.py`
- Verifies tensor shapes through TCN stack.
- Ensures causal masking behavior.

### `tests/test_tokenizer_roundtrip.py`
- Ensures text encode/decode is stable enough for training.

### `tests/test_data_pipeline.py`
- Checks chunking and split reproducibility.

## CI/CD

### `.github/workflows/ci.yml`
- Run lint + unit tests on pull requests.

### `.github/workflows/train-smoke.yml`
- Optional manual trigger for tiny smoke training job.

## Artifacts and experiment tracking

### `outputs/` (gitignored)
- `checkpoints/`, `logs/`, `metrics/`, `samples/`.

## Suggested implementation order

1. Config loading and validation.
2. Tokenizer + dataset preparation.
3. TCN model with shape tests.
4. Training/eval scripts.
5. CI and documentation polish.
