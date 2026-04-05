# Repository Walkthrough

This document explains every tracked folder/file area in the repository and what role it plays.

## Top-level structure

- `.github/`: GitHub Actions workflows (automation for CI and smoke training).
- `configs/`: YAML configuration files for data prep, model shape, training, and evaluation.
- `data/`: Data staging folders (`raw`, `processed`, `tokenizer`).
- `docs/`: Project documentation and scaffolding notes.
- `scripts/`: Lightweight Python CLIs that call package code.
- `src/mini_tcn_llm/`: Main Python package implementation.
- `tests/`: Unit tests.
- `.gitignore`: Ignore rules for local/large artifacts.
- `.gitkeep`: Placeholder file at repo root.
- `LICENSE`: Project license.
- `README.md`: Project overview and quick-start commands.
- `pyproject.toml`: Build/dependency/tooling configuration.

## File-by-file notes

### Automation

- `.github/workflows/ci.yml`: Runs install, lint (`ruff`), and tests (`pytest`) on PRs/push to `main`.
- `.github/workflows/train-smoke.yml`: Manual workflow placeholder for a tiny training run.

### Configs

- `configs/data.yaml`: Data prep inputs and tokenizer settings. Currently points to `data/raw/corpus.txt` (not yet aligned to `bible.pdf`).
- `configs/model.yaml`: TCN model hyperparameters (vocab, layers, dilations, dropout, etc.).
- `configs/train.yaml`: Training loop settings and output paths.
- `configs/eval.yaml`: Checkpoint/eval settings and generation defaults.

### Data folders

- `data/raw/.gitkeep`: Keeps empty folder tracked.
- `data/raw/bible.pdf`: Newly added raw corpus PDF (~20 MB).
- `data/processed/.gitkeep`: Placeholder for processed token files.
- `data/tokenizer/.gitkeep`: Placeholder for tokenizer artifacts.

### Docs

- `docs/repo_blueprint.md`: Build plan and intended role of each file.
- `docs/repo_files_created.md`: Scaffold checklist of files included.
- `docs/repo_walkthrough.md`: This practical inventory walkthrough.

### Source package (`src/mini_tcn_llm`)

- `__init__.py`: Package marker/version export.
- `config.py`: YAML loading + minimal config validation.
- `data.py`: Token window chunking helper (`build_token_windows`).
- `model.py`: Model constructor placeholder (`NotImplementedError`).
- `train.py`: Training entry placeholder (`NotImplementedError`).
- `eval.py`: Evaluation entry placeholder (`NotImplementedError`).
- `generate.py`: Sampling placeholder (`NotImplementedError`).
- `tokenizer.py`: Tokenizer training placeholder (`NotImplementedError`).
- `utils.py`: Utility helper(s), currently seed function.

### Scripts

- `scripts/prepare_data.py`: Data prep CLI skeleton (currently not implemented).
- `scripts/train.py`: Training CLI skeleton calling `train_main`.
- `scripts/eval.py`: Eval CLI skeleton calling `eval_main`.

### Tests

- `tests/test_data_pipeline.py`: Validates fixed-window chunking behavior.
- `tests/test_model_shapes.py`: Confirms model builder currently raises `NotImplementedError`.
- `tests/test_tokenizer_roundtrip.py`: Confirms tokenizer trainer currently raises `NotImplementedError`.

### Root/project metadata

- `.gitignore`: Ignores most local artifacts (`data/*` contents except `.gitkeep`, outputs, caches, logs).
- `.gitkeep`: Empty placeholder file in repo root.
- `LICENSE`: MIT license text.
- `README.md`: Project goals, milestones, and quick-start commands.
- `pyproject.toml`: Package metadata, dependencies, and tool configs.

## Important current-state observations

1. The repository is mostly a scaffold: many core ML functions intentionally raise `NotImplementedError`.
2. CI is ready for lint/tests, but end-to-end training is not yet implemented.
3. `configs/data.yaml` still references `data/raw/corpus.txt`; if you want to train from `data/raw/bible.pdf`, data prep code/config must be updated to parse PDF text first.
4. `.gitignore` excludes most files under `data/raw/`, but already-tracked files can still exist in Git history/index.
