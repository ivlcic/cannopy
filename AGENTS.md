# Repository Guidelines

## Project Structure & Module Organization
- Core Python source in `src/` (notebooks/experiments often under `train/` or `eval/`).
- Datasets, artifacts, and run outputs live in `data/`, `result/`, and `log/`; keep large files out of git.
- Tests reside in `test/`; prefer mirroring `src/` package structure for easy discovery.
- Configuration and hyperparameters typically live alongside entrypoints or in YAML/JSON within `conf/`.

## Build, Test, and Development Commands
- `pip install -r requirements.txt` to set up dependencies (use venv/conda).
- `python -m pytest test` runs the full test suite; add `-k <pattern>` to target subsets.
- `python -m pytest --maxfail=1 -q` for quick pre-push checks.
- `python -m <package> ...` or scripts in `train/`/`eval/` to run experiments; check README for entrypoints.

## Coding Style & Naming Conventions
- Follow PEP 8; use 4-space indentation and type hints where possible.
- Functions must be defined before a first call in a file or class.
- First function call sequence must be the same as definition sequence.
- Keep modules focused; prefer pure functions for preprocessing and clearly named classes for models.
- Name files and tests descriptively (e.g., `model_registry.py`, `test_model_registry.py`).
- Prefer f-strings, avoid unused imports, and keep imports sorted (stdlib, third-party, local).

## Testing Guidelines
- Use `pytest` for unit/integration tests; place fixtures in `conftest.py` near usage.
- Name tests with behavior intent (e.g., `test_predict_handles_missing_features`).
- When adding models or pipelines, cover data validation paths and shape/metric expectations.
- Aim for meaningful coverage around data transforms and experiment reproducibility.

## Commit & Pull Request Guidelines
- Write imperative, concise commit messages (e.g., `add config validation`, `fix training loop retry`).
- For PRs: describe intent, key changes, test evidence (`pytest` output), and any new config knobs.
- Link issues/tickets when available; include before/after metrics or screenshots for user-facing changes.

## Security & Configuration Tips
- Do not commit secrets or large artifacts; use environment variables or `.env` (gitignored) for credentials.
- Document dataset sources and licensing in PRs; ensure paths under `data/` are reproducible.
