# Phase 0: Setup — What Was Done

**Date:** 23 Nov 2026
**Status:** Complete, except W&B login (blocked; see [Open items](#open-items))
**Plan reference:** the original project plan (private), section 8, Phase 0

## Goal

A clean, working project skeleton with MLflow running locally.

**Exit check:** `python -m src.train --model dummy` creates a run you can see in the MLflow UI. ✅ Passed.

## Task checklist

| Task | Status | Notes |
| --- | --- | --- |
| Create GitHub repo and connect it | ✅ | `https://github.com/Aishwarya5-6/MLOPS-Forecasting`, branch `main` |
| Create virtual environment | ✅ | `.venv/`, Python 3.11.15, created with `uv` |
| Install packages from section 8 | ✅ | Plus `python-dotenv` and `pyyaml` |
| Pin versions in `requirements.txt` | ✅ | 174 packages pinned |
| Create folder structure (section 7) | ✅ | Includes `__init__.py` in `src/`, `src/models/`, `tests/` |
| Write `.gitignore` | ✅ | |
| Create `.env.example` | ✅ | Also copied to `.env` (ignored by git) |
| Sign up for W&B and `wandb login` | ⚠️ Blocked | Account disabled; support contacted. Not needed until Phase 4 |
| Start MLflow server with SQLite backend | ✅ | `http://127.0.0.1:5000` |
| Write `src/train.py` with `argparse` | ✅ | `--model`, `--config` |
| Log a dummy run to MLflow | ✅ | Experiment `forecasting-dev`, run `dummy-connection-test` |
| First commit: "Project skeleton" | ✅ | Commit `8cf43cf`, pushed to `origin/main` |

## Environment decisions

- **Python 3.11 via `uv`, not the conda Python 3.10.** The machine's default `python3` came from a conda env (`taxi_rl`, 3.10). A separate `uv` venv keeps this project isolated from other projects. 3.11 is well supported by every package in the stack.
- **Machine:** Apple M4 (arm64), CPU only. Every package installed from PyPI with no extra steps (including `xgboost` and `torch`).
- **Key pinned versions:**

| Package | Version |
| --- | --- |
| mlflow | 3.18.0 |
| dvc | 3.67.1 |
| wandb | 0.31.2 |
| torch | 2.15.0 |
| xgboost | 3.2.0 |
| statsmodels | 0.15.0 |
| pandas | 3.1.1 |

## Files created

```
MLOPS Forecasting/
├── .venv/                  # virtual environment (not in git)
├── .env                    # real secrets (not in git)
├── .env.example            # template: WANDB_API_KEY=, MLFLOW_TRACKING_URI=
├── .gitignore
├── README.md               # setup + run instructions
├── pyproject.toml          # ruff + pytest settings
├── requirements.txt        # pinned versions
├── configs/dummy.yaml      # one param for the connection test
├── data/raw/.gitkeep       # data itself goes to DVC in Phase 1
├── data/processed/.gitkeep
├── docs/
│   └── phase-0-setup.md    # this file
├── notebooks/              # EDA notebooks (Phase 1)
├── src/
│   ├── __init__.py
│   ├── models/__init__.py
│   └── train.py            # entry point
├── sweeps/.gitkeep
└── tests/
    ├── __init__.py
    └── test_train.py
```

Created at runtime, all ignored by git: `mlflow.db` (SQLite tracking and registry store) and `mlruns/` (artifacts).

### What each file does

- **`src/train.py`:** the single entry point for all training.
  - `--model` takes one of `dummy, naive, seasonal_naive, sarima, xgb, lstm`.
  - `--config` takes a YAML path. When it's omitted, `configs/<model>.yaml` is used if it exists.
  - Loads `.env` with `python-dotenv`, so `MLFLOW_TRACKING_URI` comes from the environment rather than from code.
  - `--model dummy` logs one param (`learning_rate`), one metric (`rmse=1.0`) and the tags `model_type`, `stage`.
  - Every other model raises `NotImplementedError` until its phase.
- **`.gitignore`:** excludes `.venv/`, `.env`, caches, `mlruns/`, `mlartifacts/`, `mlflow.db`, `wandb/`, and model binaries (`*.pkl`, `*.joblib`, `*.onnx`, `*.pt`). *(Updated in Phase 1: data rules were removed from the root `.gitignore`; DVC now writes a `.gitignore` inside each data folder. See [phase-1-data.md](phase-1-data.md).)*
- **`pyproject.toml`:**
  - ruff: line length 100, target py311, rules `E, F, I, UP, B`, `docs/` excluded (otherwise ruff tries to format code blocks in markdown).
  - pytest: `testpaths = ["tests"]`.
- **`tests/test_train.py`:** 3 tests.
  - Argument parsing works.
  - Unknown models are rejected.
  - A missing config file returns `{}`.

## Everyday commands

```bash
# Activate the environment (every new terminal)
cd "/Users/aishwarya/MLOPS Forecasting"
source .venv/bin/activate

# Start MLflow: run from the project root so it always uses the same mlflow.db
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
# UI: http://127.0.0.1:5000

# Train (in a second terminal)
python -m src.train --help
python -m src.train --model dummy

# Quality checks
pytest
ruff check .
ruff format --check .

# After installing a new package, re-pin
uv pip install <package> && uv pip freeze > requirements.txt

# Recreate the environment from scratch
uv venv --python 3.11 .venv && source .venv/bin/activate && uv pip install -r requirements.txt
```

## Verification results

| Check | Result |
| --- | --- |
| `python -m src.train --help` | Prints usage ✅ |
| Import `xgboost, torch, statsmodels, mlflow, wandb` | OK ✅ |
| `pytest` | 3 passed ✅ |
| `ruff check .` / `ruff format --check .` | Clean ✅ |
| MLflow `/health` | OK ✅ |
| `python -m src.train --model dummy` | Run logged to experiment `forecasting-dev` ✅ |
| Secrets / data / DB in git? | None committed ✅ |

## Gotchas and lessons

1. **Always start MLflow from the project root.** `sqlite:///mlflow.db` is a relative path. Starting the server from another folder creates a new, empty database, and your runs seem to vanish.
2. **The registry needs a database backend.** Plain file storage (`./mlruns` only) can't hold registered models or aliases. That's why SQLite is used from day one (plan, section 10).
3. **MLflow 3.x prints an "agent hint" log line** on import. It's harmless. Set `MLFLOW_DISABLE_AGENT_HINT=1` to hide it.
4. **The server must be running before training.** Otherwise `train.py` fails to connect to `http://127.0.0.1:5000`.
5. **Never commit `.env`.** Only `.env.example` goes to GitHub.

## Open items

- **W&B account disabled.** Signing in with GitHub showed "Your account has been disabled. Contact the administrator of your instance."
  - **Action:** email support@wandb.com with your account email and a screenshot, or sign up with a different email.
  - **Needed by:** Phase 4 (days 15–20). Nothing before that uses W&B.
  - **Fallback if still blocked:** run sweeps with **Optuna** and log each trial to MLflow. This covers Phase 4's goals: Bayesian search, parameter importance and parallel-coordinate plots.
- **Dataset not chosen yet** (plan, section 5). The roadmap recommends Hyderabad air quality: history from the OpenAQ archive (CPCB stations, 2019 onward). Decide at the start of Phase 1.

## Next: Phase 1 (Data + DVC)

Choose the dataset → `dvc init` → `dvc add` raw data → set up a DVC remote and `dvc push` → write `src/data.py` and `src/features.py` → `params.yaml` → `prepare` and `featurize` stages in `dvc.yaml` → `dvc repro`.
