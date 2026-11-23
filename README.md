# MLOps Forecasting Platform

An end-to-end MLOps platform that forecasts Hyderabad air quality (PM2.5). The pipeline is reproducible, every run is tracked, and models are promoted through a registry. The full plan is in [docs/project-plan.md](docs/project-plan.md).

**Status:** Part 1, Phase 2 (evaluation + baselines) complete. Phase notes: [Phase 0](docs/phase-0-setup.md), [Phase 1](docs/phase-1-data.md), [Phase 2](docs/phase-2-evaluation.md).

**Data:** hourly PM2.5 for Hyderabad from the [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api) (CAMS model estimates, CC BY 4.0), Aug 2022 onward.

## Setup

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env          # then add your WANDB_API_KEY
wandb login
```

## Data pipeline

```bash
dvc pull                              # fetch the versioned data from the DVC remote
dvc repro                             # prepare -> featurize -> evaluate (reruns only changed stages)
dvc metrics show                      # baseline scores
python -m src.download openmeteo      # refresh raw data, then: dvc add data/raw/pm25_raw.csv && dvc push
```

## Results

24-hour-ahead PM2.5 forecasts (µg/m³), averaged over 5 walk-forward folds (30-day test windows spread across Oct 2025 to Sep 2026, one forecast per day at midnight). Lower is better.

| Model | MAE | RMSE | sMAPE (%) |
| --- | --- | --- | --- |
| Naive (last value) | 10.87 | 14.42 | 36.4 |
| **Seasonal naive, 24 h (same hour yesterday)** | **7.81** | **10.36** | **29.1** |
| Seasonal naive, 168 h (same hour last week) | 12.91 | 17.05 | 43.9 |

The bar to beat is **seasonal naive 24 h: RMSE 10.36**. Reproduce with `dvc repro && dvc metrics show`.

## Run

```bash
# Terminal 1: tracking server (SQLite backend, needed for the model registry)
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --port 5000

# Terminal 2
python -m src.train --help
python -m src.train --model dummy   # logs a test run; view it at http://127.0.0.1:5000
pytest && ruff check .
```
