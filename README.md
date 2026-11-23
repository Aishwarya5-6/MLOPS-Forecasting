# MLOps Forecasting Platform

An end-to-end MLOps platform that forecasts Hyderabad air quality (PM2.5). The pipeline is reproducible, every run is tracked, and models are promoted through a registry. The full plan is in [docs/project-plan.md](docs/project-plan.md).

**Status:** Part 1 MVP complete (Phase 3): SARIMA and XGBoost tracked in MLflow, best model registered as `pm25-forecaster@champion`. Phase notes: [Phase 0](docs/phase-0-setup.md), [Phase 1](docs/phase-1-data.md), [Phase 2](docs/phase-2-evaluation.md), [Phase 3](docs/phase-3-mlflow-models.md).

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

24-hour-ahead PM2.5 forecasts (µg/m³), averaged over 5 walk-forward folds (30-day test windows spread across Oct 2025 to Sep 2026, one forecast per day at midnight). Lower is better. Best run of each model type shown; all 12 runs are in MLflow.

| Model | MAE | RMSE | sMAPE (%) | vs. seasonal naive (RMSE) |
| --- | --- | --- | --- | --- |
| **XGBoost, direct multi-horizon, log target** (`@champion`) | **6.34** | **8.31** | **23.3** | **-19.8%** |
| SARIMA(1,0,1)(1,0,1)₂₄ on log values, 42-day window | 6.26 | 8.37 | 23.6 | -19.2% |
| Seasonal naive, 24 h (same hour yesterday) | 7.81 | 10.36 | 29.1 | baseline |
| Naive (last value) | 10.87 | 14.42 | 36.4 | +39.2% |
| Seasonal naive, 168 h (same hour last week) | 12.91 | 17.05 | 43.9 | +64.6% |

XGBoost and SARIMA are within 1% of each other, which is inside fold-to-fold noise. Both clearly beat every baseline.

## Using the production model

```python
import mlflow, pandas as pd

mlflow.set_tracking_uri("http://127.0.0.1:5000")
model = mlflow.pyfunc.load_model("models:/pm25-forecaster@champion")
# history: DataFrame(timestamp, pm25) with at least 14 days of hourly values
forecast = model.predict(history, params={"horizon": 24})  # -> DataFrame(timestamp, forecast)
```

## Run

```bash
# Terminal 1: tracking server (SQLite backend, needed for the model registry)
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --port 5000

# Terminal 2
python -m src.train --help
python -m src.train --model xgb                       # evaluate on 5 folds + log to MLflow
python -m src.train --model sarima --set train_days=90  # override any config value
python -m src.train --model seasonal_naive              # baselines are logged too
pytest && ruff check .
```
