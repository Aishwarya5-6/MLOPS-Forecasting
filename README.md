# MLOps Forecasting Platform

An end-to-end MLOps platform that forecasts Hyderabad air quality (PM2.5). The pipeline is reproducible, every run is tracked, and models are promoted through a registry. The full plan is in [docs/project-plan.md](docs/project-plan.md).

**Status:** Part 1, Phase 5 complete: automated champion/challenger promotion with `promote.py`. Phase notes: [Phase 0](docs/phase-0-setup.md), [Phase 1](docs/phase-1-data.md), [Phase 2](docs/phase-2-evaluation.md), [Phase 3](docs/phase-3-mlflow-models.md), [Phase 4](docs/phase-4-lstm-sweep.md), [Phase 5](docs/phase-5-registry.md).

**Data:** hourly PM2.5 for Hyderabad from the [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api) (CAMS model estimates, CC BY 4.0), Aug 2022 onward.

## Setup

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env          # optional settings (MLflow URI etc.)
```

## Data pipeline

```bash
dvc pull                              # fetch the versioned data from the DVC remote
dvc repro                             # prepare -> featurize -> evaluate (reruns only changed stages)
dvc metrics show                      # baseline scores
python -m src.download openmeteo      # refresh raw data, then: dvc add data/raw/pm25_raw.csv && dvc push
```

## Results

24-hour-ahead PM2.5 forecasts (µg/m³), one forecast per day at midnight. Lower is better.

- **CV RMSE:** the average over 5 walk-forward folds (30-day test windows spread across Oct 2025 to Sep 2026). Used to develop and tune the models.
- **Holdout RMSE:** the last 30 days (25 Oct – 23 Nov 2026), which no model was trained or tuned on. Only `promote.py` uses it. This is the post-monsoon pollution rise, so every model scores worse here than in CV.

| Model | CV MAE | CV RMSE | Holdout RMSE | Registry |
| --- | --- | --- | --- | --- |
| **Ensemble: tuned XGBoost + SARIMA, equal weights** | **6.00** | **7.96** | **11.42** | **v5 `@champion`** |
| XGBoost, tuned by Optuna (40 trials) | 6.21 | 8.15 | 11.92 | v4, rejected (+1.4%) |
| XGBoost, direct multi-horizon, log target | 6.34 | 8.31 | 12.08 | v1, previous champion |
| SARIMA(1,0,1)(1,0,1)₂₄ on log values, 42-day window | 6.26 | 8.37 | 12.11 | |
| LSTM, 2 layers, 7-day input window | 6.61 | 8.78 | 13.01 | v3, rejected (−7.7%) |
| Seasonal naive, 24 h (same hour yesterday) | 7.81 | 10.36 | 14.66 | v2 `@baseline` |
| Naive (last value) | 10.87 | 14.42 | | |
| Seasonal naive, 168 h (same hour last week) | 12.91 | 17.05 | | |

**Headline:** the champion has **23.1% lower RMSE than the seasonal-naive baseline in CV**, and **22.1% lower on the untouched holdout**.

### Sweep findings (Optuna, TPE, 40 trials on XGBoost)

- **Learning rate dominates** (fANOVA importance 0.54). Every good trial uses 0.01–0.03; anything above 0.1 scores 8.45–9.3.
- **Small learning rates need many trees** (`n_estimators` importance 0.19): the best trials pair ~0.013 with 600–1,100 trees.
- **Leaves need little regularisation** (`min_child_weight` 0.15): 2–10 works best; 25+ is usually worse.
- **Depth, subsample and column sampling barely matter** (≈0.04 each) inside the ranges searched.
- **Diminishing returns:** the search plateaued at RMSE 8.15–8.22 from trial 14 on. The best trial is only 1.9% better than the hand-tuned config in CV, and 1.4% better on the holdout, so the 2% rule rejected it. Combining two model families (the ensemble) gained more than tuning did.

## Promotion (`promote.py`)

Production is whatever version `@champion` points to. A new version takes over only if its holdout RMSE is **at least 2% lower** than the champion's on the same window.

```bash
python -m src.promote challenger --run-id <run_id>            # register -> @challenger -> score -> promote or reject
python -m src.promote challenger --run-id <run_id> --dry-run  # score and decide; change nothing
python -m src.promote baseline --run-id <run_id>              # register seasonal naive as @baseline
python -m src.promote rollback                                # @champion -> its previous_champion
python -m src.promote rollback --to-version 5                 # @champion -> a specific version
python -m src.promote status                                  # aliases and their tags
```

Every decision is written to the model version tags:
- `holdout_rmse` (also MAE and sMAPE), `holdout_window`, `holdout_vs_champion`, `holdout_vs_baseline`
- on promotion: `promoted_on`, `previous_champion`, `promotion_reason`
- on rejection: `rejected_reason`, `rejected_on`
- on rollback: `rolled_back_on` / `rolled_back_to` and `restored_on` / `restored_from`

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
python -m src.train --model xgb                         # evaluate on 5 folds + log to MLflow
python -m src.train --model sarima --set train_days=90  # override any config value
python -m src.train --model ensemble                    # configs/ensemble.yaml lists the members
python -m src.sweep sweeps/xgb.yaml                     # Optuna sweep, nested MLflow runs
python -m src.promote challenger --run-id <run_id>      # try to promote the run
pytest && ruff check .
```
