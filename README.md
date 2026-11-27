# MLOps Forecasting Platform

An end-to-end MLOps platform that forecasts Hyderabad air quality (PM2.5). The pipeline is reproducible, every run is tracked, and models are promoted through a registry.

**Status:** Part 1, Phase 5 complete (automated champion/challenger promotion with `promote.py`), plus a pre-registered study of promotion rules written up as a paper ([below](#paper)). Phase notes: [Phase 0](docs/phase-0-setup.md), [Phase 1](docs/phase-1-data.md), [Phase 2](docs/phase-2-evaluation.md), [Phase 3](docs/phase-3-mlflow-models.md), [Phase 4](docs/phase-4-lstm-sweep.md), [Phase 5](docs/phase-5-registry.md), [Paper study](docs/phase-paper-study.md).

**Data:** hourly PM2.5 from two sources; see [Data and attribution](#data-and-attribution).

## Paper

**How Strict Should Automated Model Promotion Be? A Replay Study of Champion–Challenger Rules for PM2.5 Forecasting.** The question: when a newly trained model scores better on recent data, how much better must it be before it replaces the production model?

- **Design:** 52 weekly promotion decisions are replayed per dataset, on 5 datasets: Hyderabad (CAMS estimates) and CPCB measurements for Hyderabad, Delhi, Bengaluru and Chennai. Each decision is judged on the following, unseen week.
- **Rules compared:** thresholds of 0–10%, a Diebold–Mariano test rule and never-switch, each with 7-, 14- and 30-day validation windows.
- **Pre-registration:** the protocol was fixed before any result: [docs/study-protocol.md](docs/study-protocol.md), first commit `9f5db7a`, with all later deviations recorded in the same file.
- **Main finding:** promoting on any improvement swapped the model 30–44 times a year, and 36–55% of those swaps made the next week worse. A 2–5% threshold on a 30-day window removed most swaps with no measurable loss in accuracy.

**Reproduce the paper's numbers, tables and figures:**

The raw data are not stored in this repository. Both sources are openly licensed and can be re-downloaded exactly:

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate && uv pip install -r requirements.txt

# 1. CAMS series: fixed end date -> byte-identical to the version used (MD5 4f244f3c..., verified 28 Nov 2026)
python -m src.download openmeteo --end 2026-11-22

# 2. CPCB series: download city_hour.csv (CC0) from Kaggle into data/raw/station/
#    https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india   (MD5 6d9c5b1a...)

# 3. Check both files match the versions used, then rebuild everything
dvc status data/raw/pm25_raw.csv.dvc data/raw/station/city_hour.csv.dvc   # should report no changes
dvc repro     # prepare -> forecast caches -> replays -> hypotheses -> paper/figures, tables, numbers.tex
```

`dvc repro` takes about 1.5 hours on a laptop CPU, almost all of it LSTM training. No MLflow server is needed: the few numbers taken from MLflow are stored in `reports/study/tracking_export.json`. Tree models, SARIMA and the baselines are deterministic; the seeded LSTM may differ slightly across hardware and thread counts.

The LaTeX source is in [paper/](paper/). Every number in the text is a macro from the generated `paper/numbers.tex`. Set `\extendedtrue` / `\extendedfalse` at the top of `paper/main.tex` for the extended or the conference version.

## Data and attribution

| Dataset | Source | Licence | Used for |
| --- | --- | --- | --- |
| Hyderabad PM2.5, Aug 2022 – Nov 2026 | [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api), CAMS global atmospheric-composition model estimates | CC BY 4.0 | Platform (Phases 1–5) and the paper's CAMS dataset |
| City-level PM2.5 for Hyderabad, Delhi, Bengaluru, Chennai, 2015–2020 | [Air Quality Data in India (2015–2020)](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india), compiled from Central Pollution Control Board (CPCB) data | CC0 1.0 (public domain) | The paper's CPCB datasets |

- **CAMS attribution:** contains modified Copernicus Atmosphere Monitoring Service information, 2026. Neither the European Commission nor ECMWF is responsible for any use of this information. Data access is provided by [Open-Meteo](https://open-meteo.com/).
- **CPCB data:** download `city_hour.csv` from the Kaggle page above, save it to `data/raw/station/city_hour.csv`, then run `dvc repro`. The dataset is released under CC0 (public domain); please still credit CPCB and the dataset's compiler.

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

## Licence

Code: [MIT](LICENSE). The data sources keep their own licences; see [Data and attribution](#data-and-attribution).
