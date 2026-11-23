# Phase 3: Models in MLflow (MVP) — What Was Done

**Date:** 23 Nov 2026
**Status:** Complete. Every item on the MVP checklist is ticked, so **the Part 1 MVP is done.**
**Plan reference:** [project-plan.md](project-plan.md), section 8, Phase 3, and the section 4 MVP checklist

## Goal

SARIMA and XGBoost trained, tracked and compared in MLflow, with one registered champion.

## MVP acceptance checklist (section 4)

| Item | Status | Evidence |
| --- | --- | --- |
| `dvc repro` runs the whole pipeline from raw data to metrics | ✅ | prepare → featurize → evaluate → `reports/metrics.json` |
| MLflow UI shows at least 3 runs per model type, sortable by RMSE | ✅ | 4 SARIMA, 5 XGBoost, 3 baseline runs in experiment `forecasting-hyd-pm25` |
| Every run has its git commit and DVC data hash logged | ✅ | Tags `git_commit`, `git_dirty` (all `false`), `data_version`, `prepared_data_version` |
| At least one model beats the seasonal-naive baseline | ✅ | Best RMSE 8.307 vs 10.362 (**-19.8%**). All 9 model runs beat it |
| `mlflow.pyfunc.load_model("models:/<name>@champion")` returns predictions | ✅ | `pm25-forecaster@champion` loaded from a folder *outside* the repo and forecast the next 24 h |

**Name change from the plan:** the registered model is `pm25-forecaster`, not `demand-forecaster`, following the plan's own `<dataset>-forecaster` convention. The experiment is `forecasting-hyd-pm25`.

## Results: all 12 runs

5-fold walk-forward, 24-hour horizon. Units are µg/m³, sMAPE in %. "Eval s" is the time to score all 5 folds.

| Run | Type | MAE | RMSE | sMAPE | Eval s |
| --- | --- | --- | --- | --- | --- |
| **`xgb-d6-lr0.05-n400-log`** ← `@champion` | xgb | 6.34 | **8.307** | 23.3 | 9 |
| `xgb-d4-lr0.03-n800-log` | xgb | 6.39 | 8.357 | 23.9 | 12 |
| `sarima-101-101x24-42d-log` | sarima | **6.26** | 8.372 | 23.6 | 11 |
| `xgb-d4-lr0.03-n800` | xgb | 6.53 | 8.513 | 24.5 | 12 |
| `sarima-101-101x24-90d-log` | sarima | 6.42 | 8.548 | 24.3 | 21 |
| `sarima-201-101x24-42d-log` | sarima | 6.39 | 8.661 | 23.6 | 370 |
| `sarima-101-101x24-42d` | sarima | 6.45 | 8.664 | 24.6 | 7 |
| `xgb-d6-lr0.05-n400` | xgb | 6.61 | 8.665 | 24.4 | 9 |
| `xgb-d8-lr0.05-n400` | xgb | 6.57 | 8.680 | 24.3 | 12 |
| `seasonal_naive_24` | baseline | 7.81 | 10.362 | 29.1 | 0 |
| `naive` | baseline | 10.87 | 14.423 | 36.4 | 0 |
| `seasonal_naive_168` | baseline | 12.91 | 17.052 | 43.9 | 0 |

### RMSE per fold: best XGBoost vs. best SARIMA

| Fold | Season | XGBoost | SARIMA |
| --- | --- | --- | --- |
| 0 | Post-monsoon (Oct–Nov 2025) | **7.75** | 8.87 |
| 1 | Winter (Jan–Feb 2026) | 11.79 | **11.07** |
| 2 | Summer (Mar–Apr 2026) | 9.57 | **9.01** |
| 3 | Monsoon onset (Jun 2026) | 8.32 | 8.34 |
| 4 | Monsoon (Aug–Sep 2026) | **4.10** | 4.57 |

### What we learned

1. **Both real models beat every baseline by about 19%.** This is a solid result, because "same hour yesterday" is a strong baseline here.
2. **The best XGBoost and the best SARIMA are within 0.8% of each other.** That's well inside fold-to-fold noise. SARIMA has the lower MAE and wins winter and summer; XGBoost has the lower RMSE and wins post-monsoon and monsoon. The champion was picked by the plan's rule (lowest mean RMSE), but **"XGBoost is better" is not a strong claim.** An ensemble of the two might beat both; that's a stretch idea for later.
3. **A log transform helps both models**, by about 0.36 RMSE for XGBoost and 0.29 for SARIMA. PM2.5 errors grow with the level, and the log scale evens them out.
4. **Bigger isn't better:**
   - SARIMA trained on 90 days was *worse* than on 42 days, because recent behaviour matters more.
   - SARIMA(2,0,1) was worse *and* 34 times slower (370 s).
   - XGBoost with depth 8 was worse than depth 6 or 4.
5. **Feature importance (champion):** `seas_24` (same hour yesterday) accounts for about 50% of the gain, followed by `roll_mean_6`, `roll_mean_24` and `lag_1`. That's "daily shape plus recent level", which is exactly the gap between the two baselines that Phase 2 pointed to. `step`, `dayofweek` and `is_weekend` add almost nothing (no weekly cycle).
6. **Error by hours ahead (champion):** about 5 one hour ahead, peaking around 11 at hours 6–8 (morning) and 12 at hours 22–24 (night). The midday hours are easiest (about 6.5). Because forecasts start at midnight, hours ahead = hour of day.
7. **Residuals are centred** (mean -0.02, sd 8.7), so there's no systematic over- or under-forecasting.

These comparisons are hand-picked, one change at a time. The systematic search is Phase 4's W&B (or Optuna) sweep.

## Model design

### SARIMA (`src/models/sarima.py`, `configs/sarima.yaml`)

- `SARIMAX(order, seasonal_order, trend="c")` on `log1p(pm25)` (`log_transform: true`).
- **`fit`** uses only the last `train_days` (42) of the fold's training data. Fitting on years of hourly data with a 24-hour season would be very slow, and older data helped less anyway.
- **`predict`** re-applies the fitted parameters to the last `history_days` (14) with `result.apply(...)`, which runs the Kalman filter without refitting, then forecasts 24 steps, undoes the log and clips at 0. This takes about 10 ms per forecast.
- Note: `expm1` of a log-scale forecast gives roughly the *median*, not the mean. The fact that it still lowered RMSE suggests the gain from stabilising variance outweighs the bias.

### XGBoost: direct multi-horizon (`src/models/xgb.py`, `configs/xgb.yaml`)

**The Phase 1/2 open question, decided:** one model forecasts all 24 steps, with **`step` (1–24) as a feature**. No predictions are fed back in, so there's no compounding error and only one model to manage.

Training rows: one per (forecast origin, step), with an origin every `train_origin_every` (2) hours across the fold's training data. Each fold has about 330k rows and takes about 2 seconds to fit.

Features come from `direct_features()` in `src/features.py` and are **all computed from data known at the forecast origin**:

| Feature | Meaning |
| --- | --- |
| `step` | hours ahead (1–24) |
| `lag_k`, k in 1, 2, 3, 6, 12, 24 | k-th most recent value at the origin |
| `roll_mean_w`, `roll_std_w`, w in 6, 24, 168 | stats over the last w hours before the origin |
| `seas_24`, `seas_168` | value 24 / 168 h before the **target** hour (NaN if not yet known) |
| `hour`, `dayofweek`, `month`, `is_weekend`, `is_holiday` | of the **target** hour |

`log_target: true` trains on `log1p(pm25)` and inverts with `expm1`.

**Relation to the `featurize` stage:** `features.parquet` holds lags relative to the *target hour*, which suits 1-step models. XGBoost can't use it for 24-hour-ahead forecasts. The same features have to be computed from raw history at serving time anyway, so they're built inside the model from the series. `featurize` stays in the pipeline (EDA, and possibly LSTM in Phase 4).

### Final model vs. evaluation

- **Evaluation:** a fresh model per fold, fitted on that fold's training data only.
- **Logged model:** after evaluation, the model is **refitted on all data up to the holdout start** (25 Oct 2026). That's what gets saved and registered. The holdout stays unseen for `promote.py` in Phase 5.

## MLflow tracking

### `log_run()` (`src/tracking.py`)

One function writes every run. For each run it logs:

| What | Content |
| --- | --- |
| Tags | `model_type`, `git_commit`, `git_dirty`, `stage=dev`, `data_version` (raw md5 from the `.dvc` file), `prepared_data_version` (from `dvc.lock`) |
| Params | model config (`model.*`), all forecast/fold settings (`forecast.*`), `seed` |
| Metrics | `mae`, `rmse`, `smape` (means over folds), `eval_seconds` |
| Step metrics | `fold_rmse`, `fold_mae` (step = fold), `rmse_by_horizon` (step = hours ahead - 1). In the UI, click a metric to see these as charts |
| Plots | `plots/forecast_vs_actual.png` (5 panels), `plots/residuals.png`, `plots/feature_importance.png` (XGBoost) |
| Files | `config/params.yaml`, `config/<model>.yaml` |
| Model | pyfunc (see below). In MLflow 3 it appears under the experiment's **Models** tab, not in the run's Artifacts list |

### `pyfunc` wrapper (`src/models/pyfunc_wrapper.py`)

Every model type is saved the same way, so loading and serving never depend on the model type (plan, section 10).

- **Input:** DataFrame `timestamp` (datetime), `pm25` (double). Give it **at least 14 days** of hourly history (SARIMA uses 14 days; XGBoost needs 169 hours).
- **Params:** `{"horizon": 24}` (XGBoost supports up to 24).
- **Output:** DataFrame `timestamp`, `forecast` for the hours after the history.
- **Saved with it:**
  - the fitted model (`joblib`)
  - a copy of `src/` (`code_paths`), so it loads anywhere
  - a signature and a 14-day input example
  - exact pinned versions of 10 packages (also covers "log package versions per run")

### Registry

```python
mlflow.register_model(f"runs:/{run_id}/model", "pm25-forecaster")
client.set_registered_model_alias("pm25-forecaster", "champion", version)
```

- `pm25-forecaster` **v1** = run `9c305ef0…` (`xgb-d6-lr0.05-n400-log`), alias **`@champion`**.
- Version tags: `cv_rmse=8.307`, `model_type=xgb`, `data_version`, `git_commit`, `promoted_by=manual (Phase 3)`.
- The champion was set **by hand**, as the plan specifies for Phase 3. Phase 5's `promote.py` automates this with holdout scoring.

### Loading the champion

```python
model = mlflow.pyfunc.load_model("models:/pm25-forecaster@champion")
model.predict(history_df, params={"horizon": 24})
```

Tested from a folder outside the repo. With history ending 23 Nov 04:00 it forecast about 38 → 31 (afternoon low) → 43 µg/m³ (night peak). That's the daily shape the EDA found.

## `train.py` usage

```bash
python -m src.train --model xgb                         # uses configs/xgb.yaml
python -m src.train --model xgb --set log_target=true   # override any key (YAML-parsed; repeatable)
python -m src.train --model sarima --set "order=[2, 0, 1]"
python -m src.train --model seasonal_naive --set season=168
python -m src.train --model xgb --no-log-model          # skip saving the model (faster experiments)
python -m src.train --model dummy                       # Phase 0 connection test
```

Run names are built from the config, e.g. `xgb-d6-lr0.05-n400-log` or `sarima-101-101x24-42d-log`. Pass `--run-name` to override.

**Commit before running.** Runs record `git_dirty`. Every run in this phase was made from a clean commit (`git_dirty=false`), so each result maps to exact code.

## Files added or changed

```
src/models/sarima.py          # SarimaModel (new)
src/models/xgb.py             # XGBModel, direct multi-horizon (new)
src/models/pyfunc_wrapper.py  # ForecastPyfunc, log_pyfunc() (new)
src/tracking.py               # log_run(), git/DVC tags (new)
src/plots.py                  # forecast-vs-actual, residuals, feature importance (new)
src/features.py               # + direct_features(), holiday_flags()
src/evaluate.py               # + return_forecasts option
src/train.py                  # real training entry point
configs/sarima.yaml, configs/xgb.yaml, configs/seasonal_naive.yaml
tests/test_models.py          # 7 tests (new); 34 total
README.md                     # results table, "using the production model"
```

### New tests (`tests/test_models.py`)

- Direct features have the right shape and values.
- **Leakage:** changing data at or after the forecast origin leaves the features unchanged.
- A seasonal lag that isn't known yet is NaN.
- XGBoost learns a synthetic daily cycle (RMSE < 5).
- SARIMA fits, predicts 24 values and never goes negative.
- The pyfunc returns a timestamped forecast that starts 1 hour after the history.
- `--set` values are parsed as YAML.

## Gotchas

1. **zsh doesn't split `$args`.** `for a in "--model naive"; do python -m src.train $a` passes one argument. Use separate commands or arrays.
2. **MLflow 3 stores models as "logged models".** `runs:/<id>/model` still works for registering, with a harmless warning that it resolved to `models:/m-…`.
3. **Harmless warning:** "Add type hints to the `predict` method". MLflow raises it through its own handler, which ignores normal warning filters. We pass an explicit signature, so it doesn't matter. I tried suppressing it and reverted, because it wasn't worth the extra code.
4. **SARIMA's fit time grows quickly with the order:** (2,0,1) took 34 times longer than (1,0,1). Keep sweeps on small orders.
5. **The MLflow server must be running** (`mlflow server ...` from the project root), or `train.py` fails at the logging step *after* evaluation. You'd lose about 10 seconds, not much.

## Next: Phase 4 (LSTM + sweeps)

1. LSTM (PyTorch, sliding windows, early stopping, fixed seed) using the same `ForecastModel` interface and the same `log_run()`.
2. W&B mirroring in `log_run()`, if the W&B account is restored. Otherwise **Optuna + MLflow** for the sweep: 30–50 trials, with parameter importance and parallel-coordinates plots.
3. Retrain the best sweep config as a normal tracked run.
