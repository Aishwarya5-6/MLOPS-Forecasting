# Phase 2: Evaluation + Baselines — What Was Done

**Date:** 23 Nov 2026
**Status:** Complete. The exit check passes.
**Plan reference:** the original project plan (private), section 8, Phase 2

## Goal

One shared, leakage-free evaluation function, plus baseline scores every model must beat.

**Exit check:** `pytest` passes, and `dvc metrics show` prints baseline MAE, RMSE and sMAPE. ✅
- `pytest`: 27 passed.
- `dvc metrics show` prints all three metrics for all three baselines (table below).

## Baseline results

Each fold's score covers 24-hour-ahead forecasts made every day at midnight across a 30-day test window. That is 30 forecasts per fold, 150 in total. The scores below are averaged over the 5 folds. Units are µg/m³ (sMAPE in %).

| Model | MAE | RMSE | sMAPE |
| --- | --- | --- | --- |
| `naive` (last value) | 10.87 | 14.42 | 36.4 |
| **`seasonal_naive_24`** (same hour yesterday) | **7.81** | **10.36** | **29.1** |
| `seasonal_naive_168` (same hour last week) | 12.91 | 17.05 | 43.9 |

**The number to beat: RMSE 10.36** (seasonal naive, 24 h). The plan's promotion rule (section 9) needs a model to be at least 2% better, so below about 10.15.

### RMSE per fold

| Fold | Test window | Season | naive | seasonal_naive_24 | seasonal_naive_168 |
| --- | --- | --- | --- | --- | --- |
| 0 | 25 Oct – 23 Nov 2025 | Post-monsoon | 14.0 | 10.0 | 22.0 |
| 1 | 6 Jan – 4 Feb 2026 | Winter | 17.1 | 13.3 | 26.2 |
| 2 | 20 Mar – 18 Apr 2026 | Summer | 14.8 | 12.8 | 14.5 |
| 3 | 1 – 30 Jun 2026 | Monsoon onset | 16.9 | 10.2 | 15.6 |
| 4 | 13 Aug – 11 Sep 2026 | Monsoon | 9.4 | 5.5 | 7.0 |

### What the baselines tell us

- **Yesterday's same hour is a strong forecast.** It wins in every fold, which matches the EDA (the value 24 hours earlier correlates 0.82 with the current one).
- **The weekly baseline is worse than the plain naive one.** This confirms the EDA finding that there's no weekly cycle in this data.
- **Winter is hardest (RMSE 13.3); monsoon is easiest (5.5).** Errors grow with the level of pollution. Fold averages give each season equal weight, so a model can't win by doing well only in easy months.
- **RMSE by hours ahead:**
  - naive: 3.1 one hour ahead, 17.4 twelve hours ahead.
  - seasonal naive 24: 13.1 at 1 h, 7.5 at 12 h, 13.2 at 24 h.
  - The two are complementary: naive is best for the next few hours, seasonal naive for further out. A good model should combine "recent level" with "daily shape" (XGBoost with lags and the hour of day, SARIMA).
  - Because every forecast starts at midnight, *hours ahead* is the same as *hour of day* here. A curve over horizon also reflects which hours are calm (midday) and which are volatile (night).

## Evaluation design

### Splits: `make_folds()` in `src/data.py`

```
2022-08 ..................................... 2025-10   2025-10 → 2026-09 (5 folds)            2026-10-25 → 2026-11-23
[ training data only, expanding ]           [F0]...[F1]...[F2]...[F3]...[F4]...             [ HOLDOUT ]
```

- **Holdout:** the last ~30 days (`holdout_hours: 720`), with its start rounded up to midnight. Currently 25 Oct 2026 00:00 → 23 Nov 2026 04:00. It is **returned separately and never used in Phase 2**; it's reserved for `promote.py` in Phase 5.
- **5 folds:** each has a **30-day test window** (`fold_test_hours: 720`). The windows start **73 days apart** (`fold_step_hours: 1752`), so together they cover a full year of seasons and end right before the holdout.
- **Expanding window:** each fold trains on *everything* before its test window. Fold 0 has about 3.2 years of training data.
- **Origins:** inside a test window, a forecast is made **every 24 hours at midnight** (`origin_every_hours: 24`) for the next 24 hours. The last origin is the last one whose full 24 hours fit in the window.

### Model interface: `src/models/base.py`

```python
class ForecastModel(ABC):
    def fit(self, train: pd.Series) -> ForecastModel: ...            # once per fold, train only
    def predict(self, history: pd.Series, horizon: int) -> np.ndarray: ...  # forecast after history
```

**Why `predict(history, horizon)` instead of the plan's `predict(horizon)`:** with 30 forecast times per fold, a model has to forecast from different points without being refitted each time. Passing `history` makes it explicit what the model is allowed to see: training data plus the test hours already observed before the forecast time, never anything after it. Refitting 150 times would be too slow for SARIMA and LSTM.

### Evaluation loop: `src/evaluate.py`

- `mae`, `rmse`, `smape`: missing actual values are skipped. sMAPE is in percent (0–200), and a pair where both values are 0 counts as 0 error.
- `forecast_fold(model, y, fold, horizon)`: fits on `y[:train_end]`, then at each forecast time passes `y[:origin - 1h]` to `predict`.
- `evaluate(model, y, folds, horizon)` returns:
  - `folds`: scores per fold
  - `mean`: the plain average over folds; **this is the comparison number**
  - `rmse_by_horizon`: 24 values, one per hour ahead
- **Every model in Phases 3–5 goes through this same function.** That makes the comparison fair, which is success criterion #3.

### Baselines: `src/models/baseline.py`

| Class | Forecast |
| --- | --- |
| `NaiveModel` | last observed value for all 24 hours |
| `SeasonalNaiveModel(season=24)` | the last 24 hours, repeated |
| `SeasonalNaiveModel(season=168)` | the same hours one week earlier |

Both fill any missing values in the history forward first, so a gap can't produce an empty forecast.

### DVC stage: `evaluate`

```
prepare -> featurize
        -> evaluate  (deps: pm25_hourly.parquet + code; params: forecast)
                     -> reports/metrics.json          (DVC metric; dvc metrics show / diff)
                     -> reports/baseline_details.json (per-fold + per-horizon detail)
```

Both report files are `cache: false`, so git stores them and `dvc metrics diff` works across commits.

## New parameters (`params.yaml` → `forecast`)

| Key | Value | Meaning |
| --- | --- | --- |
| `holdout_hours` | 720 | last ~30 days, reserved for promotion |
| `n_folds` | 5 | number of walk-forward folds |
| `fold_test_hours` | 720 | test window per fold (30 days) |
| `fold_step_hours` | 1752 | gap between fold starts (73 days → 5 folds cover 1 year) |
| `origin_every_hours` | 24 | forecast once a day at midnight |

## Tests (16 new, 27 total)

| File | What it proves |
| --- | --- |
| `tests/test_folds.py` | 5 folds are produced. **Test always comes after train**, with no gap or overlap. Test windows **never overlap** and are in order. The training window expands. Folds never touch the holdout. The holdout starts at midnight and ends at the last hour. Forecast times start at midnight and fit inside the window. Too many folds for the data raises an error. |
| `tests/test_evaluate.py` | MAE, RMSE and sMAPE **match hand calculations**. sMAPE handles 0/0. Missing actual values are skipped. Naive and seasonal naive give the right outputs. Seasonal naive gets **RMSE = 0 on a perfect daily cycle**. A **spy model** proves the evaluator fits only on `train_end` and never shows `predict` anything at or after the forecast time. |

## Files added or changed

```
params.yaml                 # + fold settings under forecast:
dvc.yaml / dvc.lock         # + evaluate stage
src/data.py                 # + Fold, make_folds(), folds_from_params()
src/models/base.py          # ForecastModel interface (new)
src/models/baseline.py      # NaiveModel, SeasonalNaiveModel (new)
src/evaluate.py             # metrics, forecast_fold(), evaluate(), baseline stage (new)
reports/metrics.json        # baseline means (in git, DVC metric)
reports/baseline_details.json
tests/test_folds.py, tests/test_evaluate.py
README.md                   # results table
```

## Notes and gotchas

1. **Editing `src/data.py` reruns `prepare`.** The fold code lives in `data.py`, as the plan specifies. When it changed, `dvc repro` reran `prepare`. The output was identical, so `featurize` was then skipped. That's harmless, but if fold edits get frequent, move the fold code to `src/splits.py`.
2. **The holdout moves when data is refreshed.** It is always "the last ~30 days", so after `python -m src.download openmeteo` the folds and holdout shift forward and the baseline numbers change. Compare models only on the same data version (same `dvc.lock`). MLflow will record the data hash per run in Phase 3.
3. **Forecasts always start at midnight.** This is simple and matches a "tomorrow's forecast" use case. The live API in Part 3 will forecast from *any* hour. If we want scores that match that, set `origin_every_hours` to 6 or 1 later; it costs more compute for SARIMA and LSTM.

## Everyday commands

```bash
dvc repro                       # runs evaluate if code/params/data changed
dvc metrics show                # baseline table
dvc metrics diff                # compare with the last commit
pytest -q                       # 27 tests
```

## Next: Phase 3 (models in MLflow, the MVP)

1. `src/tracking.py` `log_run()`: params, metrics, artifacts, git commit and DVC data hash.
2. SARIMA (`statsmodels`) and XGBoost, each following `ForecastModel` and scored with `evaluate()`.
3. A `pyfunc` wrapper.
4. Forecast-vs-actual and residual plots.
5. 3+ runs per model.
6. Register the best model as the champion.

Open decision for XGBoost (noted in Phase 1): lags shorter than the forecast horizon aren't known when forecasting 24 hours ahead. Use either recursive prediction or a direct per-horizon setup.
