# Study protocol: how strict should automated model promotion be?

**Status:** fixed before any replay result was computed. Any later change goes in [Deviations](#deviations), with its date and reason. The git commit that adds this file is cited in the paper.
**Written:** 27 Nov 2026

## Research question

In automated champion/challenger promotion for time-series forecasting, how do the **promotion rule** and the **validation-window length** affect:
- (a) **harmful model swaps**, and
- (b) the **accuracy of the model actually deployed**?

## Data

| Name | Source | Preparation |
| --- | --- | --- |
| `cams` | Open-Meteo / CAMS hourly PM2.5, Hyderabad city centre, Aug 2022 – Nov 2026 (`data/processed/pm25_hourly.parquet`, the current DVC version) | Existing `prepare` stage |
| `station` | CPCB measurements: Kaggle "Air Quality Data in India (2015–2020)", `city_hour.csv`, rows with `City == "Hyderabad"`, column `PM2.5` (the city-level hourly value; timestamps are already IST) | The same `clean()` as `cams`: valid range (0, 999], hourly means, interpolate gaps ≤ 6 h, longer gaps stay missing |

The city-level series is used instead of a single station so that no station is cherry-picked.

## Replay design

All times are local (IST). Forecasts are 24 hours ahead, made once a day at midnight.

- **Replay window:** the last 365 days of each dataset. `replay_end` is the last midnight that still has all 24 following hours in the data, and `replay_start = replay_end − 364 days`.
- **Forecast cache:** for every midnight origin in `[replay_start − 30 d, replay_end]`, every candidate's 24-hour forecast is computed and stored with the actual values.
- **Refits:** every 30 days, starting at `replay_start − 30 d`.
  - At each refit, every candidate is fit on all data strictly before the refit time. Models that use only a recent window (SARIMA 42 days, LSTM 730 days) take it from that data.
  - Forecasts from a refit cover the origins until the next refit.
  - At every origin, a model sees only the history strictly before that origin.
- **Decision points:** `replay_start + 7k` days, for k = 0…51 (52 decisions).
- **Deployment weeks:** each decision is followed by a 7-day deployment week, with the 7 origins after the decision point. The champion's forecasts in that week are what was "served".

### Candidates (6)

| Key | Model | Config |
| --- | --- | --- |
| `snaive` | Seasonal naive, 24 h | `configs/seasonal_naive.yaml` |
| `sarima` | SARIMA(1,0,1)(1,0,1)₂₄, log, 42-day fit | `configs/sarima.yaml` |
| `xgb` | XGBoost d6 lr 0.05 n400, log target | `configs/xgb.yaml` + `log_target: true` |
| `xgb_tuned` | XGBoost, Optuna trial 17 | `configs/xgb_tuned.yaml` |
| `lstm` | LSTM, 2 layers, 7-day window | `configs/lstm.yaml` + `num_layers: 2` |
| `ensemble` | Equal-weight mean of `xgb_tuned` and `sarima` | computed from the cached member forecasts (identical to `EnsembleModel`) |

Hyperparameters are the ones chosen in Phases 3–4 on `cams` CV folds. They are **not re-tuned** for `station`.

### Each decision point *t*

1. **Validation window:** the last *V* origins before *t*.
2. **Score:** compute every candidate's RMSE over its forecasts in the window, pooling all hours and skipping missing actual values.
3. **Challenger:** the best non-champion candidate by validation RMSE.
4. **Policy:** decide whether the challenger replaces the champion.
5. **Skip rule:** if fewer than 50% of the window's actual hours are present, no decision is made and the champion is kept. This is reported as a skipped decision.

**Initial champion,** the same for every policy and every *V*: the candidate with the lowest RMSE on the 30 origins before `replay_start`.

### Policies

| Policy | Rule |
| --- | --- |
| `tau0`, `tau1`, `tau2`, `tau5`, `tau10` | Promote if (champion RMSE − challenger RMSE) / champion RMSE ≥ τ, for τ = 0, 1, 2, 5, 10%. This is `src/promote.py::decide()`. |
| `dm` | One-sided Diebold–Mariano test with the Harvey–Leybourne–Newbold correction. Promote if p < 0.05 that the challenger's loss is lower. The input is the **daily** loss differential (mean squared error per origin across its 24 hours), so the test has *V* observations. Daily forecasts don't overlap in time. |
| `never` | Keep the initial champion all year. |
| `oracle` | For each deployment week, serve the candidate with the lowest RMSE on that week. This uses hindsight and is a reference bound, not a policy. |

**Validation windows:** *V* ∈ {7, 14, 30} days.
**Primary configuration:** `tau2`, *V* = 30, the rule `promote.py` uses today.

## Metrics

For each dataset, policy and *V*:

- **Deployed RMSE:** RMSE over all served forecasts in the 52 deployment weeks, all hours pooled.
- **Swaps:** the number of decisions that changed the champion.
- **Harmful swap:** a swap at *t* after which, over the deployment week that follows, the new champion's RMSE is **higher** than the old champion's would have been.
  - **Harmful-swap rate** = harmful / swaps (undefined when there are 0 swaps).
- **Excess over the oracle:** deployed RMSE / oracle RMSE − 1.
- **Uncertainty:** a week-block bootstrap resamples the 52 deployment weeks with replacement (2,000 resamples, seed 42). It gives 95% percentile CIs for:
  - deployed RMSE
  - the difference in deployed RMSE against `tau0`
  - the difference in deployed RMSE against `never`
- **Significance:** a two-sided Diebold–Mariano test (HLN correction) on daily deployed losses, each policy vs `tau0` and vs `never`.

## Hypotheses

| | Hypothesis | Supported if |
| --- | --- | --- |
| **H1** | A zero threshold swaps too eagerly | `tau0` has more harmful swaps than `tau2` and `tau5` for most (dataset, *V*) cells, **and** `tau2`'s deployed RMSE is not significantly worse than `tau0`'s (DM two-sided p ≥ 0.05, or `tau2` is better) |
| **H2** | A significance test beats fixed thresholds | `dm` has a lower harmful-swap rate than `tau2`, **and** its deployed-RMSE difference vs `tau2` has a bootstrap CI that includes 0 or favours `dm` |
| **H3** | Longer validation windows help | for τ ∈ {0, 1, 2}%, harmful swaps don't increase from *V* = 7 → 14 → 30 |

With 52 decisions per cell, the swap counts are small. The harmful-swap results are reported descriptively, with counts, and the formal tests are on deployed accuracy. Every cell is reported, including any that contradict the hypotheses.

## Known features of the data (stated in advance)

- **`station`:** the last 12 months fall in mid-2019 – mid-2020 and include the COVID-19 lockdown (from 25 Mar 2020). That's a sharp regime shift. It is **kept**, not excluded, and reported as a stress test.
- **`cams`:** these are model estimates, not measurements. The series is smooth and has no gaps.
- **Overlap with Phase 5:** the `cams` replay window overlaps the Phase 5 holdout. The study doesn't use Phase 5's decisions; it recomputes everything from the cache.

## Deviations

None so far.
