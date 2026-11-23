# Phase 4: LSTM + Hyperparameter Sweep — What Was Done

**Date:** 23 Nov 2026
**Status:** Complete. The exit check passes.
**Plan reference:** [project-plan.md](project-plan.md), section 8, Phase 4

## Goal

A third model type, and one systematic hyperparameter sweep.

**Change from the plan:** the W&B account is still disabled, so **Optuna replaces W&B Sweeps** and **MLflow is the only tracker**. The plan's goals are all still met:
- Bayesian search: Optuna's TPE sampler.
- 30–50 trials: 40.
- 4–6 hyperparameters: 6.
- Parameter-importance and parallel-coordinates plots.
- A sweep dashboard: in MLflow, as a parent run with 40 nested child runs.

**Exit check:** "you can explain, with the plot, which 2–3 hyperparameters mattered most and why." ✅ See [Sweep findings](#sweep-findings).

## Task checklist

| Plan task | Status | Notes |
| --- | --- | --- |
| LSTM: sliding windows, 1–2 layers, early stopping, fixed seed | ✅ | `src/models/lstm.py`, `configs/lstm.yaml` |
| Same pyfunc interface, logged through `log_run()` | ✅ | 3 runs logged with model artifacts (rerun cleanly, see [Problems](#problems-hit-and-fixes)) |
| Extend `log_run()` to mirror to W&B | ⛔ Skipped | W&B blocked; I didn't add code that can't be tested |
| Check one run looks the same in MLflow and W&B | ⛔ Skipped | same reason |
| Sweep spec: Bayesian, minimise RMSE, 4–6 hyperparameters | ✅ | `sweeps/xgb.yaml`: TPE, 6 parameters |
| Run 30–50 trials | ✅ | 40 trials, 11.9 min |
| Study parameter importance and parallel coordinates; write findings | ✅ | README + below |
| Retrain the best sweep config as a normal tracked run | ✅ | `xgb-tuned-optuna-t17`: RMSE 8.150 (sweep: 8.149) |

## Results

| Model | MAE | RMSE | sMAPE | vs. seasonal naive |
| --- | --- | --- | --- | --- |
| **XGBoost tuned** (`xgb-tuned-optuna-t17`) | **6.21** | **8.150** | 23.3 | **-21.3%** |
| XGBoost Phase 3 (`@champion`) | 6.34 | 8.307 | 23.3 | -19.8% |
| SARIMA Phase 3 best | 6.26 | 8.372 | 23.6 | -19.2% |
| LSTM 2 layers, 7-day window | 6.61 | 8.781 | 24.3 | -15.3% |
| LSTM 1 layer, 14-day window | 6.67 | 8.814 | 24.5 | -14.9% |
| LSTM 1 layer, 7-day window | 6.78 | 8.824 | 24.9 | -14.8% |
| Seasonal naive 24 h | 7.81 | 10.362 | 29.1 | — |

### RMSE per fold

| Fold | Season | XGB tuned | XGB champion | LSTM l2 |
| --- | --- | --- | --- | --- |
| 0 | Post-monsoon | **7.57** | 7.75 | 9.37 |
| 1 | Winter | **11.31** | 11.79 | 10.60 ← best |
| 2 | Summer | **9.55** | 9.57 | 10.17 |
| 3 | Monsoon onset | **8.19** | 8.32 | 9.49 |
| 4 | Monsoon | 4.12 | **4.10** | 4.27 |

The tuned model wins 4 of 5 folds. The LSTM is worst overall but **best in winter (fold 1)**, the hardest season. That's a hint that an ensemble could help, which is a stretch idea.

## LSTM

### Design (`src/models/lstm.py`)

| Aspect | Choice | Why |
| --- | --- | --- |
| Input | last `window` (168) hours; per hour: scaled `log1p(pm25)` + sin/cos of hour and day of week | log stabilises variance (it helped in Phase 3); sin/cos lets midnight sit next to 23:00 |
| Output | one linear head → all 24 hours at once (direct multi-output) | same idea as XGBoost: no feeding predictions back in |
| Network | LSTM (64 hidden, 1–2 layers) → dropout → linear | small, so it trains on a laptop CPU |
| Scaling | mean/std of log values from the **fold's training data only** | no leakage |
| Training data | last `train_days` (730) of each fold's training set, a window every 2 hours | about 8,700 windows |
| Early stopping | the last 10% of windows (in time order) are validation; patience 5, max 50 epochs | validation is inside the training period, so the test window is never used. Training windows whose targets overlap the validation period are dropped |
| Reproducibility | `torch.manual_seed`, a seeded shuffle `Generator`, CPU only | a test proves two fits with the same seed give the same predictions |

### Findings

- **The LSTM beats the baseline by about 15% but trails XGBoost and SARIMA by about 6%.** That's normal for a small dataset with strong, simple seasonality, where hand-built features (same hour yesterday, rolling means) do much of the work an LSTM would have to learn.
- **The variants barely differ** (8.78–8.82). Neither a second layer nor a 14-day window changed much, so the plan's advice to time-box LSTM tuning applies.
- **Speed (clean reruns):** 221 s for 1 layer with a 7-day window, 472 s for 2 layers, 465 s for a 14-day window. That's about 45–95 s per fold, against 2 s per fold for XGBoost. A sweep over the LSTM would take hours, which is why the sweep is on XGBoost.
- **Reproducible:** each LSTM config was run twice, on different days' commits and with the laptop sleeping during one set. Every metric matched to 6 decimal places, fold by fold. This is the first hard evidence for success criterion #1 (same code + data + seed gives the same result).

## Sweep

### How it works (`src/sweep.py`, `sweeps/xgb.yaml`)

```bash
python -m src.sweep sweeps/xgb.yaml [--trials 40]
```

1. Loads the spec: `base_config` (configs/xgb.yaml) + `fixed` (`log_target: true`) + `search_space`.
2. Opens a **parent MLflow run** `sweep-xgb-tpe` (tag `stage=sweep`).
3. Each Optuna trial samples a config, runs the **same `evaluate()` on the same 5 folds**, and logs a **nested child run** `xgb-tpe-trial-NNN` with params, mae/rmse/smape and `fold_rmse`.
4. At the end, the parent gets:
   - `best_rmse`, `best.*` params, `importance.*` metrics (fANOVA)
   - plots: optimisation history, parameter importances, parallel coordinates, slice
   - `trials.csv`
   - the spec and tuned config
5. It writes **`configs/xgb_tuned.yaml`** (the full best config, with the run ID in its header) so it can be retrained with `python -m src.train --model xgb --config configs/xgb_tuned.yaml`.

| Spec field | Value |
| --- | --- |
| Sampler | TPE, seed 42, 10 random startup trials |
| Objective | mean RMSE over the 5 folds (minimise) |
| `max_depth` | int 3–10 |
| `learning_rate` | float 0.01–0.3 (log scale) |
| `n_estimators` | int 100–1200, step 100 |
| `min_child_weight` | float 1–100 (log scale) |
| `subsample` | float 0.5–1.0 |
| `colsample_bytree` | float 0.4–1.0 |
| Fixed | `log_target: true` (Phase 3 showed it helps) |

**In the MLflow UI:** open experiment `forecasting-hyd-pm25` and expand the `sweep-xgb-tpe` row to see its 40 children. Sort the children by `rmse`. Select several and click **Compare** for MLflow's own parallel-coordinates view. The Optuna plots are under the parent run's Artifacts → `plots/`.

### Sweep findings

**Best trial #17: RMSE 8.149.** Its settings are `learning_rate 0.0131`, `n_estimators 1000`, `max_depth 5`, `min_child_weight 4.28`, `subsample 0.69`, `colsample_bytree 0.55`.

| Hyperparameter | fANOVA importance |
| --- | --- |
| `learning_rate` | **0.538** |
| `n_estimators` | **0.194** |
| `min_child_weight` | **0.145** |
| `colsample_bytree` | 0.042 |
| `subsample` | 0.041 |
| `max_depth` | 0.040 |

**The 3 that mattered, and why:**
1. **`learning_rate` (54%).** Every good trial (RMSE ≤ 8.25) has a learning rate between 0.01 and 0.03. Every trial above 0.1 scores 8.45–9.3. With a strong signal like `seas_24`, big steps overfit the noise quickly. Small steps average over many trees, giving smoother, more general corrections.
2. **`n_estimators` (19%).** This goes hand in hand with the learning rate: a small rate needs many trees. The best trials pair about 0.013 with 600–1,100 trees. Low tree counts (100–300) only work with higher rates, and those are worse anyway.
3. **`min_child_weight` (15%).** Values of 2–10 work best, and 25 or more is usually worse. Larger minimum leaf sizes stop the trees from learning rarer conditions, such as winter peaks and festival spikes.

**What didn't matter:** within the searched ranges, depth (3–10), row subsampling and column subsampling move RMSE by less than about 0.05. Any depth from 5 to 7 is fine.

**Convergence:** TPE found the 8.15–8.22 plateau by trial 14. The last 25 trials all land within 0.07 of each other, so 40 trials was enough. Differences of a few hundredths are noise, not signal.

### Caveats (important for Phase 5)

1. **Selection bias:** 8.149 is the *best of 40* on the *same 5 folds used to choose it*, so it's optimistic. Retraining confirmed the number is reproducible (8.150), but not that it generalises. The holdout window (25 Oct → 23 Nov 2026), which the sweep never saw, is the honest test.
2. **The gain is small:** the tuned model is 1.9% better than the Phase 3 champion, **below the 2% promotion threshold** in section 9. Phase 5's `promote.py` will decide on the holdout. "Tuning gave less than 2%" is itself a valid finding.
3. **I did not register the tuned model** as champion. Phase 5 is where a challenger gets promoted by rule, not by hand, and it will be the first real test of `promote.py`.

## Problems hit and fixes

1. **torch and xgboost OpenMP clash (segfault).** Both packages ship their own `libomp`. Loading both in one Python process crashes on macOS, whichever is imported first:
   - torch after xgboost: the segfault happens in LSTM backward.
   - xgboost after torch: the segfault happens in `DMatrix.set_label`.

   **Fix:** the LSTM tests run in a **fresh Python process** (`tests/test_lstm.py` uses `subprocess`). Real runs are unaffected: `train.py`, the sweep and each saved pyfunc load only one model type per process.

   **Keep in mind for Part 3:** a single API process must never load an XGBoost and an LSTM model together. One serving process should serve one model type.
2. **The laptop slept during the first LSTM runs.** Two of them showed wall times of 6,918 s and 3,776 s, although they used only about 15 minutes of CPU. Only `eval_seconds` was affected, not the metrics. **Fix:** long jobs now run under **`caffeinate -i`**, a macOS command that prevents idle sleep (e.g. `caffeinate -i python -m src.sweep sweeps/xgb.yaml`).
3. **The first LSTM runs were tagged `git_dirty=true`.** I was editing sweep files while they trained, and `log_run()` checks git at the *end* of a run. Their LSTM code was in the tagged commit, so the results were valid, but the record wasn't clean.
   - **Resolution:** all 3 were **rerun from a clean commit (`9ece807`) under `caffeinate`**.
   - The reruns are `git_dirty=false`, with realistic times, and their metrics are **identical to 6 decimal places**.
   - The old runs are tagged `stage=superseded` and `superseded_by=<new run id>`. The new runs are tagged `reproduces=<old run id>`. The old runs were kept, not deleted, so the history stays complete.

   | Config | Original run | Clean rerun | RMSE (both) |
   | --- | --- | --- | --- |
   | 1 layer, 7-day window | `37d2407c` | `d9145ff6` | 8.824108 |
   | 2 layers, 7-day window | `a6128a2d` | `9fcaaa80` | 8.781359 |
   | 1 layer, 14-day window | `0710cadf` | `ff6a39f7` | 8.814436 |

   **Lesson:** don't edit tracked files while a run is in progress. Commit first, or work on a branch.
4. **`timeout` isn't available on macOS.** Used `perl -e 'alarm N; exec @ARGV' <cmd>` to cap test runs.
5. **Cosmetic:** matplotlib warns that the Optuna slice plot "includes Axes not compatible with tight_layout", and its colourbar slightly overlaps the last panel. It's readable, so I left it alone.

## Files added or changed

```
src/models/lstm.py        # LSTMModel (new)
src/sweep.py              # Optuna sweep tracked in MLflow (new)
sweeps/xgb.yaml           # sweep spec (new)
configs/lstm.yaml         # (new)
configs/xgb_tuned.yaml    # best sweep config, written by src.sweep (new)
src/tracking.py           # + nested runs, run_tags(), MLFLOW_EXPERIMENT_NAME override, W&B note
src/train.py              # + lstm in build_model
src/models/pyfunc_wrapper.py  # + per-model extra pip packages (torch for LSTM)
tests/test_lstm.py        # 2 LSTM tests in a subprocess (new)
tests/test_models.py      # + sweep search-space test
requirements.txt          # + optuna 5.1.0
README.md                 # results + sweep findings
```

Tests: **37 pass**.

## MLflow state after Phase 4

| What | Count |
| --- | --- |
| Experiment `forecasting-hyd-pm25` runs | 12 (Phase 3) + 3 LSTM + 3 LSTM superseded + 1 tuned XGB + 1 sweep parent + 40 trials = **60** |
| Registry `pm25-forecaster` | v1 = Phase 3 XGBoost, `@champion` (unchanged) |

Tip: in the runs table, filter with `tags.stage = 'dev'` to hide the 40 sweep trials and the 3 superseded LSTM runs.

## Everyday commands

```bash
caffeinate -i python -m src.train --model lstm                 # ~4 min; keep the Mac awake
caffeinate -i python -m src.sweep sweeps/xgb.yaml              # ~12 min, 40 trials
python -m src.sweep sweeps/xgb.yaml --trials 5                 # quick sweep
MLFLOW_EXPERIMENT_NAME=scratch python -m src.sweep sweeps/xgb.yaml --trials 2   # throwaway test
python -m src.train --model xgb --config configs/xgb_tuned.yaml
```

## Next: Phase 5 (registry automation)

1. `promote.py`: register a run's model as `@challenger`, score challenger and champion on the **holdout**, and promote only if RMSE improves by at least 2%. Tag every decision (`promoted_on`, `holdout_rmse`, `previous_champion` / `rejected_reason`).
2. Register seasonal naive as `@baseline`.
3. A one-command rollback.
4. Demo both paths. Natural candidates:
   - **tuned XGBoost** (1.9% better on CV, so it's genuinely uncertain whether it passes on the holdout)
   - **LSTM** (should be rejected)
