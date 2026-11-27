# Phase 5: Registry Automation — What Was Done

**Date:** 27 Nov 2026
**Status:** Complete. The exit check passes, but the registry screenshots still need to be taken by hand (see [Open items](#open-items)).
**Plan reference:** [project-plan.md](project-plan.md), section 8 Phase 5, and section 9 (promotion policy)

## Goal

Promotion to production happens through a script with a clear rule, not by hand.

**Exit check:** "a better model reaches production with one command and no code change; rollback works with one command." ✅
- `python -m src.promote challenger --run-id dd2479f4…` moved `@champion` from v1 to v5.
- `python -m src.promote rollback` moved it back to v1. `rollback --to-version 5` restored v5.
- The same line, `mlflow.pyfunc.load_model("models:/pm25-forecaster@champion")`, loaded the ensemble, then the old XGBoost, then the ensemble again, with no code change ([demo below](#alias-switch-demo)).

## Task checklist

| Plan task | Status | Notes |
| --- | --- | --- |
| `src/promote.py` takes a run ID, registers it, sets `@challenger` | ✅ | Reuses the existing version if that run is already registered |
| Score `@challenger` and `@champion` on the holdout with `evaluate.py` | ✅ | `PyfuncForecaster` adapts the loaded pyfunc to `ForecastModel`, so `evaluate()` scores it exactly like CV |
| Promotion rule: at least 2% lower RMSE | ✅ | `decide()`, a pure function; `--min-improvement` to change it |
| On promotion: move `@champion`; tag `promoted_on`, `holdout_rmse`, `previous_champion` | ✅ | Also `promotion_reason`, `promoted_by`; `@challenger` is cleared |
| On rejection: tag `rejected_reason` | ✅ | Also `rejected_on`; the version keeps `@challenger` |
| Edge case: no champion yet | ✅ | First challenger becomes champion (`previous_champion=none`); tested |
| Register seasonal naive as `@baseline` | ✅ | v2; its holdout score is printed next to every decision |
| `rollback` option | ✅ | `rollback` follows `previous_champion`; `--to-version N` for any version |
| Demo one promotion and one rejection | ✅ | Two rejections and one promotion, below |
| Screenshots of the registry | ⏳ | To be taken by you in the MLflow UI |

Extras that weren't in the plan:
- `--dry-run`: score and decide without touching the registry.
- `status`: lists each alias with its key tags.
- **Data-version guard:** refuses a challenger trained on a different data version (explained below).
- A new model, the XGBoost + SARIMA ensemble, which is the genuine promotion in the demo.

## The registry after Phase 5

`pm25-forecaster`, holdout = 25 Oct 2026 00:00 → 23 Nov 2026 04:00 (29 daily forecasts of 24 h):

| Version | Model | CV RMSE | Holdout RMSE | vs champion | Outcome | Alias |
| --- | --- | --- | --- | --- | --- | --- |
| v1 | XGBoost d6, log target (Phase 3) | 8.307 | 12.081 | | champion until v5 | |
| v2 | Seasonal naive 24 h | 10.362 | 14.662 | | reference | `@baseline` |
| v3 | LSTM, 2 layers | 8.781 | 13.013 | −7.7% | **rejected** | |
| v4 | XGBoost tuned by Optuna | 8.150 | 11.916 | +1.4% | **rejected** (under 2%) | |
| v5 | Ensemble: tuned XGBoost + SARIMA | 7.964 | 11.425 | **+5.4%** | **promoted** | `@champion` |

The current champion is 22.1% better than the baseline on the holdout (and 23.1% better in CV).

### What the demo shows

- **Rejection because it was worse (v3, LSTM).** It was 6% worse than XGBoost in CV and 7.7% worse on the holdout. Both evaluations agree.
- **Rejection because the gain was too small (v4, tuned XGBoost).** This is the case the 2% rule exists for. It was 1.9% better in CV and 1.4% better on the holdout: real but small, and inside the noise between folds (fold RMSEs range from about 4 to 11). Swapping production for that gain isn't worth it.
- **Promotion (v5, ensemble).** It was 4.1% better than v1 in CV and 5.4% better on the holdout. Averaging two different model families gained more than 40 trials of tuning. XGBoost and SARIMA make different errors, so averaging cancels part of them.
- **Holdout scores are higher than CV scores** for every model (about 12 vs 8). The holdout is late October to November, when pollution rises after the monsoon. The CV folds average over all seasons, including the calm monsoon. The *ranking* is the same in both, which is the important part.

### How the ensemble was chosen (and an honesty note)

After the dry runs showed that no existing candidate cleared 2% on the holdout, I added the ensemble as a new candidate. It was justified by CV first: CV RMSE 7.964 beat every single model before it was sent to the holdout.

Still, the holdout had now been looked at for about 8 models in this phase. Each look leaks a little information: if you keep trying candidates until one wins on the holdout, its score becomes optimistic. That's acceptable here (one new candidate, CV-backed, a 5.4% margin), but the rule for the future is to **choose candidates on CV, and send only the chosen one to `promote.py`**. The holdout also moves forward with every data refresh, so the next comparison uses fresh, unseen data.

## How `promote.py` works

```
promote.py challenger --run-id R
  1. check R's prepared_data_version == current dvc.lock   (else: "retrain on current data")
  2. register runs:/R/model -> version V; set @challenger; tag registered_on, cv_rmse, ...
  3. score @challenger, @champion, @baseline on the holdout   (each in its own subprocess)
  4. decide(challenger_rmse, champion_rmse, 0.02)
  5a. promote: @champion -> V; tag promoted_on, previous_champion, promotion_reason; drop @challenger
  5b. reject:  tag rejected_reason, rejected_on; @champion unchanged
```

### Design decisions

- **Score the registered artifact, not a retrained model.** The holdout score comes from loading `models:/pm25-forecaster@challenger` and calling the pyfunc. That is exactly what Part 3 will serve. The model was trained on data up to the holdout start (`train.py` does this), so the holdout is unseen. `PyfuncForecaster.fit` does nothing; `evaluate()` is reused unchanged.
  - Check: seasonal naive scored through the pyfunc gives RMSE 14.662095846014005, and scored directly in Python gives the same, to every digit.
- **Each model is scored in its own Python process.** torch (LSTM) and xgboost each bundle their own libomp. Loading both into one process segfaults on macOS (see Phase 4). The `score` subcommand prints one `HOLDOUT_SCORE_JSON:` line that the parent reads. Each subprocess takes about 1–5 s.
- **Both models are scored on the same window in the same call.** The champion is re-scored every time, rather than reusing an old `holdout_rmse` tag, because the holdout window moves when the data is refreshed. Its tags are updated with the new score.
- **Data-version guard.** If the data has been refreshed since a run was trained, the holdout has moved and might now overlap that run's training data. Promoting it would be an unfair comparison, so `promote.py` refuses and asks for a retrain. The champion doesn't need this check: an older champion was trained on older data, which ends before any newer holdout.
- **2% threshold with a tiny tolerance.** Exactly 2.0% counts as passing. Without the tolerance, floating-point maths turns (10 − 9.8) / 10 into 0.0199999.
- **Rollback is an undo stack.** Each promotion records `previous_champion`, so repeated rollbacks walk back through history. v1 was set by hand in Phase 3 and has no `previous_champion`, so rolling back from v1 refuses with a clear message and suggests `--to-version`.
- **Versions are compared as strings.** MLflow returns `ModelVersion.version` as an int with a SQLite store but as a string over REST. A test caught this.

### Alias switch demo

```
champion = ens(xgb-d5-lr0.013094-n1000-log+sarima-101-101x24-42d-log) | next 3h: [37.1, 40.5, 41.8]
$ python -m src.promote rollback
@champion -> v1 (rolled back from v5)
champion = xgb-d6-lr0.05-n400-log | next 3h: [37.9, 43.9, 44.0]
$ python -m src.promote rollback --to-version 5
@champion -> v5 (was v1)
champion = ens(xgb-d5-lr0.013094-n1000-log+sarima-101-101x24-42d-log) | next 3h: [37.1, 40.5, 41.8]
```

The loading code was the same each time: `mlflow.pyfunc.load_model("models:/pm25-forecaster@champion")`. (At the time of the demo the first message read "rolled back from v1" for the roll-forward too; it now prints "was v1".)

## The ensemble model

- `src/models/ensemble.py` `EnsembleModel(members, weights=None)`: fits every member on the same training data and returns the weighted average of their forecasts. `weights=None` means equal weights.
- `configs/ensemble.yaml` lists members as `{model, config}` pairs. `train.py` builds each member with the same `build_model()`, so any model type can be a member.
- Run `ens-xgbtuned-sarima` (`dd2479f418d04f55bf8330fb3f18db6f`), clean commit `0e805c6`:
  - CV MAE 6.00, RMSE 7.964, sMAPE 22.8
  - fold RMSEs 7.76 / 10.81 / 9.05 / 8.07 / 4.14
  - 29 s to evaluate
- The weights weren't tuned. Tuning them on CV is possible, but equal weights are a robust default and avoid more overfitting.

## Tests (11 new, 48 total)

| File | What it proves |
| --- | --- |
| `tests/test_promote.py` | **Decision rule:** no champion → promote; exactly 2% → promote; 1.9% → reject; worse → reject; NaN → reject; threshold is configurable. **Registry flow** (throwaway SQLite MLflow, fixed fake holdout scores): first challenger becomes champion → 1% better is rejected with `rejected_reason` → 5% better is promoted with `previous_champion` → rollback restores v1 → a second rollback refuses. `--dry-run` changes nothing. A run trained on other data is refused. Re-promoting the champion's own run doesn't create a duplicate version. |
| `tests/test_models.py` | Ensemble returns the weighted average of its members and rejects bad weights. |

The registry tests use a temporary database and never touch the real `mlflow.db`. The whole suite runs in about 8 s.

## Files added or changed

```
src/promote.py              # challenger | baseline | rollback | status | score (new)
src/models/ensemble.py      # EnsembleModel (new)
configs/ensemble.yaml       # tuned XGBoost + SARIMA, equal weights (new)
src/train.py                # + "ensemble" model type
tests/test_promote.py       # (new)
tests/test_models.py        # + ensemble test
README.md                   # holdout column, promotion section, setup without W&B
```

## Everyday commands

```bash
python -m src.promote status
python -m src.promote challenger --run-id <id> --dry-run    # see the decision first
python -m src.promote challenger --run-id <id>              # do it
python -m src.promote rollback                              # undo the last promotion
python -m src.promote rollback --to-version <N>
```

In the MLflow UI: **Models → pm25-forecaster** shows the aliases. Click a version to see its tags (decision, reason, scores).

## Open items

- **Screenshots:** open http://127.0.0.1:5000 → Models → `pm25-forecaster`. Capture:
  - the version list with aliases
  - v5's tags (promotion)
  - v4's tags (rejection with reason)

  Save them in `docs/images/` for the Phase 6 write-up.
- **Rejection wording:** a model that is *worse* (v3) gets the reason "not enough improvement…". The numbers make it clear, but a separate "worse than champion" message would read better. It's cosmetic and was left as-is so the recorded tags match the code.
- **Holdout re-use:** see the honesty note above. In Part 4 (CI), `promote.py` should run only on the candidate the training workflow picked.
- **DVC remote is still local:** this needs to change before Phase 6's fresh-machine reproducibility test.

## Next: Phase 6 (proof + write-up)

1. Fresh-clone reproducibility test: `dvc pull` → `dvc repro` → the same metrics. This needs a cloud DVC remote, or a test on this machine with a clean clone.
2. Remaining tests, plus a final `pytest` and `ruff` pass.
3. README for a recruiter:
   - architecture diagram
   - results table
   - screenshots (MLflow runs, sweep, registry)
   - the "X% better than baseline" line: **22% lower RMSE on the untouched holdout**
4. Tick the plan's success criteria, with evidence for each.
