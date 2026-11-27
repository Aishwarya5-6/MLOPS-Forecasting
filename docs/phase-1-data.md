# Phase 1: Data + DVC — What Was Done

**Date:** 23 Nov 2026
**Status:** Complete. Both exit checks pass.
**Plan reference:** the original project plan (private), section 8, Phase 1

## Goal

The dataset is versioned and turned into features by a reproducible DVC pipeline.

**Exit checks**

| Check | Result |
| --- | --- |
| Changing a parameter and running `dvc repro` reruns only the affected stages | ✅ Changing `featurize.lags` reran only `featurize`; `prepare` was skipped. Reverting restored the old output from cache with no recompute. |
| `git checkout <old commit>` + `dvc checkout` restores the old data exactly | ✅ Tested in a throwaway clone: v1 (37,729 lines, md5 `4f244f3c…`) → v2 (30,001 lines) → checkout v1 → md5 `4f244f3c…` again. |

## Task checklist

| Task | Status | Notes |
| --- | --- | --- |
| Choose the dataset | ✅ | Hyderabad PM2.5, hourly, from Open-Meteo (see [Data source decision](#data-source-decision)) |
| Download into `data/raw/` | ✅ | `data/raw/pm25_raw.csv`, 37,728 rows, 4 Aug 2022 → 22 Nov 2026 (UTC) |
| EDA notebook | ✅ | [notebooks/01_eda.ipynb](../notebooks/01_eda.ipynb), executed with outputs saved |
| `dvc init`, `dvc add`, commit `.dvc` file | ✅ | `data/raw/pm25_raw.csv.dvc` |
| DVC remote + `dvc push` | ✅ | Local folder `~/dvc-storage/mlops-forecasting` |
| `src/data.py` (load, parse dates, frequency, missing values) | ✅ | |
| `src/features.py` (lags, rolling, calendar, holidays) | ✅ | |
| `params.yaml` | ✅ | |
| `prepare` and `featurize` stages in `dvc.yaml` | ✅ | |
| `dvc repro`, commit `dvc.yaml` + `dvc.lock` | ✅ | |
| Param-change test | ✅ | See exit checks |

## Data source decision

**Plan:** OpenAQ archive (real CPCB/TSPCB station readings).
**Used:** **Open-Meteo Air Quality API**, hourly `pm2_5` at Hyderabad city centre (17.385 N, 78.4867 E).

**Why we switched**
- OpenAQ's S3 archive is public, but finding the Hyderabad station IDs needs the v3 API.
- The v3 API needs a key, and the key needs email verification.
- No verification email arrived from either of the two accounts we tried. A web search found station names but no OpenAQ IDs.

**What Open-Meteo is:** CAMS global atmospheric-composition model estimates (Copernicus), on a grid of roughly 0.4°, about 40 km. Free, no key, licence CC BY 4.0, so archiving it under DVC is allowed. PM2.5 history for this point starts on **4 Aug 2022**; earlier dates return nulls.

**Trade-offs to be honest about (put these in the README and in interviews)**
- These are **model estimates, not sensor readings.** The series is smoother and has **no gaps or spikes**, so the cleaning code currently does nothing.
- **No weekly cycle** (0.9 µg/m³ swing). The model doesn't see weekday traffic well.
- Only about 4.3 years of history, which covers 4 winters and 4 monsoons. That is enough.
- **Still fine for the project:** real seasons, real winter drift, and a live feed for Parts 4–5.

**Switching back to station data later:** `src/download.py` still has the OpenAQ commands (`stations`, `history`). Both sources write the same raw format (`datetime,value` with a UTC offset). Switching needs these steps:
1. Put a working key in `.env` as `OPENAQ_API_KEY`.
2. Run `python -m src.download stations` and pick a station.
3. Run `python -m src.download history --location-id <id>`.
4. Set `data.source` in `params.yaml`.
5. Run `dvc add`, then `dvc repro`.

The cleaning code (gap filling, spike removal) will then actually have work to do.

## Files added or changed

```
.dvc/config                 # DVC config: remote "storage" (default), autostage = true
.dvcignore
params.yaml                 # all shared settings (DVC watches these)
dvc.yaml                    # pipeline: prepare -> featurize
dvc.lock                    # exact hashes of every input/output (commit with code)
data/raw/pm25_raw.csv.dvc   # pointer to the raw data version (in git)
data/raw/.gitignore         # written by DVC: ignores pm25_raw.csv
data/processed/.gitignore   # written by DVC: ignores the parquet outputs
src/config.py               # load_params(): one place that reads params.yaml
src/download.py             # openmeteo | stations | history
src/data.py                 # prepare stage
src/features.py             # featurize stage
notebooks/01_eda.ipynb      # EDA with findings
tests/test_data.py          # 4 tests
tests/test_features.py      # 4 tests (incl. leakage test)
```

New packages: `holidays`, `requests` (already present), `ipykernel`, `nbconvert`, `nbformat`. `requirements.txt` was re-pinned.

## Pipeline

```
data/raw/pm25_raw.csv  --(prepare: src/data.py)-->  data/processed/pm25_hourly.parquet
                       --(featurize: src/features.py)-->  data/processed/features.parquet
```

### `prepare` (`src/data.py`)
1. Read the raw CSV, parse timestamps as UTC, convert to **IST (Asia/Kolkata)**, and drop the timezone. Calendar features then use local hours.
2. Drop invalid readings: `<= 0` or `> 999` µg/m³.
3. Average all readings within each hour.
4. Reindex to a complete hourly range, so every hour exists.
5. Fill gaps up to **6 hours** with time interpolation and flag them `is_imputed`. Longer gaps stay empty and are flagged `is_missing`. Values are never extrapolated past either end.

Output columns: `pm25`, `is_imputed`, `is_missing`. Current data: 37,728 hours, 0% imputed, 0% missing.

### `featurize` (`src/features.py`)

| Feature | Definition | Leakage-safe because |
| --- | --- | --- |
| `lag_k` for k in 1, 2, 3, 6, 12, 24, 48, 168 | value k hours ago | `shift(k)` |
| `roll_mean_w`, `roll_std_w` for w in 6, 24, 168 | mean / std of the previous w hours | `shift(1)` before `rolling` excludes the current hour |
| `hour`, `dayofweek`, `month`, `is_weekend` | from the timestamp | known in advance |
| `is_holiday` | Indian national holidays (incl. Diwali) from the `holidays` package | known in advance |

Output: 37,728 rows × 22 columns. Rows at the start have NaN lags; models drop or handle them.

**Note for Phase 3:** these lags are relative to the *target* hour. To forecast 24 hours ahead, `lag_1`…`lag_12` aren't known yet for most target hours. XGBoost will need either a recursive strategy or "direct" features that only use lags ≥ the horizon step. Decide in Phase 3.

### `params.yaml` (current values)

| Section | Key | Value | Meaning |
| --- | --- | --- | --- |
| `seed` | | 42 | |
| `data` | `source` | open-meteo | |
| | `latitude`, `longitude` | 17.385, 78.4867 | Hyderabad centre |
| | `start_date` | 2022-08-01 | |
| | `raw_file` | data/raw/pm25_raw.csv | |
| | `timezone` | Asia/Kolkata | |
| `prepare` | `min_value` / `max_value` | 0 / 999 | valid range |
| | `max_gap_hours` | 6 | longest gap to interpolate |
| `featurize` | `lags` | [1, 2, 3, 6, 12, 24, 48, 168] | |
| | `rolling_windows` | [6, 24, 168] | |
| `forecast` | `horizon` | 24 | hours ahead |
| | `seasonal_period` | 24 | |
| | `n_folds` | 5 | Phase 2 |
| | `holdout_hours` | 720 | last 30 days, Phase 5 |

## EDA findings

From [notebooks/01_eda.ipynb](../notebooks/01_eda.ipynb).

| Finding | Number | Decision |
| --- | --- | --- |
| Yearly cycle dominates | Month swing 34.8 µg/m³ (Jul ~9, Nov–Jan ~42–44) | `month` feature; walk-forward folds should span seasons |
| Seasons | Monsoon mean 14.8, winter 40.2, post-monsoon 38.3 (max 138) | Oct–Nov rise is the reliable drift event for Part 5 |
| Clear daily cycle | Hour swing 12.3; low ~15:00, peak 20:00–22:00, small bump ~07:00 | `seasonal_period: 24` |
| No weekly cycle | Day-of-week swing 0.9 | Keep `lag_168` / `dayofweek`, but expect little from them |
| Diwali is noisy | 7 days after vs. before: +51%, +110%, -1%, -74%, +124% (2022–2026) | Keep `is_holiday`; don't oversell Diwali |
| Stationary | ADF on last year p < 0.001 | SARIMA can start with d = 0 |
| Strong autocorrelation | lag 1 = 0.98, lag 24 = 0.82 (> lag 12 = 0.70), lag 168 = 0.48 | Seasonal-naive (24 h) will be a strong baseline; judge models on the full 24 h horizon |

## Problems hit and fixes

1. **OpenAQ verification email never arrived.** Switched the source to Open-Meteo. The OpenAQ code is kept for later.
2. **`dvc push` silently skipped the raw data.** Only 2 of 3 files reached the remote, and a fresh clone's `dvc pull` said "Everything is up to date" with no raw file.
   - **Cause:** the root `.gitignore` had `data/*` / `data/raw/*` rules with `!` exceptions. Git understood them, but DVC treated `pm25_raw.csv.dvc` as ignored, so it wasn't part of the pipeline.
   - **Fix:** removed all data rules from the root `.gitignore`. DVC now writes its own `.gitignore` inside `data/raw/` and `data/processed/`, which is the standard DVC layout.
   - **Lesson:** after `dvc push`, check the file count, and test with a fresh clone + `dvc pull`. `dvc data status` showing "deleted"/"new" for a file that exists was the warning sign.
3. **Pandas 3 / statsmodels deprecation warnings:**
   - `DatetimeIndex.dayofweek` → `day_of_week`
   - `adfuller(..., result_object=False)`
   - replaced `infer_freq` with an explicit "all steps are 1 hour" check

## Everyday commands

```bash
source .venv/bin/activate

dvc repro                          # run the pipeline (only changed stages)
dvc status                         # what is out of date?
dvc status -c                      # is the remote in sync with the local cache?
dvc push / dvc pull                # upload / download data versions
dvc dag                            # show the pipeline graph

# Refresh data to yesterday, then version it
python -m src.download openmeteo
dvc add data/raw/pm25_raw.csv && dvc repro && dvc push
git add data/raw/pm25_raw.csv.dvc dvc.lock && git commit -m "Data refresh <date>"

# Go back to an old data version
git checkout <commit> && dvc checkout

# Re-run the EDA notebook
cd notebooks && jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb
```

## Verification

| Check | Result |
| --- | --- |
| `pytest` | 11 passed ✅ |
| `ruff check .` / `ruff format --check .` | Clean ✅ |
| `dvc status` | Up to date ✅ |
| `dvc status -c` | Cache and remote in sync ✅ |
| Fresh clone + `dvc pull` | 3 files fetched ✅ |
| EDA notebook runs top to bottom | 0 warnings ✅ |

## Open items

- **DVC remote is a local folder** (`~/dvc-storage/mlops-forecasting`). A fresh clone on *another* machine can't `dvc pull`, which matters for Phase 6 (reproducibility test) and Part 4 (CI). Before then, move to a cloud remote; Google Drive needs your own OAuth client, and S3 or GCS is simpler. You can copy the existing storage folder across.
- **OpenAQ key:** if verification ever works, consider switching to station data (steps above).
- **W&B:** still blocked (see Phase 0). Needed in Phase 4.

## Next: Phase 2 (evaluation + baselines)

1. `make_folds()` with 5 expanding-window folds plus a 720-hour holdout.
2. `src/evaluate.py` with MAE, RMSE, sMAPE.
3. A `fit` / `predict(horizon)` interface.
4. Naive and seasonal-naive (24 h) baselines.
5. Tests.
6. An `evaluate` stage writing `metrics.json`.
