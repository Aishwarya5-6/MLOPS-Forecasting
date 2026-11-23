# End-to-End MLOps Forecasting Platform — Project Documentation

Nov 23, 2026 · @Aishwarya

## 0. Platform roadmap

The major project is a live forecasting service that tracks, optimizes, serves, monitors and retrains its own models, built in five parts over about 14 weeks part-time. Part 1 (sections 1–11) is the foundation; Parts 2–5 (section 12) turn it into a running system.

&#91;embedded content: platform roadmap · 5 parts, 5 gates, about 14 weeks\]

Gates keep the project from becoming five half-finished pieces: each part ends with something that works and can be shown.

### Recommended use case: Hyderabad air quality

Forecast PM2.5 (or AQI) for the next 24 hours at one or more Hyderabad monitoring stations.

- **Real, updating data:** new readings arrive hourly, so ingestion, scheduled forecasts and drift monitoring are meaningful rather than simulated.
- **Real drift:** monsoon, winter inversions and festival periods such as Diwali shift the data, which is exactly what Part 5 must catch.
- **Public value:** a live dashboard anyone in Hyderabad can open, which is easy to explain in interviews.
- **Data sources (checked 23 Nov 2026):** history from the OpenAQ archive (hourly CPCB data for Hyderabad stations, 2019 onward) is reliable. The live feed is the risk: OpenAQ's CPCB readings were reported stalled from 7 Oct 2026, so test it with a real query before Part 4. Backups: CPCB's own station feed, and the WAQI API (live only; its terms forbid archiving or redistributing the data). Build ingestion with a fallback source and a stale-data alert.

### Why this counts as a major project

| What reviewers look for | Where this project shows it |
| --- | --- |
| Something live that people can use | Public API and dashboard (Part 3) |
| Real, messy, changing data | Hourly ingestion with data checks (Part 4) |
| Engineering beyond the model | Registry, CI/CD, containers, monitoring (Parts 1, 3–5) |
| Numbers you can quote | RMSE vs. baseline, latency speedup, load-test results, drift-to-retrain time |
| Systems thinking | Automatic, logged, reversible promotion decisions (Parts 1 and 5) |

**Resume line for the full platform:** *Built and deployed an end-to-end MLOps platform forecasting Hyderabad air quality: DVC + MLflow tracking and registry, ONNX/TensorRT-optimized models served via FastAPI and Docker, GitHub Actions CI/CD, and Evidently + Grafana drift monitoring with automated retraining.* Add your real numbers once you have them.

## 1. Project overview

A reproducible forecasting pipeline that trains, tracks, compares and versions three models (SARIMA, XGBoost, LSTM) on a public time-series dataset, using MLflow, Weights & Biases and DVC. It is Part 1 of a five-part platform (section 0) that ends as a live, monitored, self-retraining forecasting service on a real data feed. Sections 1–11 detail Part 1; section 12 covers Parts 2–5.

**Why this project**

- **Resume:** shows production ML skills (experiment tracking, data versioning, model registry), not only model building.
- **Learning:** hands-on practice with the tools MLOps, ML engineering and forward deployed engineering roles ask for.
- **Foundation:** every later project in the series reuses this repo, its tracking setup and its registry.

**In one line:** any model in production can be traced back to the exact code, data, parameters and metrics that produced it, and swapped with a single alias change.

**Estimated effort:** Part 1 about 4 weeks; the full platform about 14 weeks part-time (2–3 hours a day).

## 2. Problem statement and success criteria

Most student ML projects end as a notebook: results can't be reproduced, runs aren't compared fairly, and nobody can say which model version is "the real one". This project fixes that for a forecasting task.

**Problem:** Given a public daily or hourly time series, forecast the next *h* steps, and manage the full model lifecycle so every result is reproducible and every deployment decision is traceable.

**Success criteria**

| # | Criterion | How it is checked |
| --- | --- | --- |
| 1 | Reproducible | Fresh clone + `dvc pull` + `dvc repro` gives the same metrics (within random-seed tolerance) |
| 2 | Fully tracked | Every run in MLflow has params, metrics, artifacts, git commit and data version |
| 3 | Fair comparison | All models use the same walk-forward splits and metric code |
| 4 | Beats baseline | At least one model beats the seasonal-naive baseline on RMSE |
| 5 | Tuned | One W&B sweep of 30+ trials, with parameter-importance analysed |
| 6 | Versioned and promotable | Best model registered; `@champion` / `@challenger` aliases set by a promotion script |
| 7 | Documented | README with architecture, results table, dashboard screenshots and a 2-minute demo video |

## 3. Requirements

### Functional requirements

| ID | Requirement |
| --- | --- |
| FR1 | Load a raw dataset, clean it and split it with walk-forward (rolling-origin) validation |
| FR2 | Version raw and processed data with DVC, stored in a DVC remote |
| FR3 | Run the pipeline as DVC stages: prepare → featurize → train → evaluate |
| FR4 | Train four model types from one entry point: seasonal-naive baseline, SARIMA, XGBoost, LSTM |
| FR5 | Configure every run from YAML (`params.yaml` / `configs/`), with no hard-coded hyperparameters |
| FR6 | Log params, metrics, plots, config, git commit and DVC data hash to MLflow for every run |
| FR7 | Mirror the same runs to W&B through one shared `log_run()` helper |
| FR8 | Run a W&B hyperparameter sweep for at least one model |
| FR9 | Save each model as an MLflow `pyfunc` with a signature and input example |
| FR10 | Register models in the MLflow Model Registry and manage them with aliases (`@champion`, `@challenger`) |
| FR11 | Provide `promote.py`, which compares challenger and champion on a holdout window and moves the alias only if the challenger is better |
| FR12 | Load the production model anywhere only through `models:/<name>@champion` |

### Non-functional requirements

- **Reproducibility:** fixed seeds, pinned package versions, every run linked to a git commit and a data version.
- **Fairness:** one shared evaluation function for every model.
- **Runs on a laptop:** CPU only, full pipeline in under 30 minutes (excluding sweeps).
- **Code quality:** modular `src/` package, type hints, basic tests with `pytest`, formatted with `ruff` or `black`.
- **No secrets in git:** API keys in a `.env` file or environment variables, listed in `.gitignore`.

### Prerequisite knowledge

- Python, pandas, NumPy, scikit-learn
- Git basics: commit, branch, checkout
- Time-series basics: seasonality, stationarity, lags (you already have this from SARIMA work)
- PyTorch basics, enough for a small LSTM
- Command line and virtual environments

### Software and accounts

| Item | Purpose | Cost |
| --- | --- | --- |
| Python 3.10+ with `venv` or `uv` | Environment | Free |
| Git + a GitHub repo | Code versioning, portfolio | Free |
| MLflow | Tracking server and model registry, run locally with SQLite | Free, open source |
| Weights & Biases account | Dashboards and sweeps | Free personal / academic tier |
| DVC | Data and pipeline versioning | Free, open source |
| DVC remote (Google Drive, or a local folder) | Stores data versions | Free |
| statsmodels, xgboost, torch, pandas, scikit-learn, matplotlib | Models and plots | Free |
| pytest, ruff | Tests and linting | Free |

**Hardware:** any laptop with 8 GB+ RAM. A GPU is not needed for this project (it is for project #2, TensorRT).

## 4. MVP definition

The MVP is the smallest version that already proves the core idea: one dataset, data in DVC, two models plus a baseline tracked in MLflow, and one registered champion. Aim to reach it by the end of week 2.

### In scope for the MVP

- One dataset, versioned with DVC
- Seasonal-naive baseline, SARIMA and XGBoost
- Shared walk-forward evaluation (MAE, RMSE, sMAPE)
- MLflow tracking server (SQLite backend) logging params, metrics, plots and models
- Best model registered, with the `@champion` alias set by hand
- A README with how to run it and a results table

### Out of scope for the MVP (added after)

- LSTM
- W&B logging and sweeps
- Automated `promote.py` with champion/challenger comparison
- Tests, linting, demo video

### MVP acceptance checklist

- [ ] `dvc repro` runs the whole pipeline from raw data to metrics
- [ ] MLflow UI shows at least 3 runs per model type, sortable by RMSE
- [ ] Every run has its git commit and DVC data hash logged
- [ ] At least one model beats the seasonal-naive baseline, or the README explains why none did
- [ ] `mlflow.pyfunc.load_model("models:/demand-forecaster@champion")` returns predictions

### After the MVP (full version)

1. Add the LSTM, wrapped as a `pyfunc`
2. Add W&B through `log_run()`, then run a sweep of 30–50 trials
3. Write `promote.py` for automated champion/challenger promotion
4. Add tests, linting and the reproducibility check from a fresh clone
5. Finish the README, screenshots and demo video

### Stretch goals (optional)

- Probabilistic forecasts (prediction intervals) logged as artifacts
- A small Streamlit page that loads `@champion` and plots forecasts
- Run the MLflow server in Docker (a head start on project #3)

## 5. Dataset and evaluation design

Use a public dataset with clear seasonality; do not use data from a past internship, since it usually belongs to the company.

### Dataset options

| Option | Frequency | Why it fits |
| --- | --- | --- |
| Indian electricity demand (Grid-India daily reports) | Daily | Strong weekly and yearly seasonality, India-relevant |
| Hyderabad air quality (CPCB station data) | Hourly / daily | Local, seasonal, has missing values to handle |
| Rossmann Store Sales (Kaggle) | Daily | Clean, well known, many stores to scale to |
| M5 Forecasting (Kaggle) | Daily | Industry-standard retail benchmark |

Open question: pick one before Phase 1. A single series is enough for the MVP.

### Splitting: walk-forward validation

- Never shuffle. Train on the past, test on the future.
- Use 3–5 rolling folds: each fold trains on all data up to a cut-off and tests on the next *h* steps.
- Keep the most recent window as a final **holdout**, used only by `promote.py`.

### Metrics

- **MAE** and **RMSE:** main comparison metrics (RMSE is the primary sort key).
- **sMAPE:** scale-free comparison; avoid plain MAPE, which breaks near zero.
- Log metrics per fold and their average.

### Baselines

- **Naive:** next value = last value.
- **Seasonal-naive:** next value = value one season ago (for example, same day last week).

### Leakage rules

- Fit scalers, encoders and imputers on training data only, per fold.
- Build lag and rolling features only from past values.
- Never tune hyperparameters on the holdout window.

## 6. System architecture and tech stack

Data flows left to right through DVC and training, every run is logged once through `log_run()`, and only the registry decides which model is in production.

&#91;embedded content: pipeline architecture · raw data to production model\]

The registry is the hand-off point: `promote.py` moves the `@champion` alias, and everything downstream loads by that alias, never by file path.

### Tool roles

| Tool | Role in this project | Not used for |
| --- | --- | --- |
| Git | Code and pipeline definitions (`dvc.yaml`, `dvc.lock`) | Data files, model files |
| DVC | Data versions and pipeline stages | Experiment comparison |
| MLflow Tracking | System of record for every run | Sweeps |
| MLflow Model Registry | Model versions and aliases | Storing raw data |
| Weights & Biases | Interactive dashboards and hyperparameter sweeps | Deciding what is in production |
| statsmodels / XGBoost / PyTorch | SARIMA / gradient boosting / LSTM | — |

## 7. Repository structure and conventions

```
mlops-forecasting/
├── data/
│   ├── raw/              # tracked by DVC, not git
│   └── processed/        # DVC pipeline output
├── src/
│   ├── data.py           # load, clean, walk-forward splits
│   ├── features.py       # lags, rolling stats, calendar features
│   ├── models/
│   │   ├── baseline.py
│   │   ├── sarima.py
│   │   ├── xgb.py
│   │   ├── lstm.py
│   │   └── pyfunc_wrapper.py
│   ├── evaluate.py       # shared metrics and walk-forward loop
│   ├── tracking.py       # log_run() → MLflow + W&B
│   ├── train.py          # entry point: python -m src.train --model xgb
│   └── promote.py        # champion / challenger promotion
├── configs/              # one YAML per model
├── sweeps/               # W&B sweep YAML files
├── tests/                # pytest
├── dvc.yaml              # pipeline stages
├── params.yaml           # shared parameters DVC watches
├── requirements.txt      # pinned versions
├── .env.example          # WANDB_API_KEY=, MLFLOW_TRACKING_URI=
├── .gitignore            # data/, mlruns/, mlflow.db, .env
└── README.md
```

### Conventions

- **Experiment name:** `forecasting-<dataset>`, one per dataset.
- **Run name:** `<model>-<short description>`, for example `xgb-lags14-depth6`.
- **Run tags:** `model_type`, `git_commit`, `data_version`, `stage` (`dev` or `sweep`).
- **Registered model name:** `demand-forecaster` (or `<dataset>-forecaster`).
- **Commits:** small and descriptive; commit `dvc.lock` together with the code that changed it.
- **Never commit:** data files, `mlruns/`, `mlflow.db`, `.env`, model binaries.

## 8. Implementation phases and timeline

Six phases over about 4 weeks; the MVP is complete at the end of Phase 3 (week 2). Each phase ends with a check you can demonstrate.

| Phase | When | Work | Done when |
| --- | --- | --- | --- |
| 0. Setup | Days 1–3 | MLflow and W&B quickstarts; create repo, venv, folder structure, `.gitignore`, pinned requirements | `python -m src.train --help` runs; MLflow server starts with SQLite |
| 1. Data + DVC | Days 4–7 | Choose dataset; `dvc init`, `dvc add`, set remote, `dvc push`; write `prepare` and `featurize` stages in `dvc.yaml` | Changing raw data and running `dvc repro` reruns only affected stages; old data restores with `git checkout` + `dvc checkout` |
| 2. Evaluation + baselines | Days 8–10 | `evaluate.py` with walk-forward folds and MAE / RMSE / sMAPE; naive and seasonal-naive baselines | Baseline scores logged; every model will call the same function |
| 3. Models in MLflow (MVP) | Days 11–14 | SARIMA and XGBoost; `log_run()` to MLflow; `pyfunc` wrapper; register best model, set `@champion` by hand | MVP checklist (section 4) all ticked |
| 4. LSTM + W&B | Days 15–20 | Add LSTM; mirror logs to W\&B; sweep of 30–50 trials; read parameter importance | Sweep dashboard shows which hyperparameters mattered |
| 5. Registry automation | Days 21–23 | `promote.py`: new model gets `@challenger`, compared with `@champion` on holdout, alias moves only if better; reasons in tags | Promoting a better model changes production with one alias move and no code change |
| 6. Proof + write-up | Days 24–28 | Tests, linting; fresh-clone reproducibility test; README, architecture diagram, screenshots, demo video | Someone else can clone and reproduce your results table |

### First commands (Phase 0–1)

```bash
python -m venv .venv && source .venv/bin/activate
pip install mlflow wandb dvc statsmodels xgboost torch pandas scikit-learn matplotlib pytest ruff
pip freeze > requirements.txt
git init && dvc init
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --port 5000
dvc add data/raw/<your_file>.csv
git add data/raw/<your_file>.csv.dvc .gitignore && git commit -m "Track raw data with DVC"
```

## Phase-by-phase guide

Work through one phase at a time and tick tasks as you go. Don't start the next phase until the current one's exit check passes.

### Phase 0: Setup (days 1–3)

**Goal:** a clean, working project skeleton with MLflow running locally.

**Learn first:** MLflow quickstart (about 1 hour), W&B quickstart (about 1 hour), Git branching basics.

**Tasks**

- [ ] Create a GitHub repo `mlops-forecasting` and clone it
- [ ] Create a virtual environment (`venv` or `uv`) and install the packages from section 8
- [ ] Pin versions with `pip freeze > requirements.txt`
- [ ] Create the folder structure from section 7, with empty `__init__.py` files in `src/`
- [ ] Write `.gitignore`: `data/`, `mlruns/`, `mlflow.db`, `.env`, `.venv/`
- [ ] Create `.env.example` with `WANDB_API_KEY=` and `MLFLOW_TRACKING_URI=http://127.0.0.1:5000`
- [ ] Sign up for W&B and run `wandb login`
- [ ] Start the MLflow server with the SQLite backend and open `http://127.0.0.1:5000`
- [ ] Write `src/train.py` with `argparse` (`--model`, `--config`) that just prints its arguments
- [ ] Log one dummy run (one param, one metric) to MLflow to prove the connection
- [ ] First commit: "Project skeleton"

**Deliverables:** repo skeleton, `requirements.txt`, a running MLflow server, one dummy run.

**Exit check:** `python -m src.train --model dummy` creates a run you can see in the MLflow UI.

### Phase 1: Data + DVC (days 4–7)

**Goal:** the dataset is versioned and turned into features by a reproducible DVC pipeline.

**Learn first:** DVC Get Started (data versioning + pipelines), basic exploratory data analysis for time series.

**Tasks**

- [ ] Choose the dataset (section 5) and download it into `data/raw/`
- [ ] Explore it in a notebook: date range, frequency, missing values, seasonality plot
- [ ] `dvc init`, then `dvc add data/raw/<file>`; commit the `.dvc` file
- [ ] Set up a DVC remote (`dvc remote add -d storage <path or gdrive://...>`) and run `dvc push`
- [ ] Write `src/data.py`: load, parse dates, set frequency, fill or flag missing values
- [ ] Write `src/features.py`: lag features, rolling means, calendar features (day of week, month, holidays)
- [ ] Put shared settings in `params.yaml`: forecast horizon, number of folds, lags, seasonal period
- [ ] Define `prepare` and `featurize` stages in `dvc.yaml`, with `deps`, `params` and `outs`
- [ ] Run `dvc repro` and commit `dvc.yaml` + `dvc.lock`
- [ ] Test it: change one parameter, rerun, and confirm only affected stages run

**Deliverables:** versioned raw data, `data/processed/` outputs, `dvc.yaml`, `dvc.lock`, an EDA notebook.

**Exit check:** `git checkout <old commit>` + `dvc checkout` restores the old data exactly.

### Phase 2: Evaluation + baselines (days 8–10)

**Goal:** one shared, leakage-free evaluation function, plus baseline scores every model must beat.

**Learn first:** walk-forward / rolling-origin validation (Hyndman's *Forecasting: Principles and Practice*, chapter on evaluating forecast accuracy).

**Tasks**

- [ ] Write `make_folds()` in `src/data.py`: 3–5 expanding-window folds, plus a final holdout window kept aside
- [ ] Write `src/evaluate.py` with `mae`, `rmse`, `smape`, and `evaluate(model, folds)` that returns per-fold and average scores
- [ ] Define one model interface every model will follow: `fit(train)` and `predict(horizon)`
- [ ] Implement `NaiveModel` and `SeasonalNaiveModel` in `src/models/baseline.py`
- [ ] Write tests: folds never overlap, test always comes after train, metrics match hand-calculated values
- [ ] Add an `evaluate` stage to `dvc.yaml` that writes `metrics.json`
- [ ] Record baseline scores in the README draft

**Deliverables:** `evaluate.py`, baseline models, tests, baseline metrics.

**Exit check:** `pytest` passes, and `dvc metrics show` prints baseline MAE, RMSE and sMAPE.

### Phase 3: Models in MLflow — MVP (days 11–14)

**Goal:** SARIMA and XGBoost trained, tracked and compared in MLflow, with one registered champion. This completes the MVP.

**Learn first:** MLflow tracking API (`log_param`, `log_metric`, `log_artifact`), MLflow `pyfunc` custom models, model signatures.

**Tasks**

- [ ] Write `src/tracking.py` with `log_run(params, metrics, artifacts, model)`; it also logs the git commit and the DVC data hash as tags
- [ ] Implement SARIMA in `src/models/sarima.py` (statsmodels `SARIMAX`), config in `configs/sarima.yaml`
- [ ] Implement XGBoost in `src/models/xgb.py` using the Phase 1 lag and calendar features, config in `configs/xgb.yaml`
- [ ] Write `src/models/pyfunc_wrapper.py` so every model is saved with the same `predict()` interface
- [ ] Log a forecast-vs-actual plot and a residual plot as artifacts for each run
- [ ] Run each model 3+ times with different settings
- [ ] In the MLflow UI, compare runs and sort by RMSE
- [ ] Register the best run as `demand-forecaster` and set its `@champion` alias by hand
- [ ] Load it with `mlflow.pyfunc.load_model("models:/demand-forecaster@champion")` and predict
- [ ] Write a first README: how to run, results table (baselines vs SARIMA vs XGBoost)

**Deliverables:** two tracked model types, a registered champion, README v1.

**Exit check:** every item in the MVP acceptance checklist (section 4) is ticked.

### Phase 4: LSTM + W&B sweeps (days 15–20)

**Goal:** a third model type, every run mirrored to W&B, and one systematic hyperparameter sweep.

**Learn first:** PyTorch LSTM for sequence forecasting (sliding windows), W&B `wandb.init` / `wandb.log`, W&B Sweeps (Bayesian search).

**Tasks**

- [ ] Implement the LSTM in `src/models/lstm.py`: sliding-window dataset, small model (1–2 layers), early stopping, fixed seed
- [ ] Wrap it with the same `pyfunc` interface and log it through `log_run()`
- [ ] Extend `log_run()` to also send params, metrics and plots to W&B (same run name in both tools)
- [ ] Check that one run looks the same in MLflow and W&B
- [ ] Write `sweeps/xgb.yaml` (or `lstm.yaml`): method `bayes`, metric `rmse` minimised, 4–6 hyperparameters with ranges
- [ ] Run `wandb sweep` and `wandb agent` for 30–50 trials
- [ ] Study the parallel-coordinates and parameter-importance plots; write 3–5 bullet findings in the README
- [ ] Retrain the best sweep config as a normal tracked run

**Deliverables:** LSTM, W&B project with all runs, sweep dashboard, written findings.

**Exit check:** you can explain, with the W&B plot, which 2–3 hyperparameters mattered most and why.

### Phase 5: Registry automation (days 21–23)

**Goal:** promotion to production happens through a script with a clear rule, not by hand.

**Learn first:** `MlflowClient` registry methods (aliases and model version tags); section 9 of this doc.

**Tasks**

- [ ] Write `src/promote.py` that takes a run ID, registers it, and sets `@challenger`
- [ ] Load `@challenger` and `@champion`, score both on the holdout window with `evaluate.py`
- [ ] Apply the promotion rule from section 9 (at least 2% lower RMSE)
- [ ] On promotion: move `@champion`, tag `promoted_on`, `holdout_rmse`, `previous_champion`
- [ ] On rejection: tag the challenger with `rejected_reason`
- [ ] Handle the edge case where no champion exists yet
- [ ] Register the seasonal-naive model and give it `@baseline`
- [ ] Write a `rollback` option that points `@champion` back to `previous_champion`
- [ ] Demo both paths: one promotion and one rejection, with screenshots of the registry

**Deliverables:** `promote.py`, registry with aliases and tags, screenshots.

**Exit check:** a better model reaches production with one command and no code change; rollback works with one command.

### Phase 6: Proof + write-up (days 24–28)

**Goal:** prove the project is reproducible and present it so a recruiter understands it in 2 minutes.

**Learn first:** writing a good README (look at 2–3 popular MLOps repos on GitHub for structure).

**Tasks**

- [ ] Add remaining tests (data loading, features, `promote.py` decision logic); all pass
- [ ] Run `ruff check .` and fix warnings
- [ ] Reproducibility test: clone into a new folder, fresh environment, `dvc pull`, `dvc repro`; compare metrics with your table
- [ ] Final README: problem, architecture diagram, setup, how to run each phase, results table, sweep findings, lessons learned
- [ ] Add screenshots: MLflow run comparison, registry with aliases, W&B sweep dashboard
- [ ] Record a 2-minute demo video (problem → pipeline → results → promotion) and link it in the README
- [ ] Write the resume bullet with your real numbers (section 11)
- [ ] Optional: a LinkedIn post about what you learned

**Deliverables:** public repo, final README, screenshots, demo video, resume bullet.

**Exit check:** every box in the Definition of done (section 11) is ticked.

## 9. Model registry and promotion policy

Production is whatever version the `@champion` alias points to; a new version only takes over by beating it on the holdout window. This uses aliases, the current MLflow approach, because fixed registry stages (Staging / Production) were deprecated in MLflow 2.9.

### Aliases

| Alias | Meaning |
| --- | --- |
| `@champion` | The version used in production; loaded by every downstream script |
| `@challenger` | The newest candidate, waiting to be compared |
| `@baseline` | The seasonal-naive model, kept as a reference point |

### Promotion rule (`promote.py`)

1. Register the new model version and give it `@challenger`.
2. Score `@challenger` and `@champion` on the same holdout window.
3. Promote only if the challenger's RMSE is lower by at least 2%, so tiny random gains don't trigger a swap.
4. If promoted: move `@champion` to the new version; tag it with `promoted_on`, `holdout_rmse` and `previous_champion`.
5. If not: leave `@champion` unchanged; tag the challenger with `rejected_reason`.
6. No champion exists yet: the first challenger becomes champion.

### Key API calls

```python
from mlflow import MlflowClient
client = MlflowClient()
client.set_registered_model_alias("demand-forecaster", "challenger", new_version)
champ = client.get_model_version_by_alias("demand-forecaster", "champion")
client.set_registered_model_alias("demand-forecaster", "champion", new_version)
client.set_model_version_tag("demand-forecaster", new_version, "holdout_rmse", str(rmse))

model = mlflow.pyfunc.load_model("models:/demand-forecaster@champion")
```

Rollback is one call: point `@champion` back at `previous_champion`.

## 10. Risks, pitfalls and mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Spending weeks tuning the LSTM | Pipeline never gets finished | Time-box models; the pipeline is the goal, not the best score |
| Data leakage in time series | Results look great but are fake | Walk-forward splits; fit transforms per fold; leakage rules in section 5 |
| Registry fails with plain file storage | Can't register or alias models | Run the MLflow server with a SQLite backend from day one |
| Large files committed to git | Bloated repo, push failures | `data/`, `mlruns/`, `mlflow.db` in `.gitignore` before the first commit |
| MLflow and W&B logs drift apart | Confusing, duplicated records | One `log_run()` helper; MLflow is the record, W&B is for dashboards and sweeps |
| Different model types load differently | Promotion and serving code breaks | Wrap every model as an MLflow `pyfunc` with one `predict()` interface |
| Unpinned package versions | Can't reproduce results later | `requirements.txt` with exact versions; log versions per run |
| API key pushed to GitHub | Account compromised | Keys only in `.env`; `.env` in `.gitignore`; commit `.env.example` instead |
| No model beats the baseline | Feels like failure | Report it honestly; a strong baseline is a valid finding |
| Using company data from an internship | Confidentiality breach | Public datasets only |
| Scope creep into serving or monitoring | Project #1 never closes | Park those ideas for projects #3–#5 |

## 11. Definition of done, deliverables and resources

### Definition of done

- [ ] All 7 success criteria in section 2 met
- [ ] Fresh clone → `dvc pull` → `dvc repro` reproduces the results table
- [ ] `promote.py` demonstrated once with a promotion and once with a rejection
- [ ] Tests pass and linting is clean
- [ ] Public GitHub repo with README, diagram, screenshots and demo video link

### Deliverables

1. GitHub repository (public)
2. README: problem, architecture diagram, how to run, results table, lessons learned
3. Screenshots: MLflow runs comparison, model registry with aliases, W&B sweep dashboard
4. 2-minute demo video
5. Optional: a short blog post on LinkedIn or Medium explaining the design choices

### Resume bullet

*Built a reproducible time-series forecasting pipeline (DVC, MLflow, Weights & Biases) comparing SARIMA, XGBoost and LSTM with walk-forward validation; automated champion/challenger promotion through the MLflow Model Registry.* Add your best result once you have it, for example "X% lower RMSE than the seasonal-naive baseline".

### Learning resources

- [MLflow Model Registry docs](https://mlflow.org/docs/latest/ml/model-registry): registering models, aliases, tags
- [MLflow issue on deprecating registry stages](https://github.com/mlflow/mlflow/issues/10336): why aliases replaced stages
- [DVC Get Started](https://dvc.org/doc/start): data versioning and pipelines
- [W&B Sweeps guide](https://docs.wandb.ai/guides/sweeps): hyperparameter search
- [Forecasting: Principles and Practice](https://otexts.com/fpp3/), Hyndman & Athanasopoulos: time-series evaluation and baselines

## 12. Parts 2–5: phase guides

Each part starts only after the previous part's gate is met (section 0). Same format as Part 1: goal, learn first, tasks, deliverables, exit check.

### Part 2: Optimize (weeks 5–6)

**Goal:** export the models to ONNX, accelerate the LSTM with TensorRT, and measure the speed vs. accuracy trade-off.

**Learn first:** ONNX format and ONNX Runtime; TensorRT basics (engines, FP16, INT8 calibration). TensorRT needs an NVIDIA GPU; a free Colab T4 works.

**Phase 2a: ONNX export (week 5)**

- [ ] Export the LSTM with `torch.onnx.export` (dynamic batch size)
- [ ] Export XGBoost with `onnxmltools`
- [ ] Check outputs match: ONNX vs. original predictions within a small tolerance (for example 1e-4)
- [ ] Run both with ONNX Runtime on CPU; log the ONNX files as MLflow artifacts on the same run

**Phase 2b: TensorRT and benchmarks (week 6)**

- [ ] Build TensorRT engines for the LSTM in FP32, FP16 and INT8
- [ ] Write `bench.py`: latency (p50, p95), throughput, memory, RMSE for every version, at batch sizes 1, 32, 256
- [ ] Log all benchmark results to MLflow as one comparison experiment
- [ ] Pick the serving format and write down why

**Deliverables:** ONNX models, TensorRT engines, benchmark table and chart.

**Exit check (gate 2):** ONNX outputs match the originals, and the README has a benchmark table such as "PyTorch 12 ms → TensorRT FP16 2 ms, RMSE change +0.3%". Expect smaller gains than on large vision models, because forecasting models are small; report what you measure.

### Part 3: Serve (weeks 7–9)

**Goal:** the `@champion` model runs as a containerized API in the cloud, with a public dashboard showing live forecasts.

**Learn first:** FastAPI, Pydantic, Docker and docker-compose, one cloud host (Render, Railway, or an AWS / GCP free-tier VM), Locust for load testing.

**Phase 3a: API (week 7)**

- [ ] `api/main.py` with `/predict` (horizon in, forecast out), `/health` and `/model-info` (version and alias)
- [ ] Load the model by `models:/<name>@champion` at startup; add a `/reload` endpoint for after a promotion
- [ ] Validate inputs with Pydantic; return clear errors
- [ ] Log every request and prediction (needed for monitoring in Part 5)
- [ ] Tests with FastAPI's `TestClient`

**Phase 3b: Containers and deploy (week 8)**

- [ ] `Dockerfile` for the API (slim base image, pinned versions)
- [ ] `docker-compose.yml` running the API, the MLflow server and a Postgres database for MLflow
- [ ] Deploy to the cloud host; store secrets in its environment settings
- [ ] Optional: serve the TensorRT engine through NVIDIA Triton if you have GPU access

**Phase 3c: Dashboard and load test (week 9)**

- [ ] Streamlit dashboard: latest forecast vs. actuals, model version in use
- [ ] Locust load test: find requests per second at p95 latency under 200 ms
- [ ] Record results in the README

**Deliverables:** live API URL, live dashboard URL, Docker setup, load-test report.

**Exit check (gate 3):** anyone with the link sees live forecasts, and switching `@champion` + calling `/reload` changes the served model with no redeploy.

### Part 4: Automate (weeks 10–11)

**Goal:** new data arrives on a schedule, and every code change is tested, trained, compared and deployed automatically.

**Learn first:** GitHub Actions (workflows, secrets, caching), Prefect (flows, schedules) or cron, the data source's API.

**Phase 4a: Live data ingestion (week 10)**

- [ ] Write `ingest.py` that pulls the newest readings from the live source, validates them and appends to the dataset
- [ ] Add data checks: schema, missing values, impossible values (for example negative PM2.5); bad batches are quarantined, not appended
- [ ] Schedule it hourly or daily with Prefect (or cron on the VM)
- [ ] Version each day's data with `dvc add` + `dvc push`
- [ ] Schedule a batch forecast job that writes the next 24 hours of forecasts for the dashboard

**Phase 4b: CI/CD (week 11)**

- [ ] CI workflow on every pull request: `ruff`, `pytest`, a small smoke-training run on sample data
- [ ] Training workflow (manual or weekly): `dvc pull` → `dvc repro` → `promote.py`
- [ ] CD workflow: when `@champion` changes, build and push the Docker image, redeploy (or call `/reload`)
- [ ] Store all keys in GitHub Secrets
- [ ] Add a status badge to the README

**Deliverables:** scheduled ingestion, scheduled forecasts, three GitHub Actions workflows.

**Exit check (gate 4):** a week passes with no manual steps: data arrives, forecasts update, and a pushed change goes from test to deployment automatically.

### Part 5: Monitor and retrain (weeks 12–14)

**Goal:** the system notices when it is getting worse and fixes itself by retraining, with a full audit trail.

**Learn first:** data drift vs. concept drift, Evidently AI reports and tests, Prometheus metrics and Grafana dashboards.

**Phase 5a: Service monitoring (week 12)**

- [ ] Expose API metrics with `prometheus-fastapi-instrumentator`: request count, latency, errors
- [ ] Grafana dashboard: latency p50 / p95, error rate, requests per minute, model version in use
- [ ] One alert: p95 latency or error rate above a threshold

**Phase 5b: Model monitoring (week 13)**

- [ ] Daily job: join logged forecasts with actuals as they arrive, compute live RMSE
- [ ] Evidently data-drift report comparing recent inputs with the training data
- [ ] Push live RMSE and drift scores to Grafana
- [ ] Save each day's Evidently report as an HTML artifact

**Phase 5c: Automatic retraining (week 14)**

- [ ] Retraining trigger: drift above threshold OR live RMSE worse than holdout RMSE by a set margin (for example 15%)
- [ ] Trigger runs the Part 4 training workflow → `promote.py` → redeploy only if the challenger wins
- [ ] Log every trigger, decision and reason (MLflow tags plus a simple events table)
- [ ] Test it: simulate drift by shifting inputs, and record the full loop from alert to promotion

**Deliverables:** Grafana dashboards, daily Evidently reports, automated retraining loop, a recorded drift-to-retrain demo.

**Exit check (gate 5):** a simulated drift event triggers retraining and a safe promotion with no manual steps, and you can show the logs that explain every decision.
