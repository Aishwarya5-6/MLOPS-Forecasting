# MLOps Forecasting Platform

An end-to-end MLOps platform that forecasts Hyderabad air quality (PM2.5). The pipeline is reproducible, every run is tracked, and models are promoted through a registry. The full plan is in [docs/project-plan.md](docs/project-plan.md).

**Status:** Part 1, Phase 1 (data + DVC) complete. Phase notes: [Phase 0](docs/phase-0-setup.md), [Phase 1](docs/phase-1-data.md).

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
dvc repro                             # prepare -> featurize (reruns only stages whose inputs changed)
python -m src.download openmeteo      # refresh raw data, then: dvc add data/raw/pm25_raw.csv && dvc push
```

## Run

```bash
# Terminal 1: tracking server (SQLite backend, needed for the model registry)
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --port 5000

# Terminal 2
python -m src.train --help
python -m src.train --model dummy   # logs a test run; view it at http://127.0.0.1:5000
pytest && ruff check .
```
