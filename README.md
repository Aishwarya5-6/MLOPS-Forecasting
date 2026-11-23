# MLOps Forecasting Platform

An end-to-end MLOps platform that forecasts Hyderabad air quality (PM2.5). The pipeline is reproducible, every run is tracked, and models are promoted through a registry. The full plan is in [docs/project-plan.md](docs/project-plan.md).

**Status:** Part 1, Phase 0 (setup) complete.

## Setup

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env          # then add your WANDB_API_KEY
wandb login
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
