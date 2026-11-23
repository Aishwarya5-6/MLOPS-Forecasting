"""Training entry point: python -m src.train --model <name> [--config path]."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

MODELS = ["dummy", "naive", "seasonal_naive", "sarima", "xgb", "lstm"]
EXPERIMENT_NAME = "forecasting-dev"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and track a forecasting model.")
    parser.add_argument("--model", required=True, choices=MODELS, help="Model type to train.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config for the model (default: configs/<model>.yaml).",
    )
    return parser.parse_args(argv)


def load_config(model: str, config: Path | None) -> dict:
    path = config or Path("configs") / f"{model}.yaml"
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def run_dummy(params: dict) -> None:
    """Log one param and one metric to prove the MLflow connection."""
    import mlflow

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="dummy-connection-test") as run:
        mlflow.set_tag("model_type", "dummy")
        mlflow.set_tag("stage", "dev")
        mlflow.log_param("learning_rate", params.get("learning_rate", 0.1))
        mlflow.log_metric("rmse", 1.0)
        print(f"Logged run {run.info.run_id} to {tracking_uri} (experiment '{EXPERIMENT_NAME}')")


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    args = parse_args(argv)
    params = load_config(args.model, args.config)
    print(f"model={args.model} config={args.config} params={params}")

    if args.model == "dummy":
        run_dummy(params)
    else:
        raise NotImplementedError(f"Model '{args.model}' is added in a later phase.")


if __name__ == "__main__":
    main()
