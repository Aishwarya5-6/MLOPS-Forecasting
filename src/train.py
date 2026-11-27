"""Training entry point.

    python -m src.train --model xgb [--config configs/xgb.yaml] [--set max_depth=8 --set ...]

Every run: walk-forward evaluation on the 5 folds (src.evaluate), then a final fit on all data
before the holdout, logged to MLflow with metrics, plots, config, git commit, data version
and the model as a pyfunc.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

MODELS = ["dummy", "naive", "seasonal_naive", "sarima", "xgb", "lstm", "ensemble"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train, evaluate and track a forecasting model.")
    parser.add_argument("--model", required=True, choices=MODELS, help="Model type to train.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config for the model (default: configs/<model>.yaml).",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override one config value, e.g. --set max_depth=8 (repeatable).",
    )
    parser.add_argument("--run-name", default=None, help="MLflow run name (default: from config).")
    parser.add_argument(
        "--no-log-model", action="store_true", help="Skip saving the model artifact (faster)."
    )
    return parser.parse_args(argv)


def load_config(model: str, config: Path | None) -> dict:
    path = config or Path("configs") / f"{model}.yaml"
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    out = dict(cfg)
    for item in overrides:
        key, _, value = item.partition("=")
        if not key or not _:
            raise SystemExit(f"--set expects KEY=VALUE, got '{item}'")
        out[key.strip()] = yaml.safe_load(value)
    return out


def build_model(name: str, cfg: dict, params: dict):
    horizon = params["forecast"]["horizon"]
    if name == "naive":
        from src.models.baseline import NaiveModel

        return NaiveModel()
    if name == "seasonal_naive":
        from src.models.baseline import SeasonalNaiveModel

        return SeasonalNaiveModel(**cfg)
    if name == "sarima":
        from src.models.sarima import SarimaModel

        return SarimaModel(**cfg)
    if name == "xgb":
        from src.models.xgb import XGBModel

        return XGBModel(horizon=horizon, seed=params["seed"], **cfg)
    if name == "lstm":
        from src.models.lstm import LSTMModel

        return LSTMModel(horizon=horizon, seed=params["seed"], **cfg)
    if name == "ensemble":
        from src.models.ensemble import EnsembleModel

        members = [
            build_model(m["model"], load_config(m["model"], Path(m["config"])), params)
            for m in cfg["members"]
        ]
        return EnsembleModel(members, cfg.get("weights"))
    raise NotImplementedError(f"Model '{name}' is added in a later phase.")


def run_dummy(params: dict) -> None:
    """Phase 0 connection test: log one param and one metric."""
    import mlflow

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("forecasting-dev")
    with mlflow.start_run(run_name="dummy-connection-test") as run:
        mlflow.set_tags({"model_type": "dummy", "stage": "dev"})
        mlflow.log_param("learning_rate", params.get("learning_rate", 0.1))
        mlflow.log_metric("rmse", 1.0)
        print(f"Logged run {run.info.run_id} to {tracking_uri}")


def train(args: argparse.Namespace) -> str:
    import matplotlib.pyplot as plt
    import pandas as pd

    from src import plots
    from src.config import PARAMS_FILE, load_params
    from src.data import PREPARED_FILE, folds_from_params
    from src.evaluate import evaluate
    from src.models.pyfunc_wrapper import history_frame, log_pyfunc
    from src.tracking import log_run

    params = load_params()
    cfg = apply_overrides(load_config(args.model, args.config), args.set)
    horizon = params["forecast"]["horizon"]
    y = pd.read_parquet(PREPARED_FILE)["pm25"]
    folds, holdout = folds_from_params(y.index, params)

    model = build_model(args.model, cfg, params)
    run_name = args.run_name or (model.short_name() if hasattr(model, "short_name") else model.name)
    print(f"[{run_name}] evaluating on {len(folds)} folds...")
    t0 = time.time()
    result = evaluate(model, y, folds, horizon, return_forecasts=True)
    eval_seconds = time.time() - t0
    m = result["mean"]
    print(f"[{run_name}] mae={m['mae']:.3f} rmse={m['rmse']:.3f} smape={m['smape']:.2f} "
          f"({eval_seconds:.0f}s)")  # fmt: skip

    # Final model: fit on everything before the holdout (the holdout stays unseen).
    final = build_model(args.model, cfg, params).fit(y.loc[: holdout.train_end])

    figures = {
        "forecast_vs_actual.png": plots.forecast_vs_actual(result["forecasts"], run_name),
        "residuals.png": plots.residuals(result["forecasts"], result["rmse_by_horizon"], run_name),
    }
    if hasattr(final, "feature_importance"):
        figures["feature_importance.png"] = plots.feature_importance(
            final.feature_importance(), f"{run_name}: feature importance (gain)"
        )

    example = history_frame(y.loc[: holdout.train_end], hours=24 * 14)
    files = [PARAMS_FILE] + ([args.config or Path("configs") / f"{args.model}.yaml"])
    run_params = {
        "model": final.get_params(),
        "forecast": params["forecast"],
        "seed": params["seed"],
    }

    def save_model(tmp: Path) -> None:
        log_pyfunc(final, example, tmp, horizon)

    run_id = log_run(
        run_name=run_name,
        model_type=args.model,
        params=run_params,
        metrics={**m, "eval_seconds": eval_seconds},
        step_metrics={
            "fold_rmse": [f["rmse"] for f in result["folds"]],
            "fold_mae": [f["mae"] for f in result["folds"]],
            "rmse_by_horizon": result["rmse_by_horizon"],
        },
        figures=figures,
        files=[f for f in files if Path(f).exists()],
        log_model_fn=None if args.no_log_model else save_model,
    )
    plt.close("all")
    print(f"[{run_name}] logged run {run_id}")
    return run_id


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    args = parse_args(argv)
    if args.model == "dummy":
        cfg = apply_overrides(load_config(args.model, args.config), args.set)
        run_dummy(cfg)
    else:
        train(args)


if __name__ == "__main__":
    main()
