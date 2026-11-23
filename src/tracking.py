"""log_run(): the one place every training run is written to MLflow (and to W&B in Phase 4)."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import mlflow
import yaml

from src.config import ROOT

EXPERIMENT_NAME = "forecasting-hyd-pm25"


def git_commit() -> tuple[str, bool]:
    """(commit hash, whether the working tree has uncommitted changes)."""

    def run(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout

    commit = run("rev-parse", "HEAD").strip() or "unknown"
    dirty = bool(run("status", "--porcelain", "--untracked-files=no").strip())
    return commit, dirty


def data_version() -> dict[str, str]:
    """DVC hashes of the raw data and the prepared series used for training."""
    raw = yaml.safe_load((ROOT / "data/raw/pm25_raw.csv.dvc").read_text())["outs"][0]["md5"]
    lock = yaml.safe_load((ROOT / "dvc.lock").read_text())
    prepared = lock["stages"]["prepare"]["outs"][0]["md5"]
    return {"data_version": raw, "prepared_data_version": prepared}


def setup_mlflow() -> None:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment(EXPERIMENT_NAME)


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


def log_run(
    run_name: str,
    model_type: str,
    params: dict,
    metrics: dict[str, float],
    *,
    step_metrics: dict[str, list[float]] | None = None,
    figures: dict | None = None,
    files: list[Path] | None = None,
    log_model_fn=None,
    stage: str = "dev",
) -> str:
    """Log one run and return its run ID.

    - params: model config + shared settings (nested dicts are flattened: forecast.horizon)
    - metrics: summary numbers (mae, rmse, smape)
    - step_metrics: series logged with a step, e.g. {"fold_rmse": [...], "rmse_by_horizon": [...]}
    - figures: {"forecast_vs_actual.png": matplotlib Figure}
    - files: extra files to attach (configs, json)
    - log_model_fn: callable(tmp_dir) that logs the model inside the run
    Tags always include model_type, git_commit, git_dirty, data_version, stage.
    """
    setup_mlflow()
    commit, dirty = git_commit()
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {
                "model_type": model_type,
                "git_commit": commit,
                "git_dirty": str(dirty).lower(),
                "stage": stage,
                **data_version(),
            }
        )
        mlflow.log_params({k: str(v) for k, v in _flatten(params).items()})
        mlflow.log_metrics(metrics)
        for name, values in (step_metrics or {}).items():
            for step, value in enumerate(values):
                mlflow.log_metric(name, float(value), step=step)
        for fname, fig in (figures or {}).items():
            mlflow.log_figure(fig, f"plots/{fname}")
        for f in files or []:
            mlflow.log_artifact(str(f), artifact_path="config")
        if log_model_fn is not None:
            with tempfile.TemporaryDirectory() as tmp:
                log_model_fn(Path(tmp))
        return run.info.run_id
