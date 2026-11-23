"""Hyperparameter sweep with Optuna, tracked in MLflow (replaces W&B Sweeps).

    python -m src.sweep sweeps/xgb.yaml [--trials 40]

- One parent MLflow run (stage=sweep); each trial is a nested child run with params,
  rmse/mae/smape and per-fold RMSE.
- The parent gets: best params and score, parameter importances (fANOVA), optimization
  history, parallel-coordinates and slice plots, and trials.csv.
- The best full config is written to configs/<model>_tuned.yaml, so it can be retrained
  as a normal tracked run: python -m src.train --model xgb --config configs/xgb_tuned.yaml
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
import warnings
from pathlib import Path

import yaml
from dotenv import load_dotenv


def suggest(trial, name: str, spec: dict):
    kind = spec["type"]
    if kind == "int":
        return trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step", 1))
    if kind == "float":
        return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
    if kind == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    raise ValueError(f"Unknown search-space type '{kind}' for {name}")


def make_sampler(spec: dict):
    import optuna

    if spec.get("type", "tpe") == "tpe":
        return optuna.samplers.TPESampler(
            seed=spec.get("seed"), n_startup_trials=spec.get("n_startup_trials", 10)
        )
    if spec["type"] == "random":
        return optuna.samplers.RandomSampler(seed=spec.get("seed"))
    raise ValueError(f"Unknown sampler '{spec['type']}'")


def _plots(study) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    from optuna.visualization import matplotlib as ov

    params = list(study.best_params)
    figs = {
        "optimization_history.png": ov.plot_optimization_history(study),
        "param_importances.png": ov.plot_param_importances(study),
        "parallel_coordinate.png": ov.plot_parallel_coordinate(study, params=params),
        "slice.png": ov.plot_slice(study, params=params),
    }
    out = {}
    for name, ax in figs.items():
        fig = ax.figure if hasattr(ax, "figure") else ax.flat[0].figure
        fig.set_size_inches(14 if name == "slice.png" else 10, 4.5 if name == "slice.png" else 6)
        fig.tight_layout()
        out[name] = fig
    return out


def run_sweep(spec_path: Path, n_trials: int | None = None) -> dict:
    import mlflow
    import optuna
    import pandas as pd

    from src.config import ROOT, load_params
    from src.data import PREPARED_FILE, folds_from_params
    from src.evaluate import evaluate
    from src.tracking import log_run, run_tags, setup_mlflow
    from src.train import build_model, load_config

    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    spec = yaml.safe_load(Path(spec_path).read_text())
    n_trials = n_trials or spec["n_trials"]
    params = load_params()
    horizon = params["forecast"]["horizon"]
    base = {**load_config(spec["model"], ROOT / spec["base_config"]), **spec.get("fixed", {})}
    y = pd.read_parquet(PREPARED_FILE)["pm25"]
    folds, _ = folds_from_params(y.index, params)

    def objective(trial):
        sampled = {k: suggest(trial, k, v) for k, v in spec["search_space"].items()}
        cfg = {**base, **sampled}
        model = build_model(spec["model"], cfg, params)
        t0 = time.time()
        result = evaluate(model, y, folds, horizon)
        m = result["mean"]
        log_run(
            run_name=f"{spec['name']}-trial-{trial.number:03d}",
            model_type=spec["model"],
            params={"model": model.get_params(), "trial": trial.number},
            metrics={**m, "eval_seconds": time.time() - t0},
            step_metrics={"fold_rmse": [f["rmse"] for f in result["folds"]]},
            stage="sweep",
            nested=True,
        )
        trial.set_user_attr("fold_rmse", [round(f["rmse"], 3) for f in result["folds"]])
        trial.set_user_attr("mae", m["mae"])
        print(f"trial {trial.number:3d}  {spec['metric']}={m[spec['metric']]:.4f}  {sampled}")
        return m[spec["metric"]]

    setup_mlflow()
    with mlflow.start_run(run_name=f"sweep-{spec['name']}") as parent:
        mlflow.set_tags({**run_tags(spec["model"], "sweep"), "sweep": spec["name"]})
        mlflow.log_params(
            {
                "n_trials": n_trials,
                "sampler": spec["sampler"].get("type", "tpe"),
                "metric": spec["metric"],
                "direction": spec["direction"],
            }
        )
        mlflow.log_artifact(str(spec_path), artifact_path="config")

        sampler = make_sampler(spec["sampler"])
        study = optuna.create_study(
            study_name=spec["name"], direction=spec["direction"], sampler=sampler
        )
        t0 = time.time()
        study.optimize(objective, n_trials=n_trials)
        elapsed = time.time() - t0

        importances = optuna.importance.get_param_importances(
            study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=42)
        )
        best_cfg = {**base, **study.best_params}
        mlflow.log_metrics({f"best_{spec['metric']}": study.best_value, "sweep_seconds": elapsed})
        mlflow.log_metrics({f"importance.{k}": v for k, v in importances.items()})
        mlflow.log_params({f"best.{k}": v for k, v in study.best_params.items()})
        mlflow.set_tag("best_trial", study.best_trial.number)
        for name, fig in _plots(study).items():
            mlflow.log_figure(fig, f"plots/{name}")

        trials = study.trials_dataframe(
            attrs=("number", "value", "params", "user_attrs", "duration")
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trials.csv"
            trials.to_csv(path, index=False)
            mlflow.log_artifact(str(path))

        tuned = ROOT / "configs" / f"{spec['model']}_tuned.yaml"
        header = (
            f"# Best config from Optuna sweep '{spec['name']}'"
            f" (MLflow run {parent.info.run_id})\n"
            f"# trial {study.best_trial.number}: {spec['metric']}={study.best_value:.4f}\n"
        )
        clean = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in best_cfg.items()}
        tuned.write_text(header + yaml.safe_dump(clean, sort_keys=False))
        mlflow.log_artifact(str(tuned), artifact_path="config")

    summary = {
        "parent_run_id": parent.info.run_id,
        "best_value": study.best_value,
        "best_trial": study.best_trial.number,
        "best_params": study.best_params,
        "importances": importances,
        "minutes": elapsed / 60,
        "tuned_config": str(tuned),
    }
    print(
        f"\nBest {spec['metric']}={study.best_value:.4f} (trial {study.best_trial.number}) "
        f"in {elapsed / 60:.1f} min"
    )
    print("Importances:", {k: round(float(v), 3) for k, v in importances.items()})
    print(f"Wrote {tuned}")
    return summary


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    parser = argparse.ArgumentParser(description="Optuna hyperparameter sweep tracked in MLflow.")
    parser.add_argument("spec", type=Path, help="Sweep YAML, e.g. sweeps/xgb.yaml")
    parser.add_argument("--trials", type=int, default=None, help="Override n_trials.")
    args = parser.parse_args(argv)
    run_sweep(args.spec, args.trials)


if __name__ == "__main__":
    main()
