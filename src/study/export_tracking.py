"""Export the MLflow numbers the extended paper cites (CV scores, sweep, registry decisions)
to reports/study/tracking_export.json, so the paper never depends on a live tracking server.

    python -m src.study.export_tracking      (needs the MLflow server; output is committed to git)
"""

from __future__ import annotations

import json

from mlflow import MlflowClient

from src.config import ROOT
from src.tracking import EXPERIMENT_NAME, setup_mlflow

# The best run of each candidate type (Phases 3-5). Run-ID prefixes are resolved below.
RUNS = {
    "snaive": "ed80e080",
    "sarima": "493cf78d",
    "xgb": "9c305ef0",
    "xgb_tuned": "d5b80a09",
    "lstm": "9fcaaa80",
    "ensemble": "dd2479f4",
}
SWEEP_PARENT = "919b2c36950c468abc0f427b1c7f33ba"
MODEL_NAME = "pm25-forecaster"
OUT = ROOT / "reports/study/tracking_export.json"


def main() -> None:
    setup_mlflow()
    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    all_runs = client.search_runs([exp.experiment_id], max_results=1000)

    def resolve(prefix: str):
        hits = [r for r in all_runs if r.info.run_id.startswith(prefix)]
        if len(hits) != 1:
            raise SystemExit(f"run prefix {prefix} matched {len(hits)} runs")
        return hits[0]

    cv = {}
    for name, prefix in RUNS.items():
        r = resolve(prefix)
        cv[name] = {
            "run_id": r.info.run_id,
            "run_name": r.info.run_name,
            **{m: r.data.metrics[m] for m in ("mae", "rmse", "smape")},
        }
    sweep = client.get_run(SWEEP_PARENT)
    children = client.search_runs(
        [exp.experiment_id], f"tags.mlflow.parentRunId = '{SWEEP_PARENT}'", max_results=500
    )
    registry = []
    for mv in sorted(client.search_model_versions(f"name='{MODEL_NAME}'"),
                     key=lambda v: int(v.version)):  # fmt: skip
        mv = client.get_model_version(MODEL_NAME, mv.version)
        keep = ("run_name", "model_type", "cv_rmse", "holdout_rmse", "holdout_vs_champion",
                "promoted_on", "previous_champion", "rejected_reason", "role")  # fmt: skip
        registry.append({"version": str(mv.version), "aliases": list(mv.aliases),
                         **{k: mv.tags[k] for k in keep if k in mv.tags}})  # fmt: skip
    out = {
        "cv": cv,
        "sweep": {
            "run_id": SWEEP_PARENT,
            "n_trials": len(children),
            "best_rmse": min(c.data.metrics["rmse"] for c in children if "rmse" in c.data.metrics),
            "importance": {
                k.removeprefix("importance."): v
                for k, v in sweep.data.metrics.items()
                if k.startswith("importance.")
            },  # fmt: skip
        },
        "registry": registry,
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}: {len(cv)} CV runs, {out['sweep']['n_trials']} sweep trials, "
          f"{len(registry)} registry versions")  # fmt: skip


if __name__ == "__main__":
    main()
