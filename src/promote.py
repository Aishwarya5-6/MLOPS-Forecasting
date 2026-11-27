"""Champion / challenger promotion through the MLflow Model Registry (plan section 9).

    python -m src.promote challenger --run-id <id>    # register -> @challenger -> holdout -> decide
    python -m src.promote challenger --run-id <id> --dry-run   # score + decide, change nothing
    python -m src.promote baseline --run-id <id>      # register seasonal naive as @baseline
    python -m src.promote rollback [--to-version N]   # @champion -> its previous_champion
    python -m src.promote status                      # aliases and their tags

Rule: the challenger takes @champion only if its holdout RMSE is at least 2% lower than the
champion's on the *same* holdout window. With no champion, the first challenger wins.
Every model is scored the same way: load the registered pyfunc, forecast every holdout origin
with `evaluate()`, no refitting. Each model is scored in its own subprocess, because torch and
xgboost bundle different libomp builds that segfault when loaded into one process on macOS.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from src.models.base import ForecastModel

MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "pm25-forecaster")
MIN_IMPROVEMENT = 0.02  # challenger RMSE must be >= 2% lower
HISTORY_HOURS = 24 * 42  # history sent to the pyfunc at each origin (covers every model's needs)
SCORE_MARKER = "HOLDOUT_SCORE_JSON:"


# --------------------------------------------------------------------------- decision rule
@dataclass(frozen=True)
class Decision:
    promote: bool
    reason: str
    improvement: float | None  # relative RMSE reduction vs champion; None if no champion


def decide(
    challenger_rmse: float, champion_rmse: float | None, min_improvement: float = MIN_IMPROVEMENT
) -> Decision:
    """Pure promotion rule. `improvement` > 0 means the challenger is better."""
    if not np.isfinite(challenger_rmse):
        return Decision(False, "challenger holdout RMSE is not finite", None)
    if champion_rmse is None:
        return Decision(True, "no champion yet: first challenger becomes champion", None)
    improvement = (champion_rmse - challenger_rmse) / champion_rmse
    detail = (
        f"holdout RMSE {challenger_rmse:.3f} vs champion {champion_rmse:.3f} "
        f"({improvement:+.1%}; rule needs >= {min_improvement:.0%})"
    )
    if improvement >= min_improvement - 1e-9:  # tolerance so exactly 2% counts as 2%
        return Decision(True, f"better on holdout: {detail}", improvement)
    return Decision(False, f"not enough improvement: {detail}", improvement)


# --------------------------------------------------------------------------- holdout scoring
class PyfuncForecaster(ForecastModel):
    """Adapts a logged pyfunc to the ForecastModel interface so `evaluate()` can score it.

    `fit` does nothing: the registered model is already trained (on data before the holdout).
    """

    name = "pyfunc"

    def __init__(self, pyfunc_model, history_hours: int = HISTORY_HOURS):
        self.pyfunc = pyfunc_model
        self.history_hours = history_hours

    def fit(self, train: pd.Series) -> PyfuncForecaster:
        return self

    def predict(self, history: pd.Series, horizon: int) -> np.ndarray:
        from src.models.pyfunc_wrapper import history_frame

        frame = history_frame(history, self.history_hours)
        out = self.pyfunc.predict(frame, params={"horizon": horizon})
        return out["forecast"].to_numpy(dtype=float)

    def get_params(self) -> dict:
        return {"history_hours": self.history_hours}


def holdout_window():
    from src.config import load_params
    from src.data import PREPARED_FILE, folds_from_params

    params = load_params()
    y = pd.read_parquet(PREPARED_FILE)["pm25"]
    _, holdout = folds_from_params(y.index, params)
    return y, holdout, params["forecast"]["horizon"]


def score_in_process(model_uri: str) -> dict:
    import mlflow

    from src.evaluate import evaluate

    y, holdout, horizon = holdout_window()
    model = PyfuncForecaster(mlflow.pyfunc.load_model(model_uri))
    result = evaluate(model, y, [holdout], horizon)
    return {
        **result["mean"],
        "rmse_by_horizon": result["rmse_by_horizon"],
        "n_origins": len(holdout.origins),
        "window": f"{holdout.test_start} -> {holdout.test_end}",
    }


def score_isolated(model_uri: str) -> dict:
    """Score `model_uri` on the holdout in a fresh Python process (see module docstring)."""
    proc = subprocess.run(
        [sys.executable, "-m", "src.promote", "score", model_uri],
        capture_output=True,
        text=True,
        env={**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1"},
    )
    for line in proc.stdout.splitlines():
        if line.startswith(SCORE_MARKER):
            return json.loads(line[len(SCORE_MARKER) :])
    tail = proc.stderr[-3000:]
    raise RuntimeError(f"scoring {model_uri} failed (exit {proc.returncode}):\n{tail}")


# --------------------------------------------------------------------------- registry helpers
def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _client():
    from mlflow import MlflowClient

    from src.tracking import setup_mlflow

    setup_mlflow()
    return MlflowClient()


def alias_version(client, alias: str):
    """The ModelVersion behind `alias`, or None."""
    from mlflow.exceptions import MlflowException

    try:
        return client.get_model_version_by_alias(MODEL_NAME, alias)
    except MlflowException:
        return None


def register(client, run_id: str):
    """Register the run's model (or reuse the version already registered for it)."""
    import mlflow

    run = client.get_run(run_id)  # fails early on a bad run ID
    existing = client.search_model_versions(f"name='{MODEL_NAME}' and run_id='{run_id}'")
    if existing:
        mv = max(existing, key=lambda v: int(v.version))
        print(f"Run {run_id[:8]} is already version {mv.version}; reusing it.")
        return mv
    mv = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
    tags = {
        "model_type": run.data.tags.get("model_type", "unknown"),
        "run_name": run.info.run_name,
        "cv_rmse": run.data.metrics.get("rmse"),
        "data_version": run.data.tags.get("data_version"),
        "prepared_data_version": run.data.tags.get("prepared_data_version"),
        "git_commit": run.data.tags.get("git_commit"),
        "registered_on": _now(),
    }
    for k, v in tags.items():
        if v is not None:
            client.set_model_version_tag(MODEL_NAME, mv.version, k, str(v))
    return client.get_model_version(MODEL_NAME, mv.version)


def check_data_version(client, run_id: str) -> None:
    """A challenger must be trained on the current data version. Otherwise its training data
    may overlap the current holdout window, and the comparison would be unfair."""
    from src.tracking import data_version

    current = data_version()["prepared_data_version"]
    trained_on = client.get_run(run_id).data.tags.get("prepared_data_version")
    if trained_on != current:
        raise SystemExit(
            f"Run {run_id[:8]} was trained on data {trained_on}, but the current data is "
            f"{current}. Retrain on the current data (python -m src.train ...) and promote "
            "that run."
        )


def _tag_scores(client, version: str, s: dict, prefix: str = "holdout") -> None:
    for m in ("rmse", "mae", "smape"):
        client.set_model_version_tag(MODEL_NAME, version, f"{prefix}_{m}", f"{s[m]:.4f}")
    client.set_model_version_tag(MODEL_NAME, version, f"{prefix}_window", s["window"])


# --------------------------------------------------------------------------- commands
def cmd_challenger(
    run_id: str, min_improvement: float = MIN_IMPROVEMENT, dry_run: bool = False, scorer=None
) -> Decision:
    scorer = scorer or score_isolated
    client = _client()
    check_data_version(client, run_id)
    champion = alias_version(client, "champion")

    if dry_run:
        challenger_uri, version = f"runs:/{run_id}/model", None
    else:
        mv = register(client, run_id)
        version = str(mv.version)
        if champion and str(champion.version) == version:
            raise SystemExit(f"Version {version} is already @champion; nothing to compare.")
        client.set_registered_model_alias(MODEL_NAME, "challenger", version)
        client.set_model_version_tag(MODEL_NAME, version, "challenger_since", _now())
        challenger_uri = f"models:/{MODEL_NAME}@challenger"
        print(f"Registered run {run_id[:8]} as {MODEL_NAME} v{version} -> @challenger")

    print("Scoring challenger on the holdout...")
    ch = scorer(challenger_uri)
    print(f"  challenger: rmse={ch['rmse']:.3f} mae={ch['mae']:.3f} smape={ch['smape']:.2f}  "
          f"[{ch['window']}, {ch['n_origins']} origins]")  # fmt: skip
    champ_score = None
    if champion:
        print(f"Scoring champion v{champion.version} on the same holdout...")
        champ_score = scorer(f"models:/{MODEL_NAME}@champion")
        print(f"  champion:   rmse={champ_score['rmse']:.3f} mae={champ_score['mae']:.3f} "
              f"smape={champ_score['smape']:.2f}")  # fmt: skip
    baseline = alias_version(client, "baseline")
    base_score = None
    if baseline:
        base_score = scorer(f"models:/{MODEL_NAME}@baseline")
        print(f"  baseline:   rmse={base_score['rmse']:.3f} (seasonal naive, reference only)")

    decision = decide(ch["rmse"], champ_score["rmse"] if champ_score else None, min_improvement)
    verdict = "PROMOTE" if decision.promote else "REJECT"
    print(f"Decision: {verdict} - {decision.reason}")
    if dry_run:
        print("Dry run: registry not changed.")
        return decision

    _tag_scores(client, version, ch)
    if decision.improvement is not None:
        client.set_model_version_tag(
            MODEL_NAME, version, "holdout_vs_champion", f"{decision.improvement:+.4f}"
        )
    if base_score:
        improvement_vs_base = (base_score["rmse"] - ch["rmse"]) / base_score["rmse"]
        client.set_model_version_tag(
            MODEL_NAME, version, "holdout_vs_baseline", f"{improvement_vs_base:+.4f}"
        )
    if champion:  # refresh the champion's score on the current holdout window
        _tag_scores(client, champion.version, champ_score)

    if decision.promote:
        client.set_registered_model_alias(MODEL_NAME, "champion", version)
        client.delete_registered_model_alias(MODEL_NAME, "challenger")
        for k, v in {
            "promoted_on": _now(),
            "previous_champion": str(champion.version) if champion else "none",
            "promotion_reason": decision.reason,
            "promoted_by": "promote.py",
        }.items():
            client.set_model_version_tag(MODEL_NAME, version, k, v)
        old = f" (was v{champion.version})" if champion else ""
        print(f"@champion -> v{version}{old}")
    else:
        client.set_model_version_tag(MODEL_NAME, version, "rejected_reason", decision.reason)
        client.set_model_version_tag(MODEL_NAME, version, "rejected_on", _now())
        print(f"@champion unchanged (v{champion.version}); v{version} stays @challenger.")
    return decision


def cmd_baseline(run_id: str, scorer=None) -> str:
    scorer = scorer or score_isolated
    client = _client()
    check_data_version(client, run_id)
    mv = register(client, run_id)
    client.set_registered_model_alias(MODEL_NAME, "baseline", mv.version)
    client.set_model_version_tag(MODEL_NAME, mv.version, "role", "baseline")
    s = scorer(f"models:/{MODEL_NAME}@baseline")
    _tag_scores(client, mv.version, s)
    print(f"@baseline -> v{mv.version}; holdout rmse={s['rmse']:.3f}")
    return mv.version


def cmd_rollback(to_version: str | None = None) -> str:
    client = _client()
    champion = alias_version(client, "champion")
    if champion is None:
        raise SystemExit("No @champion to roll back.")
    current = str(champion.version)
    target = str(to_version or champion.tags.get("previous_champion") or "")
    if target in ("", "none"):
        raise SystemExit(
            f"Champion v{champion.version} has no previous_champion tag; "
            "pass --to-version N to choose a version explicitly."
        )
    client.get_model_version(MODEL_NAME, target)  # fails if the version doesn't exist
    if target == current:
        raise SystemExit(f"v{target} is already @champion.")
    now = _now()
    client.set_registered_model_alias(MODEL_NAME, "champion", target)
    client.set_model_version_tag(MODEL_NAME, current, "rolled_back_on", now)
    client.set_model_version_tag(MODEL_NAME, current, "rolled_back_to", target)
    client.set_model_version_tag(MODEL_NAME, target, "restored_on", now)
    client.set_model_version_tag(MODEL_NAME, target, "restored_from", current)
    print(f"@champion -> v{target} (rolled back from v{current})")
    return target


def cmd_status() -> None:
    client = _client()
    model = client.get_registered_model(MODEL_NAME)
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    print(f"{MODEL_NAME}: {len(versions)} versions")
    keys = ("run_name", "cv_rmse", "holdout_rmse", "promoted_on", "previous_champion",
            "rejected_reason")  # fmt: skip
    for alias, version in sorted(model.aliases.items()):
        mv = client.get_model_version(MODEL_NAME, version)
        print(f"  @{alias:<10} v{version}")
        for k in keys:
            if k in mv.tags:
                print(f"      {k}: {mv.tags[k]}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("challenger", help="register a run as @challenger and maybe promote it")
    c.add_argument("--run-id", required=True)
    c.add_argument("--min-improvement", type=float, default=MIN_IMPROVEMENT)
    c.add_argument("--dry-run", action="store_true", help="score and decide; change nothing")
    b = sub.add_parser("baseline", help="register a run as @baseline")
    b.add_argument("--run-id", required=True)
    r = sub.add_parser("rollback", help="point @champion back at previous_champion")
    r.add_argument("--to-version", default=None)
    sub.add_parser("status", help="show aliases and tags")
    s = sub.add_parser("score", help="(internal) score a model URI on the holdout, print JSON")
    s.add_argument("model_uri")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    from dotenv import load_dotenv

    load_dotenv()
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    args = parse_args(argv)
    if args.command == "challenger":
        cmd_challenger(args.run_id, args.min_improvement, args.dry_run)
    elif args.command == "baseline":
        cmd_baseline(args.run_id)
    elif args.command == "rollback":
        cmd_rollback(args.to_version)
    elif args.command == "status":
        cmd_status()
    elif args.command == "score":
        from src.tracking import setup_mlflow

        setup_mlflow()
        print(SCORE_MARKER + json.dumps(score_in_process(args.model_uri)))


if __name__ == "__main__":
    main()
