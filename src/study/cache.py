"""Forecast cache for the promotion study: every candidate's 24 h forecast at every midnight
origin of the replay period, with periodic refits. Policies are then replayed offline from it.

    python -m src.study.cache --dataset cams                 # all candidates (one subprocess each)
    python -m src.study.cache --dataset cams --only xgb      # one candidate, in this process

Output: data/study/<dataset>/<candidate>.parquet with columns
    refit, origin, step, forecast, actual
Leakage rule: a refit at time r fits on y[: r - 1h]; a forecast at origin o sees y[: o - 1h].
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ROOT, load_params

HOUR = pd.Timedelta(hours=1)
DAY = pd.Timedelta(days=1)


@dataclass(frozen=True)
class Schedule:
    cache_start: pd.Timestamp  # first cached origin
    replay_start: pd.Timestamp  # first decision point
    replay_end: pd.Timestamp  # last origin with a full horizon of data
    refits: list[pd.Timestamp]
    origins: list[pd.Timestamp]  # every midnight in [cache_start, replay_end]


def make_schedule(index: pd.DatetimeIndex, study: dict, horizon: int = 24) -> Schedule:
    last_origin = (index[-1] - (horizon - 1) * HOUR).floor("D")
    replay_start = last_origin - (study["replay_days"] - 1) * DAY
    cache_start = replay_start - study["max_window_days"] * DAY
    if cache_start - HOUR < index[0]:
        raise ValueError("not enough data before the replay window")
    origins = list(pd.date_range(cache_start, last_origin, freq="D"))
    refits = list(pd.date_range(cache_start, last_origin, freq=f"{study['refit_every_days']}D"))
    return Schedule(cache_start, replay_start, last_origin, refits, origins)


def forecast_segment(model, y: pd.Series, refit: pd.Timestamp, origins, horizon: int) -> list:
    """Fit on data before `refit`, then forecast each origin from history strictly before it."""
    model.fit(y.loc[: refit - HOUR])
    rows = []
    for o in origins:
        pred = np.asarray(model.predict(y.loc[: o - HOUR], horizon), float)
        actual = y.reindex(pd.date_range(o, periods=horizon, freq="h")).to_numpy(dtype=float)
        for s in range(horizon):
            rows.append((refit, o, s + 1, pred[s], actual[s]))
    return rows


def build_candidate(spec: dict, params: dict):
    from src.train import apply_overrides, build_model, load_config

    cfg = load_config(spec["model"], Path(spec["config"]))
    cfg = apply_overrides(cfg, [f"{k}={v}" for k, v in (spec.get("set") or {}).items()])
    return build_model(spec["model"], cfg, params)


def cache_dir(dataset: str) -> Path:
    return ROOT / "data/study" / dataset


def run_candidate(dataset: str, name: str, params: dict | None = None) -> Path:
    params = params or load_params()
    study, horizon = params["study"], params["forecast"]["horizon"]
    y = pd.read_parquet(ROOT / study["datasets"][dataset])["pm25"]
    sched = make_schedule(y.index, study, horizon)
    model = build_candidate(study["candidates"][name], params)
    rows, t0 = [], time.time()
    for i, refit in enumerate(sched.refits):
        nxt = sched.refits[i + 1] if i + 1 < len(sched.refits) else sched.replay_end + DAY
        seg = [o for o in sched.origins if refit <= o < nxt]
        rows += forecast_segment(model, y, refit, seg, horizon)
        print(f"[{dataset}/{name}] refit {i + 1}/{len(sched.refits)} {refit.date()} "
              f"({len(seg)} origins, {time.time() - t0:.0f}s)", flush=True)  # fmt: skip
    out = cache_dir(dataset) / f"{name}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["refit", "origin", "step", "forecast", "actual"]).to_parquet(out)
    return out


def load_cache(dataset: str, params: dict | None = None) -> pd.DataFrame:
    """All candidates (plus derived ensembles) in one long frame with a `model` column."""
    study = (params or load_params())["study"]
    frames = {n: pd.read_parquet(cache_dir(dataset) / f"{n}.parquet") for n in study["candidates"]}
    for name, members in (study.get("derived") or {}).items():
        base = frames[members[0]].copy()
        base["forecast"] = np.mean([frames[m]["forecast"].to_numpy() for m in members], axis=0)
        frames[name] = base
    return pd.concat([f.assign(model=n) for n, f in frames.items()], ignore_index=True)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--only", default=None, help="one candidate, in this process")
    args = p.parse_args(argv)
    if args.only:
        print(f"wrote {run_candidate(args.dataset, args.only)}")
        return
    # One subprocess per candidate: torch and xgboost can't share a process on macOS (libomp).
    for name in load_params()["study"]["candidates"]:
        cmd = [sys.executable, "-m", "src.study.cache", "--dataset", args.dataset, "--only", name]
        subprocess.run(cmd, check=True, env={**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1"})


if __name__ == "__main__":
    main()
