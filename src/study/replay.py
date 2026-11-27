"""Replay a year of weekly promotion decisions from the forecast cache (docs/study-protocol.md).

    python -m src.study.replay --replay cams      # -> reports/study/cams/results.json, summary.csv

Every (policy, validation window V) starts from the same initial champion. At each weekly
decision point the best non-champion candidate on the last V days is the challenger; the policy
decides; the champion then serves the next 7 days (the real future), which is what gets scored.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import ROOT, load_params
from src.study.policies import daily_loss, policies_from_params, pooled_rmse
from src.study.stats import bootstrap_rmse, ci, dm_test

DAY = pd.Timedelta(days=1)


def error_matrices(cache: pd.DataFrame) -> tuple[pd.DatetimeIndex, dict[str, np.ndarray]]:
    """model -> squared errors shaped (n_origins, horizon); NaN where the actual is missing."""
    origins = pd.DatetimeIndex(sorted(cache["origin"].unique()))
    err = {}
    for name, g in cache.groupby("model", sort=False):
        m = g.pivot(index="origin", columns="step", values="forecast").reindex(origins)
        a = g.pivot(index="origin", columns="step", values="actual").reindex(origins)
        err[name] = ((m - a) ** 2).to_numpy(dtype=float)
    return origins, err


@dataclass
class Replay:
    policy: str
    window: int
    champions: list[str] = field(default_factory=list)  # champion served in each week
    swaps: int = 0
    harmful: int = 0
    skipped: int = 0
    week_sse: list[float] = field(default_factory=list)
    week_n: list[int] = field(default_factory=list)
    daily: list[float] = field(default_factory=list)  # daily loss of the served forecasts
    events: list[dict] = field(default_factory=list)

    @property
    def rmse(self) -> float:
        return float(np.sqrt(np.sum(self.week_sse) / np.sum(self.week_n)))


def _week_stats(e: np.ndarray) -> tuple[float, int]:
    return float(np.nansum(e)), int((~np.isnan(e)).sum())


def initial_champion(origins, err, replay_start, init_days: int) -> str:
    sel = (origins >= replay_start - init_days * DAY) & (origins < replay_start)
    return min(err, key=lambda m: pooled_rmse(err[m][sel]))


def replay_policy(
    policy, window: int, origins: pd.DatetimeIndex, err: dict[str, np.ndarray],
    decisions: list[pd.Timestamp], init: str, min_valid: float = 0.5, week_days: int = 7,
) -> Replay:  # fmt: skip
    r = Replay(policy=policy.name, window=window)
    champ = init
    for t in decisions:
        val = (origins >= t - window * DAY) & (origins < t)
        dep = (origins >= t) & (origins < t + week_days * DAY)
        old = champ
        valid_frac = np.mean(~np.isnan(err[champ][val])) if val.any() else 0.0
        if valid_frac < min_valid:
            r.skipped += 1
        else:
            others = [m for m in err if m != champ]
            chall = min(others, key=lambda m: pooled_rmse(err[m][val]))
            if policy(err[champ][val], err[chall][val]):
                champ = chall
                r.swaps += 1
                new_rmse, old_rmse = pooled_rmse(err[champ][dep]), pooled_rmse(err[old][dep])
                harmful = bool(new_rmse > old_rmse)
                r.harmful += harmful
                r.events.append({"t": str(t.date()), "from": old, "to": champ,
                                 "next_week_rmse_new": new_rmse, "next_week_rmse_old": old_rmse,
                                 "harmful": harmful})  # fmt: skip
        sse, n = _week_stats(err[champ][dep])
        r.champions.append(champ)
        r.week_sse.append(sse)
        r.week_n.append(n)
        r.daily.extend(daily_loss(err[champ][dep]).tolist())
    return r


def replay_oracle(origins, err, decisions, week_days: int = 7) -> Replay:
    r = Replay(policy="oracle", window=0)
    for t in decisions:
        dep = (origins >= t) & (origins < t + week_days * DAY)
        best = min(err, key=lambda m: pooled_rmse(err[m][dep]))
        if r.champions and best != r.champions[-1]:
            r.swaps += 1
        sse, n = _week_stats(err[best][dep])
        r.champions.append(best)
        r.week_sse.append(sse)
        r.week_n.append(n)
        r.daily.extend(daily_loss(err[best][dep]).tolist())
    return r


def run(replay: str, params: dict | None = None) -> dict:
    """Replay `replay` (a key of study.replays: which cache, and optionally which candidates)."""
    from src.study.cache import load_cache, make_schedule

    params = params or load_params()
    study = params["study"]
    spec = study["replays"][replay]
    dataset = spec["cache"]
    y = pd.read_parquet(ROOT / study["datasets"][dataset])["pm25"]
    sched = make_schedule(y.index, study, params["forecast"]["horizon"])
    origins, err = error_matrices(load_cache(dataset, params, spec.get("candidates")))
    decisions = [sched.replay_start + k * study["decide_every_days"] * DAY
                 for k in range(study["n_decisions"])]  # fmt: skip
    init = initial_champion(origins, err, sched.replay_start, study["init_window_days"])

    replays = [replay_policy(p, v, origins, err, decisions, init, study["min_valid_fraction"],
                             study["decide_every_days"])
               for v in study["windows"] for p in policies_from_params(study)]  # fmt: skip
    oracle = replay_oracle(origins, err, decisions, study["decide_every_days"])

    # Static reference: each candidate served all year
    end = decisions[-1] + study["decide_every_days"] * DAY
    dep_all = (origins >= decisions[0]) & (origins < end)
    static = {m: pooled_rmse(e[dep_all]) for m, e in err.items()}

    key = lambda r: f"{r.policy}@V{r.window}"  # noqa: E731
    sse = {key(r): np.array(r.week_sse) for r in [*replays, oracle]}
    cnt = {key(r): np.array(r.week_n, float) for r in [*replays, oracle]}
    boot = bootstrap_rmse(sse, cnt, study["bootstrap_n"], params["seed"])

    rows = []
    for r in replays:
        k, ref0, refn = key(r), f"tau0@V{r.window}", f"never@V{r.window}"
        d0 = np.array(r.daily) - np.array(next(x for x in replays if key(x) == ref0).daily)
        dn = np.array(r.daily) - np.array(next(x for x in replays if key(x) == refn).daily)
        rows.append({
            "policy": r.policy, "window": r.window, "deployed_rmse": r.rmse,
            "rmse_ci": ci(boot[k]),
            "swaps": r.swaps, "harmful": r.harmful,
            "harmful_rate": r.harmful / r.swaps if r.swaps else None,
            "skipped": r.skipped,
            "excess_vs_oracle": r.rmse / oracle.rmse - 1,
            "diff_vs_tau0": r.rmse - float(np.sqrt(sse[ref0].sum() / cnt[ref0].sum())),
            "diff_vs_tau0_ci": ci(boot[k] - boot[ref0]),
            "dm_p_vs_tau0": dm_test(d0)[1] if r.policy != "tau0" else None,
            "diff_vs_never_ci": ci(boot[k] - boot[refn]),
            "diff_vs_tau2_ci": ci(boot[k] - boot[f"tau2@V{r.window}"]),  # needed for H2
            "dm_p_vs_never": dm_test(dn)[1] if r.policy != "never" else None,
            "champions": r.champions, "events": r.events,
        })  # fmt: skip
    return {
        "dataset": replay,
        "cache": dataset,
        "candidates": list(err),
        "replay_start": str(sched.replay_start.date()),
        "replay_end": str((decisions[-1] + (study["decide_every_days"] - 1) * DAY).date()),
        "n_decisions": len(decisions),
        "initial_champion": init,
        "static_rmse": static,
        "oracle": {
            "deployed_rmse": oracle.rmse,
            "swaps": oracle.swaps,
            "champions": oracle.champions,
            "rmse_ci": ci(boot["oracle@V0"]),
        },  # fmt: skip
        "results": rows,
    }


def summary_table(res: dict) -> pd.DataFrame:
    cols = ["policy", "window", "deployed_rmse", "swaps", "harmful", "harmful_rate", "skipped",
            "excess_vs_oracle", "diff_vs_tau0", "dm_p_vs_tau0", "dm_p_vs_never"]  # fmt: skip
    return pd.DataFrame(res["results"])[cols]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--replay", required=True, help="a key of params study.replays")
    args = p.parse_args(argv)
    res = run(args.replay)
    out = ROOT / "reports/study" / args.replay
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float))
    table = summary_table(res)
    table.to_csv(out / "summary.csv", index=False, float_format="%.4f")
    print(f"{args.replay}: {res['replay_start']} -> {res['replay_end']}, "
          f"initial champion {res['initial_champion']}")  # fmt: skip
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
