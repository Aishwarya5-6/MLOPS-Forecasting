"""Score H1-H3 mechanically, using the criteria written in docs/study-protocol.md.

    python -m src.study.hypotheses     # -> reports/study/hypotheses.json (+ printed table)

H1  tau0 has more harmful swaps than tau2 AND tau5 (per dataset x V cell), and tau2's deployed
    RMSE is not significantly worse than tau0's (DM two-sided p >= 0.05, or tau2 better).
H2  dm has a lower harmful-swap rate than tau2, and the bootstrap CI of (dm - tau2) deployed RMSE
    includes 0 or favours dm (lower bound <= 0).
H3  for tau in {0, 1, 2}%: harmful swaps do not increase from V=7 -> 14 -> 30.
"""

from __future__ import annotations

import json

import pandas as pd

from src.config import ROOT, load_params

CONTROLS = {"station_nolstm"}


def _cells(res: dict) -> pd.DataFrame:
    return pd.DataFrame(res["results"]).set_index(["policy", "window"])


def h1(df: pd.DataFrame, v: int) -> bool:
    harm = lambda p: df.loc[(p, v), "harmful"]  # noqa: E731
    more_harm = harm("tau0") > harm("tau2") and harm("tau0") > harm("tau5")
    t2 = df.loc[("tau2", v)]
    not_worse = t2["diff_vs_tau0"] <= 0 or (t2["dm_p_vs_tau0"] or 1.0) >= 0.05
    return bool(more_harm and not_worse)


def h2(df: pd.DataFrame, v: int) -> bool | None:
    dm, t2 = df.loc[("dm", v)], df.loc[("tau2", v)]
    if dm["harmful_rate"] is None or t2["harmful_rate"] is None:
        return None  # undefined: a policy made no swaps
    return bool(dm["harmful_rate"] < t2["harmful_rate"] and dm["diff_vs_tau2_ci"][0] <= 0)


def h3(df: pd.DataFrame, tau: str, windows: list[int]) -> bool:
    h = [df.loc[(tau, v), "harmful"] for v in sorted(windows)]
    return all(a >= b for a, b in zip(h, h[1:], strict=False))


def evaluate(results: dict[str, dict], windows: list[int]) -> dict:
    out = {}
    for name, res in results.items():
        df = _cells(res)
        out[name] = {
            "H1": {v: h1(df, v) for v in windows},
            "H2": {v: h2(df, v) for v in windows},
            "H3": {t: h3(df, t, windows) for t in ("tau0", "tau1", "tau2")},
        }
    # Controls (e.g. station_nolstm re-uses the station cache) are reported but not counted.
    counted = {k: d for k, d in out.items() if k not in CONTROLS}
    total = lambda h: [x for d in counted.values() for x in d[h].values()]  # noqa: E731
    summary = {}
    for h in ("H1", "H2", "H3"):
        vals = total(h)
        summary[h] = {"holds": sum(x is True for x in vals), "fails": sum(x is False for x in vals),
                      "undefined": sum(x is None for x in vals), "cells": len(vals)}  # fmt: skip
    return {"per_dataset": out, "summary": summary, "controls": sorted(CONTROLS)}


def main() -> None:
    study = load_params()["study"]
    results = {}
    for name in study["replays"]:
        f = ROOT / "reports/study" / name / "results.json"
        if f.exists():
            results[name] = json.loads(f.read_text())
    ev = evaluate(results, study["windows"])
    (ROOT / "reports/study/hypotheses.json").write_text(json.dumps(ev, indent=1))
    for name, d in ev["per_dataset"].items():
        print(f"{name:18s}", {h: d[h] for h in d})
    print(ev["summary"])


if __name__ == "__main__":
    main()
