"""Paper figures and numbers from reports/study/<dataset>/results.json.

    python -m src.study.figures      # -> paper/figures/*.pdf and paper/numbers.tex

Colours: the dataviz reference palette (light mode, print). Categorical slots in fixed order;
validation windows use slots 1-3 (pass all-pairs CVD checks); models use slots 1-6 in a timeline
(adjacent segments only).
"""

from __future__ import annotations

import json
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import ROOT, load_params  # noqa: E402

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
POLICY_ORDER = ["tau0", "tau1", "tau2", "tau5", "tau10", "dm", "never"]
POLICY_LABEL = {"tau0": "τ=0%", "tau1": "τ=1%", "tau2": "τ=2%", "tau5": "τ=5%",
                "tau10": "τ=10%", "dm": "DM test", "never": "never"}  # fmt: skip
MODEL_ORDER = ["snaive", "sarima", "xgb", "xgb_tuned", "lstm", "ensemble"]
MODEL_LABEL = {"snaive": "seasonal naive", "sarima": "SARIMA", "xgb": "XGBoost",
               "xgb_tuned": "XGBoost (tuned)", "lstm": "LSTM", "ensemble": "ensemble"}  # fmt: skip
DATASET_LABEL = {
    "cams": "Hyderabad, CAMS", "station": "Hyderabad, CPCB",
    "station_nolstm": "Hyderabad, CPCB, no LSTM",
    "station_delhi": "Delhi, CPCB", "station_bengaluru": "Bengaluru, CPCB",
    "station_chennai": "Chennai, CPCB",
}  # fmt: skip
PRIMARY = ["cams", "station"]  # main-text figures; the rest go in the multi-city figure
FIG_DIR = ROOT / "paper/figures"


def _style():
    plt.rcParams.update({
        "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
        "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight",
    })  # fmt: skip


def load_results(datasets) -> dict:
    out = {}
    for d in datasets:
        f = ROOT / "reports/study" / d / "results.json"
        if f.exists():
            out[d] = json.loads(f.read_text())
    return out


def _frame(res: dict) -> pd.DataFrame:
    return pd.DataFrame(res["results"]).set_index(["policy", "window"])


def fig_swaps(results: dict, windows) -> None:
    fig, axes = plt.subplots(len(results), len(windows), figsize=(7.2, 1.7 * len(results)),
                             sharey="row", squeeze=False)  # fmt: skip
    pols = [p for p in POLICY_ORDER if p != "never"]
    for r, (d, res) in enumerate(results.items()):
        df = _frame(res)
        for c, v in enumerate(windows):
            ax = axes[r][c]
            harmful = np.array([df.loc[(p, v), "harmful"] for p in pols])
            good = np.array([df.loc[(p, v), "swaps"] for p in pols]) - harmful
            x = np.arange(len(pols))
            ax.bar(x, good, width=0.7, color=SLOTS[0], label="helpful swap",
                   edgecolor="white", linewidth=1)  # fmt: skip
            ax.bar(x, harmful, width=0.7, bottom=good, color=SLOTS[1], label="harmful swap",
                   edgecolor="white", linewidth=1)  # fmt: skip
            ax.set_xticks(x, [POLICY_LABEL[p] for p in pols], rotation=40, ha="right",
                          fontsize=7)  # fmt: skip
            ax.grid(axis="x", visible=False)
            ax.set_title(f"{DATASET_LABEL.get(d, d)}, V = {v} d", fontsize=7.5, color=INK)
            if c == 0:
                ax.set_ylabel("swaps (of 52)")
    axes[0][0].legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "swaps.pdf")
    plt.close(fig)


def fig_timeline(results: dict, window: int, fname: str = "timeline.pdf") -> None:
    pols = ["oracle", *POLICY_ORDER]
    fig, axes = plt.subplots(len(results), 1, figsize=(7.2, 1.55 * len(results)), squeeze=False)
    colour = {m: SLOTS[i] for i, m in enumerate(MODEL_ORDER)}
    for ax, (d, res) in zip(axes[:, 0], results.items(), strict=True):
        df = _frame(res)
        for row, p in enumerate(pols):
            if p == "oracle":
                champs = res["oracle"]["champions"]
            else:
                champs = df.loc[(p, window), "champions"]
            for w, m in enumerate(champs):
                ax.barh(row, 1, left=w, height=0.8, color=colour[m], edgecolor="white", lw=0.5)
        labels = ["oracle" if p == "oracle" else POLICY_LABEL[p] for p in pols]
        ax.set_yticks(range(len(pols)), labels)
        ax.invert_yaxis()
        ax.set_xlim(0, res["n_decisions"])
        ax.grid(False)
        ax.set_title(f"{DATASET_LABEL.get(d, d)}: champion served each week (V = {window} d)",
                     fontsize=7.5, color=INK)  # fmt: skip
    axes[-1][0].set_xlabel("week of replay")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colour[m]) for m in MODEL_ORDER]
    fig.legend(handles, [MODEL_LABEL[m] for m in MODEL_ORDER], loc="lower center", ncol=6,
               fontsize=7, bbox_to_anchor=(0.5, -0.06))  # fmt: skip
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname)
    plt.close(fig)


def fig_cities(results: dict, window: int) -> None:
    """Every replay: each policy's deployed-RMSE change vs tau0 (paired CI) and harmful swaps."""
    names = list(results)
    pols = POLICY_ORDER  # tau0 included: zero by definition on top, the reference count below
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 3.6), sharex=True)
    x = np.arange(len(pols))
    width = 0.8 / len(names)
    for i, d in enumerate(names):
        df = _frame(results[d])
        base = df.loc[("tau0", window), "deployed_rmse"]
        rel = [100 * (df.loc[(p, window), "deployed_rmse"] / base - 1) for p in pols]
        ci_lo = [100 * df.loc[(p, window), "diff_vs_tau0_ci"][0] / base for p in pols]
        ci_hi = [100 * df.loc[(p, window), "diff_vs_tau0_ci"][1] / base for p in pols]
        xs = x - 0.4 + width * (i + 0.5)
        axes[0].errorbar(xs, rel, yerr=[np.subtract(rel, ci_lo), np.subtract(ci_hi, rel)],
                         fmt="o", ms=3, lw=0.8, color=SLOTS[i],
                         label=DATASET_LABEL.get(d, d))  # fmt: skip
        harmful = [df.loc[(p, window), "harmful"] for p in pols]
        axes[1].bar(xs, harmful, width=width * 0.9, color=SLOTS[i])
    axes[0].axhline(0, color=INK2, lw=0.8)
    axes[0].set_ylabel("deployed RMSE\nvs τ=0 (%)")
    axes[1].set_ylabel("harmful swaps\n(count, of 52)")
    axes[1].set_xticks(x, [POLICY_LABEL[p] for p in pols])
    for ax in axes:
        ax.grid(axis="x", visible=False)
    axes[0].legend(fontsize=6.5, ncol=3, loc="upper left")
    axes[0].set_title(f"All datasets, V = {window} d", fontsize=8, color=INK)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cities.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------- numbers.tex
def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 0.1 else f"{v:.2f}"
    return str(v)


def numbers_tex(results: dict, params: dict) -> str:
    vals: dict[str, str] = {}
    for d, res in results.items():
        vals[f"{d}/init"] = res["initial_champion"].replace("_", r"\_")
        vals[f"{d}/start"], vals[f"{d}/end"] = res["replay_start"], res["replay_end"]
        vals[f"{d}/oracle/rmse"] = _fmt(res["oracle"]["deployed_rmse"])
        vals[f"{d}/oracle/swaps"] = _fmt(res["oracle"]["swaps"])
        for m, r in res["static_rmse"].items():
            vals[f"{d}/static/{m}"] = _fmt(r)
        for row in res["results"]:
            k = f"{d}/{row['policy']}/{row['window']}"
            for f in ("deployed_rmse", "swaps", "harmful", "skipped", "dm_p_vs_tau0",
                      "dm_p_vs_never", "diff_vs_tau0"):  # fmt: skip
                vals[f"{k}/{f}"] = _fmt(row[f])
            rate = row["harmful_rate"]
            vals[f"{k}/harmful_pct"] = "--" if rate is None else f"{100 * rate:.0f}\\%"
            vals[f"{k}/excess_pct"] = f"{100 * row['excess_vs_oracle']:.1f}\\%"
            lo, hi = row["diff_vs_tau0_ci"]
            vals[f"{k}/diff_tau0_ci"] = f"[{lo:+.2f}, {hi:+.2f}]"
            base = row["deployed_rmse"] - row["diff_vs_tau0"]
            vals[f"{k}/rel_tau0_pct"] = f"{100 * row['diff_vs_tau0'] / base:+.1f}\\%"
            vals[f"{k}/rel_tau0_ci"] = f"[{100 * lo / base:+.1f}, {100 * hi / base:+.1f}]\\%"
            lo, hi = row["rmse_ci"]
            vals[f"{k}/rmse_ci"] = f"[{lo:.2f}, {hi:.2f}]"
    for d, path in params["study"]["datasets"].items():
        f = ROOT / path
        if f.exists():
            s = pd.read_parquet(f)
            vals[f"{d}/hours"] = f"{len(s):,}".replace(",", "{,}")
            vals[f"{d}/missing_pct"] = f"{100 * s['is_missing'].mean():.1f}\\%"
    hyp = ROOT / "reports/study/hypotheses.json"
    if hyp.exists():
        ev = json.loads(hyp.read_text())
        for h, c in ev["summary"].items():
            vals[f"{h}/holds"], vals[f"{h}/cells"] = str(c["holds"]), str(c["cells"])
        for h, c in ev.get("summary_with_controls", {}).items():
            vals[f"{h}/holds_all"], vals[f"{h}/cells_all"] = str(c["holds"]), str(c["cells"])
    for d, res in results.items():  # exploratory: best single model kept all year
        st = res["static_rmse"]
        best = min(st, key=st.get)
        vals[f"{d}/static_best"] = best.replace("_", r"\_")
        vals[f"{d}/static_best_rmse"] = _fmt(st[best])
        rmses = [r["deployed_rmse"] for r in res["results"] if r["window"] == 30]
        vals[f"{d}/spread_pct"] = f"{100 * (max(rmses) / min(rmses) - 1):.1f}\\%"
    # Cross-dataset ranges (controls excluded), e.g. \R{range/tau0/7/swaps} -> "30--44"
    from src.study.hypotheses import CONTROLS

    counted = [r for k, r in results.items() if k not in CONTROLS]
    vals["n_datasets"] = str(len(counted))
    for pol in POLICY_ORDER:
        for v in params["study"]["windows"]:
            rows = [next(x for x in r["results"] if x["policy"] == pol and x["window"] == v)
                    for r in counted]  # fmt: skip
            for f in ("swaps", "harmful"):
                xs = [x[f] for x in rows]
                vals[f"range/{pol}/{v}/{f}"] = f"{min(xs)}--{max(xs)}"
            rates = [x["harmful_rate"] for x in rows if x["harmful_rate"] is not None]
            if rates:
                vals[f"range/{pol}/{v}/harmful_pct"] = (
                    f"{100 * min(rates):.0f}--{100 * max(rates):.0f}\\%"
                )
            rel = [x["diff_vs_tau0"] / (x["deployed_rmse"] - x["diff_vs_tau0"]) for x in rows]
            vals[f"range/{pol}/{v}/rel_tau0_pct"] = (
                f"{100 * min(rel):+.1f}\\% to {100 * max(rel):+.1f}\\%"
            )
    static_wins = sum(
        r["static_rmse"][min(r["static_rmse"], key=r["static_rmse"].get)]
        < min(x["deployed_rmse"] for x in r["results"])
        for r in counted
    )
    vals["static_beats_all"] = str(static_wins)
    gaps = [min(x["deployed_rmse"] for x in r["results"]) / r["oracle"]["deployed_rmse"] - 1
            for r in counted]  # fmt: skip
    vals["range/oracle_gap_pct"] = f"{100 * min(gaps):.1f}--{100 * max(gaps):.1f}\\%"
    osw = [r["oracle"]["swaps"] for r in counted]
    vals["range/oracle_swaps"] = f"{min(osw)}--{max(osw)}"
    n_ens = sum(min(r["static_rmse"], key=r["static_rmse"].get) == "ensemble" for r in counted)
    vals["ensemble_best_static"] = str(n_ens)
    # Worked example: the harmful swap with the largest relative worsening, CAMS, tau0, V=7.
    ex_row = next(x for x in results["cams"]["results"]
                  if x["policy"] == "tau0" and x["window"] == 7)  # fmt: skip
    bad = [e for e in ex_row["events"] if e["harmful"]]
    if bad:
        e = max(bad, key=lambda e: e["next_week_rmse_new"] / e["next_week_rmse_old"])
        vals["ex/date"] = e["t"]
        vals["ex/from"] = MODEL_LABEL[e["from"]]
        vals["ex/to"] = MODEL_LABEL[e["to"]]
        vals["ex/new"] = f"{e['next_week_rmse_new']:.2f}"
        vals["ex/old"] = f"{e['next_week_rmse_old']:.2f}"
        vals["ex/worse_pct"] = (
            f"{100 * (e['next_week_rmse_new'] / e['next_week_rmse_old'] - 1):.0f}\\%"
        )
    exp_file = ROOT / "reports/study/tracking_export.json"
    if exp_file.exists():
        export = json.loads(exp_file.read_text())
        for m, c in export["cv"].items():
            vals[f"cv/{m}/rmse"] = f"{c['rmse']:.2f}"
        vals["sweep/trials"] = str(export["sweep"]["n_trials"])
        vals["sweep/best"] = f"{export['sweep']['best_rmse']:.3f}"
        for k, v in export["sweep"]["importance"].items():
            vals[f"sweep/imp/{k}"] = f"{v:.2f}"
        reg = {r["version"]: r for r in export["registry"]}
        for v, r in reg.items():
            if "holdout_rmse" in r:
                vals[f"reg/v{v}/holdout"] = f"{float(r['holdout_rmse']):.2f}"
            if "holdout_vs_champion" in r:
                vals[f"reg/v{v}/vs"] = f"{100 * float(r['holdout_vs_champion']):+.1f}\\%"
    import yaml

    for key, dvcfile in (("cams", "data/raw/pm25_raw.csv.dvc"),
                         ("station", "data/raw/station/city_hour.csv.dvc")):  # fmt: skip
        md5 = yaml.safe_load((ROOT / dvcfile).read_text())["outs"][0]["md5"]
        vals[f"md5/{key}"] = md5[:8]
    raw = pd.read_csv(ROOT / params["data"]["raw_file"], usecols=["datetime"])
    vals["cams/raw_start"] = params["data"]["start_date"]
    vals["cams/raw_end"] = str(pd.to_datetime(raw["datetime"]).max().date())
    log = subprocess.run(["git", "log", "--format=%h", "--", "docs/study-protocol.md"],
                         cwd=ROOT, capture_output=True, text=True).stdout.split()  # fmt: skip
    vals["protocol_commit"] = log[-1] if log else "unknown"  # first version (pre-registration)
    vals["deviation_commit"] = log[0] if log else "unknown"  # latest version (deviation D2)
    vals["protocol_history"] = ", ".join(rf"\texttt{{{c}}}" for c in reversed(log))
    from importlib.metadata import version

    for pkg in ("mlflow", "dvc", "xgboost", "statsmodels", "torch", "optuna", "pandas"):
        try:
            vals[f"ver/{pkg}"] = version(pkg)
        except Exception:
            vals[f"ver/{pkg}"] = "n/a"
    lines = ["% Generated by `python -m src.study.figures`. Do not edit by hand.",
             r"\providecommand{\R}[1]{\csname r@#1\endcsname}"]  # fmt: skip
    lines += [rf"\expandafter\def\csname r@{k}\endcsname{{{v}}}" for k, v in sorted(vals.items())]
    return "\n".join(lines) + "\n"


def main() -> None:
    params = load_params()
    study = params["study"]
    results = load_results(study["replays"])
    if not results:
        raise SystemExit("no results yet: run python -m src.study.replay --dataset <name>")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _style()
    primary = {k: results[k] for k in PRIMARY if k in results}
    fig_swaps(primary, study["windows"])
    fig_timeline(primary, max(study["windows"]))
    fig_cities(results, max(study["windows"]))
    from src.study.hypotheses import CONTROLS
    from src.study.tables import write_all

    counted = {k: v for k, v in results.items() if k not in CONTROLS}
    fig_timeline(counted, max(study["windows"]), "timeline_all.pdf")
    write_all(results, DATASET_LABEL, study["windows"])
    (ROOT / "paper/numbers.tex").write_text(numbers_tex(results, params))
    print(f"figures for {list(results)} -> {FIG_DIR}; numbers -> paper/numbers.tex")


if __name__ == "__main__":
    main()
