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
DATASET_LABEL = {"cams": "CAMS (reanalysis)", "station": "CPCB station"}
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


def fig_deployed_rmse(results: dict, windows) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(3.5 * len(results), 2.4), squeeze=False)
    for ax, (d, res) in zip(axes[0], results.items(), strict=True):
        df = _frame(res)
        x = np.arange(len(POLICY_ORDER))
        width = 0.24
        for i, v in enumerate(windows):
            rmse = [df.loc[(p, v), "deployed_rmse"] for p in POLICY_ORDER]
            lo = [df.loc[(p, v), "rmse_ci"][0] for p in POLICY_ORDER]
            hi = [df.loc[(p, v), "rmse_ci"][1] for p in POLICY_ORDER]
            xs = x + (i - 1) * width
            ax.errorbar(xs, rmse, yerr=[np.subtract(rmse, lo), np.subtract(hi, rmse)], fmt="o",
                        ms=4, lw=1, capsize=0, color=SLOTS[i], label=f"V = {v} d")  # fmt: skip
        ax.axhline(res["oracle"]["deployed_rmse"], color=INK2, lw=1, ls="--")
        ax.text(len(POLICY_ORDER) - 0.5, res["oracle"]["deployed_rmse"], "oracle", color=INK2,
                ha="right", va="bottom", fontsize=7)  # fmt: skip
        ax.set_xticks(x, [POLICY_LABEL[p] for p in POLICY_ORDER], rotation=30, ha="right")
        ax.set_title(DATASET_LABEL.get(d, d), fontsize=8, color=INK)
        ax.set_ylabel("deployed RMSE (µg/m³)")
        ax.grid(axis="x", visible=False)
    axes[0][0].legend(loc="upper left", fontsize=7, ncol=3)
    fig.savefig(FIG_DIR / "deployed_rmse.pdf")
    plt.close(fig)


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


def fig_timeline(results: dict, window: int) -> None:
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
    fig.savefig(FIG_DIR / "timeline.pdf")
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
            lo, hi = row["rmse_ci"]
            vals[f"{k}/rmse_ci"] = f"[{lo:.2f}, {hi:.2f}]"
    for d, path in params["study"]["datasets"].items():
        f = ROOT / path
        if f.exists():
            s = pd.read_parquet(f)
            vals[f"{d}/hours"] = f"{len(s):,}".replace(",", "{,}")
            vals[f"{d}/missing_pct"] = f"{100 * s['is_missing'].mean():.1f}\\%"
    commit = subprocess.run(["git", "log", "--format=%h", "-1", "--", "docs/study-protocol.md"],
                            cwd=ROOT, capture_output=True, text=True).stdout.strip()  # fmt: skip
    vals["protocol_commit"] = commit or "unknown"
    lines = ["% Generated by `python -m src.study.figures`. Do not edit by hand.",
             r"\providecommand{\R}[1]{\csname r@#1\endcsname}"]  # fmt: skip
    lines += [rf"\expandafter\def\csname r@{k}\endcsname{{{v}}}" for k, v in sorted(vals.items())]
    return "\n".join(lines) + "\n"


def main() -> None:
    params = load_params()
    study = params["study"]
    results = load_results(study["datasets"])
    if not results:
        raise SystemExit("no results yet: run python -m src.study.replay --dataset <name>")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _style()
    fig_deployed_rmse(results, study["windows"])
    fig_swaps(results, study["windows"])
    fig_timeline(results, max(study["windows"]))
    (ROOT / "paper/numbers.tex").write_text(numbers_tex(results, params))
    print(f"figures for {list(results)} -> {FIG_DIR}; numbers -> paper/numbers.tex")


if __name__ == "__main__":
    main()
