"""LaTeX tables for the extended paper, generated from the study results and the MLflow export.

Called by src/study/figures.py (DVC stage study_figures). Writes paper/tables/*.tex.
"""

from __future__ import annotations

import json

from src.config import ROOT

TABLE_DIR = ROOT / "paper/tables"
POLICIES = ["tau0", "tau1", "tau2", "tau5", "tau10", "dm", "never"]
POLICY_TEX = {"tau0": r"$\tau=0$", "tau1": r"$\tau=1\%$", "tau2": r"$\tau=2\%$",
              "tau5": r"$\tau=5\%$", "tau10": r"$\tau=10\%$", "dm": "DM test",
              "never": "never"}  # fmt: skip
MODEL_TEX = {"snaive": "Seasonal naive", "sarima": "SARIMA", "xgb": "XGBoost",
             "xgb_tuned": "XGBoost (tuned)", "lstm": "LSTM", "ensemble": "Ensemble"}  # fmt: skip
YES, NO, NA = r"\checkmark", r"$\times$", "--"


def _esc(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%")


def results_table(name: str, label: str, res: dict, windows: list[int]) -> str:
    rows = {(r["policy"], r["window"]): r for r in res["results"]}
    head = " & ".join(f"$V={v}$ d" for v in windows)
    lines = [
        r"\begin{table}[ht]", r"\centering", r"\footnotesize",
        rf"\caption{{{label}: deployed RMSE (swaps/harmful) for each policy and validation window "
        rf"$V$. Replay {res['replay_start']} to {res['replay_end']}; initial champion: "
        rf"{MODEL_TEX.get(res['initial_champion'], _esc(res['initial_champion']))}; "
        rf"oracle RMSE {res['oracle']['deployed_rmse']:.2f}.}}",
        rf"\label{{tab:full-{name}}}",
        r"\begin{tabular}{@{}l" + "l" * len(windows) + "@{}}", r"\toprule",
        rf"Policy & {head} \\", r"\midrule",
    ]  # fmt: skip
    for p in POLICIES:
        cells = []
        for v in windows:
            r = rows[(p, v)]
            cells.append(f"{r['deployed_rmse']:.2f} ({r['swaps']}/{r['harmful']})")
        lines.append(f"{POLICY_TEX[p]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def hypotheses_table(ev: dict, labels: dict, windows: list[int]) -> str:
    def mark(x):
        return NA if x is None else (YES if x else NO)

    ws = [str(v) for v in windows]
    lines = [
        r"\begin{table*}[ht]", r"\centering", r"\footnotesize",
        r"\caption{Pre-registered hypotheses in every cell (\checkmark{} holds, $\times$ fails). "
        r"H1 and H2 are scored per validation window $V$ (days); H3 per threshold. "
        r"The control is reported but not counted.}",
        r"\label{tab:hyp}",
        r"\begin{tabular}{@{}l" + "c" * (2 * len(ws) + 3) + "@{}}", r"\toprule",
        rf" & \multicolumn{{{len(ws)}}}{{c}}{{H1}} & \multicolumn{{{len(ws)}}}{{c}}{{H2}} "
        r"& \multicolumn{3}{c}{H3} \\",
        "Dataset & " + " & ".join([f"$V={v}$" for v in ws] * 2)
        + r" & $\tau=0$ & $\tau=1\%$ & $\tau=2\%$ \\", r"\midrule",
    ]  # fmt: skip
    for name, d in ev["per_dataset"].items():
        lab = labels.get(name, _esc(name)) + (
            " (control)" if name in ev.get("controls", []) else ""
        )
        cells = [mark(d["H1"][w]) for w in ws] + [mark(d["H2"][w]) for w in ws]
        cells += [mark(d["H3"][t]) for t in ("tau0", "tau1", "tau2")]
        lines.append(lab + " & " + " & ".join(cells) + r" \\")
    s = ev["summary"]

    def span(h: str, n: int) -> str:
        return rf"\multicolumn{{{n}}}{{c}}{{{s[h]['holds']}/{s[h]['cells']}}}"

    lines += [r"\midrule",
              f"Holds (counted) & {span('H1', len(ws))} & {span('H2', len(ws))} & "
              f"{span('H3', 3)} " + r"\\",
              r"\bottomrule", r"\end{tabular}", r"\end{table*}"]  # fmt: skip
    return "\n".join(lines) + "\n"


def static_table(results: dict, labels: dict) -> str:
    models = ["snaive", "sarima", "xgb", "xgb_tuned", "lstm", "ensemble"]
    lines = [
        r"\begin{table}[ht]", r"\centering", r"\footnotesize",
        r"\caption{RMSE over the replay year if one candidate is served all year (no promotion), "
        r"with the best policy and the hindsight oracle for reference. -- = not a candidate.}",
        r"\label{tab:static}", r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}l" + "r" * (len(models) + 2) + "@{}}", r"\toprule",
        "Dataset & " + " & ".join(["SN", "SAR", "XGB", "XGB-t", "LSTM", "Ens"])
        + r" & Best pol. & Oracle \\", r"\midrule",
    ]  # fmt: skip
    for name, res in results.items():
        st = res["static_rmse"]
        best = min(st, key=st.get)
        cells = []
        for m in models:
            if m not in st:
                cells.append(NA)
            else:
                v = f"{st[m]:.2f}"
                cells.append(rf"\textbf{{{v}}}" if m == best else v)
        bp = min(r["deployed_rmse"] for r in res["results"])
        cells += [f"{bp:.2f}", f"{res['oracle']['deployed_rmse']:.2f}"]
        lines.append(labels.get(name, _esc(name)) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def cv_table(export: dict) -> str:
    order = ["snaive", "sarima", "xgb", "xgb_tuned", "lstm", "ensemble"]
    lines = [
        r"\begin{table}[ht]", r"\centering", r"\footnotesize",
        r"\caption{Cross-validation on CAMS (5 expanding-window folds, 30-day test windows, "
        r"daily 24\,h forecasts). Run IDs refer to the MLflow tracking store.}",
        r"\label{tab:cv}",
        r"\begin{tabular}{@{}lrrrl@{}}", r"\toprule",
        r"Model & MAE & RMSE & sMAPE (\%) & MLflow run \\", r"\midrule",
    ]  # fmt: skip
    for m in order:
        c = export["cv"][m]
        lines.append(f"{MODEL_TEX[m]} & {c['mae']:.2f} & {c['rmse']:.2f} & {c['smape']:.1f} & "
                     rf"\texttt{{{c['run_id'][:8]}}} \\")  # fmt: skip
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def registry_table(export: dict) -> str:
    lines = [
        r"\begin{table}[ht]", r"\centering", r"\footnotesize",
        r"\caption{Registry state after the Phase~5 demonstration (holdout: the last 30 days of "
        r"CAMS data). The 2\% rule rejected v4 despite a real but small improvement.}",
        r"\label{tab:registry}", r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}llrrrl@{}}", r"\toprule",
        r"Ver. & Model & CV & Holdout & vs.\ champ. & Outcome \\", r"\midrule",
    ]  # fmt: skip
    type_name = {"xgb": "XGBoost", "seasonal_naive": "Seasonal naive", "lstm": "LSTM",
                 "ensemble": "Ensemble", "sarima": "SARIMA"}  # fmt: skip

    def name_of(r):
        n = type_name.get(r.get("model_type", ""), _esc(r.get("model_type", "?")))
        tuned = r.get("model_type") == "xgb" and "tuned" in r.get("run_name", "")
        return n + (" (tuned)" if tuned else "")

    for r in export["registry"]:
        v = r["version"]
        cv = f"{float(r['cv_rmse']):.2f}" if "cv_rmse" in r else NA
        ho = f"{float(r['holdout_rmse']):.2f}" if "holdout_rmse" in r else NA
        vs = (f"{100 * float(r['holdout_vs_champion']):+.1f}\\%"
              if "holdout_vs_champion" in r else NA)  # fmt: skip
        if "champion" in r["aliases"]:
            out = r"promoted, \texttt{@champion}"
        elif "baseline" in r["aliases"]:
            out = r"\texttt{@baseline}"
        elif "rejected_reason" in r:
            out = "rejected"
        else:
            out = "previous champion"
        lines.append(f"v{v} & {name_of(r)} & {cv} & {ho} & {vs} & {out} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def write_all(results: dict, labels: dict, windows: list[int]) -> list[str]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    written = []

    def put(fname: str, text: str):
        (TABLE_DIR / fname).write_text("% Generated by src/study/tables.py. Do not edit.\n" + text)
        written.append(fname)

    for name, res in results.items():
        put(f"full_{name}.tex", results_table(name, labels.get(name, _esc(name)), res, windows))
    hyp = ROOT / "reports/study/hypotheses.json"
    if hyp.exists():
        ev = json.loads(hyp.read_text())
        ev["per_dataset"] = {k: {h: {str(kk): vv for kk, vv in d.items()} for h, d in v.items()}
                             for k, v in ev["per_dataset"].items()}  # fmt: skip
        put("hypotheses.tex", hypotheses_table(ev, labels, windows))
    put("static.tex", static_table(results, labels))
    exp = ROOT / "reports/study/tracking_export.json"
    if exp.exists():
        export = json.loads(exp.read_text())
        put("cv.tex", cv_table(export))
        put("registry.tex", registry_table(export))
    return written
