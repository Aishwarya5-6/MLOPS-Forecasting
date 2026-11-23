"""Shared metrics and the walk-forward evaluation loop. Every model is scored by this code.

python -m src.evaluate   (run by `dvc repro` as the `evaluate` stage: scores the baselines)
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import ROOT, load_params
from src.data import PREPARED_FILE, Fold, folds_from_params
from src.models.base import ForecastModel

METRICS_FILE = ROOT / "reports/metrics.json"
DETAILS_FILE = ROOT / "reports/baseline_details.json"
METRICS = ("mae", "rmse", "smape")


def _clean(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    """Drop positions where the actual value is missing."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = ~np.isnan(y_true)
    return y_true[mask], y_pred[mask]


def mae(y_true, y_pred) -> float:
    t, p = _clean(y_true, y_pred)
    return float(np.mean(np.abs(t - p)))


def rmse(y_true, y_pred) -> float:
    t, p = _clean(y_true, y_pred)
    return float(np.sqrt(np.mean((t - p) ** 2)))


def smape(y_true, y_pred) -> float:
    """Symmetric MAPE in percent (0-200). Pairs where both values are 0 count as 0 error."""
    t, p = _clean(y_true, y_pred)
    denom = np.abs(t) + np.abs(p)
    ratio = np.divide(2 * np.abs(t - p), denom, out=np.zeros_like(denom), where=denom != 0)
    return float(100 * np.mean(ratio))


def score(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }


def forecast_fold(
    model: ForecastModel, y: pd.Series, fold: Fold, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """Fit on the fold's training data, then forecast from every origin.

    Returns (actuals, forecasts), each shaped (n_origins, horizon).
    """
    model.fit(y.loc[: fold.train_end])
    actuals, forecasts = [], []
    hour = pd.Timedelta(hours=1)
    for origin in fold.origins:
        history = y.loc[: origin - hour]  # strictly before the origin: no peeking
        target = y.loc[origin : origin + (horizon - 1) * hour].to_numpy()
        pred = np.asarray(model.predict(history, horizon), float)
        if pred.shape != (horizon,):
            raise ValueError(f"{model.name}.predict returned shape {pred.shape}, need ({horizon},)")
        actuals.append(target)
        forecasts.append(pred)
    return np.array(actuals), np.array(forecasts)


def evaluate(
    model: ForecastModel,
    y: pd.Series,
    folds: list[Fold],
    horizon: int,
    return_forecasts: bool = False,
) -> dict:
    """Score a model on every fold.

    Returns {"folds": [...per-fold scores...], "mean": {...}, "rmse_by_horizon": [...]}.
    "mean" is the unweighted average over folds; this is the number used to compare models.
    With return_forecasts=True, also returns "forecasts": [(fold, actual, pred), ...] for plots.
    """
    per_fold, sq_err, forecasts = [], [], []
    for fold in folds:
        actual, pred = forecast_fold(model, y, fold, horizon)
        forecasts.append((fold, actual, pred))
        per_fold.append(
            {
                "fold": fold.index,
                "test_start": str(fold.test_start),
                "n_origins": len(fold.origins),
                **score(actual.ravel(), pred.ravel()),
            }
        )
        sq_err.append((actual - pred) ** 2)
    mean = {m: float(np.mean([f[m] for f in per_fold])) for m in METRICS}
    rmse_by_h = np.sqrt(np.nanmean(np.concatenate(sq_err), axis=0))
    out = {"folds": per_fold, "mean": mean, "rmse_by_horizon": rmse_by_h.round(3).tolist()}
    if return_forecasts:
        out["forecasts"] = forecasts
    return out


def baseline_models(params: dict) -> list[ForecastModel]:
    from src.models.baseline import NaiveModel, SeasonalNaiveModel

    m = params["forecast"]["seasonal_period"]
    return [NaiveModel(), SeasonalNaiveModel(m), SeasonalNaiveModel(7 * m)]


def main() -> None:
    params = load_params()
    y = pd.read_parquet(PREPARED_FILE)["pm25"]
    folds, holdout = folds_from_params(y.index, params)
    horizon = params["forecast"]["horizon"]

    print(f"Holdout (not used here): {holdout.test_start} -> {holdout.test_end}")
    for f in folds:
        print(
            f"Fold {f.index}: train ..{f.train_end:%Y-%m-%d}  "
            f"test {f.test_start:%Y-%m-%d} -> {f.test_end:%Y-%m-%d}  ({len(f.origins)} origins)"
        )

    metrics, details = {}, {}
    for model in baseline_models(params):
        result = evaluate(model, y, folds, horizon)
        metrics[model.name] = {k: round(v, 3) for k, v in result["mean"].items()}
        details[model.name] = result
        print(f"{model.name:<20} " + "  ".join(f"{k}={v:.2f}" for k, v in result["mean"].items()))

    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    METRICS_FILE.write_text(json.dumps(metrics, indent=2) + "\n")
    DETAILS_FILE.write_text(json.dumps(details, indent=2) + "\n")


if __name__ == "__main__":
    main()
