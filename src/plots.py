"""Diagnostic plots logged with every run."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ACTUAL, FORECAST, INK2 = "#52514e", "#2a78d6", "#52514e"
plt.rcParams.update(
    {
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": "#d6d5d0",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e8e7e3",
        "axes.titlelocation": "left",
        "axes.titleweight": "bold",
        "axes.titlesize": 10,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.labelcolor": INK2,
    }
)


def _stitch(fold, pred: np.ndarray) -> pd.Series:
    """Forecasts from consecutive origins, laid end to end on the time axis."""
    horizon = pred.shape[1]
    idx = [o + pd.Timedelta(hours=h) for o in fold.origins for h in range(horizon)]
    return pd.Series(pred.ravel(), index=pd.DatetimeIndex(idx))


def forecast_vs_actual(forecasts, title: str):
    """One panel per fold: actual PM2.5 vs the model's 24 h forecasts."""
    n = len(forecasts)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.2 * n), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (fold, actual, pred) in zip(axes, forecasts, strict=True):
        a, p = _stitch(fold, actual), _stitch(fold, pred)
        ax.plot(a.index, a, color=ACTUAL, linewidth=1.2, label="Actual")
        ax.plot(p.index, p, color=FORECAST, linewidth=1.5, label="Forecast")
        rmse = float(np.sqrt(np.nanmean((actual - pred) ** 2)))
        ax.set_title(f"Fold {fold.index}: {fold.test_start:%d %b %Y}, RMSE {rmse:.2f}")
        ax.set_ylabel("µg/m³")
    axes[0].legend(loc="upper right", frameon=False, ncols=2)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return fig


def residuals(forecasts, rmse_by_horizon: list[float], title: str):
    """Residual distribution and error growth by hours ahead."""
    res = np.concatenate([(a - p).ravel() for _, a, p in forecasts])
    res = res[~np.isnan(res)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.5))
    ax1.hist(res, bins=60, color=FORECAST, edgecolor="#fcfcfb", linewidth=0.5)
    ax1.axvline(0, color=INK2, linewidth=1)
    ax1.set_title(f"Residuals (actual - forecast): mean {res.mean():.2f}, sd {res.std():.2f}")
    ax1.set_xlabel("µg/m³")
    steps = np.arange(1, len(rmse_by_horizon) + 1)
    ax2.bar(steps, rmse_by_horizon, color=FORECAST, width=0.7)
    ax2.set_title("RMSE by hours ahead")
    ax2.set_xlabel("hours ahead")
    ax2.grid(axis="x", visible=False)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return fig


def feature_importance(importance: pd.Series, title: str):
    top = importance.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(top.index, top.to_numpy(), color=FORECAST, height=0.7)
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig
