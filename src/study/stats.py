"""Forecast-comparison statistics: Diebold-Mariano test (HLN-corrected) and week-block bootstrap."""

from __future__ import annotations

import numpy as np
from scipy import stats


def dm_test(d, h: int = 1, alternative: str = "two-sided") -> tuple[float, float]:
    """Diebold-Mariano test on a loss differential d_t = loss_A,t - loss_B,t.

    Uses the Harvey-Leybourne-Newbold (1997) small-sample correction and a t(n-1) reference.
    alternative: "two-sided", "less" (A has lower loss), or "greater".
    Returns (statistic, p-value). With fewer than 2 values or zero variance: (nan, 1.0).
    """
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 2:
        return float("nan"), 1.0
    dbar = d.mean()
    # long-run variance: autocovariances up to lag h-1 (population form, as in DM 1995)
    gamma = [np.mean((d[k:] - dbar) * (d[: n - k] - dbar)) for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if not np.isfinite(var) or var <= 0:
        return float("nan"), 1.0
    dm = dbar / np.sqrt(var)
    dm *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)  # HLN correction
    t = stats.t(df=n - 1)
    if alternative == "less":
        p = t.cdf(dm)
    elif alternative == "greater":
        p = t.sf(dm)
    elif alternative == "two-sided":
        p = 2 * t.sf(abs(dm))
    else:
        raise ValueError(f"unknown alternative {alternative!r}")
    return float(dm), float(p)


def bootstrap_rmse(
    sse: dict[str, np.ndarray], count: dict[str, np.ndarray], n_boot: int = 2000, seed: int = 42
) -> dict[str, np.ndarray]:
    """Week-block bootstrap of pooled RMSE.

    sse[name][w] / count[name][w]: sum of squared errors and number of scored hours in week w.
    The same resampled weeks are used for every name, so differences are paired.
    Returns name -> array of n_boot resampled RMSEs.
    """
    names = list(sse)
    n_weeks = len(sse[names[0]])
    idx = np.random.default_rng(seed).integers(0, n_weeks, size=(n_boot, n_weeks))
    return {k: np.sqrt(sse[k][idx].sum(axis=1) / count[k][idx].sum(axis=1)) for k in names}


def ci(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    a = (1 - level) / 2
    lo, hi = np.quantile(samples, [a, 1 - a])
    return float(lo), float(hi)
