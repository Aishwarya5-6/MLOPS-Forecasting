"""Promotion policies compared in the study (docs/study-protocol.md).

A policy sees the validation-window squared errors of the champion and the challenger
(arrays shaped (n_origins, horizon), NaN where the actual value is missing) and returns
True to promote the challenger.
"""

from __future__ import annotations

import numpy as np

from src.promote import decide
from src.study.stats import dm_test


def pooled_rmse(sq_err: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean(sq_err)))


def daily_loss(sq_err: np.ndarray) -> np.ndarray:
    """Mean squared error per origin (one value per day); NaN if the day has no actuals."""
    with np.errstate(invalid="ignore"):
        valid = (~np.isnan(sq_err)).sum(axis=1)
        return np.where(valid > 0, np.nansum(sq_err, axis=1) / np.maximum(valid, 1), np.nan)


class Threshold:
    """Promote if the challenger's RMSE is at least `tau` (relative) lower. Same as promote.py."""

    def __init__(self, tau: float):
        self.tau = tau
        self.name = f"tau{round(tau * 100):g}"

    def __call__(self, champ_err: np.ndarray, chall_err: np.ndarray) -> bool:
        return decide(pooled_rmse(chall_err), pooled_rmse(champ_err), self.tau).promote


class DieboldMariano:
    """Promote if a one-sided DM test on daily losses says the challenger is better (p < alpha)."""

    name = "dm"

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def __call__(self, champ_err: np.ndarray, chall_err: np.ndarray) -> bool:
        d = daily_loss(chall_err) - daily_loss(champ_err)
        _, p = dm_test(d, h=1, alternative="less")
        return p < self.alpha


class Never:
    name = "never"

    def __call__(self, champ_err: np.ndarray, chall_err: np.ndarray) -> bool:
        return False


def policies_from_params(study: dict) -> list:
    thresholds = [Threshold(t) for t in study["thresholds"]]
    return [*thresholds, DieboldMariano(study["dm_alpha"]), Never()]
