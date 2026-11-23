"""Baselines every real model must beat."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.base import ForecastModel


class NaiveModel(ForecastModel):
    """Every future hour = the last observed value."""

    name = "naive"

    def predict(self, history: pd.Series, horizon: int) -> np.ndarray:
        last = history.ffill().iloc[-1]
        return np.full(horizon, last, dtype=float)


class SeasonalNaiveModel(ForecastModel):
    """Every future hour = the value one season earlier (e.g. same hour yesterday for m=24)."""

    def __init__(self, season: int = 24):
        self.season = season
        self.name = f"seasonal_naive_{season}"

    def predict(self, history: pd.Series, horizon: int) -> np.ndarray:
        last_season = history.ffill().to_numpy()[-self.season :]
        if len(last_season) < self.season:
            raise ValueError(f"Need at least {self.season} hours of history")
        reps = int(np.ceil(horizon / self.season))
        return np.tile(last_season, reps)[:horizon].astype(float)

    def get_params(self) -> dict:
        return {"season": self.season}
