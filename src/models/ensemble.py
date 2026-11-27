"""Weighted average of several ForecastModels (e.g. XGBoost + SARIMA).

Different model families make different errors, so averaging them usually cancels some error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.base import ForecastModel


class EnsembleModel(ForecastModel):
    name = "ensemble"

    def __init__(self, members: list[ForecastModel], weights: list[float] | None = None):
        if not members:
            raise ValueError("ensemble needs at least one member")
        w = np.ones(len(members)) if weights is None else np.asarray(weights, float)
        if len(w) != len(members) or (w < 0).any() or w.sum() == 0:
            raise ValueError("weights must be non-negative, one per member, not all zero")
        self.members = members
        self.weights = w / w.sum()

    @property
    def extra_pip(self) -> list[str]:
        return sorted({p for m in self.members for p in getattr(m, "extra_pip", [])})

    def fit(self, train: pd.Series) -> EnsembleModel:
        for m in self.members:
            m.fit(train)
        return self

    def predict(self, history: pd.Series, horizon: int) -> np.ndarray:
        preds = np.array([np.asarray(m.predict(history, horizon), float) for m in self.members])
        return self.weights @ preds

    def short_name(self) -> str:
        names = [m.short_name() if hasattr(m, "short_name") else m.name for m in self.members]
        return "ens(" + "+".join(names) + ")"

    def get_params(self) -> dict:
        return {
            "weights": [round(float(x), 4) for x in self.weights],
            **{f"m{i}_{m.name}": m.get_params() for i, m in enumerate(self.members)},
        }
