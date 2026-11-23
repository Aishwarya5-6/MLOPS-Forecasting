"""The one interface every forecasting model follows."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class ForecastModel(ABC):
    """fit() once on training data; predict() from any later point without refitting.

    `predict(history, horizon)` forecasts the `horizon` hours right after the last
    timestamp in `history`. During evaluation, `history` contains the training data
    plus any test hours already observed before the forecast origin, never anything after.
    """

    name: str = "base"

    def fit(self, train: pd.Series) -> ForecastModel:
        return self

    @abstractmethod
    def predict(self, history: pd.Series, horizon: int) -> np.ndarray: ...

    def get_params(self) -> dict:
        return {}
