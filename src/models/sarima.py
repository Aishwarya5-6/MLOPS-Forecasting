"""Seasonal ARIMA (statsmodels SARIMAX) with a 24-hour season."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.models.base import ForecastModel


class SarimaModel(ForecastModel):
    """Fit once on the last `train_days` of training data.

    At each forecast origin the fitted parameters are re-applied (Kalman filter, no
    refit) to the last `history_days` of history, then forecast `horizon` steps.
    """

    name = "sarima"

    def __init__(
        self,
        order=(1, 0, 1),
        seasonal_order=(1, 0, 1, 24),
        trend: str = "c",
        train_days: int = 42,
        history_days: int = 14,
        log_transform: bool = True,
    ):
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)
        self.trend = trend
        self.train_days = train_days
        self.history_days = history_days
        self.log_transform = log_transform
        self.result_ = None

    def _fwd(self, x: np.ndarray) -> np.ndarray:
        return np.log1p(x) if self.log_transform else x

    def _inv(self, x: np.ndarray) -> np.ndarray:
        return np.expm1(x) if self.log_transform else x

    def fit(self, train: pd.Series) -> SarimaModel:
        y = self._fwd(train.to_numpy(dtype=float)[-self.train_days * 24 :])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                y, order=self.order, seasonal_order=self.seasonal_order, trend=self.trend
            )
            self.result_ = model.fit(disp=False)
        return self

    def predict(self, history: pd.Series, horizon: int) -> np.ndarray:
        if self.result_ is None:
            raise RuntimeError("fit() must be called before predict()")
        y = self._fwd(history.to_numpy(dtype=float)[-self.history_days * 24 :])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = self.result_.apply(y).forecast(horizon)
        return np.clip(self._inv(np.asarray(fc)), 0, None)

    @property
    def min_history(self) -> int:
        return self.history_days * 24

    def get_params(self) -> dict:
        return {
            "order": list(self.order),
            "seasonal_order": list(self.seasonal_order),
            "trend": self.trend,
            "train_days": self.train_days,
            "history_days": self.history_days,
            "log_transform": self.log_transform,
        }

    def short_name(self) -> str:
        p, d, q = self.order
        P, D, Q, s = self.seasonal_order
        log = "-log" if self.log_transform else ""
        return f"sarima-{p}{d}{q}-{P}{D}{Q}x{s}-{self.train_days}d{log}"
