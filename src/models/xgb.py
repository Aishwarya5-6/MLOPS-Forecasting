"""XGBoost, direct multi-horizon: one model forecasts all 24 steps, with `step` as a feature."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from src.features import direct_features
from src.models.base import ForecastModel


class XGBModel(ForecastModel):
    name = "xgb"

    def __init__(
        self,
        horizon: int = 24,
        lags=(1, 2, 3, 6, 12, 24),
        rolling_windows=(6, 24, 168),
        seasonal_lags=(24, 168),
        train_origin_every: int = 2,
        log_target: bool = False,
        seed: int = 42,
        **xgb_params,
    ):
        self.horizon = horizon
        self.lags = list(lags)
        self.rolling_windows = list(rolling_windows)
        self.seasonal_lags = list(seasonal_lags)
        self.train_origin_every = train_origin_every
        self.log_target = log_target
        self.seed = seed
        self.xgb_params = {
            "n_estimators": 400,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            **xgb_params,
        }
        self.model_: xgb.XGBRegressor | None = None

    @property
    def min_history(self) -> int:
        return max([*self.lags, *self.rolling_windows, *self.seasonal_lags]) + 1

    def _features(self, y: np.ndarray, start: pd.Timestamp, positions) -> pd.DataFrame:
        return direct_features(
            y, start, positions, self.horizon, self.lags, self.rolling_windows, self.seasonal_lags
        )

    def fit(self, train: pd.Series) -> XGBModel:
        y = train.to_numpy(dtype=float)
        positions = np.arange(self.min_history, len(y), self.train_origin_every)
        X = self._features(y, train.index[0], positions)
        target = np.repeat(positions, self.horizon) + X["step"].to_numpy() - 1
        t = np.full(len(X), np.nan)
        inside = target < len(y)
        t[inside] = y[target[inside]]
        keep = ~np.isnan(t)  # drops targets past the end of training data and missing hours
        X, t = X[keep], t[keep]
        if self.log_target:
            t = np.log1p(t)
        self.model_ = xgb.XGBRegressor(
            **self.xgb_params, tree_method="hist", random_state=self.seed, n_jobs=-1
        )
        self.model_.fit(X, t)
        return self

    def predict(self, history: pd.Series, horizon: int) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("fit() must be called before predict()")
        if horizon > self.horizon:
            raise ValueError(f"Model was trained for up to {self.horizon} steps ahead")
        hist = history.iloc[-self.min_history :]
        X = self._features(hist.to_numpy(dtype=float), hist.index[0], [len(hist)])
        pred = self.model_.predict(X)[:horizon]
        if self.log_target:
            pred = np.expm1(pred)
        return np.clip(pred, 0, None)

    def feature_importance(self) -> pd.Series:
        names = self.model_.get_booster().feature_names
        return pd.Series(self.model_.feature_importances_, index=names).sort_values(ascending=False)

    def get_params(self) -> dict:
        return {
            "horizon": self.horizon,
            "lags": self.lags,
            "rolling_windows": self.rolling_windows,
            "seasonal_lags": self.seasonal_lags,
            "train_origin_every": self.train_origin_every,
            "log_target": self.log_target,
            "seed": self.seed,
            **self.xgb_params,
        }

    def short_name(self) -> str:
        p = self.xgb_params
        log = "-log" if self.log_target else ""
        return f"xgb-d{p['max_depth']}-lr{p['learning_rate']}-n{p['n_estimators']}{log}"
