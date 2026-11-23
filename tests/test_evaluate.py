import math

import numpy as np
import pandas as pd
import pytest

from src.data import make_folds
from src.evaluate import evaluate, mae, rmse, smape
from src.models.base import ForecastModel
from src.models.baseline import NaiveModel, SeasonalNaiveModel

# --- metrics vs. hand calculations --------------------------------------------------------

T = [10.0, 20.0, 30.0]
P = [12.0, 18.0, 33.0]  # errors 2, -2, 3


def test_mae_by_hand():
    assert mae(T, P) == pytest.approx((2 + 2 + 3) / 3)


def test_rmse_by_hand():
    assert rmse(T, P) == pytest.approx(math.sqrt((4 + 4 + 9) / 3))


def test_smape_by_hand():
    expected = 100 / 3 * (2 * 2 / 22 + 2 * 2 / 38 + 2 * 3 / 63)
    assert smape(T, P) == pytest.approx(expected)


def test_smape_both_zero_is_zero_and_bounded():
    assert smape([0.0, 0.0], [0.0, 0.0]) == 0.0
    assert smape([0.0], [5.0]) == pytest.approx(200.0)


def test_missing_actuals_are_ignored():
    assert mae([10.0, np.nan], [12.0, 999.0]) == pytest.approx(2.0)


# --- baselines --------------------------------------------------------------------------


def _hourly(values, start="2025-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="h"), dtype=float)


def test_naive_repeats_last_value():
    assert NaiveModel().predict(_hourly([1, 2, 3]), 4).tolist() == [3, 3, 3, 3]


def test_seasonal_naive_repeats_last_season():
    hist = _hourly(list(range(48)))
    pred = SeasonalNaiveModel(24).predict(hist, 30)
    assert pred[:24].tolist() == list(range(24, 48))
    assert pred[24:].tolist() == list(range(24, 30))


# --- evaluation loop --------------------------------------------------------------------


def _periodic_series(days=400):
    hours = np.arange(days * 24)
    values = 30 + 10 * np.sin(2 * np.pi * hours / 24)
    return _hourly(values, "2024-01-01")


def _folds(y):
    return make_folds(
        y.index,
        horizon=24,
        n_folds=3,
        test_hours=240,
        step_hours=480,
        holdout_hours=240,
        origin_every=24,
    )


def test_seasonal_naive_is_perfect_on_a_perfect_daily_cycle():
    y = _periodic_series()
    folds, _ = _folds(y)
    result = evaluate(SeasonalNaiveModel(24), y, folds, horizon=24)
    assert result["mean"]["rmse"] == pytest.approx(0.0, abs=1e-9)
    assert len(result["folds"]) == 3
    assert len(result["rmse_by_horizon"]) == 24


class SpyModel(ForecastModel):
    """Records what the evaluator shows it, to prove it never sees the future."""

    name = "spy"

    def __init__(self):
        self.train_ends, self.history_ends = [], []

    def fit(self, train):
        self.train_ends.append(train.index[-1])
        return self

    def predict(self, history, horizon):
        self.history_ends.append(history.index[-1])
        return np.zeros(horizon)


def test_evaluator_never_leaks_future_data():
    y = _periodic_series()
    folds, _ = _folds(y)
    spy = SpyModel()
    evaluate(spy, y, folds, horizon=24)
    assert spy.train_ends == [f.train_end for f in folds]
    origins = [o for f in folds for o in f.origins]
    assert all(
        h == o - pd.Timedelta(hours=1) for h, o in zip(spy.history_ends, origins, strict=True)
    )
