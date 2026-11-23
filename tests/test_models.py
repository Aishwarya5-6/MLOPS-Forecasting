import numpy as np
import pandas as pd
import pytest

from src.features import direct_features
from src.models.pyfunc_wrapper import ForecastPyfunc, history_frame
from src.models.sarima import SarimaModel
from src.models.xgb import XGBModel
from src.train import apply_overrides


def _series(days=60, seed=0):
    rng = np.random.default_rng(seed)
    hours = np.arange(days * 24)
    values = 30 + 10 * np.sin(2 * np.pi * hours / 24) + rng.normal(0, 1, len(hours))
    return pd.Series(values, index=pd.date_range("2025-01-01", periods=len(hours), freq="h"))


FEAT_KW = dict(horizon=24, lags=[1, 2, 24], windows=[6, 24], seasonal_lags=[24, 168])


def test_direct_features_shape_and_values():
    y = np.arange(400, dtype=float)
    X = direct_features(y, pd.Timestamp("2025-01-01"), np.array([200]), **FEAT_KW)
    assert len(X) == 24
    assert X["step"].tolist() == list(range(1, 25))
    assert (X["lag_1"] == 199).all() and (X["lag_24"] == 176).all()
    # same hour yesterday for target 200 + step - 1
    assert X["seas_24"].tolist() == [176 + s for s in range(24)]
    assert X["roll_mean_6"].iloc[0] == np.mean(range(194, 200))


def test_direct_features_never_use_data_at_or_after_origin():
    y = np.arange(400, dtype=float)
    changed = y.copy()
    changed[200:] = 9999.0
    start = pd.Timestamp("2025-01-01")
    a = direct_features(y, start, np.array([200]), **FEAT_KW)
    b = direct_features(changed, start, np.array([200]), **FEAT_KW)
    pd.testing.assert_frame_equal(a, b)


def test_seasonal_lag_unknown_beyond_its_period_is_nan():
    y = np.arange(400, dtype=float)
    X = direct_features(y, pd.Timestamp("2025-01-01"), np.array([300]), 30, [1], [6], [24])
    assert X["seas_24"].iloc[:24].notna().all()
    assert X["seas_24"].iloc[24:].isna().all()


def test_xgb_fits_and_predicts():
    y = _series()
    model = XGBModel(horizon=24, n_estimators=30, train_origin_every=6).fit(y.iloc[:-24])
    pred = model.predict(y.iloc[:-24], 24)
    assert pred.shape == (24,)
    assert np.sqrt(np.mean((pred - y.iloc[-24:].to_numpy()) ** 2)) < 5  # learned the cycle


def test_sarima_fits_and_predicts():
    y = _series(days=30)
    model = SarimaModel(train_days=14, history_days=7).fit(y.iloc[:-24])
    pred = model.predict(y.iloc[:-24], 24)
    assert pred.shape == (24,)
    assert (pred >= 0).all()


def test_pyfunc_wrapper_returns_timestamped_forecast():
    y = _series()
    wrapper = ForecastPyfunc()
    wrapper.model = XGBModel(horizon=24, n_estimators=10, train_origin_every=12).fit(y)
    out = wrapper.predict(None, history_frame(y, 24 * 14), {"horizon": 6})
    assert list(out.columns) == ["timestamp", "forecast"]
    assert len(out) == 6
    assert out["timestamp"].iloc[0] == y.index[-1] + pd.Timedelta(hours=1)


def test_apply_overrides_parses_yaml_values():
    cfg = apply_overrides({"max_depth": 6}, ["max_depth=8", "lags=[1, 24]", "log_target=true"])
    assert cfg == {"max_depth": 8, "lags": [1, 24], "log_target": True}
    with pytest.raises(SystemExit):
        apply_overrides({}, ["oops"])


def test_sweep_suggest_respects_search_space():
    import optuna

    from src.sweep import suggest

    space = {
        "max_depth": {"type": "int", "low": 3, "high": 5},
        "lr": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "n": {"type": "int", "low": 100, "high": 300, "step": 100},
        "flag": {"type": "categorical", "choices": [True, False]},
    }
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    for _ in range(20):
        trial = study.ask()
        v = {k: suggest(trial, k, s) for k, s in space.items()}
        assert 3 <= v["max_depth"] <= 5
        assert 0.01 <= v["lr"] <= 0.3
        assert v["n"] in (100, 200, 300)
        assert v["flag"] in (True, False)
        study.tell(trial, 0.0)
