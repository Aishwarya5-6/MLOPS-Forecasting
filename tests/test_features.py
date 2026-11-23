import numpy as np
import pandas as pd

from src.features import build_features


def _series(n=400):
    idx = pd.date_range("2024-10-01", periods=n, freq="h")
    return pd.DataFrame({"pm25": np.arange(n, dtype=float)}, index=idx)


def test_lags_use_only_past_values():
    out = build_features(_series(), lags=[1, 24], windows=[6])
    assert out["lag_1"].iloc[100] == 99.0
    assert out["lag_24"].iloc[100] == 76.0


def test_rolling_excludes_current_hour():
    out = build_features(_series(), lags=[1], windows=[6])
    # mean of hours 94..99 for row 100
    assert out["roll_mean_6"].iloc[100] == np.mean(range(94, 100))


def test_no_feature_changes_when_future_changes():
    """Leakage check: altering the future must not change features at earlier times."""
    base = _series()
    changed = base.copy()
    changed.iloc[300:, 0] = 9999.0
    f1 = build_features(base, lags=[1, 24, 168], windows=[6, 24]).drop(columns="pm25")
    f2 = build_features(changed, lags=[1, 24, 168], windows=[6, 24]).drop(columns="pm25")
    pd.testing.assert_frame_equal(f1.iloc[:301], f2.iloc[:301])


def test_calendar_and_diwali_holiday():
    out = build_features(_series(24 * 40), lags=[1], windows=[6])
    assert out.loc["2024-10-31 12:00", "is_holiday"] == 1  # Diwali 2024
    assert out.loc["2024-10-05 12:00", "is_weekend"] == 1  # Saturday
    assert set(out["hour"].unique()) == set(range(24))
