"""Featurize stage: hourly PM2.5 series -> lag, rolling and calendar features.

    python -m src.features   (run by `dvc repro` as the `featurize` stage)

Every feature at time t uses only values from before t (no leakage).
"""

from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

from src.config import ROOT, load_params
from src.data import PREPARED_FILE

FEATURES_FILE = ROOT / "data/processed/features.parquet"


def add_lag_features(df: pd.DataFrame, lags: list[int], col: str = "pm25") -> pd.DataFrame:
    out = df.copy()
    for lag in lags:
        out[f"lag_{lag}"] = out[col].shift(lag)
    return out


def add_rolling_features(df: pd.DataFrame, windows: list[int], col: str = "pm25") -> pd.DataFrame:
    """Rolling mean/std over the previous `w` hours, excluding the current hour."""
    out = df.copy()
    past = out[col].shift(1)
    for w in windows:
        roll = past.rolling(w, min_periods=max(1, w // 2))
        out[f"roll_mean_{w}"] = roll.mean()
        out[f"roll_std_{w}"] = roll.std()
    return out


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = out.index
    out["hour"] = idx.hour
    out["dayofweek"] = idx.day_of_week
    out["month"] = idx.month
    out["is_weekend"] = (idx.day_of_week >= 5).astype(int)
    out["is_holiday"] = holiday_flags(idx)
    return out


def holiday_flags(idx: pd.DatetimeIndex) -> np.ndarray:
    """1 where the date is an Indian national holiday (incl. Diwali), else 0."""
    india = holidays.India(years=range(idx.year.min(), idx.year.max() + 1))
    return pd.Index(idx.date).isin(list(india.keys())).astype(int)


def build_features(df: pd.DataFrame, lags: list[int], windows: list[int]) -> pd.DataFrame:
    out = add_lag_features(df, lags)
    out = add_rolling_features(out, windows)
    return add_calendar_features(out)


# ---------------------------------------------------------------------------
# Direct multi-horizon features (used by XGBoost, Phase 3)
# ---------------------------------------------------------------------------


def _take(y: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """y[idx], with NaN wherever idx falls outside the array."""
    ok = (idx >= 0) & (idx < len(y))
    out = np.full(idx.shape, np.nan)
    out[ok] = y[idx[ok]]
    return out


def direct_features(
    y: np.ndarray,
    start: pd.Timestamp,
    positions: np.ndarray,
    horizon: int,
    lags: list[int],
    windows: list[int],
    seasonal_lags: list[int],
) -> pd.DataFrame:
    """One row per (forecast origin, step ahead), using only data known at the origin.

    `y` is a regular hourly series starting at `start`. An origin at position `pos` means
    the model has seen y[:pos] and forecasts y[pos], ..., y[pos + horizon - 1].

    Features per row:
      - step: 1..horizon (how far ahead)
      - lag_k: y[pos - k], the k-th most recent value at the origin
      - roll_mean_w / roll_std_w: over the last w hours before the origin
      - seas_s: the value s hours before the *target* hour (e.g. same hour yesterday),
        set to NaN if that hour is not yet known at the origin
      - hour, dayofweek, month, is_weekend, is_holiday of the *target* hour
    """
    positions = np.asarray(positions, dtype=int)
    n = len(positions)
    pos = np.repeat(positions, horizon)
    step = np.tile(np.arange(1, horizon + 1), n)
    target = pos + step - 1

    cols: dict[str, np.ndarray] = {"step": step}
    for k in lags:
        cols[f"lag_{k}"] = _take(y, pos - k)
    series = pd.Series(y)
    for w in windows:
        roll = series.rolling(w, min_periods=max(1, w // 2))
        cols[f"roll_mean_{w}"] = _take(roll.mean().to_numpy(), pos - 1)
        cols[f"roll_std_{w}"] = _take(roll.std().to_numpy(), pos - 1)
    for s in seasonal_lags:
        src_idx = target - s
        vals = _take(y, src_idx)
        vals[src_idx >= pos] = np.nan  # not known yet at the origin
        cols[f"seas_{s}"] = vals

    times = pd.DatetimeIndex(start + pd.to_timedelta(target, unit="h"))
    cols["hour"] = times.hour.to_numpy()
    cols["dayofweek"] = times.day_of_week.to_numpy()
    cols["month"] = times.month.to_numpy()
    cols["is_weekend"] = (times.day_of_week >= 5).astype(int)
    cols["is_holiday"] = holiday_flags(times)
    return pd.DataFrame(cols)


def main() -> None:
    params = load_params()["featurize"]
    df = pd.read_parquet(PREPARED_FILE)
    out = build_features(df, params["lags"], params["rolling_windows"])
    out.to_parquet(FEATURES_FILE)
    print(f"featurize: {len(out):,} rows x {out.shape[1]} columns -> {FEATURES_FILE.name}")


if __name__ == "__main__":
    main()
