"""Featurize stage: hourly PM2.5 series -> lag, rolling and calendar features.

    python -m src.features   (run by `dvc repro` as the `featurize` stage)

Every feature at time t uses only values from before t (no leakage).
"""

from __future__ import annotations

import holidays
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
    india = holidays.India(years=range(idx.year.min(), idx.year.max() + 1))
    out["is_holiday"] = pd.Index(idx.date).isin(list(india.keys())).astype(int)
    return out


def build_features(df: pd.DataFrame, lags: list[int], windows: list[int]) -> pd.DataFrame:
    out = add_lag_features(df, lags)
    out = add_rolling_features(out, windows)
    return add_calendar_features(out)


def main() -> None:
    params = load_params()["featurize"]
    df = pd.read_parquet(PREPARED_FILE)
    out = build_features(df, params["lags"], params["rolling_windows"])
    out.to_parquet(FEATURES_FILE)
    print(f"featurize: {len(out):,} rows x {out.shape[1]} columns -> {FEATURES_FILE.name}")


if __name__ == "__main__":
    main()
