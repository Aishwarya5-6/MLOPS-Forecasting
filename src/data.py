"""Prepare stage: raw OpenAQ readings -> clean, regular hourly PM2.5 series.

python -m src.data   (run by `dvc repro` as the `prepare` stage)
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config import ROOT, load_params

PREPARED_FILE = ROOT / "data/processed/pm25_hourly.parquet"


def load_raw(path, timezone: str) -> pd.DataFrame:
    """Read the raw archive CSV and return (timestamp, value) in local time, tz-naive."""
    df = pd.read_csv(path, usecols=["datetime", "value"])
    ts = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(timezone).dt.tz_localize(None)
    return pd.DataFrame({"timestamp": ts, "value": df["value"].astype(float)})


def clean(
    df: pd.DataFrame,
    freq: str = "h",
    min_value: float = 0.0,
    max_value: float = 999.0,
    max_gap_hours: int = 6,
) -> pd.DataFrame:
    """Return a regular hourly series with columns: pm25, is_imputed, is_missing.

    - Invalid readings (<= min_value or > max_value) are dropped.
    - Multiple readings in one hour are averaged.
    - Gaps up to max_gap_hours are filled by time interpolation and flagged `is_imputed`.
    - Longer gaps stay NaN and are flagged `is_missing`; models decide how to handle them.
    """
    valid = df[(df["value"] > min_value) & (df["value"] <= max_value)]
    hourly = valid.set_index("timestamp")["value"].resample(freq).mean()
    full_index = pd.date_range(hourly.index.min(), hourly.index.max(), freq=freq)
    hourly = hourly.reindex(full_index)

    was_missing = hourly.isna()
    # limit_area="inside" never extrapolates; mask out filled values that belong to long gaps.
    gap_id = (~was_missing).cumsum()
    gap_len = was_missing.groupby(gap_id).transform("sum")
    filled = hourly.interpolate(method="time", limit_area="inside")
    filled[was_missing & (gap_len > max_gap_hours)] = float("nan")

    out = pd.DataFrame({"pm25": filled})
    out["is_imputed"] = was_missing & filled.notna()
    out["is_missing"] = filled.isna()
    out.index.name = "timestamp"
    return out


# ---------------------------------------------------------------------------
# Walk-forward splits
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Fold:
    """One walk-forward split. Train is everything up to and including `train_end`.

    The test window is [test_start, test_end]. Forecasts are made at each `origin`
    for the `horizon` hours starting at that origin.
    """

    index: int
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    origins: tuple[pd.Timestamp, ...]


def _origins(start: pd.Timestamp, end: pd.Timestamp, horizon: int, every: int) -> tuple:
    """Forecast origins in [start, end] whose full horizon still fits inside the window."""
    last = end - pd.Timedelta(hours=horizon - 1)
    return tuple(pd.date_range(start, last, freq=f"{every}h"))


def make_folds(
    index: pd.DatetimeIndex,
    horizon: int,
    n_folds: int,
    test_hours: int,
    step_hours: int,
    holdout_hours: int,
    origin_every: int,
) -> tuple[list[Fold], Fold]:
    """Expanding-window folds plus a final holdout window.

    - The holdout covers roughly the last `holdout_hours`, starting at midnight. It is
      returned separately and used only by promote.py, never for tuning.
    - Fold k's test window starts at holdout_start - (n_folds - k) * step_hours, so the folds
      end just before the holdout and are spread out to cover different seasons.
    - Training data for each fold is everything before its test window (expanding window).
    """
    if test_hours > step_hours:
        raise ValueError("fold_test_hours must be <= fold_step_hours so test windows don't overlap")
    hour = pd.Timedelta(hours=1)
    holdout_start = (index[-holdout_hours]).ceil("D")
    holdout = Fold(
        index=-1,
        train_end=holdout_start - hour,
        test_start=holdout_start,
        test_end=index[-1],
        origins=_origins(holdout_start, index[-1], horizon, origin_every),
    )
    folds = []
    for k in range(n_folds):
        start = holdout_start - pd.Timedelta(hours=(n_folds - k) * step_hours)
        end = start + pd.Timedelta(hours=test_hours) - hour
        if start - hour < index[0]:
            raise ValueError(f"Fold {k} would start before the data does; use fewer folds")
        folds.append(
            Fold(
                index=k,
                train_end=start - hour,
                test_start=start,
                test_end=end,
                origins=_origins(start, end, horizon, origin_every),
            )
        )
    return folds, holdout


def folds_from_params(index: pd.DatetimeIndex, params: dict | None = None):
    f = (params or load_params())["forecast"]
    return make_folds(
        index,
        horizon=f["horizon"],
        n_folds=f["n_folds"],
        test_hours=f["fold_test_hours"],
        step_hours=f["fold_step_hours"],
        holdout_hours=f["holdout_hours"],
        origin_every=f["origin_every_hours"],
    )


def main() -> None:
    params = load_params()
    raw = load_raw(ROOT / params["data"]["raw_file"], params["data"]["timezone"])
    p = params["prepare"]
    out = clean(raw, p["freq"], p["min_value"], p["max_value"], p["max_gap_hours"])
    PREPARED_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PREPARED_FILE)
    print(
        f"prepare: {len(out):,} hours ({out.index.min()} -> {out.index.max()}), "
        f"imputed {out['is_imputed'].mean():.1%}, still missing {out['is_missing'].mean():.1%}"
    )


if __name__ == "__main__":
    main()
