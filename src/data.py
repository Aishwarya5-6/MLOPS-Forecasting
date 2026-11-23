"""Prepare stage: raw OpenAQ readings -> clean, regular hourly PM2.5 series.

python -m src.data   (run by `dvc repro` as the `prepare` stage)
"""

from __future__ import annotations

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
