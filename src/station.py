"""Prepare the CPCB station dataset for the promotion study.

Source: Kaggle "Air Quality Data in India (2015-2020)", file city_hour.csv (downloaded by hand into
data/raw/station/city_hour.csv, versioned with `dvc add`). Columns used: City, Datetime, PM2.5.
Timestamps are already local time (IST). Cleaning is the same `clean()` as the CAMS series.

    python -m src.station [--city Delhi]      (DVC stage `prepare_station@<city>`)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import ROOT, load_params
from src.data import clean

RAW = ROOT / "data/raw/station/city_hour.csv"
OUT = ROOT / "data/processed/station_pm25_hourly.parquet"  # Hyderabad (the primary dataset)
CITY = "Hyderabad"


def output_path(city: str) -> Path:
    return OUT if city == CITY else ROOT / f"data/processed/station_{city.lower()}.parquet"


def load_station(path=RAW, city: str = CITY) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["City", "Datetime", "PM2.5"])
    df = df[df["City"] == city]
    if df.empty:
        raise ValueError(f"no rows for City == {city!r} in {path}")
    return pd.DataFrame(
        {"timestamp": pd.to_datetime(df["Datetime"]), "value": df["PM2.5"].astype(float)}
    ).dropna(subset=["timestamp"])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default=CITY)
    city = ap.parse_args(argv).city
    p = load_params()["prepare"]
    raw = load_station(city=city)
    out = clean(raw, p["freq"], p["min_value"], p["max_value"], p["max_gap_hours"])
    path = output_path(city)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path)
    print(f"{city}: ", end="")
    imputed, missing = out["is_imputed"].mean(), out["is_missing"].mean()
    print(f"{len(out):,} hours {out.index[0]} -> {out.index[-1]}; "
          f"imputed {imputed:.1%}, missing {missing:.1%}")  # fmt: skip


if __name__ == "__main__":
    main()
