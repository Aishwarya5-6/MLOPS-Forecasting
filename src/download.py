"""Download Hyderabad PM2.5 data into data/raw/.

    # Open-Meteo (current source; no key needed): hourly CAMS PM2.5 estimates, Aug 2022 -> now
    python -m src.download openmeteo [--start 2022-08-01] [--end 2026-11-22]

OpenAQ station data (kept for later; blocked until an OPENAQ_API_KEY works):

    # 1. List PM2.5 stations near Hyderabad (needs OPENAQ_API_KEY in .env)
    python -m src.download stations

    # 2. Download the full hourly history of one station from the public OpenAQ S3 archive
    python -m src.download history --location-id 12345 [--start-year 2019]

The archive is public (no key needed): one gzipped CSV per location per day, at
s3://openaq-data-archive/records/csv.gz/locationid=<id>/year=<yyyy>/month=<mm>/
"""

from __future__ import annotations

import argparse
import gzip
import io
import os
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from src.config import ROOT, load_params

API_URL = "https://api.openaq.org/v3"
ARCHIVE_URL = "https://openaq-data-archive.s3.amazonaws.com"
OPENMETEO_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
HYDERABAD = (17.385, 78.4867)  # lat, lon of city centre
S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def list_stations(radius_m: int = 25_000) -> pd.DataFrame:
    """Return PM2.5 monitoring stations within radius_m of Hyderabad, best coverage first."""
    key = os.getenv("OPENAQ_API_KEY")
    if not key:
        raise SystemExit("OPENAQ_API_KEY missing: add it to .env (free key at explore.openaq.org)")
    resp = requests.get(
        f"{API_URL}/locations",
        params={
            "coordinates": f"{HYDERABAD[0]},{HYDERABAD[1]}",
            "radius": radius_m,
            "parameters_id": 2,  # 2 = pm25 in OpenAQ v3
            "limit": 100,
        },
        headers={"X-API-Key": key},
        timeout=30,
    )
    resp.raise_for_status()
    rows = []
    for loc in resp.json()["results"]:
        rows.append(
            {
                "location_id": loc["id"],
                "name": loc["name"],
                "provider": (loc.get("provider") or {}).get("name"),
                "first": (loc.get("datetimeFirst") or {}).get("local"),
                "last": (loc.get("datetimeLast") or {}).get("local"),
                "is_monitor": loc.get("isMonitor"),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("first").reset_index(drop=True)


def _list_keys(prefix: str) -> list[str]:
    """List every object key under an S3 prefix (handles pagination)."""
    keys: list[str] = []
    token = None
    while True:
        params = {"list-type": 2, "prefix": prefix}
        if token:
            params["continuation-token"] = token
        resp = requests.get(ARCHIVE_URL, params=params, timeout=30)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        keys += [k.text for k in root.findall("s3:Contents/s3:Key", S3_NS)]
        if root.findtext("s3:IsTruncated", namespaces=S3_NS) != "true":
            return keys
        token = root.findtext("s3:NextContinuationToken", namespaces=S3_NS)


def _fetch_csv(key: str) -> pd.DataFrame:
    resp = requests.get(f"{ARCHIVE_URL}/{key}", timeout=60)
    resp.raise_for_status()
    with gzip.open(io.BytesIO(resp.content)) as f:
        return pd.read_csv(f)


def download_history(location_id: int, start_year: int, parameter: str) -> pd.DataFrame:
    keys = [
        k
        for k in _list_keys(f"records/csv.gz/locationid={location_id}/")
        if int(k.split("year=")[1][:4]) >= start_year
    ]
    if not keys:
        raise SystemExit(f"No archive files for location {location_id} since {start_year}")
    print(f"Downloading {len(keys)} daily files for location {location_id}...")
    with ThreadPoolExecutor(max_workers=16) as pool:
        frames = list(pool.map(_fetch_csv, keys))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["parameter"] == parameter]
    return df.sort_values("datetime").reset_index(drop=True)


def download_openmeteo(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """Hourly PM2.5 from the Open-Meteo air-quality API, one request per calendar year.

    Times are requested in UTC and written as ISO strings with a `Z` suffix, so
    `src.data.load_raw` handles them the same way as OpenAQ data.
    """
    frames = []
    for year_start in pd.date_range(pd.Timestamp(start).to_period("Y").start_time, end, freq="YS"):
        chunk_start = max(pd.Timestamp(start), year_start)
        chunk_end = min(pd.Timestamp(end), year_start + pd.offsets.YearEnd(0))
        resp = requests.get(
            OPENMETEO_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": "pm2_5",
                "start_date": chunk_start.date().isoformat(),
                "end_date": chunk_end.date().isoformat(),
                "timezone": "GMT",
            },
            timeout=60,
        )
        resp.raise_for_status()
        hourly = resp.json()["hourly"]
        frames.append(pd.DataFrame({"datetime": hourly["time"], "value": hourly["pm2_5"]}))
    df = pd.concat(frames, ignore_index=True).dropna(subset=["value"])
    df["datetime"] = df["datetime"] + ":00Z"
    return df.reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    om = sub.add_parser("openmeteo", help="Download hourly PM2.5 from Open-Meteo (no key)")
    om.add_argument("--start", default=None, help="YYYY-MM-DD (default: params.yaml)")
    om.add_argument("--end", default=None, help="YYYY-MM-DD (default: yesterday)")
    sub.add_parser("stations", help="List PM2.5 stations near Hyderabad")
    hist = sub.add_parser("history", help="Download one station's history from the archive")
    hist.add_argument("--location-id", type=int, required=True)
    hist.add_argument("--start-year", type=int, default=2019)
    args = parser.parse_args(argv)

    if args.cmd == "stations":
        with pd.option_context("display.width", 200, "display.max_columns", None):
            print(list_stations())
        return

    params = load_params()["data"]
    if args.cmd == "openmeteo":
        end = args.end or (pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=1)).date()
        df = download_openmeteo(
            params["latitude"], params["longitude"], args.start or params["start_date"], str(end)
        )
    else:
        df = download_history(args.location_id, args.start_year, params["parameter"])
    out = ROOT / params["raw_file"]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} rows ({df['datetime'].min()} -> {df['datetime'].max()}) to {out}")


if __name__ == "__main__":
    main()
