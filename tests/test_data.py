import numpy as np
import pandas as pd

from src.data import clean


def _raw(values, start="2024-01-01 00:00"):
    ts = pd.date_range(start, periods=len(values), freq="h")
    return pd.DataFrame({"timestamp": ts, "value": values})


def test_invalid_values_dropped_then_interpolated():
    out = clean(_raw([10.0, -5.0, 30.0]), max_gap_hours=6)
    assert out["pm25"].tolist() == [10.0, 20.0, 30.0]
    assert out["is_imputed"].tolist() == [False, True, False]


def test_long_gaps_stay_missing():
    values = [10.0] + [np.nan] * 8 + [20.0]
    out = clean(_raw(values), max_gap_hours=6)
    assert out["pm25"].iloc[1:9].isna().all()
    assert out["is_missing"].sum() == 8
    assert not out["is_imputed"].any()


def test_short_gap_filled_long_gap_not():
    values = [10.0, np.nan, 12.0] + [np.nan] * 7 + [5.0]
    out = clean(_raw(values), max_gap_hours=6)
    assert out["pm25"].iloc[1] == 11.0
    assert out["pm25"].iloc[3:10].isna().all()


def test_sub_hourly_readings_averaged_and_index_regular():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-01 00:10", "2024-01-01 00:40", "2024-01-01 02:00"]
            ),
            "value": [10.0, 20.0, 40.0],
        }
    )
    out = clean(df)
    assert out.index.freq is None or out.index.freqstr == "h"
    assert len(out) == 3
    assert out["pm25"].iloc[0] == 15.0


def test_load_station_filters_city(tmp_path):
    from src.station import load_station

    p = tmp_path / "city_hour.csv"
    p.write_text(
        "City,Datetime,PM2.5,PM10\n"
        "Delhi,2019-01-01 00:00:00,200,300\n"
        "Hyderabad,2019-01-01 00:00:00,40.5,80\n"
        "Hyderabad,2019-01-01 01:00:00,,80\n"
    )
    df = load_station(p)
    assert list(df["value"].fillna(-1)) == [40.5, -1]
    assert df["timestamp"].iloc[0] == pd.Timestamp("2019-01-01 00:00")
