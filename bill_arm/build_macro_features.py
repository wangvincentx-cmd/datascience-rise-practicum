"""
Fetch macroeconomic climate indicators from FRED's public CSV endpoint (no
API key needed) and build a daily "as-of-introduction" feature table.

Leakage discipline: government statistics are revised and released with a
lag after the period they describe. A bill introduced on a series'
observation date does NOT have that value available yet. Each series below
is shifted forward by its typical publication lag before being joined to
the calendar, so the value attached to a given day is the value that would
actually have been public knowledge by then -- consistent with this
project's introduction-time-only leakage rule (see README).

USREC (NBER recession indicator) is a special case: the NBER Business Cycle
Dating Committee has historically announced recession start/end dates
6-21 months after the fact, not in real time. It is shifted by a
conservative 365-day lag to approximate this. Treat recession_flag as a
backward-looking label, not a real-time signal, and disclose this in the
writeup.

Output: data/macro_daily.csv, one row per calendar day covering the bill
corpus's date range (2002-01-01..2025-06-01), each series forward-filled to
its last known, lag-adjusted value.

Usage:
  python build_macro_features.py
"""

import argparse
import io
from pathlib import Path

import pandas as pd
import requests

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"

# (series_id, output column, publication lag in days)
# Lags are approximate typical release schedules, not per-observation exact:
#   UNRATE     monthly, BLS releases ~1st Friday of the following month
#   USREC      NBER recession dating, announced with a long real-world lag
#   GDPC1      quarterly, BEA advance estimate ~1 month after quarter end
#              (observation_date is the quarter's first day, hence +120)
#   CPIAUCSL   monthly, BLS releases ~2 weeks after month end
#   UMCSENT    monthly, U. Michigan final reading at month end
#   ICSA       weekly initial jobless claims, released ~5 days after week end
SERIES = [
    ("UNRATE", "unemployment_rate", 45),
    ("USREC", "recession_flag", 365),
    ("GDPC1", "real_gdp", 120),
    ("CPIAUCSL", "cpi", 45),
    ("UMCSENT", "consumer_sentiment", 30),
    ("ICSA", "initial_claims", 7),
]


def fetch_series(series_id):
    resp = requests.get(FRED_CSV.format(series=series_id), timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = ["date", series_id]
    df["date"] = pd.to_datetime(df["date"])
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
    return df.dropna()


def build_macro_daily(start="2002-01-01", end="2025-06-01"):
    raw = {series_id: fetch_series(series_id) for series_id, _, _ in SERIES}

    # Derived growth rates, computed at native frequency before lagging.
    raw["GDPC1"]["gdp_growth_yoy"] = raw["GDPC1"]["GDPC1"].pct_change(4) * 100
    raw["CPIAUCSL"]["cpi_inflation_yoy"] = raw["CPIAUCSL"]["CPIAUCSL"].pct_change(12) * 100

    columns = [
        ("UNRATE", "unemployment_rate", 45),
        ("USREC", "recession_flag", 365),
        ("gdp_growth_yoy", "gdp_growth_yoy", 120),
        ("cpi_inflation_yoy", "cpi_inflation_yoy", 45),
        ("UMCSENT", "consumer_sentiment", 30),
        ("ICSA", "initial_claims", 7),
    ]

    daily = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
    for src_col, out_col, lag_days in columns:
        series_id = "GDPC1" if src_col == "gdp_growth_yoy" else (
            "CPIAUCSL" if src_col == "cpi_inflation_yoy" else src_col)
        s = raw[series_id][["date", src_col]].dropna().copy()
        s["date"] = s["date"] + pd.Timedelta(days=lag_days)
        s = s.rename(columns={src_col: out_col}).sort_values("date")
        daily = pd.merge_asof(daily.sort_values("date"), s, on="date", direction="backward")

    return daily


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2002-01-01")
    ap.add_argument("--end", default="2025-06-01")
    ap.add_argument("--out", default="data/macro_daily.csv")
    args = ap.parse_args()

    daily = build_macro_daily(args.start, args.end)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(args.out, index=False)
    print(f"{len(daily)} days -> {args.out}")
    print(daily.describe())


if __name__ == "__main__":
    main()
