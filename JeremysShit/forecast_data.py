"""
Unified forecast table across US professional forecasters -- Greenbook (Fed staff),
Livingston (private economists), SPF (individual professional forecasters) --
harmonized onto ONE provenance-tagged schema for the forecast-credibility model.
See forecast_credibility_PLAN.md.

Each row = one forecaster's ~1-year-ahead directional call at one date, scored on
the SAME INDPRO/NBER ground truth as the newspapers (score_claims.realized_direction,
12-month window). v1 covers real GDP (`variable_canonical == "real_gdp"`) across all
three sources -- the clean apples-to-apples comparison; other variables are a small
extension (SPF Individual_<VAR>, Greenbook gIP/UNEMP, Livingston sheets).

PROVENANCE: every row keeps `source`, `forecaster_id`, `source_file`,
`variable_native`, `horizon_native` so nothing blends silently. A merge audit
(rows x source x era, hit rate per source) prints on every build.

Run:  python forecast_data.py   ->  forecasts.csv  + audit
"""

import functools
import io
import re
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from score_claims import fred, realized_direction
from spf_benchmark import direction_label
from greenbook_benchmark import load_editions

BAND = 1.0                      # annualized-% no-change band, shared with SPF/Greenbook
SPF_DIR = Path("data/forecasts/spf")
LIV_GROWTH = "MedianGrowthRate.xlsx"

CORE = ["source", "forecaster_id", "variable_canonical", "forecast_date",
        "horizon_native", "pred_growth", "pred_direction", "realized_direction",
        "hit", "source_file", "variable_native"]


def read_xlsx_robust(path, sheet_name=0):
    """Philly Fed .xlsx loader (strips the docProps datetime that trips openpyxl),
    generalized to any sheet. Same fix as spf_benchmark.read_xlsx_robust."""
    raw = Path(path).read_bytes()
    zin = zipfile.ZipFile(io.BytesIO(raw))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "docProps/core.xml":
                data = re.sub(rb"<dcterms:(created|modified)[^>]*>[^<]*</dcterms:\1>",
                              b"", data)
            zout.writestr(item, data)
    buf.seek(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.read_excel(buf, engine="openpyxl", sheet_name=sheet_name)


def realized_fn():
    """Memoized realized-direction lookup on the newspapers' INDPRO/NBER rule."""
    cpi, indpro, unrate = fred("CPIAUCNS"), fred("INDPRO"), fred("UNRATE")

    @functools.lru_cache(maxsize=None)
    def f(ts):
        return realized_direction("general_business", "", "", pd.Timestamp(ts), 12,
                                  cpi, indpro, unrate)[0]
    return lambda d: f(pd.Timestamp(d))


def _finish(df, realized):
    df["pred_direction"] = df["pred_growth"].map(lambda g: direction_label(g, BAND))
    df["realized_direction"] = df["forecast_date"].map(realized)
    df["hit"] = [int(p == r) if (p and r) else np.nan
                 for p, r in zip(df["pred_direction"], df["realized_direction"])]
    # All three are scored on the SAME concept -- realized general-business
    # direction (INDPRO/NBER) -- even though the native forecast variable differs
    # (Greenbook/SPF real GDP growth, Livingston industrial production). This
    # RGDP-vs-INDPRO cross is the deliberate, already-established scoring in
    # spf_benchmark ("holds ground truth identical across forecasters"). The
    # native series each forecaster actually reported is kept in variable_native.
    df["variable_canonical"] = "general_business"
    return df[CORE]


def load_greenbook(realized):
    """One ~1yr call per Greenbook edition (mean of the 4 quarters after nowcast)."""
    ed = load_editions("gRGDP").rename(columns={"forecast_growth": "pred_growth"})
    ed["source"] = "greenbook"
    ed["forecaster_id"] = "fed_staff"
    ed["variable_native"] = "gRGDP"
    ed["horizon_native"] = "+1..+4Q"
    return _finish(ed, realized)


def load_spf(realized, var="RGDP"):
    """One ~1yr call per (forecaster x survey quarter): annualized growth from the
    nowcast level (<VAR>2) to the +4-quarter level (<VAR>6). Keeps forecaster ID."""
    df = read_xlsx_robust(SPF_DIR / f"Individual_{var}.xlsx")
    c2, c6 = f"{var}2", f"{var}6"
    df = df.dropna(subset=[c2, c6]).copy()
    df = df[pd.to_numeric(df[c2], errors="coerce") > 0]
    df["pred_growth"] = 100 * (df[c6].astype(float) / df[c2].astype(float) - 1)
    df["forecast_date"] = pd.PeriodIndex.from_fields(
        year=df["YEAR"].astype(int), quarter=df["QUARTER"].astype(int),
        freq="Q").start_time
    df["source"] = "spf"
    df["forecaster_id"] = "spf_" + df["ID"].astype(int).astype(str)
    df["variable_native"] = var
    df["horizon_native"] = "nowcast->+4Q"
    df["source_file"] = f"Individual_{var}.xlsx"
    return _finish(df, realized)


def load_livingston(realized):
    """One ~1yr call per semiannual survey: median growth from the base period (last
    known actual) to 12 months ahead (G_BP_To_12M) off the INDUSTRIAL PRODUCTION
    sheet -- Livingston's long-running series, populated back to 1946 (real GDP
    only starts 1992, which would throw away Livingston's whole reason for being
    here: pre-1967 recession episodes). IP also matches the INDPRO ruler directly.
    Source = the median panel (repo has medians/dispersion, not individual
    Livingston responses)."""
    df = read_xlsx_robust(LIV_GROWTH, sheet_name="IP").rename(
        columns={"Date": "forecast_date"})
    df = df.dropna(subset=["G_BP_To_12M"]).copy()
    df["forecast_date"] = pd.to_datetime(df["forecast_date"])
    df["pred_growth"] = df["G_BP_To_12M"].astype(float)
    df["source"] = "livingston"
    df["forecaster_id"] = "livingston_median"
    df["variable_native"] = "IP"
    df["horizon_native"] = "BP->12M"
    df["source_file"] = LIV_GROWTH
    return _finish(df, realized)


# Real-activity variables (growth up = economy improving) -- all legitimately
# scored on the same INDPRO/general-business ruler. Inflation/interest-rate
# variables are EXCLUDED here (different sign convention / different concept).
SPF_ACTIVITY = ["RGDP", "INDPROD", "EMP", "RCONSUM", "RNRESIN"]
GB_ACTIVITY = ["gRGDP", "gIP"]


def load_greenbook_var(realized, var):
    ed = load_editions(var).rename(columns={"forecast_growth": "pred_growth"})
    ed["source"] = "greenbook"
    ed["forecaster_id"] = "fed_staff"
    ed["variable_native"] = var
    ed["horizon_native"] = "+1..+4Q"
    return _finish(ed, realized)


def build_multivar():
    """Robustness pool: multiple real-activity variables per source (SPF/Greenbook),
    Livingston IP as-is. More activity measures of 'is the economy strengthening',
    same INDPRO ruler; variable_native distinguishes them."""
    realized = realized_fn()
    parts = [load_livingston(realized)]
    for v in GB_ACTIVITY:
        try:
            parts.append(load_greenbook_var(realized, v))
        except Exception as e:
            print(f"  (skip greenbook {v}: {e})")
    for v in SPF_ACTIVITY:
        try:
            parts.append(load_spf(realized, v))
        except Exception as e:
            print(f"  (skip spf {v}: {e})")
    df = pd.concat(parts, ignore_index=True)
    df["forecast_date"] = pd.to_datetime(df["forecast_date"])
    df["year"] = df["forecast_date"].dt.year
    df = df.sort_values(["source", "variable_native", "forecast_date"]).reset_index(drop=True)
    df.to_csv("forecasts_multivar.csv", index=False)
    print("=== MULTIVAR pool ===")
    print(df.groupby(["source", "variable_native"]).size().to_string())
    print(f"total {len(df)} rows -> forecasts_multivar.csv")
    return df


def build():
    realized = realized_fn()
    parts = [load_greenbook(realized), load_livingston(realized), load_spf(realized)]
    df = pd.concat(parts, ignore_index=True)
    df["forecast_date"] = pd.to_datetime(df["forecast_date"])
    df["year"] = df["forecast_date"].dt.year
    df = df.sort_values(["source", "forecast_date"]).reset_index(drop=True)
    df.to_csv("forecasts.csv", index=False)
    audit(df)
    return df


def audit(df):
    print("=== MERGE AUDIT: forecasts.csv ===")
    print(f"total rows: {len(df)}   date range: {df['year'].min()}-{df['year'].max()}")
    scored = df.dropna(subset=["hit"])
    g = scored.groupby("source")
    tab = pd.DataFrame({
        "rows": df.groupby("source").size(),
        "scored": g.size(),
        "forecasters": df.groupby("source")["forecaster_id"].nunique(),
        "yr_min": df.groupby("source")["year"].min(),
        "yr_max": df.groupby("source")["year"].max(),
        "hit_rate": g["hit"].mean().round(3),
        "worsen_share": g.apply(lambda x: (x["pred_direction"] == "worsen").mean(),
                                include_groups=False).round(3),
    })
    print(tab.to_string())
    print("\nvariable_canonical:", df["variable_canonical"].unique().tolist())
    print("provenance columns present:", all(c in df.columns for c in CORE))
    print("forecasts.csv written")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--multivar", action="store_true",
                    help="also build forecasts_multivar.csv (real-activity pool)")
    a = ap.parse_args()
    build()
    if a.multivar:
        build_multivar()
