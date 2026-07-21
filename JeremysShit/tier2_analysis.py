"""
Tier 2 analyses: uncertainty, geography, and a household-sentiment benchmark.

1. EPU — Baker-Bloom-Davis US *Historical* Economic Policy Uncertainty index
   (newspaper-based, monthly, 1900-2014, policyuncertainty.com). Tests the team's
   third goal directly: did predictions fail when policy uncertainty was high?

2. GEOGRAPHY — hit rate by Census region, financial-center states (NY/IL/MA/PA),
   and DC as its own political-hub bucket. Did papers near the money predict
   better?

3. MICHIGAN — "Michigan" is shorthand for the University of Michigan Survey
   Research Center's Surveys of Consumers (FRED series UMCSENT, 1952- ), the
   same index referenced when people say "consumer sentiment" in financial
   news. It is a NATIONAL survey of US households (nationally representative
   random-digit-dial sample) -- the name refers to the institution that
   administers it, not to Michigan residents or Michigan the state. Already
   answers "all of America's households"; there's no separate more-national
   version to switch to.
   Households' implied directional call (sentiment rising = "improve") scored
   against realized industrial production, giving a third benchmark:
   newspapers vs. expert economists vs. ordinary households.
   Caveat to state on the poster: sentiment is an *implied* forecast, not a
   stated one — it's the closest thing to a continuous household poll.

Usage:
    python tier2_analysis.py                    # needs claims_scored.csv
    python tier2_analysis.py --claims other.csv

Outputs: printed tables, results_by_region.csv, figures/fig_epu_vs_accuracy.png,
figures/fig_geography.png, figures/fig_three_way_benchmark.png
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from score_claims import fred, livingston_directional, FRED_HEADERS

CACHE = Path("cache")
FIGDIR = Path("figures")
EPU_URL = "https://www.policyuncertainty.com/media/US_Historical_EPU_data.xlsx"

REGIONS = {
    "northeast": ["connecticut", "maine", "massachusetts", "new hampshire", "rhode island",
                  "vermont", "new jersey", "new york", "pennsylvania"],
    "midwest": ["illinois", "indiana", "michigan", "ohio", "wisconsin", "iowa", "kansas",
                "minnesota", "missouri", "nebraska", "north dakota", "south dakota"],
    "south": ["delaware", "florida", "georgia", "maryland", "north carolina", "south carolina",
              "virginia", "district of columbia", "west virginia", "alabama", "kentucky",
              "mississippi", "tennessee", "arkansas", "louisiana", "oklahoma", "texas"],
    "west": ["arizona", "colorado", "idaho", "montana", "nevada", "new mexico", "utah",
             "wyoming", "alaska", "california", "hawaii", "oregon", "washington"],
}
STATE_TO_REGION = {s: r for r, states in REGIONS.items() for s in states}
FIN_CENTERS = {"new york", "illinois", "massachusetts", "pennsylvania"}
POLITICAL_HUBS = {"district of columbia"}


def epu_series():
    """Monthly historical EPU index (1900-2014) as a Period-indexed Series."""
    CACHE.mkdir(exist_ok=True)
    f = CACHE / "US_Historical_EPU_data.xlsx"
    if not f.exists():
        for attempt in range(3):
            try:
                r = requests.get(EPU_URL, headers=FRED_HEADERS, timeout=90)
                r.raise_for_status()
                f.write_bytes(r.content)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(5)
    raw = pd.read_excel(f)
    # Robust to header naming: find year/month columns, then the EPU value column.
    cols = {str(c).strip().lower(): c for c in raw.columns}
    ycol = next(c for k, c in cols.items() if "year" in k)
    mcol = next(c for k, c in cols.items() if "month" in k)
    vcol = next(c for k, c in cols.items() if c not in (ycol, mcol)
                and pd.api.types.is_numeric_dtype(raw[c]))
    df = raw[[ycol, mcol, vcol]].dropna()
    idx = pd.to_datetime({"year": df[ycol].astype(int), "month": df[mcol].astype(int),
                          "day": 1}).dt.to_period("M")
    return pd.Series(df[vcol].values, index=idx, name="epu").sort_index()


def load_scored(path):
    s = pd.read_csv(path, parse_dates=["date"]).dropna(subset=["hit"])
    s["hit"] = s["hit"].astype(int)
    return s


def epu_analysis(s, plt):
    print("\n=== 1. POLICY UNCERTAINTY: did predictions fail when uncertainty was high? ===")
    epu = epu_series()
    s = s.copy()
    s["epu"] = s["date"].dt.to_period("M").map(epu)
    s = s.dropna(subset=["epu"])
    r = np.corrcoef(s["epu"], s["hit"])[0, 1]
    s["epu_tercile"] = pd.qcut(s["epu"], 3, labels=["low EPU", "mid EPU", "high EPU"])
    tab = s.groupby("epu_tercile", observed=True).agg(
        n=("hit", "size"), hit_rate=("hit", "mean"), mean_epu=("epu", "mean")).round(3)
    print(tab.to_string())
    print(f"point-biserial correlation(EPU, hit) = {r:+.3f}  (n={len(s)})")

    if plt:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        tab["hit_rate"].plot(kind="bar", ax=ax, color=["seagreen", "goldenrod", "crimson"],
                             alpha=.85, rot=0)
        for i, (n, hr) in enumerate(zip(tab["n"], tab["hit_rate"])):
            ax.text(i, hr + 0.02, f"n={n}", ha="center", fontsize=9)
        ax.axhline(0.5, color="gray", ls="--", lw=1)
        ax.set_ylim(0, 1); ax.set_ylabel("hit rate")
        ax.set_title("Prediction accuracy vs. policy uncertainty at the time of the claim\n"
                     "(Baker-Bloom-Davis historical EPU, 1900-2014)")
        plt.tight_layout(); plt.savefig(FIGDIR / "fig_epu_vs_accuracy.png", dpi=200); plt.close()
    return s  # with epu column, for model.py reuse


def geography_analysis(s, plt):
    print("\n=== 2. GEOGRAPHY: did papers near the money predict better? ===")
    s = s.copy()
    s["state"] = s["state"].fillna("").str.lower().str.strip()
    s["region"] = s["state"].map(STATE_TO_REGION).fillna("unknown")
    s["fin_center"] = np.select(
        [s["state"].isin(FIN_CENTERS), s["state"].isin(POLITICAL_HUBS)],
        ["financial-center state", "political hub (DC)"],
        default="elsewhere")
    reg = s[s["region"] != "unknown"].groupby("region").agg(
        n=("hit", "size"), hit_rate=("hit", "mean")).round(3)
    fin = s.groupby("fin_center").agg(n=("hit", "size"), hit_rate=("hit", "mean")).round(3)
    print(reg.to_string()); print(); print(fin.to_string())
    reg.to_csv("results_by_region.csv")

    if plt:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        reg["hit_rate"].plot(kind="bar", ax=axes[0], color="steelblue", alpha=.85, rot=20)
        axes[0].set_title("Hit rate by Census region"); axes[0].set_ylim(0, 1)
        axes[0].axhline(0.5, color="crimson", ls="--", lw=1)
        for i, (n, hr) in enumerate(zip(reg["n"], reg["hit_rate"])):
            axes[0].text(i, hr + 0.02, f"n={n}", ha="center", fontsize=8)
        fin["hit_rate"].plot(kind="bar", ax=axes[1],
                             color=["steelblue", "darkgoldenrod", "seagreen"],
                             alpha=.85, rot=15)
        axes[1].set_title("Financial-center states (NY/IL/MA/PA) vs. DC vs. elsewhere")
        axes[1].set_ylim(0, 1); axes[1].axhline(0.5, color="crimson", ls="--", lw=1)
        for i, (n, hr) in enumerate(zip(fin["n"], fin["hit_rate"])):
            axes[1].text(i, hr + 0.02, f"n={n}", ha="center", fontsize=8)
        plt.tight_layout(); plt.savefig(FIGDIR / "fig_geography.png", dpi=200); plt.close()


def michigan_directional(band=2.0):
    """Households' implied directional calls from Michigan sentiment (FRED UMCSENT).

    Implied prediction at t: sentiment rose over the last 6 months -> "improve",
    fell -> "worsen". Realized: INDPRO % change t -> t+12m with the same +/-2% band
    used for newspaper claims. Returns a DataFrame of (date, pred, act).
    """
    ics = fred("UMCSENT")
    indpro = fred("INDPRO")
    rows = []
    for p in ics.index:
        prev, fwd = p - 6, p + 12
        if prev not in ics.index or p not in indpro.index or fwd not in indpro.index:
            continue
        d_s = ics[p] - ics[prev]
        pred = "improve" if d_s > 0 else "worsen" if d_s < 0 else "no_change"
        chg = (indpro[fwd] / indpro[p] - 1) * 100
        act = "improve" if chg > band else "worsen" if chg < -band else "no_change"
        rows.append({"period": p, "pred": pred, "act": act})
    df = pd.DataFrame(rows)
    df["hit"] = (df["pred"] == df["act"]).astype(int)
    return df


def three_way_benchmark(s, plt):
    print("\n=== 3. THREE-WAY BENCHMARK: newspapers vs. economists vs. households ===")
    rows = []
    news = s[(s["date"] >= "1946-01-01") & (s["date"] < "1964-01-01")]
    if len(news):
        rows.append(("Newspapers (1946-63)", news["hit"].mean(), len(news)))
    liv = livingston_directional()
    rows.append(("Livingston economists (1946-63)", (liv["pred"] == liv["act"]).mean(), len(liv)))
    mich = michigan_directional()
    m5363 = mich[(mich["period"] >= pd.Period("1953-01", "M")) &
                 (mich["period"] <= pd.Period("1963-12", "M"))]
    if len(m5363):
        rows.append(("US households, Michigan SRC survey (1953-63)", m5363["hit"].mean(), len(m5363)))
    rows.append(("US households, Michigan SRC survey (1953-2010)",
                 mich[mich["period"] <= pd.Period("2010-12", "M")]["hit"].mean(),
                 len(mich[mich["period"] <= pd.Period("2010-12", "M")])))
    tab = pd.DataFrame(rows, columns=["forecaster", "hit_rate", "n"]).set_index("forecaster").round(3)
    print(tab.to_string())
    print("(\"Michigan SRC\" = University of Michigan Survey Research Center's Surveys of"
          "\n Consumers (UMCSENT) -- a NATIONAL survey of US households, not Michigan-specific;"
          "\n named for the administering institution. Sentiment change is an implied direction,"
          "\n not a stated forecast; overlapping-window observations are serially correlated,"
          "\n so n overstates precision.)")

    if plt:
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        tab["hit_rate"].plot(kind="barh", ax=ax,
                             color=["steelblue", "seagreen", "goldenrod", "darkgoldenrod"][:len(tab)],
                             alpha=.85)
        ax.axvline(0.5, color="crimson", ls="--", lw=1, label="coin flip")
        ax.set_xlim(0, 1); ax.set_xlabel("directional hit rate"); ax.legend()
        for i, (hr, n) in enumerate(zip(tab["hit_rate"], tab["n"])):
            ax.text(hr + 0.01, i, f"n={n}", va="center", fontsize=8)
        ax.set_title("Who called the economy's direction right?")
        plt.tight_layout(); plt.savefig(FIGDIR / "fig_three_way_benchmark.png", dpi=200)
        plt.close()


def main(args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        FIGDIR.mkdir(exist_ok=True)
    except ImportError:
        plt = None
    s = load_scored(args.claims)
    print(f"{len(s)} scored claims loaded from {args.claims}")
    epu_analysis(s, plt)
    geography_analysis(s, plt)
    three_way_benchmark(s, plt)
    print("\nDone. Figures in figures/, table in results_by_region.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--claims", default="claims_scored.csv")
    args = ap.parse_args()
    main(args)
