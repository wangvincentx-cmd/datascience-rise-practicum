"""
Build the monthly press-expectations index -- the poster's headline artifact.

Aggregates extracted forecast claims to one row per calendar month, so press
expectations can be plotted as a continuous time series (1900-1963) and tested
against what the economy actually did. This is the deliverable the continuous
monthly corpus exists for; the crisis corpus cannot produce it (decade gaps).

Every series is a SHARE or a RATE, never a raw count. Newspaper digitisation
varies by an order of magnitude across these decades, so a raw monthly claim
count would track how much of that month was digitised, not the press. The
denominator (pages sampled that month) comes from the page corpus.

Series produced (month x):
  n_claims        raw count, for reference / weighting only
  attention       forecasts per 100 pages sampled           RATE
  net_direction   share(improve) - share(worsen)            [-1, +1]
  share_worsen    share predicting the economy worsens
  hedge_rate      share hedged (LLM label; see caveat)      SHARE
  disagreement    1 - |net_direction|, among directional    [0, 1]  (BBD-style)
  mean_horizon    mean stated/inferred horizon in months
  share_named     share with a named forecaster (specificity proxy)
  share_expert / share_official / share_journalist          voice mix

Scope: only `national` (and unlabelled) claims enter the direction series --
foreign/regional/industry forecasts are not about the US economy the index
represents. They still count toward `attention` (the press WAS forecasting).

Usage:
    python build_press_index.py --claims claims_monthly.jsonl \\
        --pages data/monthly/pages_monthly.jsonl --out data/press_index.csv
    # rehearsal on the crisis corpus (sparse, but proves the aggregation):
    python build_press_index.py --claims claims_v2.jsonl --pages data/pages.jsonl \\
        --out data/press_index_cached.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_claims(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    df = pd.DataFrame(rows)
    df["month"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M")
    return df[df["month"].notna()].copy()


def pages_per_month(pages_path):
    """Distinct pages sampled per month -- the denominator for `attention`."""
    if not pages_path or not Path(pages_path).exists():
        return None
    rows = [json.loads(l) for l in open(pages_path, encoding="utf-8") if l.strip()]
    p = pd.DataFrame(rows)
    if "month" in p.columns:
        m = p["month"].astype(str)
    else:
        m = pd.to_datetime(p["date"], errors="coerce").dt.to_period("M").astype(str)
    key = "page_id" if "page_id" in p.columns else p.columns[0]
    return p.assign(_m=m).groupby("_m")[key].nunique()


def build(claims_path, pages_path=None):
    df = load_claims(claims_path)
    df["dir"] = df.get("direction", "").astype(str)
    df["is_improve"] = (df["dir"] == "improve").astype(int)
    df["is_worsen"] = (df["dir"] == "worsen").astype(int)
    df["is_dir"] = df["is_improve"] + df["is_worsen"]
    df["hedged"] = (df.get("confidence", "").astype(str) == "hedged").astype(int)
    df["named"] = (df.get("speaker_name", "na").astype(str).str.lower().ne("na")).astype(int)
    df["horizon_m"] = pd.to_numeric(df.get("horizon_months", np.nan),
                                    errors="coerce")  # "vague" -> NaN naturally
    for v in ["expert", "official", "journalist"]:
        df[f"v_{v}"] = (df.get("voice", "").astype(str) == v).astype(int)

    scope = df.get("scope", "national").astype(str)
    nat = df[scope.isin(["national", "na", ""]) | scope.isna()].copy()

    def agg(g):
        d = g["is_dir"].sum()
        imp, wor = g["is_improve"].sum(), g["is_worsen"].sum()
        net = (imp - wor) / d if d else np.nan
        return pd.Series({
            "n_claims": len(g),
            "n_directional": int(d),
            "net_direction": net,
            "share_worsen": wor / d if d else np.nan,
            "share_improve": imp / d if d else np.nan,
            "disagreement": (1 - abs(net)) if d else np.nan,
            "hedge_rate": g["hedged"].mean(),
            "mean_horizon": g["horizon_m"].mean(),
            "share_named": g["named"].mean(),
            "share_expert": g["v_expert"].mean(),
            "share_official": g["v_official"].mean(),
            "share_journalist": g["v_journalist"].mean(),
        })

    idx = nat.groupby(nat["month"].astype(str)).apply(agg, include_groups=False)

    # attention = ALL forecasts (any scope) per 100 pages sampled that month
    total_by_month = df.groupby(df["month"].astype(str)).size()
    ppm = pages_per_month(pages_path)
    idx["n_all_claims"] = total_by_month.reindex(idx.index)
    if ppm is not None:
        idx["n_pages"] = ppm.reindex(idx.index)
        idx["attention"] = 100.0 * idx["n_all_claims"] / idx["n_pages"]
    else:
        idx["attention"] = np.nan  # no page denominator available

    idx.index.name = "month"
    return idx.reset_index()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--claims", required=True)
    ap.add_argument("--pages", default=None,
                    help="page corpus, for the attention denominator")
    ap.add_argument("--out", default="data/press_index.csv")
    args = ap.parse_args()

    idx = build(args.claims, args.pages)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    idx.to_csv(args.out, index=False)
    print(f"{len(idx)} months -> {args.out}")
    dir_cov = idx["net_direction"].notna().sum()
    print(f"  months with a directional signal: {dir_cov}")
    if dir_cov:
        print(f"  net_direction range: {idx['net_direction'].min():+.2f} "
              f"to {idx['net_direction'].max():+.2f}")
        print(f"  hedge_rate mean: {idx['hedge_rate'].mean():.2f}")
    if idx["attention"].notna().any():
        print(f"  attention (fcasts/100pg) mean: {idx['attention'].mean():.1f}")
    print("\n  sample rows:")
    cols = ["month", "n_claims", "net_direction", "disagreement", "hedge_rate", "attention"]
    print(idx[cols].head(8).to_string(index=False))
