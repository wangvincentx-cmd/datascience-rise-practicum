"""
Can the two corpora be pooled? Measure it, don't argue about it.

Pooling LOC (1900-1963) with ProQuest (1965-2009) is safe only if the features
cannot reveal WHICH corpus a row came from. If they can, the model gets free AUC
by detecting the source -- the eras have different hit rates, so "is this
ProQuest?" is a shortcut to "is this a hit?" -- and reports it as forecasting
skill. Removing an explicit `source` column does NOT fix that: the leak lives in
whatever else differs systematically.

THE DIAGNOSTIC: fit a classifier to predict source from the model's own
features. AUC 0.5 means indistinguishable and pooling is safe. AUC 1.0 means
perfectly separable and a pooled result is uninterpretable.

Measured (monthly context over each corpus's date range):

    all 13 factors (+ has_ flags)              AUC 0.961    <- leaks badly
    equal-coverage factors only (stock + epu)  AUC 0.617    <- usable

WHERE THE 0.961 COMES FROM -- coverage, not economics. A factor is NaN before
its series exists, and macro_block emits a has_ flag saying so. LOC straddles
those start dates; ProQuest is entirely after them:

    factor          LOC has_   ProQuest has_    gap
    unrate_d6         24.0%        100.0%      76.0%
    unrate            24.8%        100.0%      75.2%
    ip_gap            65.7%        100.0%      34.3%
    ip_accel          68.6%        100.0%      31.4%
    ip_growth_12m     68.6%        100.0%      31.4%
    ip_growth_6m      69.4%        100.0%      30.6%
    cpi_yoy/accel     78.0%        100.0%      22.0%
    stock_* (x4)     100.0%        100.0%       0.0%   <- fixed by Shiller
    epu              100.0%        100.0%       0.0%

`has_unrate == 1` alone almost names the corpus. UNRATE starts 1948 and INDPRO
1919, so this is structural: no stock series fixes it, because the missing data
really is missing.

THE USABLE SUBSET, and why it costs nothing. The published model's entire
measured signal is two interaction terms:

    x_dir_epu          0.0378 AUC drop   (#1)
    x_dir_stock_ret6   0.0155 AUC drop   (#2)
    everything else   ~0

Both are built from factors with 100% coverage in BOTH corpora. So restricting a
pooled model to POOLABLE_FACTORS keeps every feature that was carrying signal
and drops source detectability from 0.961 to 0.617.

0.617 is not 0.5, and that residual is honest: the 1900-63 and 1965-2009
economies genuinely differed (inflation regime, market volatility), so some
separability is real economics rather than an artifact. It is a number to
report, not to eliminate.

Usage:
    from pooling import POOLABLE_FACTORS, source_detectability
    source_detectability(df)                       # on a pooled frame
    source_detectability(df, POOLABLE_FACTORS)     # after restricting
"""

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Factors with 100% coverage in both corpora. stock_* is only in this list
# because new_model swapped in Shiller's series; under the published series it
# had 8.9% ProQuest coverage and belonged with the rest.
POOLABLE_FACTORS = ["stock_ret6", "stock_ret12", "stock_vol6", "stock_drawdown",
                    "epu"]

# Interpretation thresholds for the diagnostic. Deliberately coarse: this is a
# go/no-go check, not an estimate of anything.
SAFE, MARGINAL = 0.60, 0.75


def coverage_gap(df, factors=None):
    """Per-factor coverage by source, and the gap. The gap is the leak."""
    if "source" not in df.columns:
        raise ValueError("frame has no `source` column -- nothing to compare")
    factors = factors or [c for c in df.columns
                          if f"has_{c}" in df.columns or c in POOLABLE_FACTORS]
    rows = []
    for f in factors:
        if f not in df.columns:
            continue
        by = df.groupby("source")[f].apply(lambda s: s.notna().mean())
        rows.append({"factor": f, **by.to_dict(),
                     "gap": float(by.max() - by.min()) if len(by) > 1 else 0.0})
    return pd.DataFrame(rows).sort_values("gap", ascending=False)


def source_detectability(df, factors=None, cv=5, verbose=True):
    """AUC for predicting `source` from the macro features. Lower is safer.

    Uses the same macro_block the model uses, so it measures the leak the model
    would actually see -- including the has_ flags, which are where most of it
    lives."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if "source" not in df.columns or df["source"].nunique() < 2:
        if verbose:
            print("only one source present -- nothing to detect")
        return float("nan")

    import hit_predictor as hp
    factors = list(factors or hp.FACTORS)
    missing = [f for f in factors if f not in df.columns]
    if missing:
        raise ValueError(f"frame is missing factor columns: {missing}")

    saved = hp.FACTORS
    try:
        hp.FACTORS = factors
        X = hp.macro_block(df[factors].copy())
    finally:
        hp.FACTORS = saved

    y = (df["source"] == sorted(df["source"].unique())[-1]).astype(int).values
    pipe = Pipeline([("s", StandardScaler()),
                     ("c", LogisticRegression(max_iter=2000))])
    auc = float(cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean())

    if verbose:
        verdict = ("SAFE -- sources are hard to tell apart" if auc < SAFE else
                   "MARGINAL -- report this number alongside any pooled result"
                   if auc < MARGINAL else
                   "LEAKS -- a pooled result here is not interpretable")
        print(f"source detectability: AUC {auc:.3f}   {verdict}")
        print(f"  ({len(factors)} factors, {X.shape[1]} columns incl. has_ flags, "
              f"n={len(df):,})")
        if auc >= SAFE:
            gaps = coverage_gap(df, factors)
            worst = gaps[gaps["gap"] > 0.10]
            if len(worst):
                print("  driven by coverage gaps in:")
                for _, r in worst.head(6).iterrows():
                    print(f"    {r['factor']:<16} gap {r['gap']:.1%}")
                print(f"  try: source_detectability(df, POOLABLE_FACTORS)")
    return auc


def restrict_to_poolable(df):
    """Drop factor columns that are not equally covered in both corpora.

    Returns a copy with the unequal factors and their has_ flags removed, so a
    pooled model physically cannot use them."""
    drop = []
    import hit_predictor as hp
    for f in hp.FACTORS:
        if f not in POOLABLE_FACTORS:
            drop += [c for c in (f, f"has_{f}", f"m_{f}", f"x_dir_{f}")
                     if c in df.columns]
    return df.drop(columns=drop)


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT / "src"))

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", nargs="+", required=True,
                    help="context CSVs to pool, e.g. new_model/data/macro_context_*.csv")
    args = ap.parse_args()

    frames = [pd.read_csv(t, low_memory=False) for t in args.tables]
    for t, f in zip(args.tables, frames):
        if "source" not in f.columns:
            f["source"] = "proquest" if "proquest" in t else "loc"
    df = pd.concat(frames, ignore_index=True, sort=False)
    print(f"pooled {len(df):,} rows from {len(frames)} table(s): "
          f"{dict(df['source'].value_counts())}\n")

    print("--- all factors ---")
    source_detectability(df)
    print("\n--- restricted to equal-coverage factors ---")
    source_detectability(df, POOLABLE_FACTORS)
    print("\n--- coverage by source ---")
    print(coverage_gap(df).to_string(index=False))
