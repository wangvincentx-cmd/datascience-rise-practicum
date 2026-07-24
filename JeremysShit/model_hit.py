"""
Does anything about HOW a forecast was written predict whether it came true --
beyond what the state of the economy already implied?

This is the "attempted model" for the poster. It is deliberately framed as a
nested comparison, not a single accuracy number, because a single number would
be misleading: the economy is autocorrelated, so a model can score decently on
`hit` by learning the business cycle and ignoring the forecast entirely. The
only honest quantity is the INCREMENT from adding claim features on top of a
macro-only baseline.

    1. base rate            -- the null null
    2. macro-only           -- what real-time economic data alone predicts
    3. claim-only           -- what the forecast's own wording predicts
    4. claim + macro        -- the model
    5. gradient boosting    -- same features, non-linear robustness check

Reported: ROC-AUC for each, the macro->full DELTA (the thing that matters), a
permutation test on the delta, and calibration (Brier). A null result --
"whether a forecast came true is governed by the economy, not by how it was
written" -- is a legitimate, publishable outcome and is stated as such.

Leakage discipline, non-negotiable:
- macro features are publication-LAGGED (what was knowable at print time), so
  the model cannot peek at data released after the forecast
- NBER recession status is NOT a feature (announced 6-21 months late); it is
  only ever an outcome, upstream in score_predictions
- validation is GROUPED by episode/period, never a random split, because claims
  in one window share wire copy and one macro reality

Input: the CSV from score_predictions.py (scorable claims only).

Usage:
    python score_predictions.py --claims claims_v2.jsonl --out scored_v2.csv --scorable-only
    python model_hit.py --scored scored_v2.csv
    python model_hit.py --scored scored_v2.csv --rigid    # real-horizon claims only
"""

import argparse
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from truth_data import load_fred

warnings.filterwarnings("ignore")

# Publication lag in months applied to every macro series before it is read as
# a feature. A statistic describing month M is not public until ~M+lag, so a
# forecast printed in M can only "see" data through M-lag. One conservative
# uniform lag keeps the leakage story simple for a poster; per-series lags
# (bill_arm/build_macro_features.py) are the refinement if this becomes a paper.
MACRO_LAG_M = 2
TOKEN_RE = re.compile(r"[a-z0-9]+")
NUM_RE = re.compile(r"\d")


def macro_features(dates):
    """Publication-lagged trailing macro state as of each claim's print date.

    For every claim date, look up what INDPRO / CPI / UNRATE had done over the
    trailing 6 and 12 months USING ONLY values that would have been public by
    then (shifted forward by MACRO_LAG_M). Missing (pre-coverage) -> NaN, filled
    with a flag column so early claims still contribute claim features."""
    indpro = load_fred("INDPRO").shift(MACRO_LAG_M)
    cpi = load_fred("CPIAUCNS").shift(MACRO_LAG_M)
    unrate = load_fred("UNRATE").shift(MACRO_LAG_M)

    def trail(series, p, months):
        p0 = p - months
        if p0 < series.index.min() or p > series.index.max():
            return np.nan
        v0, v1 = series.asof(p0), series.asof(p)
        if pd.isna(v0) or pd.isna(v1) or v0 == 0:
            return np.nan
        return 100.0 * (v1 - v0) / v0

    rows = []
    for d in dates:
        p = pd.Timestamp(d).to_period("M")
        ip6, ip12 = trail(indpro, p, 6), trail(indpro, p, 12)
        cp12 = trail(cpi, p, 12)
        ur = unrate.asof(p) if unrate.index.min() <= p <= unrate.index.max() else np.nan
        ur6 = (ur - unrate.asof(p - 6)) if not pd.isna(ur) and (p - 6) >= unrate.index.min() else np.nan
        rows.append({
            "m_indpro_g6": ip6, "m_indpro_g12": ip12,
            "m_indpro_accel": (ip6 - ip12) if not (pd.isna(ip6) or pd.isna(ip12)) else np.nan,
            "m_cpi_yoy": cp12, "m_unrate": ur, "m_unrate_d6": ur6,
        })
    m = pd.DataFrame(rows)
    m["m_has_indpro"] = m["m_indpro_g12"].notna().astype(int)
    m["m_has_unrate"] = m["m_unrate"].notna().astype(int)
    return m.fillna(0.0)


def claim_features(df):
    """Features known at print time from the forecast itself."""
    out = pd.DataFrame(index=df.index)
    for c in ["direction", "topic", "voice", "scope"]:
        out[f"c_{c}"] = df.get(c, "na").fillna("na").astype(str)
    out["c_hedged"] = (df.get("confidence", "").astype(str) == "hedged").astype(int)
    out["c_quoted"] = df.get("is_quoted_forecaster", False).astype(str).isin(
        ["True", "true", "1"]).astype(int)
    out["c_named"] = (df.get("speaker_name", "na").astype(str)
                      .str.lower().ne("na")).astype(int)
    q = df.get("quote", "").astype(str)
    out["c_has_number"] = q.str.contains(NUM_RE).astype(int)
    out["c_len"] = q.str.split().apply(len).clip(0, 80)
    out["c_horizon"] = pd.to_numeric(df.get("horizon_used"), errors="coerce").fillna(12)
    return out


CLAIM_CAT = ["c_direction", "c_topic", "c_voice", "c_scope"]
CLAIM_NUM = ["c_hedged", "c_quoted", "c_named", "c_has_number", "c_len", "c_horizon"]
MACRO_NUM = ["m_indpro_g6", "m_indpro_g12", "m_indpro_accel", "m_cpi_yoy",
             "m_unrate", "m_unrate_d6", "m_has_indpro", "m_has_unrate"]


def make_pipe(cat, num):
    steps = []
    if cat:
        steps.append(("cat", OneHotEncoder(handle_unknown="ignore",
                                           min_frequency=15), cat))
    if num:
        steps.append(("num", StandardScaler(), num))
    return Pipeline([("pre", ColumnTransformer(steps)),
                     ("clf", LogisticRegression(penalty="l2", C=0.5, max_iter=2000))])


def grouped_auc(X, y, groups, cat, num, clf=None):
    """Out-of-fold ROC-AUC under LeaveOneGroupOut. Pooled OOF predictions so a
    single AUC is computed across all held-out claims, not averaged over folds
    of wildly different size."""
    oof = np.full(len(y), np.nan)
    logo = LeaveOneGroupOut()
    for tr, te in logo.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        pipe = make_pipe(cat, num)
        if clf is not None:
            pipe.set_params(clf=clf)
        pipe.fit(X.iloc[tr], y[tr])
        oof[te] = pipe.predict_proba(X.iloc[te])[:, 1]
    ok = ~np.isnan(oof)
    if len(np.unique(y[ok])) < 2:
        return np.nan, oof
    return roc_auc_score(y[ok], oof[ok]), oof


def run(args):
    df = pd.read_csv(args.scored)
    df = df[df["scorable"] == True].copy()
    if args.rigid and "horizon_basis" in df:
        df = df[df["horizon_basis"] != "default"].copy()
    df = df[df["hit"].isin([0, 1])].reset_index(drop=True)
    y = df["hit"].astype(int).values

    # Grouping unit: episode if present (crisis corpus), else print quarter.
    if "episode" in df and df["episode"].notna().any():
        groups = df["episode"].fillna("na").values
    else:
        groups = pd.to_datetime(df["date"]).dt.to_period("Q").astype(str).values

    print(f"claims: {len(df)}  hit rate: {y.mean():.3f}  "
          f"groups: {len(set(groups))}")
    if len(df) < 100 or len(set(groups)) < 3:
        print("WARNING: too few claims or groups for a trustworthy grouped AUC.")

    # Descriptive hit-rate-by-feature -- robust on ANY corpus (no CV, no macro
    # baseline), and the honest headline when the nested model is not usable.
    print("\n=== descriptive: hit rate by claim feature ===")
    for col in ["confidence", "voice", "topic", "predicted_norm"]:
        if col not in df:
            continue
        g = df.groupby(df[col].fillna("na")).agg(n=("hit", "size"),
                                                 hit=("hit", "mean"))
        g = g[g["n"] >= 25].sort_values("hit", ascending=False)
        if len(g):
            print(f"  by {col}:")
            for k, row in g.iterrows():
                print(f"    {str(k):<18} n={int(row['n']):<5} hit={row['hit']:.3f}")

    cf = claim_features(df)
    mf = macro_features(df["date"])
    X = pd.concat([cf.reset_index(drop=True), mf.reset_index(drop=True)], axis=1)

    print("\n=== nested ROC-AUC (LeaveOneGroupOut, out-of-fold) ===")
    print(f"  1. base rate                {max(y.mean(), 1-y.mean()):.3f}  "
          f"(always-majority accuracy, for reference)")
    auc_macro, _ = grouped_auc(X, y, groups, [], MACRO_NUM)
    print(f"  2. macro-only               AUC {auc_macro:.3f}")
    auc_claim, _ = grouped_auc(X, y, groups, CLAIM_CAT, CLAIM_NUM)
    print(f"  3. claim-only               AUC {auc_claim:.3f}")
    auc_full, oof_full = grouped_auc(X, y, groups, CLAIM_CAT, CLAIM_NUM + MACRO_NUM)
    print(f"  4. claim + macro            AUC {auc_full:.3f}")
    auc_gb, _ = grouped_auc(X, y, groups, CLAIM_CAT, CLAIM_NUM + MACRO_NUM,
                            clf=GradientBoostingClassifier(random_state=0))
    print(f"  5. gradient boosting        AUC {auc_gb:.3f}")

    delta = auc_full - auc_macro
    print(f"\n  DELTA (claim features add, over macro alone): {delta:+.3f}")

    # Honesty guard: if the macro baseline is at or below chance, the nested
    # comparison is not interpretable. This happens on the CRISIS-ONLY corpus
    # under leave-one-episode-out, because each fold removes an entire macro
    # regime, so the macro model predicts a held-out episode it has no
    # comparable training data for and the relationship inverts. The fix is not
    # in the code -- it is data: the macro-incremental question needs the
    # CONTINUOUS monthly corpus with many months per regime and time-blocked CV.
    if auc_macro <= 0.52:
        print(f"\n  ** WARNING: macro baseline AUC {auc_macro:.3f} is at/below chance.")
        print(f"     The nested delta and its p-value are NOT interpretable here.")
        print(f"     Cause: leave-one-group-out removes a whole macro regime per")
        print(f"     fold. This model belongs on the continuous monthly corpus,")
        print(f"     not crisis-only data. On THIS corpus, report the descriptive")
        print(f"     hit-rate-by-feature breakdown and the claim-only model instead.")
        return

    # Permutation test on the delta: shuffle y WITHIN groups, refit, see how often
    # a delta this large arises by chance. Within-group shuffle preserves the
    # macro-cluster structure so the test is about the CLAIM features specifically.
    rng = np.random.default_rng(0)
    null = []
    for _ in range(args.perm):
        yp = y.copy()
        for g in set(groups):
            idx = np.where(groups == g)[0]
            yp[idx] = rng.permutation(yp[idx])
        am, _ = grouped_auc(X, yp, groups, [], MACRO_NUM)
        af, _ = grouped_auc(X, yp, groups, CLAIM_CAT, CLAIM_NUM + MACRO_NUM)
        if not (np.isnan(am) or np.isnan(af)):
            null.append(af - am)
    if null:
        p = (1 + sum(d >= delta for d in null)) / (1 + len(null))
        print(f"  permutation p (delta >= observed, {len(null)} shuffles): {p:.3f}")

    ok = ~np.isnan(oof_full)
    if ok.sum():
        print(f"\n  calibration (full model): Brier {brier_score_loss(y[ok], oof_full[ok]):.3f}")

    print("\n  Interpretation: a small, non-significant delta is the EXPECTED and "
          "\n  honest outcome -- it says forecast accuracy is driven by the economy, "
          "\n  not by how the forecast was written. A significant positive delta "
          "\n  would be the surprising, publishable finding.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scored", required=True, help="score_predictions.py CSV")
    ap.add_argument("--rigid", action="store_true",
                    help="real-horizon claims only (drops defaulted windows)")
    ap.add_argument("--perm", type=int, default=200)
    args = ap.parse_args()
    run(args)
