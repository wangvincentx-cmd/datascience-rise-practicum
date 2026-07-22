"""
Validate the political-climate x year/epu interaction lead from
model_interactions.py (see CHANGELOG's "Not done / next up": that script's
result was a SINGLE train/test split, exploratory only. This script holds it
to the same bar every other headline number in this arm was held to
(LOGIT_C, the GB-tuning rejection, local_disagreement): leave-one-episode-out
CV, then a permutation test on the statistic itself.

THE STATISTIC: model_interactions.py's single split found 8 political-climate
x year/epu SHAP-interaction pairs standing out despite each of those
categorical features (president_party, unified_government) having a null
MARGINAL effect in model.py. Recomputing on one split can't tell a real
interaction from a fold-specific fluke, so here each of the 8 pairs is
recomputed OUT-OF-FOLD across all 19 LeaveOneGroupOut(episode) folds (fit on
18 episodes, take SHAP interaction values on the held-out episode's claims
only -- never train and score the same claim), pooled into one real
statistic per pair -- the interaction-value analog of model.py's pooled LOEO
accuracy.

THEN a permutation test (identical machinery to model.py's
permutation_test()): shuffle `hit` globally, rerun the exact same LOEO/SHAP
procedure, repeat, and see where the real pooled statistic falls in the
resulting null distribution. A pair only counts as a real finding if it
clears p<0.05 here, same bar as everything else in this arm.

Usage:
    python validate_interactions.py                    # n_perm=200 (~15-20 min)
    python validate_interactions.py --n-perm 50 --claims claims_scored.csv
"""

import argparse
import time

import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import LeaveOneGroupOut

import model as m
import model_interactions as mi

# The 8 pairs model_interactions.py's single split (test = TEST_EPISODES_DEFAULT)
# flagged as the "political climate matters in combination, not alone" lead
# (CHANGELOG: "direction x year, direction x epu, and notably president_party
# x year / unified_government x year|epu"). Numeric-numeric pairs from that
# same output (year x epu, year x months, epu x months, epu x
# local_disagreement) are excluded -- they're not part of the political-
# climate story this script is built to check.
TARGET_PAIRS = [
    ("direction_worsen", "year"),
    ("direction_improve", "epu"),
    ("direction_improve", "year"),
    ("president_party_R", "year"),
    ("president_party_D", "year"),
    ("unified_government_False", "year"),
    ("unified_government_True", "year"),
    ("unified_government_False", "epu"),
]


def pooled_interactions(df, pairs):
    """One LeaveOneGroupOut(episode) pass: fit the structural GB model on 18
    episodes, take SHAP interaction values on the 19th (held-out) episode's
    claims only, for every fold. Returns, per pair, the pooled mean |SHAP
    interaction| across every out-of-fold claim that had both columns present
    (a rare category can land in OneHotEncoder's min_frequency=5 "infrequent"
    bucket in one fold's training data and not another -- skip those claims
    for that pair rather than pretend the column existed)."""
    episodes = sorted(set(df["episode"]))
    pooled = {pair: [] for pair in pairs}
    coverage = {pair: 0 for pair in pairs}

    for held in episodes:
        train = df[df["episode"] != held]
        test = df[df["episode"] == held]
        if train["hit"].nunique() < 2:
            continue  # GB can't fit a single-class fold

        pipe = mi.build_struct_pipeline()
        pipe.fit(train, train["hit"])
        names = mi.struct_feature_names(pipe)
        name_idx = {n: i for i, n in enumerate(names)}

        X_test = pipe.named_steps["pre"].transform(test)
        if hasattr(X_test, "toarray"):
            X_test = X_test.toarray()
        explainer = shap.TreeExplainer(pipe.named_steps["clf"])
        interactions = explainer.shap_interaction_values(X_test)

        for a, b in pairs:
            if a in name_idx and b in name_idx:
                vals = np.abs(interactions[:, name_idx[a], name_idx[b]])
                pooled[(a, b)].extend(vals.tolist())
                coverage[(a, b)] += len(vals)

    return {pair: (np.mean(vals) if vals else np.nan) for pair, vals in pooled.items()}, coverage


def build_struct_df(claims_path):
    df = m.build(pd.read_csv(claims_path))
    for c in mi.STRUCT_CAT:
        df[c] = df[c].fillna("unknown").astype(str)
    for c in mi.STRUCT_NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df


def main(args):
    df = build_struct_df(args.claims)
    n_episodes = df["episode"].nunique()
    print(f"{len(df)} scored claims across {n_episodes} episodes")

    t0 = time.time()
    real, coverage = pooled_interactions(df, TARGET_PAIRS)
    print(f"real LOEO pass took {time.time() - t0:.1f}s")

    rng = np.random.default_rng(args.seed)
    null = {pair: np.empty(args.n_perm) for pair in TARGET_PAIRS}

    t0 = time.time()
    for i in range(args.n_perm):
        shuffled = df.copy()
        shuffled["hit"] = rng.permutation(df["hit"].values)
        stats, _ = pooled_interactions(shuffled, TARGET_PAIRS)
        for pair in TARGET_PAIRS:
            null[pair][i] = stats[pair]
        if (i + 1) % max(1, args.n_perm // 10) == 0:
            elapsed = time.time() - t0
            print(f"  permutation {i + 1}/{args.n_perm} ({elapsed:.0f}s elapsed)")

    print(f"\n=== VALIDATION: political-climate x year/epu interactions, "
          f"LOEO CV + permutation test ({args.n_perm} shuffles) ===")
    print("(real = pooled mean |SHAP interaction| across all out-of-fold claims; "
          "a pair only counts as real if p<0.05 here, same bar as every other "
          "headline number in this arm)\n")
    for pair in TARGET_PAIRS:
        r = real[pair]
        nulls = null[pair]
        if np.isnan(r):
            print(f"  {pair[0]:28s} x {pair[1]:8s}  SKIPPED (column absent from "
                  f"every fold's encoding -- category too rare)")
            continue
        p = (np.sum(nulls >= r) + 1) / (args.n_perm + 1)
        verdict = "SIGNIFICANT (p<0.05)" if p < 0.05 else "not significant"
        print(f"  {pair[0]:28s} x {pair[1]:8s}  real {r:.4f}  "
              f"null mean {nulls.mean():.4f} SD {nulls.std():.4f} max {nulls.max():.4f}  "
              f"p={p:.4f}  [{verdict}]  (n={coverage[pair]} out-of-fold claims)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--claims", default="claims_scored.csv")
    ap.add_argument("--n-perm", type=int, default=200,
                    help="number of label shuffles (each shuffle re-runs the "
                         "full 19-fold LEOO/SHAP pass -- ~5s/shuffle, so 200 "
                         "is ~15-20 min)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    main(args)
