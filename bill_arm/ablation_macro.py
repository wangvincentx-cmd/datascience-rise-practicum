"""
Ablation: does the ECONOMY (macro features) improve prediction of a POLITICAL
outcome (whether a bill becomes law)?

This is the (C) question — "how the economy affects political predictions" —
operationalized on bill_arm. The 6 macroeconomic features are already joined
into features.csv (unemployment, recession flag, GDP growth, CPI inflation,
consumer sentiment, initial claims), each shifted by its real publication lag
(no leakage), but model.py's NUMS list does NOT use them. So we compare, on the
identical Congress-based split:

    A) structural features only          (model.py's current NUMS)
    B) structural + 6 macro features

Same model (calibrated logistic + gradient boosting), same held-out test
Congresses, same text. The PR-AUC lift of B over A — with a bootstrap 95% CI —
is the answer: if the interval clears 0, the economic climate measurably helps
predict bill passage; if it straddles 0, it doesn't.

Usage:  python ablation_macro.py [--test-congresses 117 118] [--text-features 500]
"""

import argparse
import numpy as np
from sklearn.metrics import average_precision_score

from model import (load_features, split_by_congress, fit_and_score,
                   bootstrap_pr_auc_delta, CATS, NUMS)

MACRO = ["unemployment_rate", "recession_flag", "gdp_growth_yoy",
         "cpi_inflation_yoy", "consumer_sentiment", "initial_claims"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default="data/features.csv")
    ap.add_argument("--test-congresses", type=int, nargs="+", default=[117, 118])
    ap.add_argument("--text-features", type=int, default=500,
                    help="TF-IDF vocab size; identical in both arms so it cancels out")
    args = ap.parse_args()

    df = load_features(args.features)
    for c in MACRO:
        df[c] = df[c].astype(float)
    train, test = split_by_congress(df, args.test_congresses)

    # median-impute macro NaNs on TRAIN stats only (passthrough numerics can't
    # take NaN for the logistic arm); applied identically to both arms.
    for c in MACRO:
        med = train[c].median()
        train[c] = train[c].fillna(med)
        test[c] = test[c].fillna(med)

    print(f"\nbase rate P(became_law) train={train.y.mean():.4f} test={test.y.mean():.4f} "
          f"({test.y.sum()} positives in test)\n")

    # A: structural only ; B: structural + macro
    _, proba_a, _ = fit_and_score(train, test, CATS, NUMS,
                                  "A structural", args.text_features, verbose=False)
    _, proba_b, _ = fit_and_score(train, test, CATS, NUMS + MACRO,
                                  "B structural+macro", args.text_features, verbose=False)

    print(f"{'model':22s}{'PR-AUC A':>11s}{'PR-AUC B':>11s}{'delta':>9s}   95% CI (bootstrap)")
    print("-" * 78)
    for m in ("logistic_regression", "gradient_boosting"):
        pa = average_precision_score(test.y, proba_a[m])
        pb = average_precision_score(test.y, proba_b[m])
        ci = bootstrap_pr_auc_delta(test.y, proba_a[m], proba_b[m])
        sig = "" if (ci["ci_low"] <= 0 <= ci["ci_high"]) else "  *sig"
        print(f"{m:22s}{pa:11.4f}{pb:11.4f}{pb-pa:+9.4f}   "
              f"[{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}]{sig}")

    print("\nbaseline (prevalence) PR-AUC =", round(test.y.mean(), 4))
    print("* = 95% CI excludes 0 (macro features significantly change PR-AUC).")
    print("Reading: delta > 0 with CI clearing 0 => the economic climate measurably")
    print("improves prediction of bill passage. CI straddling 0 => no measurable effect.")


if __name__ == "__main__":
    main()
