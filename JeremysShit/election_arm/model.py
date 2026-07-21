"""
Stage 4: predict whether a claim will be correct, from its features.
Feature importances answer the project's core question: which factors made
predictions accurate or inaccurate.

Works on either arm:
  python model.py --arm elections --test-from 1980
  python model.py --arm economy   --test-windows recession_1948,recession_1957,calm_1955

THE SPLIT RULE (do not change to a random split): claims within one election
cycle or one crisis episode share the same outcome. A random split leaks that
outcome into the test set and inflates accuracy. Elections split by cycle
(train earlier, test later). Economy splits by whole windows (entire episodes
held out).
"""

import argparse

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss, classification_report,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def load_elections():
    df = pd.read_csv("data/scored_claims.csv")
    df = df[df["scope"] == "national"].dropna(subset=["correct"]).copy()
    df["y"] = df["correct"].astype(int)
    cats = ["source_type", "publisher_state", "source"]
    nums = ["cycle", "hedged"]
    return df, cats, nums


def load_economy():
    df = pd.read_csv("data/scored_economy.csv").dropna(subset=["hit"]).copy()
    df["y"] = df["hit"].astype(int)
    cats = ["voice", "source", "window_kind", "predicted_state_at_horizon"]
    nums = ["horizon_months", "hedged"]
    if "epu" in df.columns:   # policy uncertainty at claim time (see analyze_economy.load_epu)
        nums.append("epu")
    return df, cats, nums


def build(model, cats, nums):
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cats),
        ("num", "passthrough", nums),
        ("txt", TfidfVectorizer(max_features=500, ngram_range=(1, 2)), "claim_text"),
    ])
    return Pipeline([("pre", pre), ("clf", model)])


def report_top_features(pipe, cats, nums, k=15):
    """Print the strongest logistic-regression coefficients, human-readable."""
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return
    names = list(pre.named_transformers_["cat"].get_feature_names_out(cats))
    names += nums
    names += [f"word:{w}" for w in
              pre.named_transformers_["txt"].get_feature_names_out()]
    coefs = pd.Series(clf.coef_[0], index=names)
    print("\nTop features pushing toward CORRECT predictions:")
    print(coefs.sort_values(ascending=False).head(k).round(3))
    print("\nTop features pushing toward WRONG predictions:")
    print(coefs.sort_values().head(k).round(3))


def fit_calibrated(pipe, train, max_cv=3):
    """Wrap pipe in CalibratedClassifierCV so predict_proba is a meaningful
    probability, not just a rank score -- same rationale and same edge-case
    handling as bill_arm/factor_analysis.py's fit_calibrated: small test-set
    splits (a whole election cycle or crisis window held out) can easily have
    fewer than `cv` positives, so degrade to an uncalibrated fit with a
    warning instead of crashing."""
    n_pos = int(train.y.sum())
    cv = min(max_cv, n_pos, len(train) - n_pos)
    if cv < 2:
        print(f"  WARNING: only {n_pos} positive / {len(train) - n_pos} negative example(s) "
              f"in this training set; skipping probability calibration (need >=2 for "
              f"cross-validated calibration). Fitting uncalibrated.")
        pipe.fit(train, train.y)
        return pipe
    cal = CalibratedClassifierCV(pipe, method="sigmoid", cv=cv)
    cal.fit(train, train.y)
    return cal


def _unwrap(fitted):
    """Get back the underlying Pipeline (for coef_/feature-name access) whether
    or not fit_calibrated actually calibrated it."""
    if hasattr(fitted, "calibrated_classifiers_"):
        return fitted.calibrated_classifiers_[0].estimator
    return fitted


def report_metrics(name, y_true, proba):
    pr_auc = average_precision_score(y_true, proba)
    try:
        auc = roc_auc_score(y_true, proba)
    except ValueError:
        auc = float("nan")
    brier = brier_score_loss(y_true, proba)
    base_rate = y_true.mean()
    print(f"=== {name} ===")
    print(f"accuracy {accuracy_score(y_true, proba > 0.5):.3f}   roc_auc {auc:.3f}   "
          f"pr_auc {pr_auc:.3f}  (baseline/prevalence {base_rate:.3f})   "
          f"brier {brier:.3f}  (naive-prevalence brier {base_rate * (1 - base_rate):.3f})")
    print(classification_report(y_true, proba > 0.5, zero_division=0))

    n_bins = min(10, len(set(proba)))
    if n_bins >= 2:
        frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=n_bins, strategy="quantile")
        print("calibration curve (predicted vs. observed):")
        for mp, fp in zip(mean_pred, frac_pos):
            print(f"  predicted {mp:.3f}  observed {fp:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["elections", "economy"], required=True)
    ap.add_argument("--test-from", type=int,
                    help="elections: cycle where the test set begins")
    ap.add_argument("--test-windows",
                    help="economy: comma-separated window_ids to hold out")
    args = ap.parse_args()

    if args.arm == "elections":
        df, cats, nums = load_elections()
        cycles = sorted(df["cycle"].unique())
        split = args.test_from or cycles[len(cycles) * 2 // 3]
        train, test = df[df["cycle"] < split], df[df["cycle"] >= split]
        print(f"train cycles {sorted(train.cycle.unique())} ({len(train)})")
        print(f"test  cycles {sorted(test.cycle.unique())} ({len(test)})")
    else:
        df, cats, nums = load_economy()
        windows = sorted(df["window"].unique())
        held = (args.test_windows.split(",") if args.test_windows
                else windows[-max(2, len(windows) // 3):])
        train = df[~df["window"].isin(held)]
        test = df[df["window"].isin(held)]
        print(f"train windows {sorted(train.window.unique())} ({len(train)})")
        print(f"test  windows {held} ({len(test)})")

    if train.empty or test.empty:
        raise SystemExit("Split leaves an empty side; pick another split point.")

    def prep(frame):
        frame = frame.copy()
        frame["claim_text"] = frame["claim_text"].fillna("")
        frame["hedged"] = frame["hedged"].fillna(False).astype(bool).astype(int)
        for c in cats:
            frame[c] = frame[c].fillna("unknown").astype(str)
        for c in nums:
            if c != "hedged":
                frame[c] = pd.to_numeric(frame[c], errors="coerce").fillna(0)
        return frame

    train, test = prep(train), prep(test)

    print(f"baseline (always guess majority): "
          f"{max(test.y.mean(), 1 - test.y.mean()):.3f}\n")

    # GradientBoostingClassifier left at sklearn defaults -- NOT tuned, unlike
    # JeremysShit/model.py's LOGIT_C/GB or bill_arm's XGB_PARAMS. Both of
    # this script's real inputs (data/scored_claims.csv, data/scored_economy.csv)
    # are generated by analyze_elections.py/analyze_economy.py and weren't
    # present in the working tree as of 2026-07-21 -- tuning against synthetic
    # data would just be a fabricated result. Grid-search this the same way
    # (GridSearchCV/LeaveOneGroupOut grouped by cycle or window, scoring=
    # "accuracy" or "pr_auc", require a win in EVERY fold before adopting)
    # once those scored CSVs exist for real.
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000,
                                                  class_weight="balanced"),
        "gradient_boosting": GradientBoostingClassifier(random_state=0),
    }
    for name, clf in models.items():
        pipe = build(clf, cats, nums)
        fitted = fit_calibrated(pipe, train)
        proba = fitted.predict_proba(test)[:, 1]
        report_metrics(name, test["y"], proba)
        if name == "logistic_regression":
            report_top_features(_unwrap(fitted), cats, nums)
        print()


if __name__ == "__main__":
    main()
