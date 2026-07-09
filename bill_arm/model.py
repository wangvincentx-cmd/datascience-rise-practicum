"""
Model 1: predict became_law from introduction-time structural features only
(Section 7). Answers research question 1 -- this reproduces well-trodden
ground (Nay 2017, GovTrack), so it is the baseline, not the contribution.

Model 2 (only with --modeling-csv): structural + press features (Section 8),
trained and evaluated on the covered subset only. Section 9's core
experiment: does adding press features raise PR-AUC over structure alone,
on the SAME covered bills in the same held-out test Congress. Answers
research question 2. Research question 3 (was the press's own directional
call accurate) is answered directly from press_predicts_pass vs became_law,
no model needed.

THE SPLIT RULE (do not change to a random split): bills within one Congress
share the same political environment (which party controls each chamber,
what's salient that cycle). A random split leaks that context into the test
set. Always train on earlier Congresses, test on the most recent one or two.

Accuracy is not reported. About 3-4% of bills become law, so a model that
always predicts "dies" scores ~96% accuracy and is worthless. PR-AUC on the
passed class is the primary metric; see Section 5.

Usage:
  python model.py --features data/features.csv --test-congresses 118
  python model.py --features data/features.csv --modeling-csv data/modeling.csv \\
      --test-congresses 118
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             classification_report, precision_recall_curve,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

CATS = ["chamber", "bill_type", "sponsor_party", "sponsor_state", "policy_area",
       "primary_committee"]
NUMS = ["n_original_cosponsors", "bipartisan", "frac_cosponsors_majority",
       "intro_month_in_session", "title_length", "has_companion_bill",
       "sponsor_in_majority"]
CATS_PRESS = ["press_predicts_pass", "press_confidence"]
NUMS_PRESS = ["n_articles", "days_intro_to_first_coverage"]
TEXT_COL = "combined_text"


def load_features(path):
    df = pd.read_csv(path)
    df["y"] = df["became_law"].astype(int)
    df[TEXT_COL] = (df["title_text"].fillna("") + " " +
                    df["introduced_text"].fillna(""))
    for c in ["bipartisan", "has_companion_bill", "sponsor_in_majority"]:
        df[c] = df[c].map({True: 1, "True": 1, False: 0, "False": 0}).fillna(0).astype(int)
    for c in CATS:
        df[c] = df[c].fillna("unknown").astype(str)
    for c in ["n_original_cosponsors", "frac_cosponsors_majority",
             "intro_month_in_session", "title_length"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df


def load_modeling_features(path):
    """Structural features plus press features, from join_dataset.py's output."""
    df = load_features(path)
    df["has_national_coverage"] = df["has_national_coverage"].map(
        {True: 1, "True": 1, False: 0, "False": 0}).fillna(0).astype(int)
    df["press_predicts_pass"] = df["press_predicts_pass"].fillna("none").astype(str)
    df["press_confidence"] = df["press_confidence"].fillna("none").astype(str)
    df["n_articles"] = pd.to_numeric(df["n_articles"], errors="coerce").fillna(0)
    df["days_intro_to_first_coverage"] = pd.to_numeric(
        df["days_intro_to_first_coverage"], errors="coerce").fillna(-1)
    return df


def split_by_congress(df, test_congresses):
    if test_congresses:
        test_set = set(test_congresses)
    else:
        congresses = sorted(df["congress"].unique())
        test_set = set(congresses[-1:])
    train = df[~df["congress"].isin(test_set)]
    test = df[df["congress"].isin(test_set)]
    print(f"train congresses {sorted(train.congress.unique())} ({len(train)} bills, "
         f"{train.y.mean():.4f} became law)")
    print(f"test  congresses {sorted(test.congress.unique())} ({len(test)} bills, "
         f"{test.y.mean():.4f} became law)")
    return train, test


def build_preprocessor(cats, nums, max_text_features=2000):
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cats),
        ("num", "passthrough", nums),
        ("txt", TfidfVectorizer(max_features=max_text_features, ngram_range=(1, 2),
                               min_df=2), TEXT_COL),
    ])


def feature_names(pre, cats, nums):
    names = list(pre.named_transformers_["cat"].get_feature_names_out(cats))
    names += nums
    names += [f"word:{w}" for w in pre.named_transformers_["txt"].get_feature_names_out()]
    return names


def report_metrics(name, y_true, proba, verbose=True):
    pr_auc = average_precision_score(y_true, proba)
    try:
        roc_auc = roc_auc_score(y_true, proba)
    except ValueError:
        roc_auc = float("nan")
    brier = brier_score_loss(y_true, proba)
    base_rate = y_true.mean()
    print(f"\n=== {name} ===")
    print(f"PR-AUC {pr_auc:.4f}  (baseline / prevalence: {base_rate:.4f})")
    print(f"ROC-AUC {roc_auc:.4f}")
    print(f"Brier score {brier:.4f}  (lower is better; naive-prevalence Brier: "
         f"{base_rate * (1 - base_rate):.4f})")

    if not verbose:
        return {"pr_auc": pr_auc, "roc_auc": roc_auc, "brier": brier}

    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0)
    best_idx = int(np.argmax(f1))
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    pred = (proba >= best_thresh).astype(int)
    print(f"best-F1 threshold {best_thresh:.3f} (chosen on this same set; report as "
         f"descriptive, not tuned on a separate validation set)")
    print(classification_report(y_true, pred, target_names=["died", "became_law"],
                                zero_division=0))

    print("calibration curve (predicted vs. observed, 10 bins):")
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=min(10, len(np.unique(proba))),
                                            strategy="quantile")
    for mp, fp in zip(mean_pred, frac_pos):
        print(f"  predicted {mp:.3f}  observed {fp:.3f}")
    return {"pr_auc": pr_auc, "roc_auc": roc_auc, "brier": brier}


def report_logistic_importances(pipe, cats, nums, k=20):
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return
    names = feature_names(pre, cats, nums)
    coefs = pd.Series(clf.coef_[0], index=names)
    print("\nTop features pushing toward BECAME LAW:")
    print(coefs.sort_values(ascending=False).head(k).round(3))
    print("\nTop features pushing toward DIED:")
    print(coefs.sort_values().head(k).round(3))


def report_xgb_importances(pipe, cats, nums, k=20):
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    names = feature_names(pre, cats, nums)
    importances = pd.Series(clf.feature_importances_, index=names)
    print("\nTop features by gradient-boosting importance:")
    print(importances.sort_values(ascending=False).head(k).round(4))


def fit_calibrated(pipe, train, max_cv=3):
    """CalibratedClassifierCV needs at least `cv` examples of the minority
    class. Covered-subset training sets (Section 8.1's small-sample regime)
    can easily have fewer than 3 positives; degrade to an uncalibrated fit
    with a warning instead of crashing."""
    n_pos = int(train.y.sum())
    cv = min(max_cv, n_pos)
    if cv < 2:
        print(f"  WARNING: only {n_pos} positive example(s) in this training set; "
             f"skipping probability calibration (need >=2 for cross-validated "
             f"calibration). Fitting uncalibrated.")
        pipe.fit(train, train.y)
        return pipe
    cal = CalibratedClassifierCV(pipe, method="sigmoid", cv=cv)
    cal.fit(train, train.y)
    return cal


def _unwrap(fitted):
    if hasattr(fitted, "calibrated_classifiers_"):
        return fitted.calibrated_classifiers_[0].estimator
    return fitted


def fit_and_score(train, test, cats, nums, label, max_text_features=2000, verbose=True):
    """Fit calibrated logistic regression + gradient boosting, score on `test`.
    Returns (fitted_pipes, proba_by_model, metrics_by_model)."""
    n_pos, n_neg = train.y.sum(), len(train) - train.y.sum()
    scale_pos_weight = n_neg / max(n_pos, 1)

    logit = Pipeline([
        ("pre", build_preprocessor(cats, nums, max_text_features)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    logit_cal = fit_calibrated(logit, train)
    proba_logit = logit_cal.predict_proba(test)[:, 1]

    xgb = Pipeline([
        ("pre", build_preprocessor(cats, nums, max_text_features)),
        ("clf", XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                              scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
                              random_state=0)),
    ])
    xgb_cal = fit_calibrated(xgb, train)
    proba_xgb = xgb_cal.predict_proba(test)[:, 1]

    metrics = {}
    if verbose:
        metrics["logistic_regression"] = report_metrics(
            f"{label}: logistic_regression", test.y, proba_logit)
        report_logistic_importances(_unwrap(logit_cal), cats, nums)
        metrics["gradient_boosting"] = report_metrics(
            f"{label}: gradient_boosting", test.y, proba_xgb)
        report_xgb_importances(_unwrap(xgb_cal), cats, nums)

    fitted = {"logistic_regression": logit_cal, "gradient_boosting": xgb_cal}
    proba = {"logistic_regression": proba_logit, "gradient_boosting": proba_xgb}
    return fitted, proba, metrics


def bootstrap_pr_auc_delta(y_true, proba_a, proba_b, n_boot=2000, seed=0):
    """Bootstrap CI for PR-AUC(b) - PR-AUC(a) over the same held-out bills.
    Resamples with replacement; skips resamples with no positives (PR-AUC
    undefined) or no negatives."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    proba_a, proba_b = np.asarray(proba_a), np.asarray(proba_b)
    n = len(y_true)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y_true[idx]
        if yb.sum() == 0 or yb.sum() == n:
            continue
        pr_a = average_precision_score(yb, proba_a[idx])
        pr_b = average_precision_score(yb, proba_b[idx])
        deltas.append(pr_b - pr_a)
    if not deltas:
        return {"mean_delta": float("nan"), "ci_low": float("nan"),
               "ci_high": float("nan"), "n_boot_used": 0}
    deltas = np.array(deltas)
    return {
        "mean_delta": float(deltas.mean()),
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "n_boot_used": len(deltas),
    }


def research_q3_hit_rate(df):
    """Research question 3: among covered bills with a non-neutral press
    prediction, how often did the press call the outcome correctly."""
    non_neutral = df[df["press_predicts_pass"].isin(["pass", "fail"])]
    if non_neutral.empty:
        return {"n": 0, "hit_rate": float("nan")}
    hits = (((non_neutral["press_predicts_pass"] == "pass") & (non_neutral["y"] == 1)) |
           ((non_neutral["press_predicts_pass"] == "fail") & (non_neutral["y"] == 0)))
    return {"n": len(non_neutral), "hit_rate": float(hits.mean())}


def run_model1(df, test_congresses, max_text_features):
    print("\n" + "=" * 70)
    print("MODEL 1: structural features only, all bills (research question 1)")
    print("=" * 70)
    train, test = split_by_congress(df, test_congresses)
    if train.empty or test.empty:
        raise SystemExit("Split leaves an empty side; pick another --test-congresses.")
    if train.y.sum() == 0:
        raise SystemExit("No positive examples (became_law) in the training set.")
    fitted, proba, metrics = fit_and_score(train, test, CATS, NUMS,
                                           "Model 1 (structural)", max_text_features)
    return train, test, fitted, proba, metrics


def run_press_experiment(modeling_path, test_congresses, max_text_features,
                         fitted1, test1):
    print("\n" + "=" * 70)
    print("PRESS EXPERIMENT: Model 2 (structural + press) vs Model 1, same "
         "covered test bills (research questions 2-3, Section 9)")
    print("=" * 70)
    joined = load_modeling_features(modeling_path)
    train_j, test_j = split_by_congress(joined, test_congresses)
    covered_train = train_j[train_j.has_national_coverage == 1]
    covered_test = test_j[test_j.has_national_coverage == 1]
    print(f"covered train bills: {len(covered_train)}  covered test bills: {len(covered_test)}")

    if covered_train.empty or covered_test.empty or covered_train.y.sum() == 0:
        print("Not enough covered bills to run the press experiment on this split. "
             "Check coverage_report.py's gate before running this.")
        return

    cats2 = CATS + CATS_PRESS
    nums2 = NUMS + NUMS_PRESS
    fitted2, proba2, _ = fit_and_score(covered_train, covered_test, cats2, nums2,
                                       "Model 2 (structural + press)", max_text_features)

    print("\n--- Model 1 (structural only), scored on the SAME covered test bills ---")
    proba1_covered = {}
    for name, pipe in fitted1.items():
        p = pipe.predict_proba(covered_test)[:, 1]
        proba1_covered[name] = p
        report_metrics(f"Model 1 (structural, covered subset): {name}", covered_test.y, p,
                       verbose=False)

    print("\n--- PR-AUC delta, Model 2 minus Model 1, same covered test bills "
         "(bootstrap 95% CI) ---")
    for name in ("logistic_regression", "gradient_boosting"):
        boot = bootstrap_pr_auc_delta(covered_test.y.values, proba1_covered[name],
                                      proba2[name])
        print(f"{name}: delta PR-AUC = {boot['mean_delta']:+.4f}  "
             f"95% CI [{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}]  "
             f"(n_boot={boot['n_boot_used']})")

    print("\n--- Research question 3: press directional accuracy "
         "(press_predicts_pass vs became_law) ---")
    overall = research_q3_hit_rate(joined)
    test_only = research_q3_hit_rate(covered_test)
    print(f"all covered congresses with a non-neutral call: n={overall['n']}, "
         f"hit rate={overall['hit_rate']:.4f}")
    print(f"held-out test congress(es) only:                n={test_only['n']}, "
         f"hit rate={test_only['hit_rate']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features.csv")
    ap.add_argument("--modeling-csv", default=None,
                    help="data/modeling.csv from join_dataset.py; if given, also "
                        "runs the Model 2 press experiment (Section 9)")
    ap.add_argument("--test-congresses", default=None,
                    help="comma-separated congress numbers held out for test; "
                        "defaults to the single most recent congress in the data")
    ap.add_argument("--max-text-features", type=int, default=2000)
    args = ap.parse_args()

    df = load_features(args.features)
    test_congresses = ([int(c) for c in args.test_congresses.split(",")]
                       if args.test_congresses else None)

    print("Note: plain accuracy is not reported. ~3-4% of bills become law, so "
         "a model that always predicts 'dies' scores ~96% accuracy and has zero "
         "value. See PR-AUC, precision/recall on the passed class, and Brier "
         "score below.")
    train, test, fitted1, proba1, metrics1 = run_model1(df, test_congresses,
                                                         args.max_text_features)

    if args.modeling_csv:
        run_press_experiment(args.modeling_csv, test_congresses,
                            args.max_text_features, fitted1, test)


if __name__ == "__main__":
    main()
