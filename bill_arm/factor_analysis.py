"""
Shared fitting/evaluation machinery for analyzing which introduction-time
structural factors (sponsor party, cosponsors, committee, macro/political
climate, etc.) are associated with a bill becoming law -- feature
importances, calibration, PR curves. Used by make_figures.py, model_figures.py,
_ablation_figdata.py, and ablation_macro.py to produce factor-analysis
figures, not to deliver a bill-passage predictor.

(Split out from the former model.py 2026-07-17 when the bill-passage
prediction project itself -- and the whole press-coverage pipeline that fed
its Model 2 -- was dropped. This file keeps only the parts the still-wanted
figure/ablation scripts import: fitting a classifier and reading off which
features it leans on is still useful for factor analysis even without a
"run this to predict a bill's fate" deliverable.)

THE SPLIT RULE (do not change to a random split): bills within one Congress
share the same political environment (which party controls each chamber,
what's salient that cycle). A random split leaks that context into the test
set. Always train on earlier Congresses, test on the most recent one or two.
"""

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
TEXT_COL = "combined_text"

# Tuned 2026-07-18 via GridSearchCV/cross_val_score with GroupKFold(5),
# grouped by Congress (not random -- same leakage rule as everywhere else),
# scoring="average_precision" (PR-AUC, this project's primary metric).
# Confirmed with a fair apples-to-apples comparison (identical CV for both
# old and new params), not just a single lucky split:
#   logistic C: 1.0 (old default) -> 0.3017 PR-AUC; 0.1 (tuned) -> 0.3136
#   xgboost: n_estimators=300/max_depth=4/lr=0.05 (old) -> mean 0.3775
#            (per-fold [0.353,0.339,0.357,0.426,0.414]);
#            n_estimators=500/max_depth=6/lr=0.1 (tuned) -> mean 0.3921
#            (per-fold [0.371,0.352,0.377,0.440,0.421]) -- tuned beats old
#            in EVERY fold, not just on average, which is why this was
#            adopted (unlike the economy arm's gradient-boosting tuning,
#            which looked like a gain on one comparison but didn't survive
#            a permutation test -- see CHANGELOG 2026-07-18).
LOGIT_C = 0.1
XGB_PARAMS = dict(n_estimators=500, max_depth=6, learning_rate=0.1)


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
             "intro_month_in_session", "title_length", "sponsor_is_committee_chair"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
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
    class. Small training sets can easily have fewer than 3 positives;
    degrade to an uncalibrated fit with a warning instead of crashing."""
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
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=LOGIT_C)),
    ])
    logit_cal = fit_calibrated(logit, train)
    proba_logit = logit_cal.predict_proba(test)[:, 1]

    xgb = Pipeline([
        ("pre", build_preprocessor(cats, nums, max_text_features)),
        ("clf", XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
                              random_state=0, **XGB_PARAMS)),
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
