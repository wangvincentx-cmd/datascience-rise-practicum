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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
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

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000,
                                                  class_weight="balanced"),
        "gradient_boosting": GradientBoostingClassifier(random_state=0),
    }
    for name, clf in models.items():
        pipe = build(clf, cats, nums)
        pipe.fit(train, train["y"])
        pred = pipe.predict(test)
        acc = accuracy_score(test["y"], pred)
        try:
            auc = roc_auc_score(test["y"], pipe.predict_proba(test)[:, 1])
        except ValueError:
            auc = float("nan")
        print(f"=== {name} ===")
        print(f"accuracy {acc:.3f}   roc_auc {auc:.3f}")
        print(classification_report(test["y"], pred, zero_division=0))
        if name == "logistic_regression":
            report_top_features(pipe, cats, nums)
        print()


if __name__ == "__main__":
    main()
