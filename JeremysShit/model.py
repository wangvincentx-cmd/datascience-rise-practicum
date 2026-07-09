"""
Model stage (economy arm): which factors predict whether a claim was RIGHT?

Trains logistic regression (interpretable baseline) and gradient boosting on
claims_scored.csv to predict `hit` from claim features — including the level of
policy uncertainty (historical EPU) at the moment the claim was printed. The
feature importances answer the team's third research goal: which factors
impacted prediction accuracy.

THE SPLIT IS THE POINT (adopted from Vincent's election arm): claims inside one
episode share one outcome, so a random split leaks the answer into the test set.
We split by EPISODE — train on early windows, test on held-out later windows —
and also report leave-one-episode-out cross-validation.

Usage:
    python model.py                       # uses claims_scored.csv
    python model.py --claims other.csv

Outputs: printed metrics + importances, model_predictions.csv,
figures/fig_model_importances.png

Interpretation warning for the poster: with ~10 episodes, held-out accuracy has
wide error bars. The interesting output is the FEATURE IMPORTANCE ranking, not
squeezing out accuracy points.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from tier2_analysis import epu_series

FIGDIR = Path("figures")
CAT = ["kind", "topic", "voice", "confidence", "state"]
NUM = ["year", "epu"]
TEST_EPISODES_DEFAULT = ["1945 Reconversion", "1948 Recession", "1955 Calm (control)",
                         "1957 Recession"]  # hold out the post-war windows


def build(df):
    df = df.dropna(subset=["hit"]).copy()
    df["hit"] = df["hit"].astype(int)
    for c in CAT:
        df[c] = df.get(c, pd.Series("unknown", index=df.index)).fillna("unknown").astype(str)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    # Policy uncertainty at claim time (Tier 2 EPU index, 1900-2014)
    epu = epu_series()
    df["epu"] = df["date"].dt.to_period("M").map(epu).fillna(epu.median())
    df["quote"] = df["quote"].fillna("")
    return df


def pipeline(model):
    pre = ColumnTransformer([
        # min_frequency folds one-off states etc. into an "infrequent" bucket
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5), CAT),
        ("num", "passthrough", NUM),
        ("text", TfidfVectorizer(max_features=500, ngram_range=(1, 2),
                                 stop_words="english"), "quote"),
    ])
    return Pipeline([("pre", pre), ("clf", model)])


def feature_names(pipe):
    pre = pipe.named_steps["pre"]
    names = list(pre.named_transformers_["cat"].get_feature_names_out(CAT))
    names += NUM
    names += [f"word:{w}" for w in pre.named_transformers_["text"].get_feature_names_out()]
    return names


def main(args):
    df = build(pd.read_csv(args.claims))
    print(f"{len(df)} scored claims across {df['episode'].nunique()} episodes; "
          f"base rate P(hit) = {df['hit'].mean():.3f}")

    test_eps = [e for e in TEST_EPISODES_DEFAULT if e in set(df["episode"])]
    train, test = df[~df["episode"].isin(test_eps)], df[df["episode"].isin(test_eps)]
    print(f"\nEpisode split — train: {sorted(set(train['episode']))}")
    print(f"              test:  {sorted(set(test['episode']))}")

    coef_series = None
    results = {}
    for name, model in [("logistic regression", LogisticRegression(max_iter=2000, C=0.5)),
                        ("gradient boosting", GradientBoostingClassifier(random_state=0))]:
        pipe = pipeline(model)
        pipe.fit(train, train["hit"])
        proba = pipe.predict_proba(test)[:, 1]
        acc = accuracy_score(test["hit"], proba > 0.5)
        auc = roc_auc_score(test["hit"], proba) if test["hit"].nunique() > 1 else float("nan")
        maj = max(test["hit"].mean(), 1 - test["hit"].mean())
        results[name] = proba
        print(f"\n=== {name} ===")
        print(f"  held-out accuracy {acc:.3f}  (majority-class baseline {maj:.3f})   AUC {auc:.3f}")

        if name == "logistic regression":
            coef_series = pd.Series(pipe.named_steps["clf"].coef_[0], index=feature_names(pipe))
            coef_series = coef_series[coef_series.abs() > 1e-6]
            print("  most predictive of being RIGHT:")
            for f, v in coef_series.sort_values(ascending=False).head(8).items():
                print(f"    {v:+.2f}  {f}")
            print("  most predictive of being WRONG:")
            for f, v in coef_series.sort_values().head(8).items():
                print(f"    {v:+.2f}  {f}")
        else:
            # Permutation importance on held-out data: which INPUT COLUMNS matter,
            # measured honestly (drop in test AUC when a column is shuffled).
            perm = permutation_importance(pipe, test, test["hit"], n_repeats=10,
                                          random_state=0, scoring="roc_auc")
            imp = (pd.Series(perm.importances_mean, index=test.columns)
                   .loc[lambda x: x.index.isin(CAT + NUM + ["quote"])]
                   .sort_values(ascending=False))
            print("  permutation importance (held-out AUC drop when column shuffled):")
            for f, v in imp.items():
                print(f"    {v:+.4f}  {f}")

    # Predictions file: lets the team inspect what the model gets wrong
    out = test[["claim_id", "episode", "publisher", "date", "quote", "hit"]].copy()
    for name, proba in results.items():
        out[f"p_hit_{name.split()[0]}"] = np.round(proba, 3)
    out.to_csv("model_predictions.csv", index=False)

    # Leave-one-episode-out CV: the honest overall number
    pipe = pipeline(LogisticRegression(max_iter=2000, C=0.5))
    scores = cross_val_score(pipe, df, df["hit"], groups=df["episode"],
                             cv=LeaveOneGroupOut(), scoring="accuracy")
    eps = sorted(set(df["episode"]))
    print(f"\nLeave-one-episode-out accuracy: {scores.mean():.3f} "
          f"(± {scores.std():.3f} across {len(scores)} episodes)")
    for e, sc in zip(eps, scores):
        print(f"  {e:22s} {sc:.2f}")

    # Coefficient figure for the poster
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        FIGDIR.mkdir(exist_ok=True)
        top = pd.concat([coef_series.sort_values().head(10),
                         coef_series.sort_values().tail(10)])
        fig, ax = plt.subplots(figsize=(8, 6))
        top.plot(kind="barh", ax=ax,
                 color=np.where(top > 0, "seagreen", "crimson"), alpha=.85)
        ax.axvline(0, color="black", lw=1)
        ax.set_xlabel("logistic coefficient (right = associated with correct predictions)")
        ax.set_title("What made a newspaper prediction likely to be right?")
        plt.tight_layout(); plt.savefig(FIGDIR / "fig_model_importances.png", dpi=200)
        plt.close()
        print("\nWrote model_predictions.csv and figures/fig_model_importances.png")
    except ImportError:
        print("\nWrote model_predictions.csv (matplotlib missing — no figure)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--claims", default="claims_scored.csv")
    args = ap.parse_args()
    main(args)
