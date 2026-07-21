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

from disagreement import add_disagreement_features
from tier2_analysis import epu_series, STATE_TO_REGION, FIN_CENTERS, POLITICAL_HUBS

FIGDIR = Path("figures")
# region/fin_center replace raw state: state alone is noisy/sparse (many
# states have <5 claims and get folded into one "infrequent" bucket by the
# encoder, losing whatever signal they had) -- region and financial/political
# hub status are the same geographic information at a granularity the data
# can actually support. direction and months were sitting in claims_scored.csv
# unused; both are plausible predictors (the claim's own predicted direction,
# and how long a horizon it's making a call over) that were simply never wired in.
# political_lean/urban_rural (publisher_metadata.csv, hand-researched top 30
# publishers) and unified_government/president_party/election_year
# (data/political_climate.csv, 59th-111th Congress, 1905-2010, deliberately
# NOT linked to bill_arm -- see build()'s docstring-style comment) are BUILT
# and JOINED (see build()) but deliberately excluded from CAT/NUM here.
# Tested (2026-07-16): a single train/test split suggested they hurt a lot
# (GB accuracy 0.607->0.549), but the more robust LOEO CV showed all four
# combinations (neither / publisher only / political only / both) landing
# within 0.548-0.584 -- indistinguishable from each other given LOEO's own
# ~0.20 SD. Rather than ship added complexity on a difference that isn't
# distinguishable from noise, kept the simpler feature set that has an
# actual confirmed result behind it (permutation test, p=0.0099). The
# columns are still computed and available in `build()`'s output for anyone
# who wants to explore them (e.g. `df.groupby("political_lean")["hit"].mean()`)
# without them being load-bearing for the reported model.
# local_disagreement (disagreement.py, 2026-07-17): per-claim, backward-
# looking, leakage-safe measure of how split OTHER near-contemporaneous
# claims were in direction (0 = everyone agreed, 0.5 = perfectly split).
# Tested against the same permutation-test validation gate
# political_lean/urban_rural went through above (2026-07-16): LOEO accuracy
# WITH vs. WITHOUT it in NUM, 50-shuffle permutation test, both models.
# Result: logistic 0.583->0.580 (unchanged, within noise), gradient
# boosting 0.573->0.552 (measurably WORSE). Doesn't clear the bar -- dropped
# from NUM. `add_disagreement_features` is still called in build() so the
# column is computed and available for exploration
# (e.g. `df.groupby(pd.cut(df.local_disagreement, 4))["hit"].mean()`), same
# as political_lean/urban_rural, without being load-bearing for the
# reported model. (episode_disagreement_rate(), the episode-level aggregate
# version of this same idea, is a separate question -- see
# disagreement_severity.py -- a per-claim null here doesn't settle whether
# episode-level consensus predicts crisis severity.)
CAT = ["kind", "topic", "voice", "confidence", "region", "fin_center", "direction"]
NUM = ["year", "epu", "months"]
# Tuned 2026-07-18 via GridSearchCV with LeaveOneGroupOut (grouped by episode,
# not a random split -- same leakage rule as everywhere else), NOT by hand
# against the final reported number. Grid [0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
# was monotonic (smaller C = stronger regularization = better, all the way
# down): 0.05 gave 0.624 CV accuracy vs. the old default C=0.5's 0.583 --
# makes sense, TF-IDF gives up to 500 text features on only 843 examples, so
# the old default was underregularized. Confirmed with the same permutation
# test everything else here is held to: real accuracy 0.624 vs. null mean
# 0.491 (SD 0.025, max 0.533), p=0.0196 -- exceeds every one of 50 shuffles.
LOGIT_C = 0.05
# GradientBoostingClassifier hyperparameters left at sklearn's defaults
# (n_estimators=100, max_depth=3, learning_rate=0.1) -- CONFIRMED, not just
# never tried. Grid-searched 2026-07-21 via the same LeaveOneGroupOut
# (grouped by episode) CV as LOGIT_C above, scoring="accuracy", on the full
# 1,428-claim scorable corpus: 13 candidates spanning n_estimators
# 50-500, max_depth 1-4, learning_rate 0.02-0.1. Best candidate
# (n_estimators=50, same depth/lr) edged the default on mean LOEO accuracy
# (0.6191 vs 0.6172) but only won 15/19 folds, not every fold -- the bar
# this project holds tuning changes to (see LOGIT_C's own history and
# bill_arm/factor_analysis.py's XGB_PARAMS comment). No candidate cleared
# that bar. Consistent with this file's own prior finding that GB tuning
# here "looked like a gain on one comparison but didn't survive a
# permutation test" -- defaults are already near-optimal for this corpus;
# don't re-tune this without new data materially changing the picture.
TEST_EPISODES_DEFAULT = ["1945 Reconversion", "1948 Recession", "1955 Calm (control)",
                         "1957 Recession"]  # hold out the post-war windows


def load_political_climate():
    return pd.read_csv(Path(__file__).parent / "data" / "political_climate.csv")


def load_publisher_metadata():
    pm = pd.read_csv(Path(__file__).parent / "publisher_metadata.csv")
    pm["publisher"] = pm["publisher"].str.lower().str.strip()
    # Normalize casing so the CSV's own "UNKNOWN" and the fillna default
    # below for publishers outside the top 30 land in the same category.
    pm["political_lean"] = pm["political_lean"].str.lower().str.strip()
    pm["urban_rural"] = pm["urban_rural"].str.lower().str.strip()
    return pm.set_index("publisher")[["political_lean", "urban_rural"]]


def build(df):
    df = df.dropna(subset=["hit"]).copy()
    df["hit"] = df["hit"].astype(int)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df = add_disagreement_features(df)

    # Same geography derivation as tier2_analysis.py's geography_analysis,
    # so region/fin_center mean the same thing in both places.
    state_lower = df.get("state", pd.Series("", index=df.index)).fillna("").str.lower().str.strip()
    df["region"] = state_lower.map(STATE_TO_REGION).fillna("unknown")
    df["fin_center"] = np.select(
        [state_lower.isin(FIN_CENTERS), state_lower.isin(POLITICAL_HUBS)],
        ["financial-center state", "political hub (DC)"],
        default="elsewhere")

    # Political climate: year -> Congress -> president/chamber-control party.
    climate = load_political_climate()
    year_to_row = {}
    for _, r in climate.iterrows():
        for y in range(int(r["start_year"]), int(r["end_year"]) + 1):
            year_to_row[y] = r
    df["president_party"] = df["year"].map(
        lambda y: year_to_row[y]["president_party"] if y in year_to_row else "unknown")
    df["unified_government"] = df["year"].map(
        lambda y: str(y in year_to_row and
                      year_to_row[y]["president_party"] == year_to_row[y]["senate_majority_party"] ==
                      year_to_row[y]["house_majority_party"]))
    df["is_presidential_election_year"] = (df["year"] % 4 == 0).astype(int)

    # Publisher metadata: hand-researched top-30, "unknown" for the long tail.
    pm = load_publisher_metadata()
    pub_lower = df["publisher"].fillna("").str.lower().str.strip()
    df["political_lean"] = pub_lower.map(pm["political_lean"]).fillna("unknown")
    df["urban_rural"] = pub_lower.map(pm["urban_rural"]).fillna("unknown")

    for c in CAT:
        df[c] = df.get(c, pd.Series("unknown", index=df.index)).fillna("unknown").astype(str)
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


def permutation_test(df, model, model_name, n_perm, seed=0):
    """Is the LOEO accuracy actually distinguishable from chance, given the
    episode-clustered structure?

    Null hypothesis: the features have no real relationship to `hit`. Under
    that null, shuffling the hit/miss labels (globally, unrestricted -- this
    breaks any feature-target relationship while leaving the LOEO CV
    machinery, group structure, and base rate exactly as they really are)
    and rerunning the IDENTICAL LeaveOneGroupOut procedure should produce
    accuracy no better than what a model can scrounge from overfitting noise
    alone. Repeat many times to build that null distribution, then see where
    the REAL (unshuffled) LOEO accuracy falls in it.

    This is the test `model.py` never had: permutation_importance (used
    elsewhere in this file) tells you which COLUMN matters most, not whether
    the model as a whole beats chance at all.
    """
    pipe = pipeline(model)
    real_scores = cross_val_score(pipe, df, df["hit"], groups=df["episode"],
                                  cv=LeaveOneGroupOut(), scoring="accuracy")
    real_acc = real_scores.mean()

    rng = np.random.default_rng(seed)
    null_accs = np.empty(n_perm)
    shuffled = df.copy()
    for i in range(n_perm):
        shuffled["hit"] = rng.permutation(df["hit"].values)
        scores = cross_val_score(pipe, shuffled, shuffled["hit"], groups=shuffled["episode"],
                                 cv=LeaveOneGroupOut(), scoring="accuracy")
        null_accs[i] = scores.mean()

    # Standard permutation-test p-value with continuity correction.
    p = (np.sum(null_accs >= real_acc) + 1) / (n_perm + 1)
    print(f"\n=== PERMUTATION TEST: {model_name} LOEO accuracy vs. chance "
         f"({n_perm} shuffles) ===")
    print(f"  real LOEO accuracy:        {real_acc:.3f}")
    print(f"  null distribution (shuffled labels): mean {null_accs.mean():.3f}  "
         f"SD {null_accs.std():.3f}  max {null_accs.max():.3f}")
    print(f"  empirical p-value: {p:.4f}  "
         f"({'SIGNIFICANT at p<0.05' if p < 0.05 else 'NOT significant at p<0.05'} "
         f"-- {'the model beats chance' if p < 0.05 else 'cannot distinguish this model from chance-level overfitting given the data'})")
    return real_acc, null_accs, p


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
    for name, model in [("logistic regression", LogisticRegression(max_iter=2000, C=LOGIT_C)),
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
    pipe = pipeline(LogisticRegression(max_iter=2000, C=LOGIT_C))
    scores = cross_val_score(pipe, df, df["hit"], groups=df["episode"],
                             cv=LeaveOneGroupOut(), scoring="accuracy")
    eps = sorted(set(df["episode"]))
    print(f"\nLeave-one-episode-out accuracy: {scores.mean():.3f} "
          f"(± {scores.std():.3f} across {len(scores)} episodes)")
    for e, sc in zip(eps, scores):
        print(f"  {e:22s} {sc:.2f}")

    if args.permutation_test:
        permutation_test(df, LogisticRegression(max_iter=2000, C=LOGIT_C),
                         "logistic regression", args.n_perm)
        permutation_test(df, GradientBoostingClassifier(random_state=0),
                         "gradient boosting", args.n_perm)

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
    ap.add_argument("--permutation-test", action="store_true",
                    help="test whether LOEO accuracy beats chance (slow: "
                         "n_perm x 19 LOEO folds x 2 models)")
    ap.add_argument("--n-perm", type=int, default=200,
                    help="number of label shuffles for --permutation-test")
    args = ap.parse_args()
    main(args)
