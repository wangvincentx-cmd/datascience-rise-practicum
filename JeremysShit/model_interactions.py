"""
Feature-INTERACTION analysis for the economy arm (a separate question from
model.py's marginal importances).

model.py already reports which single columns matter (logistic coefficients,
gradient-boosting gain, permutation importance) -- but several structural
features were tested and dropped from the shipped model precisely because
their MARGINAL effect was null/indistinguishable from noise (political_lean,
urban_rural, unified_government, president_party -- see CHANGELOG's
"Political-climate proxy" entry; local_disagreement -- see disagreement.py).
A marginal null does not rule out an INTERACTION effect (e.g. political
climate might matter only when policy uncertainty is already high). This
script checks that, using SHAP interaction values on a gradient-boosting
model fit on the structural (non-text) features only.

Deliberately excludes the TF-IDF text columns model.py's pipeline uses:
shap_interaction_values is O(n_features^2) per sample, and 500 text features
would make the interaction tensor both too slow to compute and too sparse to
read (most word features fire on a handful of claims each). Restricting to
the ~9 structural columns keeps the interaction tensor small (columns after
one-hot encoding) and interpretable, at the cost of not seeing text-feature
interactions -- a real scope limit, not an oversight.

THE SPLIT RULE (same as model.py): claims inside one episode share one
outcome, so this fits on the same episode-based train split model.py uses
and reports interactions computed on the held-out test claims only, not the
training data -- an interaction that's just the model's memorization of a
handful of test-set-adjacent training rows to it.

Usage:
    python model_interactions.py                # uses claims_scored.csv
    python model_interactions.py --claims other.csv --top 15
"""

import argparse

import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import model as m

# Structural-only feature set -- the ones with a real "is this an
# interaction, not a marginal effect" question attached (see module
# docstring). Excludes model.py's `kind`/`topic` (weaker theoretical link to
# the political-climate/disagreement story this script is built to check)
# to keep the interaction matrix focused; add them back if a specific
# hypothesis calls for it.
STRUCT_CAT = ["region", "fin_center", "political_lean", "urban_rural",
             "unified_government", "president_party", "confidence", "voice", "direction"]
STRUCT_NUM = ["year", "epu", "months", "local_disagreement"]


def build_struct_pipeline():
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), STRUCT_CAT),
        ("num", "passthrough", STRUCT_NUM),
    ])
    return Pipeline([("pre", pre), ("clf", GradientBoostingClassifier(random_state=0))])


def struct_feature_names(pipe):
    pre = pipe.named_steps["pre"]
    names = list(pre.named_transformers_["cat"].get_feature_names_out(STRUCT_CAT))
    names += STRUCT_NUM
    return names


def top_interactions(shap_interactions, names, top=15):
    """shap_interactions: (n_samples, n_features, n_features). Returns the
    top off-diagonal (i, j) pairs by mean |interaction value| across samples,
    each pair counted once (i<j)."""
    mean_abs = np.abs(shap_interactions).mean(axis=0)
    n = len(names)
    pairs = [(mean_abs[i, j], names[i], names[j])
             for i in range(n) for j in range(i + 1, n)]
    pairs.sort(reverse=True)
    return pairs[:top]


def main(args):
    df = m.build(pd.read_csv(args.claims))
    for c in STRUCT_CAT:
        df[c] = df[c].fillna("unknown").astype(str)
    for c in STRUCT_NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    test_eps = [e for e in m.TEST_EPISODES_DEFAULT if e in set(df["episode"])]
    train, test = df[~df["episode"].isin(test_eps)], df[df["episode"].isin(test_eps)]
    print(f"train episodes {sorted(set(train['episode']))} ({len(train)})")
    print(f"test  episodes {sorted(set(test['episode']))} ({len(test)})")

    pipe = build_struct_pipeline()
    pipe.fit(train, train["hit"])
    names = struct_feature_names(pipe)

    X_test = pipe.named_steps["pre"].transform(test)
    if hasattr(X_test, "toarray"):
        X_test = X_test.toarray()

    explainer = shap.TreeExplainer(pipe.named_steps["clf"])
    print(f"\nComputing SHAP interaction values for {len(names)} structural features "
          f"x {X_test.shape[0]} held-out claims...")
    interactions = explainer.shap_interaction_values(X_test)

    print(f"\n=== Top {args.top} feature-PAIR interactions (mean |SHAP interaction|, held-out) ===")
    print("(a marginal-null feature showing up HERE means it matters in combination, "
          "not alone -- read the corresponding model.py permutation-importance entry "
          "alongside this)")
    for val, a, b in top_interactions(interactions, names, args.top):
        print(f"  {val:.4f}  {a}  x  {b}")

    mean_abs_total = np.abs(interactions).mean(axis=0)
    main_effect = np.diag(mean_abs_total)
    print(f"\n=== Main-effect strength (SHAP interaction matrix diagonal) for reference ===")
    print(pd.Series(main_effect, index=names).sort_values(ascending=False).round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--claims", default="claims_scored.csv")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()
    main(args)
