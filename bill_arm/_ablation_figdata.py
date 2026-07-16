"""Fit structural-only and +macro models, dump PR-AUC + feature-importance
data for the presentation figure. Writes /tmp/ablation_figdata.json."""
import json
import numpy as np
from sklearn.metrics import average_precision_score
from model import (load_features, split_by_congress, fit_and_score,
                   feature_names, _unwrap, CATS, NUMS)

MACRO = ["unemployment_rate", "recession_flag", "gdp_growth_yoy",
         "cpi_inflation_yoy", "consumer_sentiment", "initial_claims"]
PRETTY = {"unemployment_rate": "unemployment", "recession_flag": "recession flag",
          "gdp_growth_yoy": "GDP growth", "cpi_inflation_yoy": "CPI inflation",
          "consumer_sentiment": "consumer sentiment", "initial_claims": "jobless claims"}

df = load_features("data/features.csv")
for c in MACRO:
    df[c] = df[c].astype(float)
train, test = split_by_congress(df, [117, 118])
for c in MACRO:
    med = train[c].median()
    train[c] = train[c].fillna(med); test[c] = test[c].fillna(med)

fit_a, proba_a, _ = fit_and_score(train, test, CATS, NUMS, "A", 500, verbose=False)
fit_b, proba_b, _ = fit_and_score(train, test, CATS, NUMS + MACRO, "B", 500, verbose=False)

pr = {"baseline": float(test.y.mean())}
for m in ("logistic_regression", "gradient_boosting"):
    pr[m] = {"structural": float(average_precision_score(test.y, proba_a[m])),
             "macro": float(average_precision_score(test.y, proba_b[m]))}

# gradient-boosting importances from the +macro model
gb = _unwrap(fit_b["gradient_boosting"])
names = feature_names(gb.named_steps["pre"], CATS, NUMS + MACRO)
imp = gb.named_steps["clf"].feature_importances_
imp = imp / imp.sum()
pairs = sorted(zip(names, imp), key=lambda x: -x[1])

macro_mass = float(sum(v for n, v in zip(names, imp) if n in MACRO))
# group one-hot cats back to their base field for a readable ranking
def base(n):
    if n in MACRO: return PRETTY[n]
    if n.startswith("word:"): return "bill text (words)"
    for c in CATS:
        if n.startswith(c + "_"): return c.replace("_", " ")
    return n
agg = {}
for n, v in zip(names, imp):
    agg[base(n)] = agg.get(base(n), 0.0) + float(v)
top = sorted(agg.items(), key=lambda x: -x[1])[:12]

out = {"pr_auc": pr,
       "macro_importance_share": macro_mass,
       "top_features": [{"name": k, "importance": v,
                         "is_economy": k in PRETTY.values()} for k, v in top]}
json.dump(out, open("/tmp/ablation_figdata.json", "w"), indent=2)
print(json.dumps(out, indent=2))
