"""MODEL 1 -- Conditional (fixed-effects) logit: compare forecasts printed in
the same month.

The problem this solves
-----------------------
The pooled model in src/model_hit.py asks one question of all 14,251 claims at
once: was this forecast right because of HOW IT WAS WRITTEN, or because of WHEN
it was written? Those are entangled. A hedged forecast in 1933 and an assertive
one in 1955 differ in wording and in the economy they faced, and no amount of
adding macro columns fully separates the two -- you are relying on
`m_indpro_g6` and friends being the *correct functional form* for the
confounder.

A conditional logit does not rely on that. It adds one intercept per month and
estimates claim-feature coefficients using ONLY within-month variation. Two
forecasts printed in March 1933 faced the identical economy, the identical war
news, the identical wire copy. If one hedged and hit while the other asserted
and missed, that difference cannot be the macro regime. It has to be the claim.

Everything constant within a month -- every macro feature, the press consensus,
the season, the state of the world -- is absorbed by the intercept and drops
out. That is the point: the confounder is removed by DESIGN rather than by
model specification.

What it costs
-------------
1. Months where every claim hit, or every claim missed, contribute nothing to
   the conditional likelihood and are dropped. The count of *informative*
   months and claims is reported, and it is the honest n.
2. It cannot say anything about across-time effects. Whether the 1930s had a
   lower hit rate than the 1950s is unanswerable here, by construction. That
   question belongs to model 2.
3. Standard errors are still clustered by 3-year block, because the month
   intercepts handle the confounding but not the correlation of residuals
   across neighbouring months.

Three groupings are fitted -- month, year, 3-year block -- as a coarseness
ladder. Month is the cleanest comparison and the smallest effective sample;
block is the loosest and the largest. If a coefficient survives all three, it
is not an artefact of where the lines were drawn.

    python poster_models/m1_fixed_effects.py
    python poster_models/m1_fixed_effects.py --rigid
"""

import argparse
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.conditional_models import ConditionalLogit

import _common as C

warnings.filterwarnings("ignore")

# Features constant (or near-constant) within a month are collinear with the
# month intercept and must not enter the within-month fit. The consensus
# features are the subtle case: `x_consensus_net` and `x_disagreement` vary
# across claims only through the leave-one-out correction, which is a ~1/n
# wobble around a monthly constant -- numerically identified, substantively
# meaningless, and it produces enormous unstable coefficients. Only
# `x_against_consensus`, which genuinely differs claim-to-claim within a month,
# is kept.
DROP_WITHIN_MONTH = ["x_consensus_net", "x_disagreement"]


def against_consensus(df):
    """Does this claim swim against what the rest of its month was saying?

    Reuses the audited leave-one-out construction so the definition matches the
    poster's main model exactly."""
    from model_hit import consensus_features
    xf = consensus_features(df)
    xf.index = df.index
    return xf["x_against_consensus"].astype(float)


def fit_pooled(y, X, blocks):
    """Ordinary logit with NO fixed effects -- the confounded comparison.

    Printed alongside the fixed-effects fit so the reader can see what the
    month intercepts actually changed. If the two agree, confounding by macro
    regime was not driving the pooled result; if they diverge, it was."""
    Xc = sm.add_constant(X, has_constant="add")
    res = sm.Logit(y, Xc).fit(disp=0, cov_type="cluster",
                              cov_kwds={"groups": blocks})
    return res


# statsmodels' ConditionalLogit evaluates the exact conditional likelihood by a
# recursion over group members, so a group of a thousand claims overflows the
# stack. That limit is not just an implementation detail -- it tracks the
# statistics. Conditional logit exists to handle MANY groups with FEW
# observations each, where estimating the intercepts directly would bias the
# slopes (the incidental-parameters problem). Once a group holds hundreds of
# claims that bias is O(1/T) and negligible, and a plain logit with group
# dummies is both consistent and better behaved -- and, unlike the conditional
# fit, it accepts cluster-robust standard errors.
MAX_CONDITIONAL_GROUP = 150


def fit_fe(y, X, groups, label, blocks):
    """One intercept per group, by whichever estimator suits the group size."""
    g = pd.Series(groups)
    yy = pd.Series(y)
    # Groups with no outcome variation carry no within-group information: under
    # conditional logit they drop out of the likelihood exactly, so they are
    # removed here for the dummy fit too, keeping the two n's comparable.
    var = yy.groupby(g).transform(lambda s: s.nunique() > 1).values
    n_groups_all, n_groups_inf = g.nunique(), g[var].nunique()
    biggest = int(g[var].value_counts().max()) if var.any() else 0
    print(f"\n  [{label}] {n_groups_all} groups, {n_groups_inf} informative "
          f"(both hits and misses); {int(var.sum()):,} of {len(yy):,} claims "
          f"contribute; largest group {biggest}")
    if n_groups_inf < 3 or var.sum() < 50:
        print(f"  [{label}] too little within-group variation to fit.")
        return None, None, None

    Xf = C.drop_collinear(X.loc[var].reset_index(drop=True))
    gv = g[var].values
    yv = yy[var].values

    if biggest <= MAX_CONDITIONAL_GROUP:
        print(f"  [{label}] exact conditional logit ({n_groups_inf} intercepts "
              f"eliminated, not estimated)")
        res = ConditionalLogit(yv, Xf.values, groups=gv).fit(disp=0)
        return res.params, res.bse, Xf.columns

    print(f"  [{label}] logit with {n_groups_inf} group dummies, SE clustered "
          f"by {len(set(blocks))} time blocks")
    D = pd.get_dummies(pd.Series(gv, name="g"), prefix="fe", drop_first=True,
                       dtype=float).reset_index(drop=True)
    Xd = pd.concat([Xf, D], axis=1)
    Xd = sm.add_constant(Xd, has_constant="add")
    res = sm.Logit(yv, Xd).fit(disp=0, cov_type="cluster",
                               cov_kwds={"groups": np.asarray(blocks)[var]})
    # Report only the claim features; the group intercepts are nuisance.
    keep = [c for c in Xf.columns]
    return res.params[keep].values, res.bse[keep].values, pd.Index(keep)


def run(args):
    C.header("MODEL 1: conditional (fixed-effects) logit",
             "Claim-feature effects identified ONLY from claims printed in the "
             "same period.\nEverything constant within the period -- the whole "
             "macro regime -- is absorbed.")

    df = C.load_scored(args.scored, args.rigid)
    y = df["hit"].astype(int).values
    blocks = C.time_blocks(df, args.block_years)
    months = C.month_index(df)
    years = df["date"].dt.year.astype(str).values

    X = C.claim_design(df)
    X["x_against_consensus"] = against_consensus(df).values
    X = C.standardize(X)
    X = C.drop_collinear(X)

    print(f"\n  claims {len(df):,}   hit rate {y.mean():.3f}   "
          f"months {len(set(months))}   blocks {len(set(blocks))}")
    print(f"  claim features: {len(X.columns)}")

    # --- the confounded benchmark -------------------------------------------
    pooled = fit_pooled(y, X, blocks)
    tab_pooled = C.coef_table(
        X.columns, pooled.params[1:].values, pooled.bse[1:].values,
        "POOLED logit (no fixed effects, block-clustered SE)",
        note="This is the comparison the fixed-effects models replace. Any "
             "coefficient\n  here mixes 'this wording works' with 'this wording "
             "was common in good years'.",
        n=len(df))
    C.save(tab_pooled, "m1_pooled.csv")

    # --- the ladder of groupings --------------------------------------------
    results = {}
    for label, g in [("month", months), ("year", years), ("block", blocks)]:
        params, bse, cols = fit_fe(y, X, g, label, blocks)
        if params is None:
            continue
        tab = C.coef_table(
            cols, params, bse,
            f"FIXED EFFECTS by {label} (within-{label} comparison only)",
            note=f"Each coefficient is the effect of the feature among claims "
                 f"printed in the SAME {label}.\n  Macro regime, wire copy and "
                 f"season are differenced out, not controlled for.")
        C.save(tab, f"m1_fe_{label}.csv")
        results[label] = tab

    # --- does the conclusion survive the choice of grouping? ----------------
    if len(results) >= 2:
        print("\n=== robustness: same coefficient across groupings ===")
        print("  A feature that flips sign or loses significance as the grouping\n"
              "  coarsens was never identified by within-period variation.\n")
        terms = sorted(set().union(*[set(t["term"]) for t in results.values()]))
        wide = pd.DataFrame({"term": terms}).set_index("term")
        wide["pooled"] = tab_pooled.set_index("term")["coef"]
        for label, tab in results.items():
            t = tab.set_index("term")
            wide[f"fe_{label}"] = t["coef"]
            wide[f"p_{label}"] = t["p"]
        cols_show = ["pooled"] + [f"fe_{k}" for k in results]
        print(f"  {'term':<26}" + "".join(f"{c:>11}" for c in cols_show)
              + "   sign-stable")
        for term, r in wide.iterrows():
            vals = [r.get(c, np.nan) for c in cols_show]
            fe_vals = [v for v in vals[1:] if np.isfinite(v)]
            stable = ("yes" if fe_vals and len({np.sign(v) for v in fe_vals}) == 1
                      else "NO")
            cells = "".join(f"{v:>+11.3f}" if np.isfinite(v) else f"{'--':>11}"
                            for v in vals)
            print(f"  {term:<26}{cells}   {stable}")
        C.save(wide.reset_index(), "m1_grouping_robustness.csv")

    print("\n  How to read this: a claim feature that keeps a stable, "
          "significant\n  coefficient under month fixed effects is predicting "
          "accuracy for reasons\n  that have nothing to do with the era it was "
          "written in. A feature that is\n  significant pooled but null under "
          "fixed effects was measuring the decade.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_common_args(ap)
    run(ap.parse_args())
