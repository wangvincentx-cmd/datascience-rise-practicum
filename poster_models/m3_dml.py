"""MODEL 3 -- Double machine learning: turn the macro delta into an estimate
with a confidence interval.

The problem this solves
-----------------------
The poster's headline quantity is the INCREMENT: what do claim features add on
top of what the economy already implied? src/model_hit.py measures it as a
difference of two cross-validated AUCs (macro-only vs claim+macro) and tests it
by permutation.

That difference is not an estimator of anything. It has no standard error, no
confidence interval, and no population quantity it converges to -- so the
project's own rule ("report confidence intervals") cannot be satisfied by it,
and its permutation p-value is testing a composite null that mixes the claim
features together with the CV procedure itself.

Double machine learning replaces it with a real parameter. Assume

    hit  =  theta' * claim  +  g(macro)  +  e

where g is an arbitrary, unknown, possibly wildly non-linear function of the
macro state. Estimate g by gradient boosting, estimate each claim feature's own
dependence on macro the same way, subtract both predictions off, and regress
the leftover outcome on the leftover features. What survives is theta: the
partial effect of wording on accuracy, holding the economy fixed, WITHOUT
having assumed a functional form for how the economy matters.

Two properties make this the right tool rather than just a fancier one:

  - Neyman orthogonality. Residualising BOTH sides makes theta insensitive to
    first-order errors in the estimate of g. Controlling for macro by throwing
    the macro columns into one logit does not have this property: if the true g
    is non-linear, the claim coefficients absorb the misspecification.
  - Cross-fitting. Each observation's nuisance prediction comes from a model
    that never saw it -- and, here, never saw its TIME BLOCK -- so the flexible
    first stage cannot overfit its way into the second-stage coefficient.

Output is a coefficient, a cluster-robust standard error, a 95% CI, and a joint
chi-square test of "all claim features together add nothing over the economy."
That last p-value is the direct, properly-sized replacement for the permutation
test on the AUC delta.

    python poster_models/m3_dml.py
    python poster_models/m3_dml.py --folds 10
"""

import argparse
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sps
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import _common as C

warnings.filterwarnings("ignore")


def grouped_folds(groups, n_splits):
    """Cross-fitting folds that keep whole time blocks together.

    A random KFold here would leak: the nuisance model would learn the macro
    state of a block from some of its own claims and then predict the rest,
    making the residuals artificially small and the second-stage SE too tight."""
    n_splits = min(n_splits, len(set(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    dummy = np.zeros(len(groups))
    return list(gkf.split(dummy, groups=groups)), n_splits


def crossfit_predict(Z, target, folds, kind):
    """Out-of-fold E[target | macro], by gradient boosting.

    Boosting rather than a linear model on purpose: the whole point is that g
    may be non-linear, and a linear nuisance would leave exactly the
    misspecification DML is designed to remove."""
    pred = np.full(len(target), np.nan)
    for tr, te in folds:
        if kind == "binary" and len(np.unique(target[tr])) < 2:
            pred[te] = target[tr].mean()
            continue
        if kind == "binary":
            m = HistGradientBoostingClassifier(max_iter=200, max_depth=4,
                                               learning_rate=0.06,
                                               random_state=0)
            m.fit(Z.iloc[tr], target[tr])
            pred[te] = m.predict_proba(Z.iloc[te])[:, 1]
        else:
            m = HistGradientBoostingRegressor(max_iter=200, max_depth=4,
                                              learning_rate=0.06,
                                              random_state=0)
            m.fit(Z.iloc[tr], target[tr])
            pred[te] = m.predict(Z.iloc[te])
    return pred


def run(args):
    C.header("MODEL 3: double machine learning",
             "The macro increment as a COEFFICIENT with a confidence interval, "
             "not a difference\nof two AUCs. Cross-fitted by time block; "
             "orthogonal to the macro nuisance.")

    df = C.load_scored(args.scored, args.rigid)
    y = df["hit"].astype(int).values.astype(float)
    blocks = C.time_blocks(df, args.block_years)

    # required=True: this model's entire question is 'what does wording add
    # OVER the economy'. Without macro data there is no question to answer, so
    # it must fail rather than silently report an unconditional association.
    Z = C.macro_design(df, required=True)
    D = C.standardize(C.drop_collinear(C.claim_design(df)))

    folds, n_splits = grouped_folds(blocks, args.folds)
    print(f"\n  claims {len(df):,}   hit rate {y.mean():.3f}   "
          f"blocks {len(set(blocks))}")
    print(f"  macro nuisance features {Z.shape[1]}   claim features "
          f"{D.shape[1]}   cross-fit folds {n_splits} (grouped by block)")

    # --- stage 1: what does the economy alone predict? ----------------------
    print("\n  stage 1: cross-fitting E[hit | macro] ...")
    y_hat = crossfit_predict(Z, y, folds, "binary")
    ok = np.isfinite(y_hat)
    auc_macro = roc_auc_score(y[ok], y_hat[ok]) if len(set(y[ok])) > 1 else np.nan
    print(f"    out-of-fold macro-only AUC {auc_macro:.3f}   "
          f"(model_hit reports the same quantity)")

    print(f"  stage 1: cross-fitting E[feature | macro] for "
          f"{D.shape[1]} claim features ...")
    D_hat = pd.DataFrame(index=D.index, columns=D.columns, dtype=float)
    for c in D.columns:
        kind = "binary" if set(np.unique(D[c].values)) <= {0.0, 1.0} else "cont"
        D_hat[c] = crossfit_predict(Z, D[c].values, folds, kind)

    # --- stage 2: regress leftover outcome on leftover features -------------
    y_res = y - y_hat
    D_res = (D - D_hat)
    D_res = C.drop_collinear(D_res)
    keep = np.isfinite(y_res) & np.isfinite(D_res).all(axis=1).values
    print(f"  stage 2: OLS of residualised hit on residualised features "
          f"({keep.sum():,} rows)")

    Xr = sm.add_constant(D_res.loc[keep], has_constant="add")
    fit = sm.OLS(y_res[keep], Xr).fit(cov_type="cluster",
                                      cov_kwds={"groups": blocks[keep]})

    names = [c for c in Xr.columns if c != "const"]
    idx = [list(Xr.columns).index(c) for c in names]
    theta = fit.params.values[idx]
    se = fit.bse.values[idx]

    tab = C.coef_table(
        names, theta, se,
        "DML partial effects on P(hit), holding the economy fixed",
        note="Units are PROBABILITY, not log-odds: +0.02 means the feature "
             "raises the\n  chance a forecast came true by 2 percentage "
             "points, net of the macro state.\n  SEs clustered by 3-year block.",
        n=int(keep.sum()), odds=False)
    C.save(tab, "m3_dml_coefficients.csv")

    # --- the joint test: does WORDING add anything at all over the economy? -
    R = np.zeros((len(idx), len(fit.params)))
    for i, j in enumerate(idx):
        R[i, j] = 1.0
    wald = fit.wald_test(R, scalar=True)
    stat = float(np.squeeze(wald.statistic))
    pval = float(np.squeeze(wald.pvalue))
    dfree = len(idx)

    print("\n=== JOINT TEST: do claim features add anything over the economy? ===")
    print(f"  H0: every claim-feature partial effect is zero")
    print(f"  chi2({dfree}) = {stat:.2f}   p = {pval:.4g}   "
          f"(cluster-robust, {len(set(blocks))} blocks)")
    if pval < 0.05:
        print("  -> Reject. Wording carries information about accuracy that "
              "the economy\n     does not. This is the properly-sized version "
              "of a positive AUC delta.")
    else:
        print("  -> Do not reject. Whether a forecast came true is governed by "
              "the economy,\n     not by how it was written -- and this is now "
              "a real null with a real\n     p-value, not the absence of an "
              "AUC gap.")

    if pval < 0.05:
        # Which features carry the rejection, and are they substantive or
        # definitional? `direction` is the one to watch: predicting 'improve'
        # hits more often simply because the economy expands most of the time,
        # which is a fact about the economy rather than about writing.
        sig = tab[tab["p"] < 0.05].reindex(
            tab["coef"].abs().sort_values(ascending=False).index).dropna()
        print(f"\n  Which features carry it ({len(sig)} individually "
              f"significant):")
        for _, r in sig.head(6).iterrows():
            print(f"    {r['term']:<26} {r['coef']:+.4f}  "
                  f"95% CI [{r['lo95']:+.4f}, {r['hi95']:+.4f}]")
        print("  Check before quoting: a rejection driven by c_direction_* is "
              "largely the\n  base-rate asymmetry -- 'improve' hits ~60% "
              "because expansions are common --\n  not evidence that WORDING "
              "matters. The stylistic features (hedged, quoted,\n  named, "
              "has_number, len) are the ones that speak to how a forecast "
              "was written.")
    else:
        # An interval matters as much as the p-value: 'no effect' only means
        # something if the study could have detected one.
        largest = tab.reindex(
            tab["coef"].abs().sort_values(ascending=False).index)
        print("\n  Precision check -- the largest effects, with their intervals:")
        for _, r in largest.head(4).iterrows():
            print(f"    {r['term']:<26} {r['coef']:+.4f}  "
                  f"95% CI [{r['lo95']:+.4f}, {r['hi95']:+.4f}] "
                  f"({abs(r['hi95'] - r['lo95']) * 100:.1f} pp wide)")
        print("  A null with intervals this tight rules out effects of that "
              "size. A null with\n  wide intervals would only mean the study "
              "was underpowered -- say which it is.")

    style_stat = style_p = np.nan
    style = [c for c in ["c_hedged", "c_quoted", "c_named", "c_has_number",
                         "c_len"] if c in list(tab["term"])]
    if style:
        si = [list(Xr.columns).index(c) for c in style]
        Rs = np.zeros((len(si), len(fit.params)))
        for i, j in enumerate(si):
            Rs[i, j] = 1.0
        ws = fit.wald_test(Rs, scalar=True)
        style_stat = float(np.squeeze(ws.statistic))
        style_p = float(np.squeeze(ws.pvalue))
        print(f"\n=== narrower test: STYLE features only "
              f"({', '.join(style)}) ===")
        print(f"  chi2({len(si)}) = {style_stat:.2f}   p = {style_p:.4g}")
        print("  This drops direction and topic -- which encode WHAT was "
              "predicted -- and\n  keeps only HOW it was written. This is the "
              "poster's actual question.")
        if style_p >= 0.05 and pval < 0.05:
            print("\n  The two tests together are the headline: the full test "
                  "rejects, the style\n  test does not. What a forecast "
                  "PREDICTED carries real information about\n  whether it came "
                  "true; HOW it was written carries none that survives the\n"
                  "  economy and honest clustering.")

    summary = pd.DataFrame([{
        "n": int(keep.sum()), "n_blocks": len(set(blocks)),
        "auc_macro_oof": auc_macro, "wald_chi2": stat, "wald_df": dfree,
        "wald_p": pval,
        "style_chi2": style_stat, "style_df": len(style), "style_p": style_p,
        "max_abs_effect": float(tab["coef"].abs().max()),
        "median_ci_width_pp": float((tab["hi95"] - tab["lo95"]).median() * 100),
    }])
    C.save(summary, "m3_dml_summary.csv")

    # --- placebo: shuffle the claim features within block -------------------
    if args.placebo:
        print(f"\n  placebo: re-running the joint test with claim features "
              f"shuffled WITHIN block\n  ({args.placebo} draws). A correct "
              f"procedure rejects about 5% of the time.")
        rng = np.random.default_rng(0)
        rejects = 0
        for _ in range(args.placebo):
            perm = np.arange(len(D_res))
            for b in set(blocks[keep]):
                i = np.where(blocks[keep] == b)[0]
                perm[i] = rng.permutation(i)
            Xp = sm.add_constant(D_res.loc[keep].iloc[perm].reset_index(drop=True),
                                 has_constant="add")
            f2 = sm.OLS(y_res[keep], Xp).fit(cov_type="cluster",
                                             cov_kwds={"groups": blocks[keep]})
            R2 = np.zeros((len(idx), len(f2.params)))
            for i, j in enumerate(idx):
                R2[i, j] = 1.0
            if float(np.squeeze(f2.wald_test(R2, scalar=True).pvalue)) < 0.05:
                rejects += 1
        rate = rejects / args.placebo
        print(f"  false-positive rate {rate:.1%} at the 5% level "
              f"({rejects}/{args.placebo}).")
        print("  Near 5% means the clustered inference is correctly sized; "
              "much above it\n  would mean the standard errors are still too "
              "small and the main p-value\n  cannot be trusted.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_common_args(ap)
    ap.add_argument("--folds", type=int, default=5,
                    help="cross-fitting folds (grouped by time block)")
    ap.add_argument("--placebo", type=int, default=0,
                    help="placebo draws to check the test's false-positive rate")
    run(ap.parse_args())
