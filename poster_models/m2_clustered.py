"""MODEL 2 -- Honest uncertainty: cluster-robust and hierarchical logit.

The problem this solves
-----------------------
CLAUDE.md states the rule plainly: "the effective sample is ~21 blocks, not
14,251 claims." Forecasts printed in one era share wire copy, share a house
style, and share one realisation of the business cycle -- so they are nowhere
near 14,251 independent draws.

The default logit does not know that. It computes standard errors under the
assumption that every row is an independent observation, which makes every
interval too narrow, every p-value too small, and every finding less likely to
replicate than it looks. Nothing about the point estimates is wrong. The
uncertainty around them is.

Three progressively stronger corrections, all reported side by side:

  1. NAIVE            -- the default. Included only as the thing being corrected.
  2. CLUSTER-ROBUST   -- Liang-Zeger sandwich SEs clustered on 3-year block.
                         Point estimates identical; intervals honest. The ratio
                         of clustered to naive SE is the design effect, and the
                         implied effective sample size is the single most
                         useful number this file produces.
  3. BLOCK BOOTSTRAP  -- resample whole blocks with replacement, refit. Makes no
                         asymptotic assumption about the number of clusters,
                         which matters because 22 clusters is few: the sandwich
                         estimator is itself only asymptotically valid in the
                         number of CLUSTERS, not observations.
  4. HIERARCHICAL     -- a random intercept per block (variational Bayes). Does
                         not merely widen intervals: it PARTIALLY POOLS, so a
                         block with 200 claims is shrunk toward the grand mean
                         more than a block with 1,000. Also estimates directly
                         how much of the variation in hit rate is between eras
                         rather than between claims.

Unlike model 1, this model keeps across-time variation and controls for the
macro state instead of differencing it out -- so the two are complements, not
rivals. Read them together: model 1 says which effects survive the strictest
possible comparison, model 2 says how wide the error bars really are.

    python poster_models/m2_clustered.py
    python poster_models/m2_clustered.py --boot 500
"""

import argparse
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

import _common as C

warnings.filterwarnings("ignore")


def design(df, with_macro=True):
    """Claim features, plus publication-lagged macro controls when available."""
    X = C.claim_design(df)
    used_macro = False
    if with_macro:
        m = C.macro_design(df, required=False)
        if m is not None:
            X = pd.concat([X, m], axis=1)
            used_macro = True
    X = C.standardize(X)
    return C.drop_collinear(X), used_macro


def design_effect(naive_se, clust_se):
    """SE inflation and the sample size the data actually behaves like.

    A design effect of D means the clustered variance is D times the naive one,
    so the 14,251 claims carry as much information as 14,251/D independent
    ones. Reported per coefficient because clustering bites hardest on features
    that vary slowly over time (topic mix, house style) and barely at all on
    features that vary claim-to-claim within a month (hedging)."""
    ratio = np.divide(clust_se, naive_se,
                      out=np.full_like(clust_se, np.nan), where=naive_se > 0)
    return ratio, ratio ** 2


def block_bootstrap(y, X, blocks, n_boot, seed=0):
    """Resample whole blocks with replacement; refit; take the coefficient SD.

    Resampling BLOCKS rather than rows is what preserves the within-era
    correlation. Resampling rows would reproduce the naive standard error and
    tell us nothing."""
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(blocks)))
    idx_by_block = {b: np.where(blocks == b)[0] for b in uniq}
    Xc = sm.add_constant(X, has_constant="add")
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_block[b] for b in pick])
        yb = y[rows]
        if len(np.unique(yb)) < 2:
            continue
        try:
            r = sm.Logit(yb, Xc.iloc[rows]).fit(disp=0, maxiter=200)
            if np.all(np.isfinite(r.params.values)):
                draws.append(r.params.values)
        except Exception:
            continue
    if len(draws) < max(20, n_boot // 5):
        return None, None, len(draws)
    D = np.vstack(draws)
    lo, hi = np.percentile(D, [2.5, 97.5], axis=0)
    return D.std(axis=0), (lo, hi), len(draws)


def hierarchical(y, X, blocks):
    """Logit with a random intercept per time block, fitted by variational Bayes.

    statsmodels' BinomialBayesMixedGLM is used rather than PyMC deliberately --
    no new dependency, and with 22 groups and a single variance component the
    variational approximation is close enough for the purpose here, which is a
    partially-pooled interval rather than an exact posterior. The number worth
    reading is the between-block intercept SD: it says, on the log-odds scale,
    how much the baseline hit rate moves from era to era once claim features
    are accounted for."""
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    d = X.copy()
    d["_y"] = y
    d["_blk"] = blocks
    fixed = "_y ~ " + " + ".join(f"Q('{c}')" for c in X.columns)
    model = BinomialBayesMixedGLM.from_formula(
        fixed, {"blk": "0 + C(_blk)"}, d)
    return model.fit_vb(verbose=False)


def run(args):
    C.header("MODEL 2: cluster-robust and hierarchical logit",
             "Same coefficients as the standard model. Honest error bars, "
             "because the effective\nsample is ~22 time blocks, not 14,251 "
             "independent claims.")

    df = C.load_scored(args.scored, args.rigid)
    y = df["hit"].astype(int).values
    blocks = C.time_blocks(df, args.block_years)
    X, used_macro = design(df, with_macro=not args.no_macro)

    print(f"\n  claims {len(df):,}   hit rate {y.mean():.3f}   "
          f"blocks {len(set(blocks))}   features {len(X.columns)}"
          f"   macro controls: {'yes' if used_macro else 'NO'}")

    Xc = sm.add_constant(X, has_constant="add")
    naive = sm.Logit(y, Xc).fit(disp=0, maxiter=200)
    clust = sm.Logit(y, Xc).fit(disp=0, maxiter=200, cov_type="cluster",
                                cov_kwds={"groups": blocks})

    terms = list(Xc.columns)
    ratio, deff = design_effect(naive.bse.values, clust.bse.values)

    tab = C.coef_table(terms, clust.params.values, clust.bse.values,
                       "CLUSTER-ROBUST logit (SE clustered on 3-year block)",
                       n=len(df))
    tab["se_naive"] = naive.bse.values
    tab["se_ratio"] = ratio
    tab["design_effect"] = deff
    tab["n_eff"] = len(df) / np.where(deff > 0, deff, np.nan)

    print("\n=== what clustering costs: SE inflation per coefficient ===")
    print("  A ratio of 2 means the naive interval was half as wide as it "
          "should be.\n  n_eff is the number of INDEPENDENT claims this "
          "feature's estimate is worth.\n")
    print(f"  {'term':<26}{'se naive':>10}{'se clust':>10}{'ratio':>8}"
          f"{'n_eff':>10}   still sig?")
    order = tab.reindex(tab["se_ratio"].sort_values(ascending=False).index)
    for _, r in order.iterrows():
        was = "*" if abs(r["coef"] / r["se_naive"]) > 1.96 else " "
        now = "*" if abs(r["z"]) > 1.96 else " "
        flag = ("lost" if was == "*" and now == " " else
                "kept" if now == "*" else "-")
        print(f"  {r['term']:<26}{r['se_naive']:>10.3f}{r['se']:>10.3f}"
              f"{r['se_ratio']:>8.2f}{r['n_eff']:>10,.0f}   {flag}")

    lost = int(((np.abs(tab["coef"] / tab["se_naive"]) > 1.96) &
                (np.abs(tab["z"]) <= 1.96)).sum())
    med_deff = float(np.nanmedian(tab["design_effect"]))
    print(f"\n  {lost} coefficient(s) significant under naive SEs are NOT "
          f"significant once\n  clustering is respected. Median design effect "
          f"{med_deff:.1f}x -> the corpus behaves\n  like roughly "
          f"{len(df) / med_deff:,.0f} independent claims, not {len(df):,}.")

    # --- block bootstrap ----------------------------------------------------
    if args.boot:
        print(f"\n  running {args.boot} block-bootstrap refits "
              f"(resampling whole 3-year blocks)...")
        bse_b, ci_b, n_ok = block_bootstrap(y, X, blocks, args.boot)
        if bse_b is None:
            print(f"  bootstrap failed to converge often enough "
                  f"({n_ok} usable draws); skipping.")
        else:
            tab["se_boot"] = bse_b
            tab["boot_lo95"], tab["boot_hi95"] = ci_b
            print(f"  {n_ok} usable draws.")
            print("\n=== block bootstrap vs sandwich (do the two agree?) ===")
            print("  They should be close. Where the bootstrap is much wider, "
                  "the sandwich\n  estimator is being optimistic -- it is only "
                  "valid asymptotically in the\n  NUMBER OF CLUSTERS, and 22 "
                  "is not many.\n")
            print(f"  {'term':<26}{'se clust':>10}{'se boot':>10}"
                  f"{'bootstrap 95% CI':>22}")
            for _, r in tab.iterrows():
                ci = f"[{r['boot_lo95']:+.3f}, {r['boot_hi95']:+.3f}]"
                print(f"  {r['term']:<26}{r['se']:>10.3f}"
                      f"{r['se_boot']:>10.3f}{ci:>22}")

    C.save(tab, "m2_clustered.csv")

    # --- hierarchical / partial pooling -------------------------------------
    print("\n  fitting hierarchical model (random intercept per block)...")
    try:
        h = hierarchical(y, X, blocks)
    except Exception as e:
        print(f"  hierarchical fit failed: {e}")
        return

    names = list(h.model.exog_names)
    k = len(names)
    hp, hs = np.asarray(h.params[:k]), np.asarray(h.cov_params()[:k]) ** 0.5
    clean = [n.replace("Q('", "").replace("')", "") for n in names]
    htab = C.coef_table(clean, hp, hs,
                        "HIERARCHICAL logit (random intercept per time block)",
                        note="Posterior means with 95% credible intervals. "
                             "Blocks are partially pooled,\n  so small blocks "
                             "are shrunk toward the grand mean rather than "
                             "trusted whole.",
                        n=len(df))
    C.save(htab, "m2_hierarchical.csv")

    # Between-block variance: how much does the baseline hit rate move by era?
    vc_sd = float(np.exp(h.vcp_mean[0])) if len(h.vcp_mean) else np.nan
    if np.isfinite(vc_sd):
        icc = vc_sd ** 2 / (vc_sd ** 2 + np.pi ** 2 / 3)
        print(f"\n=== how much of the variation is BETWEEN eras? ===")
        print(f"  between-block intercept SD : {vc_sd:.3f} (log-odds)")
        print(f"  intraclass correlation     : {icc:.3f}")
        print(f"  Reading: {icc:.1%} of the residual variation in whether a "
              f"forecast came true\n  is variation between 3-year blocks, not "
              f"between claims. That is exactly the\n  correlation the naive "
              f"standard errors assume away.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_common_args(ap)
    ap.add_argument("--boot", type=int, default=300,
                    help="block-bootstrap replications (0 to skip)")
    ap.add_argument("--no-macro", action="store_true",
                    help="claim features only, no macro controls")
    run(ap.parse_args())
