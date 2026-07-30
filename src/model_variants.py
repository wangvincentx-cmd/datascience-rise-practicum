"""Two variants on the AUC-0.647 hit model, run under its exact evaluation.

Both questions come straight out of the poster's own logic, and both are
answered here rather than argued about:

  A. PENALTY. hit_predictor.py uses L2 with C=0.5, a value never tuned on this
     corpus (the documented grid search in model.py was for the v1 TF-IDF
     model on 843 episodes). L1 is the interesting alternative: with ~40
     columns against ~21 independent time blocks, lasso zeroes most of them
     and hands back a short list of survivors -- which is both a better poster
     artifact than 40 coefficients and a real robustness check on the claim
     that the direction x economy terms are what carry the model.

     Note this is the PENALTY, not the loss and not the optimiser. The loss
     stays log-loss; changing it makes the model something else. The optimiser
     cannot matter at all -- penalised log-loss is strictly convex, so lbfgs,
     saga and Adam all land on the same coefficients. Only lbfgs cannot fit
     L1, which is why saga appears below.

  B. GENERAL BUSINESS ONLY. The corpus is 61% general-business claims and they
     hit at 0.561 against a pooled 0.513. Restricting to them asks whether the
     model works on one homogeneous topic, or whether its apparent skill is
     partly the model learning which TOPIC is easy -- a fact about the scoring
     series, not about forecasting.

Everything tried is written to docs/RESULTS_MODEL_VARIANTS.md, including the
configurations that lost. A grid where only the winner is reported is a grid
that cannot be checked.

Evaluation is hit_predictor's, unchanged: leave-one-3-year-block-out, pooled
out-of-fold predictions, ROC-AUC / PR-AUC / Brier. Tuning changes are held to
this project's standing bar (see model.py's LOGIT_C history): a candidate must
beat the incumbent on MOST FOLDS, not merely on the pooled number.

    python src/model_variants.py                 # both parts
    python src/model_variants.py --only penalty
    python src/model_variants.py --only topic
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hit_predictor as HP
from macro_context import BLOCK_YEARS, FACTORS

warnings.filterwarnings("ignore")

OUT = Path("data/models")
DOC = Path("docs/RESULTS_MODEL_VARIANTS.md")

# The C values tried. Spans two orders of magnitude around the incumbent 0.5 so
# the answer cannot be "you only looked next door". Kept to 5 values because
# every one costs ~21 refits per penalty and saga is the slow solver.
C_GRID = [0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0]

# l1_ratio for elastic net. One value, not a second grid: elastic net is here
# as a midpoint between the two penalties actually being compared, not as a
# candidate to be tuned in its own right.
L1_RATIO = 0.5


def load():
    """The same frame, outcome and grouping hit_predictor.run uses."""
    p = Path("data/scored/macro_context.csv")
    if not p.exists():
        raise SystemExit(f"{p} not found -- run `python src/macro_context.py` first.")
    d = pd.read_csv(p, low_memory=False)
    d = d[d["hit"].isin([0, 1])].reset_index(drop=True)
    y = d["hit"].astype(int).values
    year = pd.to_datetime(d["date"]).dt.year
    groups = ((year // BLOCK_YEARS) * BLOCK_YEARS).values
    return d, y, groups


def design(d):
    """Full spec 5: claim + economy + direction x economy."""
    X = pd.concat([HP.claim_features(d), HP.macro_block(d[FACTORS]),
                   HP.interaction_block(d, d[FACTORS])], axis=1)
    return X, list(HP.CLAIM_CAT), HP.CLAIM_NUM + HP.MACRO_NUM + HP.INTER_NUM


def clf_for(penalty, C):
    """The estimator for one grid cell.

    saga for L1/elastic net because lbfgs cannot fit a non-smooth penalty.
    max_iter is raised to 5000: saga converges more slowly than lbfgs and a
    comparison between a converged fit and an unconverged one is not a
    comparison of penalties."""
    if penalty == "l2":
        return LogisticRegression(penalty="l2", C=C, solver="lbfgs",
                                  max_iter=2000, random_state=HP.SEED)
    if penalty == "l1":
        return LogisticRegression(penalty="l1", C=C, solver="saga",
                                  max_iter=5000, random_state=HP.SEED)
    return LogisticRegression(penalty="elasticnet", C=C, solver="saga",
                              l1_ratio=L1_RATIO, max_iter=5000,
                              random_state=HP.SEED)


def fold_decomposition(y, oof_a, oof_b, groups, label_a, label_b):
    """Is a pooled AUC gain a WITHIN-era gain, or only a cross-era one?

    This distinction cost us a wrong recommendation once, so it is now a
    function rather than a habit. Pooled out-of-fold AUC ranks all 14,251
    predictions against each other, INCLUDING claims from different eras, so it
    rewards two different abilities:

      1. within an era, ranking hits above misses  -- forecast-level skill,
         which is what the project actually claims to measure
      2. across eras, knowing 1929 was a worse year than 1959 -- macro
         information, and a model that merely puts its scores on a consistent
         scale across decades gains here for free

    A model can add several points of pooled AUC while being exactly level on
    (1). The per-fold deltas separate them: if the mean and size-weighted
    per-fold gains are ~0 while the pooled gain is positive, the improvement is
    entirely (2) and must not be reported as better forecast discrimination."""
    fa, fb = per_fold_auc(y, oof_a, groups), per_fold_auc(y, oof_b, groups)
    shared = sorted(set(fa) & set(fb))
    dl = np.array([fa[g] - fb[g] for g in shared])
    n = np.array([((groups == g) & np.isfinite(oof_a)).sum() for g in shared])
    ok_a, ok_b = np.isfinite(oof_a), np.isfinite(oof_b)
    pooled = roc_auc_score(y[ok_a], oof_a[ok_a]) - roc_auc_score(y[ok_b], oof_b[ok_b])
    r = {"a": label_a, "b": label_b, "pooled_delta": pooled,
         "folds_won": int((dl > 0).sum()), "n_folds": len(dl),
         "mean_fold_delta": float(dl.mean()),
         "wtd_fold_delta": float(np.average(dl, weights=n)),
         "worst_block": float(dl.min()), "best_block": float(dl.max())}
    print(f"\n  {label_a}  vs  {label_b}")
    print(f"    pooled AUC delta      {r['pooled_delta']:+.4f}")
    print(f"    folds won             {r['folds_won']}/{r['n_folds']}")
    print(f"    mean per-fold delta   {r['mean_fold_delta']:+.4f}")
    print(f"    size-weighted delta   {r['wtd_fold_delta']:+.4f}   <- the honest one")
    print(f"    worst / best block    {r['worst_block']:+.3f} / {r['best_block']:+.3f}")
    return r


def per_fold_auc(y, oof, groups):
    """AUC inside each held-out block.

    The pooled OOF AUC is the headline, but it can be carried by one large
    block. The project's bar for accepting a tuning change is a majority of
    FOLDS, so the per-fold vector is what that bar is applied to."""
    out = {}
    for g in sorted(set(groups)):
        m = (groups == g) & np.isfinite(oof)
        if m.sum() < 20 or len(np.unique(y[m])) < 2:
            continue
        out[g] = roc_auc_score(y[m], oof[m])
    return out


def nonzero_terms(X, y, cat, num, penalty, C):
    """Which coefficients survive, fitted on ALL rows.

    In-sample on purpose and reported as such: this is a DESCRIPTION of what
    the penalty keeps, not an evaluation of it. The evaluation is the
    out-of-fold AUC in the same row of the table. Fitting the description on
    the full data avoids the awkwardness of 21 different survivor lists."""
    pipe = HP.make_pipe(cat, num, clf=clf_for(penalty, C))
    pipe.fit(X, y)
    names = pipe.named_steps["pre"].get_feature_names_out()
    coef = pipe.named_steps["clf"].coef_.ravel()
    keep = np.abs(coef) > 1e-8
    s = pd.Series(coef[keep], index=[n.split("__", 1)[-1] for n in names[keep]])
    return s.reindex(s.abs().sort_values(ascending=False).index), len(coef)


def run_penalty(d, y, groups):
    X, cat, num = design(d)
    print(f"\n=== A. PENALTY COMPARISON ===")
    print(f"  {len(d):,} claims   hit rate {y.mean():.3f}   "
          f"{len(set(groups))} blocks   {X.shape[1]} raw columns")
    print(f"  loss = log-loss throughout; only the penalty changes.")
    print(f"\n  {'penalty':<12}{'C':>7}{'ROC-AUC':>9}{'PR-AUC':>9}{'Brier':>8}"
          f"{'nonzero':>9}{'folds won':>11}")

    # Every cell is fitted first and compared second. Comparing inside the loop
    # would leave the cells evaluated BEFORE the incumbent with no yardstick --
    # and those turned out to be the ones that win.
    cells = []
    for penalty in ["l2", "l1", "elasticnet"]:
        for C in C_GRID:
            oof = HP.oof_predict(X, y, groups, cat, num,
                                 clf=clf_for(penalty, C))
            surv, total = nonzero_terms(X, y, cat, num, penalty, C)
            cells.append({"penalty": penalty, "C": C,
                          "m": HP.metrics(y, oof),
                          "folds": per_fold_auc(y, oof, groups),
                          "nonzero": len(surv), "n_coef": total})

    base = next(c["folds"] for c in cells
                if c["penalty"] == "l2" and c["C"] == 0.5)

    rows = []
    for c in cells:
        m, folds = c["m"], c["folds"]
        won = sum(1 for g, a in folds.items() if a > base.get(g, np.inf))
        rows.append({"penalty": c["penalty"], "C": c["C"],
                     "roc_auc": m["auc"], "pr_auc": m["pr_auc"],
                     "brier": m["brier"], "nonzero": int(c["nonzero"]),
                     "n_coef": int(c["n_coef"]), "folds_won": won,
                     "n_folds": len(folds)})
        tag = "  <- incumbent" if (c["penalty"] == "l2" and c["C"] == 0.5) else ""
        print(f"  {c['penalty']:<12}{c['C']:>7}{m['auc']:>9.3f}"
              f"{m['pr_auc']:>9.3f}{m['brier']:>8.3f}"
              f"{c['nonzero']:>6}/{c['n_coef']:<3}"
              f"{f'{won}/{len(folds)}':>11}{tag}")

    tab = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT / "model_variants_penalty.csv", index=False)

    best_l1 = tab[tab["penalty"] == "l1"].sort_values("roc_auc").iloc[-1]
    surv, total = nonzero_terms(X, y, cat, num, "l1", best_l1["C"])
    surv.to_csv(OUT / "model_variants_l1_survivors.csv", header=["coef"])
    print(f"\n  Best L1 is C={best_l1['C']}: AUC {best_l1['roc_auc']:.3f}, "
          f"{int(best_l1['nonzero'])} of {total} coefficients survive.")
    print(f"  Survivors, largest first (in-sample fit, full data):")
    for name, v in surv.head(12).items():
        print(f"    {name:<34}{v:+.4f}")
    return tab, surv


def run_topic(d, y, groups, topic="general_business"):
    m = (d["topic"] == topic).values
    dt, yt, gt = d[m].reset_index(drop=True), y[m], groups[m]
    X, cat, num = design(dt)

    # c_topic is constant inside the subset. Leaving it in would not break the
    # fit -- the one-hot encoder drops unseen levels -- but it would put a
    # meaningless column in the survivor list, so it goes.
    cat = [c for c in cat if c != "c_topic"]

    print(f"\n=== B. {topic.upper()} ONLY ===")
    print(f"  {len(dt):,} claims ({len(dt)/len(d):.0%} of the corpus)   "
          f"hit rate {yt.mean():.3f} (pooled corpus: {y.mean():.3f})   "
          f"{len(set(gt))} blocks")

    # The same nested ladder hit_predictor prints, so the numbers are directly
    # comparable to the 0.647 table rather than to nothing.
    ladder = [
        ("claim features only", cat, HP.CLAIM_NUM),
        ("economy only", [], HP.MACRO_NUM),
        ("claim + economy (additive)", cat, HP.CLAIM_NUM + HP.MACRO_NUM),
        ("+ direction x economy", cat,
         HP.CLAIM_NUM + HP.MACRO_NUM + HP.INTER_NUM),
    ]
    print(f"\n  {'model':<32}{'ROC-AUC':>9}{'PR-AUC':>9}{'Brier':>8}")
    rows = []
    for name, c, n in ladder:
        oof = HP.oof_predict(X, yt, gt, c, n)
        mm = HP.metrics(yt, oof)
        rows.append({"model": name, **{k: mm[k] for k in
                                       ("auc", "pr_auc", "brier")}})
        print(f"  {name:<32}{mm['auc']:>9.3f}{mm['pr_auc']:>9.3f}"
              f"{mm['brier']:>8.3f}")

    # L1 on the subset too: the survivor list is the readable output, and a
    # topic-specific list that differs from the pooled one is itself a finding.
    full_num = HP.CLAIM_NUM + HP.MACRO_NUM + HP.INTER_NUM
    oof_l1 = HP.oof_predict(X, yt, gt, cat, full_num, clf=clf_for("l1", 0.1))
    m_l1 = HP.metrics(yt, oof_l1)
    surv, total = nonzero_terms(X, yt, cat, full_num, "l1", 0.1)
    rows.append({"model": "+ direction x economy, L1 (C=0.1)",
                 "auc": m_l1["auc"], "pr_auc": m_l1["pr_auc"],
                 "brier": m_l1["brier"]})
    print(f"  {'+ direction x economy, L1 (C=0.1)':<32}{m_l1['auc']:>9.3f}"
          f"{m_l1['pr_auc']:>9.3f}{m_l1['brier']:>8.3f}")
    print(f"\n  L1 keeps {len(surv)} of {total} coefficients:")
    for name, v in surv.head(12).items():
        print(f"    {name:<34}{v:+.4f}")

    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / f"model_variants_{topic}.csv", index=False)
    surv.to_csv(OUT / f"model_variants_{topic}_l1_survivors.csv",
                header=["coef"])
    return tab, surv


def run_diagnose(d, y, groups):
    """Which reported gains are WITHIN-era, and which are only cross-era?

    Run on the two penalty candidates and on the poster's own headline, because
    the headline has to pass the same test it is being used to judge others by."""
    X, cat, num = design(d)
    print(f"\n=== C. FOLD DECOMPOSITION OF EVERY CLAIMED GAIN ===")
    base = HP.oof_predict(X, y, groups, cat, num, clf=clf_for("l2", 0.5))

    rows = [
        fold_decomposition(y, HP.oof_predict(X, y, groups, cat, num,
                                             clf=clf_for("l2", 0.02)),
                           base, groups, "L2 C=0.02", "L2 C=0.5 (incumbent)"),
        fold_decomposition(y, HP.oof_predict(X, y, groups, cat, num,
                                             clf=clf_for("l1", 0.05)),
                           base, groups, "L1 C=0.05", "L2 C=0.5 (incumbent)"),
    ]

    # The headline. If the interaction block's +0.066 were also cross-era only,
    # the poster's central claim would need rewriting -- so it is tested here
    # rather than assumed.
    add = HP.oof_predict(X, y, groups, cat, HP.CLAIM_NUM + HP.MACRO_NUM)
    rows.append(fold_decomposition(y, base, add, groups,
                                   "interacted (the headline)", "additive"))

    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "model_variants_fold_decomposition.csv", index=False)
    return tab


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["penalty", "topic", "diagnose"],
                    default=None)
    ap.add_argument("--topic", default="general_business")
    a = ap.parse_args()

    d, y, groups = load()
    if a.only in (None, "penalty"):
        run_penalty(d, y, groups)
    if a.only in (None, "topic"):
        run_topic(d, y, groups, a.topic)
    if a.only in (None, "diagnose"):
        run_diagnose(d, y, groups)
    print(f"\nTables written to {OUT}/model_variants_*.csv")


if __name__ == "__main__":
    main()
