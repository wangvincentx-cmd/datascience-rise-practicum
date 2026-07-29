"""
Fit main's hit-predictor ladder with the newspaper-free uncertainty factors.

Runs `src/hit_predictor.py` unmodified. The only thing this driver changes is
WHICH factors exist and which of them are allowed into the direction x economy
interaction block:

  added to the interaction block   stock_vol6      (already computed by
                                                    macro_context, never
                                                    interacted until now)
  added as new factors             credit_spread   Baa - Aaa, 1919+
                                   term_spread     10y - 3m, 1953+

WHY. `epu` supplies most of the published model's lift and is built by counting
policy-uncertainty language in six newspapers -- 0% of the LOC corpus, but 94%
of the ProQuest corpus (see new_model/rate_factors.py). On ProQuest it must be
excluded, which leaves the model's uncertainty slot empty unless something fills
it. These three fill it from market prices.

`stock_vol6` is the one to watch. It is realized volatility from the price
series, so it involves no newspapers anywhere, and unlike EPU it REPLICATED on
the crisis corpus (-0.268 optimistic / +0.252 pessimistic, versus EPU's
+0.084 / -0.012). It has been sitting in FACTORS the whole time; it was simply
never in INTERACT_WITH, so the model was never allowed to multiply it by
direction -- the exact structure that made EPU work.

THE COVERAGE GATE. src/hit_predictor.py's INTERACT_WITH carries this warning:
"interacting a 1948+ series with direction would produce a term that is zero for
86% of the corpus and whose coefficient is estimated off one decade." That
applies directly to term_spread, which is 100% on ProQuest but only 1953+ on
LOC. So a factor joins the interaction block only if it is present on at least
COVERAGE_MIN of the rows in THIS corpus, and the decision is printed. It is not
a judgement call made once and forgotten; it is recomputed per corpus.

Outputs go to new_model/data/models/, never data/models/, so the published
model's artefacts cannot be overwritten by running this.

Usage (from the repo root):
    # LOC, for comparison -- epu is legitimate here (0% paper overlap)
    python new_model/fit_hit.py --context new_model/data/macro_context_loc.csv

    # ProQuest -- epu is NOT legitimate here (94% paper overlap)
    python new_model/fit_hit.py \
        --context new_model/data/macro_context_proquest.csv --exclude epu

    # what the new factors are worth: same corpus, with and without them
    python new_model/fit_hit.py --context ... --exclude epu --baseline
"""

import argparse
import sys
import warnings
from argparse import Namespace
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

import hit_predictor as hp  # noqa: E402
from rate_factors import FACTORS as RATE_FACTORS  # noqa: E402

# A factor may enter the interaction block only if it is defined on this share
# of the corpus. Below it, the interaction term is mostly structural zeros and
# its coefficient is estimated off whichever decade happens to have the data.
COVERAGE_MIN = 0.90

# Newspaper-free, and the point of this driver.
NEW_INTERACTIONS = ["stock_vol6"] + RATE_FACTORS


def patch_factors(d, baseline=False, verbose=True):
    """Extend hit_predictor's factor and interaction lists in place.

    MACRO_NUM, INTER_NUM and LADDER are module-level constants built from
    FACTORS/INTERACT_WITH at import time, so patching the two lists alone would
    silently do nothing -- all five have to be rebuilt together. `run()` reads
    FACTORS and INTERACT_WITH at call time, so patching the module attributes is
    enough once the derived constants are rebuilt."""
    factors = list(hp.FACTORS)
    interact = list(hp.INTERACT_WITH)

    if not baseline:
        for f in RATE_FACTORS:
            if f in d.columns and f not in factors:
                factors.append(f)
            elif f not in d.columns and verbose:
                print(f"  [{f} not in this context table -- rebuild it with "
                      f"new_model/build_context.py to include it]")

        for f in NEW_INTERACTIONS:
            if f not in factors or f in interact:
                continue
            cov = d[f].notna().mean()
            if cov >= COVERAGE_MIN:
                interact.append(f)
                if verbose:
                    print(f"  interact: + {f:<15} (coverage {cov:.1%})")
            elif verbose:
                print(f"  interact: - {f:<15} (coverage {cov:.1%} < "
                      f"{COVERAGE_MIN:.0%} -- factor kept, interaction dropped)")

    hp.FACTORS = factors
    hp.INTERACT_WITH = interact
    hp.MACRO_NUM = ([f"m_{f}" for f in factors] + [f"has_{f}" for f in factors])
    hp.INTER_NUM = ["x_dir_sign"] + [f"x_dir_{f}" for f in interact]
    hp.LADDER = [
        ("1. base rate (no model)", [], []),
        ("2. claim features only", hp.CLAIM_CAT, hp.CLAIM_NUM),
        ("3. economy only", [], hp.MACRO_NUM),
        ("4. claim + economy (additive)", hp.CLAIM_CAT,
         hp.CLAIM_NUM + hp.MACRO_NUM),
        ("5. + direction x economy", hp.CLAIM_CAT,
         hp.CLAIM_NUM + hp.MACRO_NUM + hp.INTER_NUM),
    ]
    return factors, interact


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--context", required=True)
    ap.add_argument("--exclude", default=None,
                    help="factor substrings to ablate, e.g. 'epu' (required "
                         "for ProQuest)")
    ap.add_argument("--baseline", action="store_true",
                    help="published factor set only, for a like-for-like diff")
    ap.add_argument("--perm", type=int, default=0)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--imp-reps", type=int, default=3)
    ap.add_argument("--no-importance", dest="importance",
                    action="store_false", default=True)
    args = ap.parse_args()

    ctx = Path(args.context)
    if not ctx.exists():
        raise SystemExit(
            f"\n*** {ctx} not found.\n"
            f"*** Build it first:\n"
            f"***   python new_model/build_context.py --scored <scored.csv> "
            f"--out {ctx}\n")

    d = pd.read_csv(ctx, low_memory=False)

    # The reason this whole folder exists. Silence here would let a ProQuest
    # model be fitted on a feature made from its own source text.
    src = str(d["source"].iloc[0]).lower() if "source" in d.columns else ""
    excluded = (args.exclude or "").split(",")
    if src == "proquest" and "epu" not in excluded and "epu" in hp.FACTORS:
        raise SystemExit(
            "\n*** REFUSING to fit a ProQuest model with `epu` included.\n"
            "*** 94% of ProQuest claims come from the six newspapers the EPU\n"
            "*** index is built from, so the feature is partly made of the\n"
            "*** articles the forecasts were extracted from.\n"
            "*** Re-run with --exclude epu (or --exclude epu,<others>).\n")

    print(f"context: {ctx}   source: {src or 'unknown'}   rows: {len(d):,}")
    print("factor set: published only" if args.baseline
          else "factor set: published + newspaper-free uncertainty")
    factors, interact = patch_factors(d, baseline=args.baseline)
    # Show the list AFTER ablation. Printing `epu` among the interactions of a
    # run invoked with --exclude epu is how a caveat gets lost.
    shown = [f for f in interact
             if not any(e and e in f for e in excluded)]
    print(f"  factors: {len(factors)}   interactions: {len(shown)}"
          f"   -> {', '.join(shown)}")

    # Outputs land under new_model/, so nothing here can overwrite the
    # published model's joblib/CSV/JSON.
    hp.MODELS = Path(__file__).resolve().parent / "data" / "models"

    missing = [f for f in factors if f not in d.columns]
    if missing:
        raise SystemExit(f"\n*** context table is missing {missing}.\n"
                         f"*** Rebuild it with new_model/build_context.py.\n")

    hp.run(Namespace(context=str(ctx), exclude=args.exclude, boot=args.boot,
                     importance=args.importance, imp_reps=args.imp_reps,
                     perm=args.perm))


if __name__ == "__main__":
    main()
