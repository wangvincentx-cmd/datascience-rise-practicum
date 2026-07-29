"""
Rebuild the scored+context table using the Shiller stock series.

Runs main's `src/macro_context.py` unmodified and swaps ONLY the stock series
underneath it, by patching `load_fred` for the one series id. Everything else --
the publication lags, the factor definitions, the leakage discipline, the block
bootstrap -- is the published code, so the two models differ in exactly one
input and any difference in the result is attributable to that.

Nothing in src/ is touched, which is the point: the published model stays
bit-reproducible, and `python tests/test_context_sources.py` still passes.

Outputs go to new_model/data/, never to data/scored/, so the old tables cannot
be overwritten by running this.

Usage (from the repo root):
    # LOC, with the new stock series
    python new_model/build_context.py --scored data/scored/monthly_scored.csv \
        --out new_model/data/macro_context_loc.csv

    # ProQuest, once its claims are scored
    python new_model/build_context.py --scored data/scored/proquest_scored.csv \
        --out new_model/data/macro_context_proquest.csv
"""

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

import macro_context as mc  # noqa: E402
import truth_data  # noqa: E402
from rate_factors import FACTORS as RATE_FACTORS  # noqa: E402
from rate_factors import build_rate_factors  # noqa: E402
from stock_series import load_stocks  # noqa: E402


def patch_stock_series(verbose=True):
    """Point macro_context's stock lookup at Shiller, leaving src/ untouched.

    macro_context.build_context calls load_fred(STOCK_SERIES) directly, so the
    swap has to happen at the module boundary. Only that one id is redirected;
    INDPRO, CPIAUCNS and UNRATE still come from the real loader, unchanged."""
    original = mc.load_fred
    target = truth_data.STOCK_SERIES
    shiller = load_stocks()

    def load_fred_patched(series_id, *a, **kw):
        if series_id == target:
            return shiller
        return original(series_id, *a, **kw)

    mc.load_fred = load_fred_patched
    if verbose:
        print(f"stock series: {target} -> Shiller S&P "
              f"({shiller.index.min()} to {shiller.index.max()})")
    return original


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scored", required=True, help="output of score_predictions.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", default=None,
                    help="value for the `source` column (default: inferred)")
    ap.add_argument("--lag-scale", type=float, default=1.0)
    ap.add_argument("--no-rate-factors", dest="rate_factors",
                    action="store_false", default=True,
                    help="omit credit_spread/term_spread (needs FRED access)")
    args = ap.parse_args()

    out = Path(args.out)
    if "scored" in out.parts and "new_model" not in out.parts:
        raise SystemExit(
            f"\n*** Refusing to write to {out}\n"
            f"*** That is the published model's directory. This script exists to\n"
            f"*** sit alongside it, not replace it. Write under new_model/data/.\n")

    patch_stock_series()

    # These four steps mirror macro_context.run() exactly -- same filter, same
    # has_ flags, same concat order -- so the only difference between this table
    # and the published one is the stock series. Reimplemented rather than
    # called because run() also prints the attribution report and takes a dozen
    # CLI args this driver does not need.
    df = pd.read_csv(args.scored, low_memory=False)
    n_all = len(df)
    df = df[(df["scorable"] == True) & (df["hit"].isin([0, 1]))].reset_index(drop=True)  # noqa: E712
    print(f"{len(df):,} scorable claims (of {n_all:,}) from {args.scored}")

    ctx = mc.build_context(df["date"], lag_scale=args.lag_scale)

    # The newspaper-free uncertainty factors. Added as extra COLUMNS rather than
    # by editing mc.FACTORS, so a table built here still reads correctly in any
    # published script that only knows about mc.FACTORS.
    factors = list(mc.FACTORS)
    if args.rate_factors:
        rates = build_rate_factors(df["date"])
        for f in RATE_FACTORS:
            ctx[f] = rates[f].values
        factors += RATE_FACTORS

    for f in factors:
        ctx[f"has_{f}"] = ctx[f].notna().astype(int)
    both = pd.concat([df.reset_index(drop=True), ctx.reset_index(drop=True)], axis=1)

    source = args.source or ("proquest" if "proquest" in Path(args.scored).name
                             else "loc")
    both["source"] = source

    out.parent.mkdir(parents=True, exist_ok=True)
    both.to_csv(out, index=False)
    print(f"source = {source}")
    print(f"wrote {len(both):,} rows x {both.shape[1]} cols -> {out}")

    have = both["stock_ret6"].notna().mean()
    print(f"stock_ret6 coverage: {have:.1%}"
          + ("   <- the whole point" if have > 0.95 else
             "   <- still incomplete, check the date range"))

    # Coverage decides eligibility for the interaction block downstream, so it is
    # printed here rather than left for fit_hit.py to discover.
    for f in (RATE_FACTORS if args.rate_factors else []):
        c = both[f].notna().mean()
        print(f"{f} coverage: {c:.1%}"
              + ("" if c >= 0.90 else
                 "   <- too thin to interact with direction on this corpus"))

    if source == "proquest":
        print("\n*** ProQuest: 94% of these claims come from the six newspapers "
              "the EPU\n*** index is built from. Fit with --exclude epu. "
              "See new_model/rate_factors.py.")


if __name__ == "__main__":
    main()
