"""
Load the scored+context table for one or more corpora -- WITHOUT merging them
on disk.

The LOC corpus (1900-1963, gpt-4.1-labelled) is the published result. ProQuest
(1965-2009, gpt-4o-mini-labelled, via the TDM Studio VM) is a second corpus that
may or may not survive scrutiny. Merging the two into one CSV would make that
decision irreversible: there would be no way back to a LOC-only number without
rebuilding, and every downstream table would silently change the day the merge
happened.

So they stay in separate files and are combined only in memory, at load time,
on request:

    load_context()                      -> LOC only. THE DEFAULT. Unchanged.
    load_context(("loc", "proquest"))   -> both, with a `source` column

THE INVARIANCE THAT MATTERS: with default arguments this returns exactly the
rows the notebook read before this module existed, and every model feature built
from it is identical. `source` is the only added column and no feature block
reads it (claim_features, macro_block and interaction_block in hit_predictor.py
touch none of it). test_context_sources.py asserts this against the real table.

WHY POOLING IS NOT THE DEFAULT, beyond reversibility -- the two corpora differ
on three axes at once, and a naive pool attributes all three to whatever the
model finds:

  1. ERA. 1900-1963 vs 1965-2009, with NO overlap. The CV holds out 3-year
     blocks, so ProQuest adds new folds rather than strengthening existing ones.
  2. LABELLER. gpt-4.1 vs gpt-4o-mini + a self-verify pass (gold F1 0.512,
     precision 0.700, direction kappa 0.81). Source is confounded with labeller.
  3. COVERAGE. The stock series (M1109BUSM293NNBR) ends 1968-12, so for
     ProQuest rows every stock factor is missing and its `has_` flag is 0 --
     constant by source, which is a free "this row is ProQuest" signal. Report
     `stock_coverage_warning()` before trusting any stock-factor result.

Usage:
    from context_sources import load_context
    d = load_context()                              # LOC only (default)
    d = load_context(("loc", "proquest"))           # pooled, opt-in
    d = load_context(("proquest",))                 # ProQuest alone
"""

from pathlib import Path

import pandas as pd

CONTEXT_FILES = {
    "loc": Path("data/scored/macro_context.csv"),
    "proquest": Path("data/scored/macro_context_proquest.csv"),
}

# Where each corpus's stock factors stop being real. See docstring point 3.
STOCK_SERIES_ENDS = "1968-12"

STOCK_FACTORS = ["stock_ret6", "stock_ret12", "stock_vol6", "stock_drawdown"]


def load_context(sources=("loc",), require=True, verbose=False):
    """Concatenate the context tables for `sources`.

    `source` is added if absent -- the LOC table predates the column, and
    back-filling it here means the file on disk does not have to change.
    """
    if isinstance(sources, str):
        sources = (sources,)
    unknown = [s for s in sources if s not in CONTEXT_FILES]
    if unknown:
        raise ValueError(f"unknown source(s) {unknown}; "
                         f"known: {sorted(CONTEXT_FILES)}")

    frames = []
    for s in sources:
        path = CONTEXT_FILES[s]
        if not path.exists():
            if require:
                raise SystemExit(
                    f"\n*** {path} does not exist.\n"
                    f"*** Build it first:\n"
                    f"***   python src/score_predictions.py --claims <{s}.jsonl> "
                    f"--out data/scored/{s}_scored.csv\n"
                    f"***   python src/macro_context.py --scored "
                    f"data/scored/{s}_scored.csv --out {path}\n")
            continue
        df = pd.read_csv(path, low_memory=False)
        if "source" not in df.columns:
            df["source"] = s
        frames.append(df)
        if verbose:
            dates = pd.to_datetime(df["date"], errors="coerce")
            print(f"  {s:<10} {len(df):>7,} rows  "
                  f"{str(dates.min())[:7]} to {str(dates.max())[:7]}")

    if not frames:
        raise SystemExit("no context tables found for " + str(sources))
    if len(frames) == 1:
        return frames[0]

    # Columns present in one corpus and not the other become NaN, which is
    # correct and visible. ProQuest has no `quote` (it cannot leave the VM) and
    # no `conditional_on`/`reasoning` (not in that prompt).
    out = pd.concat(frames, ignore_index=True, sort=False)
    if verbose:
        only = {s: sorted(set(f.columns) - set.intersection(
            *[set(g.columns) for g in frames]))
            for s, f in zip(sources, frames)}
        for s, cols in only.items():
            if cols:
                print(f"  columns only in {s}: {', '.join(cols)}")
    return out


def stock_coverage_warning(df):
    """Print, and return, how much of each source has real stock factors.

    The stock series ends 1968-12. Any corpus after that has ZERO coverage, and
    because `macro_block` emits a `has_` flag per factor, a pooled model can
    read those flags as a source indicator instead of as economics."""
    if "source" not in df.columns:
        df = df.assign(source="loc")
    rows = []
    for s, g in df.groupby("source"):
        have = g[STOCK_FACTORS[0]].notna().mean() if STOCK_FACTORS[0] in g else 0.0
        rows.append({"source": s, "n": len(g), "stock_coverage": have})
    rep = pd.DataFrame(rows)
    print(f"stock factors are only defined through {STOCK_SERIES_ENDS}:")
    for _, r in rep.iterrows():
        flag = "  <- no coverage; constant by source" if r.stock_coverage < 0.01 else ""
        print(f"  {r['source']:<10} {r['n']:>7,} rows   "
              f"coverage {r.stock_coverage:.1%}{flag}")
    if len(rep) > 1 and rep["stock_coverage"].min() < 0.01 < rep["stock_coverage"].max():
        print("  WARNING: coverage differs by source. Drop the stock factors, or\n"
              "           report per-source, before reading anything into them.")
    return rep


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", default="loc",
                    help="comma-separated: loc, proquest")
    args = ap.parse_args()
    srcs = tuple(s.strip() for s in args.sources.split(",") if s.strip())
    d = load_context(srcs, require=False, verbose=True)
    print(f"\ntotal {len(d):,} rows")
    if "hit" in d:
        scored = d[d["hit"].isin([0, 1])]
        print(f"scored {len(scored):,}  hit rate {scored['hit'].mean():.3f}")
        if "source" in d.columns and d["source"].nunique() > 1:
            print("\nby source:")
            print(scored.groupby("source")["hit"].agg(n="size", hit_rate="mean")
                  .round(3).to_string())
    print()
    stock_coverage_warning(d)
