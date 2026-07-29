"""
Two interest-rate factors, as newspaper-free replacements for EPU.

WHY THIS EXISTS. `epu` is the single largest contributor to the published hit
model (~0.060 of its 0.086 lift over claim-only), and it is built by counting
policy-uncertainty language in six newspapers: NYT, WSJ, Washington Post,
Chicago Tribune, LA Times, Boston Globe. Measured 2026-07-29:

    LOC monthly 1900-1963   0.0% of claims from those six papers
    ProQuest    1965-2009  94.1% of claims from those six papers

So on LOC, EPU is an outside instrument and the caveat in RESULTS_MACRO.md is
harsher than the evidence requires. On ProQuest it is not an outside instrument
at all: it would count anxious words in the same articles the forecasts were
extracted from, in the same papers, in the same (economy) section. The model
must not use it there.

The slot EPU fills is `uncertainty, interacted with direction`. These two fill
the same slot from market prices, which no journalist writes:

    credit_spread = BAA - AAA      corporate risk premium, monthly from 1919
    term_spread   = GS10 - TB3MS   the yield curve, monthly from 1953

`credit_spread` is the one that matters most, because it is defined on BOTH
corpora -- a feature that exists on only one side cannot support the
train-LOC/test-ProQuest transfer design. `term_spread` covers 100% of ProQuest
but only 1953+ of LOC, so it is a ProQuest-first factor (see COVERAGE_MIN in
fit_hit.py, which keeps thin factors out of the interaction block).

LEAKAGE. None of this is hindsight. Both are contemporaneous market prices,
never revised, and were on the wire the day they printed. Two deliberate
choices:

  * LAG = 1 month, not 0. FRED's GS10/BAA are *monthly averages of daily
    observations*, so month M's value is not complete until M ends -- a forecast
    printed on the 3rd of M cannot know it. macro_context gives STOCK a lag of 0
    because a same-day ticker really is same-day observable; a monthly average
    is not. One month is the conservative reading and costs almost nothing,
    since these series move slowly.
  * Level, not change-relative-to-future. Every lookup is `.asof(p)` on a
    lagged series, so nothing after the publication month can enter.

The yield curve is a known recession predictor. That is not leakage -- it was
public at print time, and "the information was on the table and went unused" is
the project's thesis, not a bug in it. It does mean a good result here should be
described as `the yield curve knew`, which is a well-established finding, rather
than as something this corpus discovered.

Usage:
    python new_model/rate_factors.py           # fetch, cache, report coverage
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from truth_data import load_fred  # noqa: E402

# Monthly averages of daily quotes -> not knowable until the month closes.
LAG = 1

# name -> (minuend, subtrahend). Both are spreads, so the level of rates (which
# drifts hugely across a century of inflation regimes) cancels out and what is
# left is the risk/slope signal that is comparable across eras.
SPREADS = {
    "credit_spread": ("BAA", "AAA"),
    "term_spread": ("GS10", "TB3MS"),
}

PRETTY = {
    "credit_spread": "credit spread (Baa - Aaa)",
    "term_spread": "yield curve (10y - 3m)",
}

FACTORS = list(SPREADS)


def _spread_or_none(name, lag=LAG, verbose=True):
    """A spread as a lagged Period-indexed Series, or None if unavailable.

    Mirrors macro_context._epu_or_none: a missing series degrades this factor to
    all-NaN (and its has_ flag to 0) rather than taking the pipeline down. The
    exception MESSAGE is printed, not just its class -- a bare '[unavailable]'
    once cost an afternoon."""
    a, b = SPREADS[name]
    try:
        sa, sb = load_fred(a), load_fred(b)
    except Exception as e:
        if verbose:
            print(f"  [{name} unavailable: {type(e).__name__}: {e}]")
            print(f"   {name} will be all-NaN; other factors unaffected, but a "
                  f"model trained without it is not comparable to one with it.")
        return None
    s = (sa - sb).dropna()
    return s.shift(lag) if lag else s


def build_rate_factors(dates, lag=LAG, verbose=True):
    """One row per date, one column per spread, aligned to publication month.

    Same contract as macro_context.build_context: `dates` in, a DataFrame with
    len(dates) rows out, NaN where a series does not reach that month. Out-of-
    range months are NaN rather than edge-filled -- `.asof` alone would happily
    carry 1953's first yield curve back to 1900."""
    series = {n: _spread_or_none(n, lag=lag, verbose=verbose) for n in FACTORS}
    rows = []
    for d in dates:
        p = pd.Timestamp(d).to_period("M")
        r = {}
        for n, s in series.items():
            r[n] = (float(s.asof(p))
                    if s is not None and s.index.min() <= p <= s.index.max()
                    and not pd.isna(s.asof(p)) else np.nan)
        rows.append(r)
    return pd.DataFrame(rows, columns=FACTORS)


def main():
    print(__doc__.strip().split("\n")[1])
    print(f"\npublication lag: {LAG} month(s)\n")
    for n in FACTORS:
        s = _spread_or_none(n)
        if s is None:
            print(f"  {n:<15} UNAVAILABLE")
            continue
        s = s.dropna()
        print(f"  {n:<15} {s.index.min()} -> {s.index.max()}   "
              f"n={len(s):,}   mean {s.mean():+.2f}   sd {s.std():.2f}")

    # Coverage against the two corpora's date ranges, which is the number that
    # decides whether a factor may enter the interaction block.
    for label, lo, hi in [("LOC 1900-1963", "1900-01", "1963-12"),
                          ("ProQuest 1965-2009", "1965-01", "2009-12")]:
        months = pd.period_range(lo, hi, freq="M")
        ctx = build_rate_factors(months.to_timestamp(), verbose=False)
        cov = ", ".join(f"{n} {ctx[n].notna().mean():6.1%}" for n in FACTORS)
        print(f"  {label:<20} {cov}")


if __name__ == "__main__":
    main()
