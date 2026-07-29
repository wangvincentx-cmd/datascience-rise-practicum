"""
A stock series that covers BOTH corpora, replacing the one that stops in 1968.

The published model scores markets against FRED's M1109BUSM293NNBR, an NBER
historical index that ends 1968-12. That was fine while the corpus was LOC
1900-1963. It is not fine now:

    coverage            M1109BUSM293NNBR    Shiller S&P
    LOC 1900-1963              76.7%           100.0%
    ProQuest 1965-2009          8.9%           100.0%

and this is not a peripheral column. In the published permutation importances
`x_dir_stock_ret6` is the SECOND strongest feature (0.0155 AUC drop, behind only
`x_dir_epu` at 0.0378). On ProQuest rows it would be identically zero -- so the
model's number-two predictor is not merely missing, it is constant by source,
which a pooled model can read as "this row is ProQuest" instead of as economics.

Robert Shiller's monthly S&P composite (1871-) is the standard long series for
exactly this literature. It is one consistent series across both eras -- no
splice, no gap. Validated against the series it replaces on their 649-month
overlap (1914-12 to 1968-12):

    level correlation        0.9963
    6m-return correlation    0.9459
    12m-return correlation   0.9538

i.e. the same signal, measured further.

BECAUSE IT ALSO RAISES LOC COVERAGE (76.7% -> 100%), SWAPPING IT CHANGES THE
LOC RESULT. That is why this lives in new_model/ and touches nothing in src/:
the published model stays bit-reproducible, and the two can be run side by side.

The parsed series is cached to data/shiller_sp500_monthly.csv and committed, so
the new model reproduces without network access. Re-fetch with --refresh.

Usage:
    python new_model/stock_series.py            # report coverage, build cache
    python new_model/stock_series.py --refresh  # re-download from Yale
"""

import argparse
import subprocess
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE / "data" / "shiller_sp500_monthly.csv"
SOURCE_XLS = HERE / "data" / "ie_data.xls"
# Shiller's own page. shillerdata.com 404s on the direct file; the Yale mirror
# is the one that serves it.
SHILLER_URL = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"


def _download(dest):
    """curl, not urllib: this repo's Python SSL path fails behind a
    TLS-inspecting proxy (see truth_data._ssl_context for the same problem),
    and curl uses the OS trust store."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {SHILLER_URL}")
    r = subprocess.run(["curl", "-sSL", "-m", "120", "-o", str(dest), SHILLER_URL],
                       capture_output=True, text=True)
    if r.returncode or not dest.exists() or dest.stat().st_size < 100_000:
        raise SystemExit(
            f"download failed ({r.stderr[:200]}).\n"
            f"Fetch {SHILLER_URL} by hand, save it to {dest}, and re-run.")
    print(f"  {dest.stat().st_size:,} bytes")


def _parse_xls(path):
    """Shiller's 'Data' sheet -> monthly PeriodIndex of the S&P composite price.

    The Date column is a FLOAT encoding YYYY.MM, so October 1871 is stored as
    1871.1, not 1871.10 -- reading the decimals as text makes every October a
    January. Multiplying the fraction by 100 and rounding is what makes the
    month unambiguous."""
    df = pd.read_excel(path, sheet_name="Data", header=7)
    df = df.dropna(subset=["Date", "P"])
    d = df["Date"].astype(float)
    year = d.apply(int)
    month = d.apply(lambda x: int(round((x - int(x)) * 100)))
    bad = sorted(set(month) - set(range(1, 13)))
    if bad:
        raise SystemExit(f"month parse produced {bad}; the sheet layout changed")
    idx = pd.PeriodIndex.from_fields(year=year, month=month, freq="M")
    s = pd.Series(pd.to_numeric(df["P"], errors="coerce").values, index=idx)
    return s.dropna().sort_index()


def build_cache(refresh=False):
    if refresh or not SOURCE_XLS.exists():
        _download(SOURCE_XLS)
    s = _parse_xls(SOURCE_XLS)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    s.rename("sp500").rename_axis("month").to_csv(CACHE)
    print(f"wrote {len(s):,} months -> {CACHE}")
    return s


def load_stocks(refresh=False):
    """Monthly S&P composite as a PeriodIndex Series -- the drop-in replacement
    for `truth_data.load_fred(STOCK_SERIES)`."""
    if refresh or not CACHE.exists():
        return build_cache(refresh=refresh)
    df = pd.read_csv(CACHE)
    return pd.Series(df["sp500"].values,
                     index=pd.PeriodIndex(df["month"], freq="M")).sort_index()


def coverage_report(old=None):
    """Print what each corpus gains. `old` is the series being replaced."""
    s = load_stocks()
    print(f"Shiller S&P : {s.index.min()} -> {s.index.max()}  ({len(s):,} months)")
    if old is not None:
        print(f"replaces    : {old.index.min()} -> {old.index.max()}  "
              f"({len(old):,} months)")
        ov = s.index.intersection(old.index)
        if len(ov) > 24:
            a, b = s.reindex(ov), old.reindex(ov)
            print(f"\noverlap {ov.min()} to {ov.max()} ({len(ov)} months):")
            print(f"  level correlation      {a.corr(b):.4f}")
            print(f"  6m-return correlation  "
                  f"{a.pct_change(6).corr(b.pct_change(6)):.4f}")
            print(f"  12m-return correlation "
                  f"{a.pct_change(12).corr(b.pct_change(12)):.4f}")
    print("\nmonthly coverage by corpus:")
    for name, lo, hi in [("LOC 1900-1963", "1900-01", "1963-12"),
                         ("ProQuest 1965-2009", "1965-01", "2009-12")]:
        rng = pd.period_range(lo, hi, freq="M")
        line = f"  {name:<20} Shiller {s.reindex(rng).notna().mean():6.1%}"
        if old is not None:
            line += f"   (was {old.reindex(rng).notna().mean():5.1%})"
        print(line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="re-download from Yale")
    args = ap.parse_args()
    build_cache(refresh=args.refresh) if args.refresh else load_stocks()
    old = None
    try:
        import sys
        sys.path.insert(0, str(HERE.parent / "src"))
        from truth_data import STOCK_SERIES, load_fred
        old = load_fred(STOCK_SERIES)
    except Exception as e:
        print(f"(could not load the old series for comparison: {e})\n")
    coverage_report(old)
