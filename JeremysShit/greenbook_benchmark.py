"""
Add the Fed Board staff's Greenbook/Tealbook forecasts as a FOURTH professional
benchmark, alongside SPF (spf_benchmark.py), the Livingston economists, and the
Michigan households. Same design as spf_benchmark.py: put a professional
forecaster on the IDENTICAL directional ground truth the newspapers are scored
on, so "did the Fed staff see it coming" becomes a matched head-to-head, not a
number computed a different way.

WHAT THIS IS (and is NOT):
  - It is a BENCHMARK/SCORING source, not a feed for model.py. Greenbook is
    structured numbers, not text -- there is nothing to LLM-extract (this is the
    good news: it skips the noisy step that gave ProQuest 0.46 precision), and it
    has none of the newspaper-claim features model.py learns from (publisher,
    voice, hedging, claim text). It is forecaster #4 on the leaderboard.
  - Coverage is Jan 1966 -> ~Dec 2020: the Fed staff forecasts began in 1966, and
    FOMC materials are released on a strict 5-YEAR confidentiality lag, so the set
    is always ~5 years behind the present and will not reach 2021-2026. This costs
    us nothing -- every post-1963 crisis window (oil_1973 ... gfc_2008, covid_2020)
    sits inside 1966-2020. covid is the showcase: did the Jan/Feb-2020 Tealbooks
    call the pandemic collapse? (Almost certainly not.)

DATA -- one manual download (the file is JS-gated on the Philly Fed site; there is
no stable direct URL to curl):
    1. Open https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/greenbook
    2. Download the "Excel -- Row Format" workbook.
    3. Save it as:  JeremysShit/cache/greenbook_row_format.xlsx
Then:
    python greenbook_benchmark.py --inspect     # FIRST: list sheets + columns, map the real names
    python greenbook_benchmark.py               # score + table + figures/fig_greenbook_benchmark.png

Row format: 16 sheets (sheet 1 = Documentation, then 15 variable sheets). Each
variable sheet has one row per Greenbook date, columns for (at most) the 4 quarters
before the nowcast, the nowcast quarter, and up to 9 quarters ahead. We score the
real-GDP-growth sheet: average the ~1-year-ahead columns (the four quarters after
the nowcast), band to improve/worsen/no_change, and compare to realized INDPRO/NBER
over 12 months -- the same rule and band the newspaper general-business claims use.

The three --sheet/--date-col/--forecast-cols defaults below are BEST GUESSES until
--inspect confirms them against the real file; they are the only thing to finalize.
"""

import argparse
import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests  # noqa: F401  (kept for parity with spf_benchmark; download is manual)

from score_claims import fred, realized_direction
from spf_benchmark import direction_label, _boot_ci

CACHE = Path("cache")
FIGDIR = Path("figures")
GB_FILE = CACHE / "greenbook_row_format.xlsx"
BAND_GDP = 1.0          # annualized-% no-change band, same default as SPF

# --- VERIFY THESE THREE WITH --inspect, then finalize -----------------------
# Real-GDP-growth sheet name; the Greenbook-date column; and the four ~1yr-ahead
# forecast columns (the quarters AFTER the nowcast). Names below are guesses.
SHEET_RGDP = "gRGDP"
DATE_COL = "GBdate"
FORECAST_COLS = ["gRGDPF1", "gRGDPF2", "gRGDPF3", "gRGDPF4"]
# ----------------------------------------------------------------------------


def read_xlsx_robust(path, sheet_name=0):
    """Load a Philly Fed .xlsx sheet. Their files carry a date-only docProps
    field that trips openpyxl; strip it in-memory first. (Generalized from
    spf_benchmark.read_xlsx_robust to take a sheet_name.)"""
    raw = Path(path).read_bytes()
    zin = zipfile.ZipFile(io.BytesIO(raw))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "docProps/core.xml":
                data = re.sub(rb"<dcterms:(created|modified)[^>]*>[^<]*</dcterms:\1>",
                              b"", data)
            zout.writestr(item, data)
    buf.seek(0)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.read_excel(buf, engine="openpyxl", sheet_name=sheet_name)


def inspect():
    """List every sheet, then the columns + first rows of the RGDP sheet, so the
    SHEET_RGDP / DATE_COL / FORECAST_COLS constants can be set to real names."""
    if not GB_FILE.exists():
        raise SystemExit(f"Download the Row-Format workbook to {GB_FILE} first "
                         f"(see this file's docstring).")
    xl = pd.ExcelFile(io.BytesIO(GB_FILE.read_bytes()), engine="openpyxl")
    print("SHEETS:")
    for s in xl.sheet_names:
        print(f"  {s}")
    guess = SHEET_RGDP if SHEET_RGDP in xl.sheet_names else next(
        (s for s in xl.sheet_names if re.search(r"rgdp|gdp", s, re.I)), None)
    if not guess:
        print("\n(no obvious real-GDP sheet -- pick one from the list above)")
        return
    df = read_xlsx_robust(GB_FILE, sheet_name=guess)
    print(f"\nRGDP-ish sheet '{guess}':\n  columns: {list(df.columns)}")
    print(df.head(6).to_string())


def score_greenbook(gb, realized_dir_fn, band=BAND_GDP):
    """Score each Greenbook row on the newspapers' ground truth."""
    rows = []
    for _, r in gb.iterrows():
        fc = np.nanmean([r[c] for c in FORECAST_COLS if c in gb.columns])
        d = pd.to_datetime(r[DATE_COL], errors="coerce")
        if pd.isna(d):
            continue
        pred = direction_label(fc, band)
        realized = realized_dir_fn(d)
        hit = int(pred == realized) if (pred and realized) else np.nan
        rows.append({"date": d, "forecast_growth": round(float(fc), 3) if fc == fc else np.nan,
                     "predicted_label": pred, "realized_label": realized, "hit": hit})
    return pd.DataFrame(rows)


def main(args):
    if not GB_FILE.exists():
        raise SystemExit(f"Download the Row-Format workbook to {GB_FILE} first "
                         f"(see this file's docstring).")
    cpi, indpro, unrate = fred("CPIAUCNS"), fred("INDPRO"), fred("UNRATE")

    def realized_dir_fn(date):
        return realized_direction("general_business", "", "", date, 12,
                                  cpi, indpro, unrate)[0]

    gb = read_xlsx_robust(GB_FILE, sheet_name=SHEET_RGDP)
    scored = score_greenbook(gb, realized_dir_fn, args.band)
    scored.to_csv("greenbook_scored.csv", index=False)
    gb_s = scored.dropna(subset=["hit"])
    era = gb_s[gb_s["date"] >= f"{args.since}-01-01"]

    def compo(s):
        return {d: (s["predicted_label"] == d).mean()
                for d in ("improve", "no_change", "worsen")}

    ci = _boot_ci(era["hit"])
    c = compo(era)
    print(f"=== Greenbook (Fed Board staff), {args.since}+, on the newspapers' "
          "ground truth ===")
    print(f"  directional hit rate: {era['hit'].mean():.1%}  "
          f"95% CI [{ci[0]:.1%}, {ci[1]:.1%}]  n={len(era)}")
    print(f"  prediction mix: improve {c['improve']:.0%} / no_change "
          f"{c['no_change']:.0%} / worsen {c['worsen']:.0%}")
    print("  Expected (like SPF): the Fed staff rarely forecast a contraction a "
          "year out\n  -- the 'failure to predict recessions'. Compare the worsen "
          "share to SPF's ~0%.")

    print("\n=== Benchmark reference ===")
    print(f"  Greenbook (INDPRO/NBER, {args.since}+): {era['hit'].mean():.1%}  n={len(era)}")
    print("  SPF economists (spf_benchmark.py):      54.1%")
    print("  Livingston economists (1946-63):        54.4%")
    print("  Michigan households (tier2):            ~55%")

    _figure(era, compo(era), args.since)
    print("\ngreenbook_scored.csv + figures/fig_greenbook_benchmark.png written")


def _figure(era, mix, since):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib missing -- no figure)")
        return
    FIGDIR.mkdir(exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    vals = [mix["improve"], mix["no_change"], mix["worsen"]]
    ax1.bar(["improve", "no_change", "worsen"], vals,
            color=["seagreen", "gray", "crimson"], alpha=.85)
    for i, v in enumerate(vals):
        ax1.text(i, v + 0.01, f"{v:.0%}", ha="center", fontsize=9)
    ax1.set_ylim(0, 1); ax1.set_ylabel("share of Greenbook forecasts")
    ax1.set_title("Does the Fed staff ever forecast a downturn a year out?")
    hr = era["hit"].mean()
    ax2.bar(["Greenbook"], [hr], color="darkgreen", alpha=.85, width=.5)
    ax2.text(0, hr + 0.01, f"{hr:.1%}\n(n={len(era)})", ha="center", fontsize=9)
    ax2.axhline(0.5, color="crimson", ls="--", lw=1, label="coin flip")
    ax2.set_ylim(0, 1); ax2.set_ylabel("directional hit rate")
    ax2.set_title(f"Greenbook directional accuracy, {since}+")
    ax2.legend()
    fig.suptitle("The Fed's Greenbook: another professional benchmark on the newspapers' ruler")
    plt.tight_layout()
    plt.savefig(FIGDIR / "fig_greenbook_benchmark.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inspect", action="store_true",
                    help="list sheets + RGDP columns so the 3 constants can be mapped")
    ap.add_argument("--band", type=float, default=BAND_GDP,
                    help="annualized-%% no-change band (default 1.0, matches SPF)")
    ap.add_argument("--since", type=int, default=1966,
                    help="first year of the era to report on")
    args = ap.parse_args()
    if args.inspect:
        inspect()
    else:
        main(args)
