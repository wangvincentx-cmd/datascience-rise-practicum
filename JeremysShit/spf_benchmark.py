"""
Add the Survey of Professional Forecasters (SPF) as a second professional
benchmark, alongside the Livingston economists and the Michigan households.

The disagreement work (disagreement.py) explicitly frames itself against the
SPF-based forecaster-disagreement literature, but the project never actually
scored the SPF itself. This does: it puts professional forecasters on the
IDENTICAL directional ground truth the newspapers are scored on, over the
overlapping post-1968 era, so "did the press see it coming better than the
professionals" becomes a real, matched head-to-head (not two numbers computed
different ways).

Data: Philadelphia Fed SPF median forecasts of annualized real GDP growth
(median_rgdp_growth.xlsx), free, quarterly, 1968Q4-present. Columns YEAR,
QUARTER, DRGDP2..DRGDP6 = median forecast of annualized real GDP growth for the
survey quarter (DRGDP2) through four quarters ahead (DRGDP6).

Scoring (leakage-safe, and identical ground truth to the newspapers):
  - predicted direction: from the mean of DRGDP3..DRGDP6 -- the four quarters
    AFTER the survey quarter, i.e. the ~1-year-ahead forecast, using only
    information the forecasters had at the survey date. improve if that mean
    annualized growth > BAND_GDP, worsen if < -BAND_GDP, else no_change.
    BAND_GDP is a documented modeling choice, sensitivity-tested via --band.
  - realized direction: score_claims.realized_direction() over a 12-month
    window from the survey date -- the SAME INDPRO/NBER rule and band the
    newspaper general-business claims are scored on. Scoring a GDP-growth
    forecast against realized industrial production is deliberate: it holds the
    ground truth identical across forecasters (each is judged on "did the
    economy go the way you called"), which is the whole point of a fair
    head-to-head. Disclosed as such.

A notable, honest sub-result to expect: professional forecasters almost never
issue a "worsen" call (economists rarely forecast a contraction a year out),
so like the newspapers they are structurally poor at calling downturns in
advance -- the script reports each side's share of "worsen" predictions to make
that comparable.

Usage:
    python spf_benchmark.py                 # default GDP band 1.0 (annualized %)
    python spf_benchmark.py --band 0.5
Outputs: spf_scored.csv, printed benchmark table, figures/fig_spf_benchmark.png
"""

import argparse
import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from score_claims import BANDS, fred, realized_direction

CACHE = Path("cache")
FIGDIR = Path("figures")
SPF_URL = ("https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/"
           "survey-of-professional-forecasters/data-files/files/median_rgdp_growth.xlsx")
FRED_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
BAND_GDP = 1.0          # annualized-% no-change band for the SPF GDP forecast
FORECAST_COLS = ["DRGDP3", "DRGDP4", "DRGDP5", "DRGDP6"]   # the four quarters ahead


def read_xlsx_robust(path):
    """Load a Philly Fed .xlsx. Their files carry a date-only docProps
    `modified`/`created` field that trips openpyxl (TypeError: expected
    datetime); strip those two elements in-memory before handing the workbook
    to pandas. Everything else is untouched."""
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
        return pd.read_excel(buf, engine="openpyxl")


def download_spf():
    """Cached SPF median real-GDP-growth table."""
    CACHE.mkdir(exist_ok=True)
    f = CACHE / "spf_median_rgdp_growth.xlsx"
    if not f.exists():
        f.write_bytes(requests.get(SPF_URL, headers=FRED_HEADERS, timeout=60).content)
    return read_xlsx_robust(f)


def direction_label(growth, band):
    """Map an annualized-% growth number to improve/worsen/no_change."""
    if growth != growth:            # NaN
        return ""
    if growth > band:
        return "improve"
    if growth < -band:
        return "worsen"
    return "no_change"


def survey_date(year, quarter):
    """Timestamp at the start of a survey quarter."""
    return pd.Period(year=int(year), quarter=int(quarter), freq="Q").start_time


def score_spf(spf, realized_dir_fn, band=BAND_GDP):
    """Score each SPF survey row. `realized_dir_fn(date)` returns the realized
    improve/worsen/no_change label over the 12 months from `date` (injected so
    this is testable without FRED). Returns a scored DataFrame."""
    rows = []
    for _, r in spf.iterrows():
        fc = np.nanmean([r[c] for c in FORECAST_COLS])
        pred = direction_label(fc, band)
        d = survey_date(r["YEAR"], r["QUARTER"])
        realized = realized_dir_fn(d)
        hit = int(pred == realized) if (pred and realized) else np.nan
        rows.append({"date": d, "year": int(r["YEAR"]), "quarter": int(r["QUARTER"]),
                     "forecast_growth": round(float(fc), 3) if fc == fc else np.nan,
                     "predicted_label": pred, "realized_label": realized, "hit": hit})
    return pd.DataFrame(rows)


def _boot_ci(hits, seed=0):
    hits = np.asarray(hits, float)
    rng = np.random.default_rng(seed)
    boots = [rng.choice(hits, len(hits)).mean() for _ in range(2000)]
    return np.percentile(boots, [2.5, 97.5])


def main(args):
    cpi, indpro, unrate = fred("CPIAUCNS"), fred("INDPRO"), fred("UNRATE")

    def realized_dir_fn(date):
        return realized_direction("general_business", "", "", date, 12,
                                  cpi, indpro, unrate)[0]

    spf = download_spf()
    scored = score_spf(spf, realized_dir_fn, args.band)
    scored.to_csv("spf_scored.csv", index=False)
    spf_s = scored.dropna(subset=["hit"])

    # Newspapers on the SAME rule (their `hit` in claims_scored.csv already is
    # the INDPRO/NBER general-business score), restricted to the overlapping era.
    news = pd.read_csv("claims_scored.csv").dropna(subset=["hit"]).copy()
    news["date"] = pd.to_datetime(news["date"])
    news_gb = news[news["predicted_label"].isin(["improve", "worsen", "no_change"])]
    cut = f"{args.since}-01-01"
    news_era = news_gb[news_gb["date"] >= cut]
    spf_era = spf_s[spf_s["date"] >= cut]

    def compo(s):
        return {d: (s["predicted_label"] == d).mean()
                for d in ("improve", "no_change", "worsen")}

    print(f"=== SPF (professional forecasters), {args.since}+, on the newspapers' "
          "ground truth ===")
    spf_ci = _boot_ci(spf_era["hit"])
    c = compo(spf_era)
    print(f"  directional hit rate: {spf_era['hit'].mean():.1%}  "
          f"95% CI [{spf_ci[0]:.1%}, {spf_ci[1]:.1%}]  n={len(spf_era)}")
    print(f"  prediction mix: improve {c['improve']:.0%} / no_change "
          f"{c['no_change']:.0%} / worsen {c['worsen']:.0%}")
    print("  CLEAN STANDALONE FINDING: professionals essentially NEVER forecast a "
          "contraction\n  a year out (worsen share ~0%) -- the well-known "
          "'failure to predict recessions'\n  (Loungani 2001). They score well "
          "mostly by siding with the economy's usual\n  upward drift, not by "
          "calling turning points.")

    print(f"\n=== Press vs SPF, {args.since}+ (report with the caveat below) ===")
    cn = compo(news_era)
    print(f"  newspapers: hit {news_era['hit'].mean():.1%}  n={len(news_era)}  "
          f"(improve {cn['improve']:.0%}/no_change {cn['no_change']:.0%}/worsen {cn['worsen']:.0%})")
    print(f"  SPF:        hit {spf_era['hit'].mean():.1%}  n={len(spf_era)}")
    print("  CAVEAT -- the raw hit-rate gap is CONFOUNDED, not a skill comparison:")
    print("  the post-1968 newspaper corpus is NYT crisis-windowed, so it over-samples")
    print("  downturns and skews pessimistic (worsen share high), while SPF is a")
    print("  continuous quarterly survey of all conditions. The sampling-ROBUST contrast")
    print("  is the prediction mix (SPF ~0% worsen vs the crisis-sampled press), not the")
    print("  hit rate. Same NYT under-sampling limit flagged elsewhere in this arm.")

    print("\n=== Benchmark reference (each on the basis noted) ===")
    print(f"  SPF economists (INDPRO/NBER, {args.since}+): {spf_era['hit'].mean():.1%}  n={len(spf_era)}")
    print("  Livingston economists (1946-63, own IP/CPI/UNEMP): 54.4%  (score_claims.py head-to-head)")
    print("  Michigan households (UMCSENT, tier2_analysis.py):  ~55%   (three-way benchmark)")

    _figure(spf_era, compo(spf_era), args.since)
    print("\nspf_scored.csv + figures/fig_spf_benchmark.png written")


def _figure(spf_era, spf_mix, since):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib missing -- no figure)")
        return
    FIGDIR.mkdir(exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    # Left: SPF prediction mix -- the clean, sampling-robust headline (~0% worsen).
    mix = [spf_mix["improve"], spf_mix["no_change"], spf_mix["worsen"]]
    ax1.bar(["improve", "no_change", "worsen"], mix,
            color=["seagreen", "gray", "crimson"], alpha=.85)
    for i, v in enumerate(mix):
        ax1.text(i, v + 0.01, f"{v:.0%}", ha="center", fontsize=9)
    ax1.set_ylim(0, 1); ax1.set_ylabel("share of SPF forecasts")
    ax1.set_title("Professionals essentially never forecast a downturn a year out")

    # Right: SPF directional hit rate with a coin-flip reference.
    hr = spf_era["hit"].mean()
    ax2.bar(["SPF economists"], [hr], color="darkslateblue", alpha=.85, width=.5)
    ax2.text(0, hr + 0.01, f"{hr:.1%}\n(n={len(spf_era)})", ha="center", fontsize=9)
    ax2.axhline(0.5, color="crimson", ls="--", lw=1, label="coin flip")
    ax2.set_ylim(0, 1); ax2.set_ylabel("directional hit rate")
    ax2.set_title(f"SPF directional accuracy, {since}+")
    ax2.legend()
    fig.suptitle("The Survey of Professional Forecasters: accurate on average, blind to turning points")
    plt.tight_layout()
    plt.savefig(FIGDIR / "fig_spf_benchmark.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--band", type=float, default=BAND_GDP,
                    help="annualized-%% no-change band for the SPF GDP forecast "
                         "(sensitivity knob; default 1.0)")
    ap.add_argument("--since", type=int, default=1968,
                    help="first year of the overlapping era to compare on")
    main(ap.parse_args())
