"""
Add the Fed Board staff's Greenbook/Tealbook forecasts as a FOURTH professional
benchmark, alongside SPF (spf_benchmark.py), Livingston, and Michigan. Same
design as spf_benchmark.py: put a professional forecaster on the IDENTICAL
directional ground truth the newspapers are scored on, so "did the Fed staff see
it coming" is a matched head-to-head, not a number computed a different way.

This is a BENCHMARK/SCORING source, not a feed for model.py (Greenbook is
structured numbers, no text to extract, and none of the newspaper-claim features).
It is also the base table the forecast-credibility model builds on
(forecast_credibility_PLAN.md).

DATA: the "All Column Format" folder (per-variable xlsx; each COLUMN is a Greenbook
edition `gRGDP_YYYYMMDD`, each ROW a target quarter `YYYY.Q`, cell = that edition's
forecast). Downloaded to cache/Gbweb_All_Column_Format/. Real GDP growth = gRGDP,
split into two era files (1967_1984, 1985_Last); 490 editions, 1967-2020.

Scoring (leakage-safe, identical to spf_benchmark):
  - predicted direction: mean of the four quarters AFTER the edition's nowcast
    quarter (the ~1-year-ahead call), banded improve/worsen/no_change at BAND_GDP.
  - realized direction: score_claims.realized_direction over 12 months from the
    edition date -- the SAME INDPRO/NBER rule/band the newspapers use.

Usage:
    python greenbook_benchmark.py                 # gRGDP, band 1.0
    python greenbook_benchmark.py --var gIP       # score a different variable
Outputs: greenbook_scored.csv, printed benchmark table, figures/fig_greenbook_benchmark.png
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from score_claims import fred, realized_direction
from spf_benchmark import read_xlsx_robust, direction_label, _boot_ci

GB_DIR = Path("cache/Gbweb_All_Column_Format")
FIGDIR = Path("figures")
BAND_GDP = 1.0          # annualized-% no-change band, same default as SPF
ERA_FILES = {"1967_1984": "{var}_1967_1984.xlsx", "1985_Last": "{var}_1985_Last.xlsx"}


def _target_qindex(label):
    """'1966.1'/1966.1 -> quarter index (year*4 + q-1). Q is the first decimal."""
    s = str(label).strip()
    if "." not in s:
        return None
    y, q = s.split(".")
    q = int(q[0])                       # 1966.10 shouldn't happen, but be safe
    if not (1 <= q <= 4):
        return None
    return int(y) * 4 + (q - 1)


def _edition_date(col):
    m = re.search(r"_(\d{8})$", str(col))
    return pd.to_datetime(m.group(1), format="%Y%m%d") if m else None


def load_editions(var="gRGDP"):
    """One (edition_date, forecast_growth_1yr, source_file) per Greenbook edition:
    the mean of the four quarters AFTER the edition's nowcast quarter."""
    rows = []
    for fname in (ERA_FILES[k].format(var=var) for k in ERA_FILES):
        path = GB_DIR / fname
        if not path.exists():
            continue
        df = read_xlsx_robust(path)
        df = df.rename(columns={df.columns[0]: "target"})
        df["qi"] = df["target"].map(_target_qindex)
        ed_cols = [c for c in df.columns if str(c).startswith(f"{var}_")]
        for c in ed_cols:
            d = _edition_date(c)
            if d is None:
                continue
            eq = d.year * 4 + (d.month - 1) // 3          # nowcast quarter index
            window = df[(df["qi"] > eq) & (df["qi"] <= eq + 4)][c]
            fc = np.nanmean(pd.to_numeric(window, errors="coerce")) if len(window) else np.nan
            rows.append({"forecast_date": d, "forecast_growth": fc,
                         "source_file": fname})
    return pd.DataFrame(rows).sort_values("forecast_date").reset_index(drop=True)


def score(ed, realized_dir_fn, band=BAND_GDP, var="gRGDP"):
    out = ed.copy()
    out["pred_direction"] = out["forecast_growth"].map(lambda g: direction_label(g, band))
    out["realized_direction"] = out["forecast_date"].map(realized_dir_fn)
    out["hit"] = [int(p == r) if (p and r) else np.nan
                  for p, r in zip(out["pred_direction"], out["realized_direction"])]
    # provenance / harmonized columns (forecast_credibility_PLAN.md schema)
    out["source"] = "greenbook"
    out["forecaster_id"] = "fed_staff"
    out["variable_native"] = var
    out["variable_canonical"] = {"gRGDP": "real_gdp", "gIP": "industrial_production",
                                 "UNEMP": "unemployment"}.get(var, var)
    out["horizon_native"] = "+1..+4Q"
    return out


def main(args):
    cpi, indpro, unrate = fred("CPIAUCNS"), fred("INDPRO"), fred("UNRATE")

    def realized_dir_fn(date):
        return realized_direction("general_business", "", "", date, 12,
                                  cpi, indpro, unrate)[0]

    ed = load_editions(args.var)
    if ed.empty:
        raise SystemExit(f"No editions for {args.var} in {GB_DIR} -- check the folder.")
    scored = score(ed, realized_dir_fn, args.band, args.var)
    scored.to_csv("greenbook_scored.csv", index=False)
    s = scored.dropna(subset=["hit"])
    era = s[s["forecast_date"] >= f"{args.since}-01-01"]

    def compo(x):
        return {d: (x["pred_direction"] == d).mean()
                for d in ("improve", "no_change", "worsen")}

    ci, c = _boot_ci(era["hit"]), compo(era)
    print(f"=== Greenbook (Fed Board staff, {args.var}), {args.since}+, "
          "newspapers' ground truth ===")
    print(f"  editions scored: {len(era)}  (of {len(scored)} total, "
          f"{scored['forecast_date'].dt.year.min()}-{scored['forecast_date'].dt.year.max()})")
    print(f"  directional hit rate: {era['hit'].mean():.1%}  "
          f"95% CI [{ci[0]:.1%}, {ci[1]:.1%}]")
    print(f"  prediction mix: improve {c['improve']:.0%} / no_change "
          f"{c['no_change']:.0%} / worsen {c['worsen']:.0%}")
    print("  (expect, like SPF: the Fed staff rarely call a contraction a year "
          "out -- worsen share near 0)")
    print("\n=== Benchmark leaderboard ===")
    print(f"  Greenbook (INDPRO/NBER, {args.since}+): {era['hit'].mean():.1%}  n={len(era)}")
    print("  SPF economists:            54.1%")
    print("  Livingston economists:     54.4%")
    print("  Michigan households:      ~55%")
    _figure(era, compo(era), args.since, args.var)
    print("\ngreenbook_scored.csv + figures/fig_greenbook_benchmark.png written")


def _figure(era, mix, since, var):
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
    ax2.set_title(f"Greenbook accuracy, {since}+")
    ax2.legend()
    fig.suptitle("The Fed's Greenbook on the newspapers' ruler")
    plt.tight_layout()
    plt.savefig(FIGDIR / "fig_greenbook_benchmark.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--var", default="gRGDP", help="Greenbook variable code (gRGDP, gIP, UNEMP, ...)")
    ap.add_argument("--band", type=float, default=BAND_GDP,
                    help="annualized-%% no-change band (default 1.0, matches SPF)")
    ap.add_argument("--since", type=int, default=1967, help="first year to report on")
    main(ap.parse_args())
