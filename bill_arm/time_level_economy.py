"""
Time-level economy -> politics test.

Bill-level ablation found the economy can't predict WHICH bill passes (it's a
constant across bills in a period). This asks the version where the economy
actually varies: does the PASSAGE RATE per period move with the economic climate?

Unit = calendar quarter of introduction (2003-2024). For each quarter:
  passage_rate = fraction of bills introduced that quarter that became law
  + the quarter's economic climate (mean of each macro series)

Honesty controls:
  - raw Pearson/Spearman AND detrended Pearson (both series residualized on a
    linear time trend) — because passage rate and the economy both drift over
    20 years, so a raw correlation can be pure co-trending, not a real link.
  - recession vs non-recession passage-rate comparison (Welch t-test): the
    cleanest, least trend-confounded cut.
  - n is small (~80 quarters, effectively ~11 Congresses of independent
    political regime) and both series autocorrelate, so p-values are optimistic.
    Treat as exploratory.

Writes /tmp/timelevel_figdata.json ; prints the table.
"""
import json
import numpy as np
import pandas as pd
from scipy import stats

MACRO = {"unemployment_rate": "unemployment", "recession_flag": "recession share",
         "gdp_growth_yoy": "GDP growth", "cpi_inflation_yoy": "CPI inflation",
         "consumer_sentiment": "consumer sentiment", "initial_claims": "jobless claims"}

df = pd.read_csv("data/features.csv", low_memory=False)
df["introduced_date"] = pd.to_datetime(df["introduced_date"], errors="coerce")
df = df.dropna(subset=["introduced_date"])
df["became_law"] = df["became_law"].astype(str).str.lower().isin(["true", "1", "1.0"])
df["q"] = df["introduced_date"].dt.to_period("Q")

g = df.groupby("q")
agg = pd.DataFrame({"n_bills": g.size(), "passage_rate": g["became_law"].mean()})
for c in MACRO:
    agg[c] = g[c].mean()
agg = agg[agg["n_bills"] >= 30].dropna().reset_index()      # drop thin quarters
agg["t"] = np.arange(len(agg))                              # linear time index

print(f"{len(agg)} quarters, {int(agg.n_bills.sum())} bills, "
      f"passage rate {agg.passage_rate.min():.1%}-{agg.passage_rate.max():.1%}\n")


def detrend(y):
    b = np.polyfit(agg["t"], y, 1)
    return y - np.polyval(b, agg["t"])


pr_res = detrend(agg["passage_rate"].values)
rows = []
print(f"{'economic indicator':22s}{'raw r':>9s}{'raw p':>8s}{'detrended r':>14s}{'p':>8s}")
print("-" * 61)
for c, pretty in MACRO.items():
    x = agg[c].values
    r_raw, p_raw = stats.pearsonr(x, agg["passage_rate"].values)
    r_det, p_det = stats.pearsonr(detrend(x), pr_res)
    rows.append({"indicator": pretty, "col": c, "r_raw": r_raw, "p_raw": p_raw,
                 "r_detrended": r_det, "p_detrended": p_det})
    print(f"{pretty:22s}{r_raw:+9.2f}{p_raw:8.3f}{r_det:+14.2f}{p_det:8.3f}")

# recession vs non-recession quarters (recession_flag mean > 0.5 => recession qtr)
rec = agg[agg["recession_flag"] > 0.5]["passage_rate"]
exp = agg[agg["recession_flag"] <= 0.5]["passage_rate"]
if len(rec) and len(exp):
    t, p = stats.ttest_ind(rec, exp, equal_var=False)
    print(f"\nrecession quarters (n={len(rec)}): passage {rec.mean():.1%}  |  "
          f"expansion (n={len(exp)}): {exp.mean():.1%}  |  Welch p={p:.3f}")

best = max(rows, key=lambda r: abs(r["r_detrended"]))
json.dump({
    "quarters": [{"q": str(q), "n": int(n), "passage_rate": float(pr),
                  **{c: float(agg.loc[i, c]) for c in MACRO}}
                 for i, (q, n, pr) in enumerate(zip(agg["q"], agg["n_bills"], agg["passage_rate"]))],
    "correlations": rows,
    "best_detrended": best,
    "recession_passage": float(rec.mean()) if len(rec) else None,
    "expansion_passage": float(exp.mean()) if len(exp) else None,
}, open("/tmp/timelevel_figdata.json", "w"), indent=2, default=str)
print("\nwrote /tmp/timelevel_figdata.json")
print("Reading: detrended r near 0 / p>0.05 => no real economy signal beyond shared drift.")
