"""
Did a newspaper's POLITICS colour its economic forecasts? Two questions:

  A. By lean: were partisan papers (Republican / Democratic / Socialist-left)
     systematically more optimistic or more accurate about the economy than
     independent papers?
  B. Partisan alignment (the sharper, literature-connected test): did a paper
     predict a ROSIER economy when its OWN party held the White House? This is
     the classic partisan-perceptual-bias result (partisans see a better
     economy under their own president; Bartels 2002, Gerber & Huber 2010),
     tested here on a century of newspaper forecasts instead of modern surveys.

Data already on disk: publisher_metadata.csv (hand-researched political lean of
the top-30 publishers) + data/political_climate.csv (president's party by year).

IMPORTANT DATA-QUALITY NOTE found while building this: the publisher names in
claims_scored.csv carry a location/date suffix ("... (seward, alaska)
1905-1914") that publisher_metadata.csv's clean names lack, so model.py's
EXACT-string join matched only 258 of 1,628 predictions -- its `political_lean`
feature was "unknown" for 84% of rows, which partly explains why the political
features read as null there (mostly MISSING, not merely weak). This script
strips the suffix before joining (`short_name`), recovering ~970 leaned claims
(Independent 633, Democratic 105, left 72, Republican 51). That is a real join
improvement, not a leakage shortcut -- lean is a fixed property of the
publisher, known at all times.

HONEST LIMITS, stated up front: partisan papers are thin and era-clustered.
The big publishers (NYT, Evening Star) are "Independent", so that bucket is
~2 papers. Each small partisan paper's claims fall in a narrow date range, so
"aligned vs opposed" partly encodes WHICH ERA a paper wrote in, not just its
politics -- read Question B as suggestive/exploratory, not a clean causal
partisan effect. Reported that way regardless of which direction it points.

Usage: python partisan_analysis.py
Outputs: partisan_by_lean.csv, printed tables, figures/fig_partisan.png
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

FIGDIR = Path("figures")
DIRECTIONS = ("improve", "worsen")
LEAN_ORDER = ["left", "democratic", "independent", "republican", "unknown"]


def short_name(publisher):
    """Publisher name with the '(location...) dates' suffix stripped, lowercased."""
    s = str(publisher).lower().strip()
    return re.split(r"\s*\(", s)[0].strip()


def simplify_lean(raw):
    """Collapse the metadata's lean labels into analysis buckets."""
    s = str(raw).lower().strip()
    if s in ("socialist", "labor/left", "left"):
        return "left"
    if s == "republican":
        return "republican"
    if s == "democratic":
        return "democratic"
    if s == "independent":
        return "independent"
    return "unknown"          # "UNKNOWN", "None stated", "", nan


def president_party_by_year(climate):
    """year -> president's party ('R'/'D'/...) from political_climate.csv."""
    out = {}
    for _, r in climate.iterrows():
        for y in range(int(r["start_year"]), int(r["end_year"]) + 1):
            out[y] = str(r["president_party"]).strip().upper()
    return out


def alignment(lean_bucket, president_party):
    """'aligned' / 'opposed' / None. Only defined for republican/democratic
    papers (independent/left/unknown have no single party to align)."""
    party = {"republican": "R", "democratic": "D"}.get(lean_bucket)
    if party is None or president_party not in ("R", "D"):
        return None
    return "aligned" if party == president_party else "opposed"


def _net_optimism(sub):
    d = sub[sub["predicted_label"].isin(DIRECTIONS)]
    n_i = int((d["predicted_label"] == "improve").sum())
    n_w = int((d["predicted_label"] == "worsen").sum())
    tot = n_i + n_w
    return (n_i - n_w) / tot if tot else np.nan, n_i, n_w


def load_lean_lookup():
    pm = pd.read_csv(Path(__file__).parent / "publisher_metadata.csv")
    pm["short"] = pm["publisher"].apply(short_name)
    pm["lean_bucket"] = pm["political_lean"].apply(simplify_lean)
    return pm.set_index("short")["lean_bucket"]


def main():
    df = pd.read_csv("claims_scored.csv")
    lean = load_lean_lookup()
    df["short"] = df["publisher"].apply(short_name)
    df["lean"] = df["short"].map(lean).fillna("unknown")

    matched = (df["lean"] != "unknown").sum()
    print(f"Publisher-lean join: {matched}/{len(df)} predictions matched a leaned "
          "top-30 publisher\n(vs 258 under model.py's exact-string join -- see the "
          "module docstring).")

    # ---- Question A: by lean ----
    print("\n=== A. Optimism and accuracy by publisher lean ===")
    print("net optimism: +1 all-improve, -1 all-worsen. Independent = ~NYT + Evening Star.")
    rows = []
    for lb in LEAN_ORDER:
        sub = df[df["lean"] == lb]
        if not len(sub):
            continue
        net, n_i, n_w = _net_optimism(sub)
        scor = sub.dropna(subset=["hit"])
        rows.append({"lean": lb, "n_pred": len(sub), "n_improve": n_i, "n_worsen": n_w,
                     "net_optimism": round(net, 3) if net == net else np.nan,
                     "n_scorable": len(scor),
                     "hit_rate": round(scor["hit"].mean(), 3) if len(scor) else np.nan})
    by_lean = pd.DataFrame(rows).set_index("lean")
    by_lean.to_csv("partisan_by_lean.csv")
    print(by_lean.to_string())

    # ---- Question B: partisan alignment ----
    print("\n=== B. Partisan alignment: rosier under your OWN party's president? ===")
    climate = pd.read_csv(Path(__file__).parent / "data" / "political_climate.csv")
    pby = president_party_by_year(climate)
    part = df[df["lean"].isin(["republican", "democratic"])].copy()
    part["date"] = pd.to_datetime(part["date"])
    part["pres_party"] = part["date"].dt.year.map(pby)
    part["alignment"] = [alignment(l, p) for l, p in zip(part["lean"], part["pres_party"])]
    part = part[part["alignment"].notna()]

    tbl = []
    for al in ("aligned", "opposed"):
        sub = part[part["alignment"] == al]
        net, n_i, n_w = _net_optimism(sub)
        scor = sub.dropna(subset=["hit"])
        tbl.append({"alignment": al, "n_pred": len(sub), "n_improve": n_i, "n_worsen": n_w,
                    "share_improve": round(n_i / (n_i + n_w), 3) if (n_i + n_w) else np.nan,
                    "net_optimism": round(net, 3) if net == net else np.nan,
                    "hit_rate": round(scor["hit"].mean(), 3) if len(scor) else np.nan})
    align_tbl = pd.DataFrame(tbl).set_index("alignment")
    print(align_tbl.to_string())

    if {"aligned", "opposed"} <= set(align_tbl.index):
        ai, aw = int(align_tbl.loc["aligned", "n_improve"]), int(align_tbl.loc["aligned", "n_worsen"])
        oi, ow = int(align_tbl.loc["opposed", "n_improve"]), int(align_tbl.loc["opposed", "n_worsen"])
        try:
            from scipy.stats import fisher_exact
            _, p = fisher_exact([[ai, aw], [oi, ow]])
            print(f"\n  Fisher exact on improve-vs-worsen (aligned vs opposed): p={p:.3f}")
        except Exception as e:
            print(f"  (Fisher test unavailable: {e})")
        diff = align_tbl.loc["aligned", "net_optimism"] - align_tbl.loc["opposed", "net_optimism"]
        print(f"  aligned minus opposed net optimism: {diff:+.3f}  "
              f"({'rosier under own party' if diff > 0 else 'not rosier'})")
    print("\n  CAVEAT: partisan n is small and era-clustered -- each small paper's "
          "claims\n  fall in a narrow window, so alignment partly encodes ERA, not "
          "just politics.\n  Read as exploratory; a clean test needs more partisan "
          "papers per era.")

    _figure(by_lean, align_tbl if {"aligned", "opposed"} <= set(align_tbl.index) else None)
    print("\npartisan_by_lean.csv + figures/fig_partisan.png written")


def _figure(by_lean, align_tbl):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib missing -- no figure)")
        return
    FIGDIR.mkdir(exist_ok=True)
    n = 2 if align_tbl is not None else 1
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.6))
    axes = np.atleast_1d(axes)

    d = by_lean.dropna(subset=["net_optimism"])
    colors = {"left": "firebrick", "democratic": "royalblue", "independent": "gray",
              "republican": "crimson", "unknown": "lightgray"}
    axes[0].bar(d.index, d["net_optimism"], color=[colors.get(i, "gray") for i in d.index], alpha=.85)
    for i, (lb, r) in enumerate(d.iterrows()):
        axes[0].text(i, r["net_optimism"] + 0.02, f"n={int(r['n_pred'])}", ha="center", fontsize=8)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_ylabel("net optimism (+improve / -worsen)")
    axes[0].set_title("Economic optimism by publisher lean")
    axes[0].tick_params(axis="x", rotation=20)

    if align_tbl is not None:
        axes[1].bar(align_tbl.index, align_tbl["net_optimism"],
                    color=["seagreen", "indianred"], alpha=.85)
        for i, al in enumerate(align_tbl.index):
            axes[1].text(i, align_tbl.loc[al, "net_optimism"] + 0.02,
                         f"n={int(align_tbl.loc[al, 'n_pred'])}", ha="center", fontsize=8)
        axes[1].axhline(0, color="black", lw=0.8)
        axes[1].set_title("Partisan papers: own party in power vs not")
        axes[1].set_ylabel("net optimism")
    fig.suptitle("Did politics colour economic forecasts? (partisan samples are thin -- exploratory)")
    plt.tight_layout()
    plt.savefig(FIGDIR / "fig_partisan.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
