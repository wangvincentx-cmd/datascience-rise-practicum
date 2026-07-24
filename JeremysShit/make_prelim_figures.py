"""
Preliminary-result figures from the cached crisis corpus (claims_v2_scored.csv).

Four figures, each answering one question a poster viewer will ask:
  1. Does the new extraction actually beat the keyword baseline?   (methods win)
  2. Which crises did forecasters miss worst?                      (hit by episode)
  3. Were the misses optimistic or pessimistic, and did that shift? (asymmetry)
  4. How much of what we extracted can honestly be scored?         (scope gate)

Design follows the dataviz method: form chosen by the data's job, one hue for
magnitude, a diverging pair for polarity, thin marks, direct value labels,
recessive axes. Palette is Okabe-Ito (CVD-safe, validated).

Usage:  python make_prelim_figures.py
Output: prelim_figures/*.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Okabe-Ito, validated CVD-safe.
BLUE, VERM, GREEN, GRAY = "#0072B2", "#D55E00", "#009E73", "#9AA0A6"
INK, MUTED = "#1a1a1a", "#6b6b6b"
OUT = Path("prelim_figures")
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#cccccc", "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": "#ececec", "grid.linewidth": 0.8,
    "font.family": "DejaVu Sans",
})


def _finish(ax, title, subtitle=None):
    # Title above the subtitle above the axes, with enough headroom that a
    # two-line subtitle never rides up into the title.
    n_sub_lines = 1 + (subtitle.count("\n") if subtitle else 0)
    ax.set_title(title, fontsize=13, fontweight="bold", loc="left",
                 pad=16 + 14 * n_sub_lines)
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=9.5,
                color=MUTED, va="bottom")


# --- 1. Extraction: keyword regex vs LLM -----------------------------------
def fig_extraction():
    # measured on the 16-page gold standard (gold_extraction/RESULTS.md)
    regex = {"precision": 0.609, "recall": 0.269}
    llm = {"precision": 0.844, "recall": 0.731}
    metrics = ["recall", "precision"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    y = range(len(metrics))
    h = 0.36
    for i, m in enumerate(metrics):
        ax.barh(i + h/2, llm[m], height=h, color=GREEN, zorder=3)
        ax.barh(i - h/2, regex[m], height=h, color=GRAY, zorder=3)
        ax.text(llm[m] + .01, i + h/2, f"{llm[m]:.0%}", va="center", fontsize=10,
                color=INK, fontweight="bold")
        ax.text(regex[m] + .01, i - h/2, f"{regex[m]:.0%}", va="center",
                fontsize=10, color=MUTED)
    ax.set_yticks(list(y))
    ax.set_yticklabels([m.capitalize() for m in metrics])
    ax.set_xlim(0, 1)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.text(llm["recall"], 1 + h/2 + .28, "LLM (Gemini)", color=GREEN,
            fontsize=10, fontweight="bold", ha="center")
    ax.text(regex["recall"], 1 - h/2 - .34, "keyword regex", color=MUTED,
            fontsize=10, ha="center")
    _finish(ax, "The keyword pipeline finds one forecast in four",
            "Extraction vs. a hand-built gold standard. Recall is the gap that matters: "
            "whole-page\nLLM reading recovers 73% of forecasts, the regex only 27%.")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUT / "fig1_extraction_gap.png", bbox_inches="tight")
    plt.close(fig)


# --- 2. Hit rate by episode -------------------------------------------------
def fig_hit_by_episode(sc):
    e = (sc.groupby("episode")
           .agg(n=("hit", "size"), hit=("hit", "mean"))
           .query("n >= 20").sort_values("hit"))
    fig, ax = plt.subplots(figsize=(7.5, 5))
    colors = [VERM if "1929" in ep else BLUE for ep in e.index]
    ax.barh(range(len(e)), e["hit"], color=colors, zorder=3, height=0.7)
    for i, (ep, row) in enumerate(e.iterrows()):
        ax.text(row["hit"] + .008, i, f"{row['hit']:.0%}", va="center",
                fontsize=9.5, color=INK,
                fontweight="bold" if "1929" in ep else "normal")
    ax.axvline(0.5, color="#bbbbbb", lw=1, ls="--", zorder=2)
    ax.text(0.5, len(e) - .25, "coin flip", fontsize=8, color=MUTED, ha="center")
    ax.set_yticks(range(len(e)))
    ax.set_yticklabels([f"{ep}  (n={int(n)})" for ep, n in
                        zip(e.index, e["n"])], fontsize=9)
    ax.set_xlim(0, 0.75)
    ax.set_xticks([0, .25, .5, .75])
    ax.set_xticklabels(["0", "25%", "50%", "75%"])
    ax.set_xlabel("directional hit rate")
    _finish(ax, "Forecasters missed the 1929 Crash most of all",
            "Share of scorable forecasts that got the direction right, by crisis. "
            "1929: 16% correct.")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_hit_by_episode.png", bbox_inches="tight")
    plt.close(fig)


# --- 3. Optimism asymmetry (the headline) ----------------------------------
def fig_asymmetry(sc):
    # chronological order
    order = ["1907 Panic", "1910 Recession", "1913 Recession", "1920 Depression",
             "1923 Recession", "1926 Recession", "1929 Crash", "1937 Recession",
             "1945 Reconversion", "1948 Recession", "1957 Recession"]
    sc = sc.copy()
    sc["opt_err"] = ((sc.predicted_norm == "improve") & (sc.hit == 0)).astype(int)
    sc["pess_err"] = ((sc.predicted_norm == "worsen") & (sc.hit == 0)).astype(int)
    g = sc.groupby("episode").agg(opt=("opt_err", "sum"), pess=("pess_err", "sum"))
    g = g.reindex([e for e in order if e in g.index])
    fig, ax = plt.subplots(figsize=(8, 5.2))
    y = range(len(g))
    ax.barh(y, g["opt"], color=VERM, zorder=3, height=0.72)
    ax.barh(y, -g["pess"], color=BLUE, zorder=3, height=0.72)
    for i, (ep, row) in enumerate(g.iterrows()):
        if row["opt"]:
            ax.text(row["opt"] + 3, i, int(row["opt"]), va="center", fontsize=9,
                    color=VERM, fontweight="bold")
        if row["pess"]:
            ax.text(-row["pess"] - 3, i, int(row["pess"]), va="center", ha="right",
                    fontsize=9, color=BLUE, fontweight="bold")
    ax.axvline(0, color="#888888", lw=1)
    ax.set_yticks(list(y))
    ax.set_yticklabels(g.index, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("<-- pessimistic errors        errors        optimistic errors -->")
    ax.set_xticks([])
    ax.text(0.98, 0.02, "optimistic error = said 'improve', it didn't",
            transform=ax.transAxes, ha="right", fontsize=8.5, color=VERM)
    ax.text(0.02, 0.02, "pessimistic = said 'worsen', it didn't",
            transform=ax.transAxes, ha="left", fontsize=8.5, color=BLUE)
    _finish(ax, "Optimistic before the war, pessimistic after",
            "Wrong forecasts by type, chronological. Pre-1930 crises are almost all "
            "optimistic\nmisses (1929: 252 to 5). Postwar, the press flips to crying "
            "wolf about downturns.")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_optimism_asymmetry.png", bbox_inches="tight")
    plt.close(fig)


# --- 4. Scope gate ----------------------------------------------------------
def fig_scope(all_claims):
    counts = pd.Series([c.get("scope") for c in all_claims]).value_counts()
    order = ["national", "industry", "regional", "foreign"]
    counts = counts.reindex([o for o in order if o in counts.index])
    total = counts.sum()
    fig, ax = plt.subplots(figsize=(7.5, 1.9))
    left = 0
    cmap = {"national": GREEN, "industry": GRAY, "regional": "#c7ccd1",
            "foreign": "#dfe3e6"}
    for scope, n in counts.items():
        ax.barh(0, n, left=left, color=cmap.get(scope, GRAY), height=0.6,
                edgecolor="white", linewidth=2, zorder=3)
        if n / total > 0.05:
            ax.text(left + n/2, 0, f"{scope}\n{n/total:.0%}", ha="center",
                    va="center", fontsize=9,
                    color="white" if scope == "national" else INK,
                    fontweight="bold" if scope == "national" else "normal")
        left += n
    ax.set_xlim(0, total)
    ax.set_ylim(-.5, .5)
    ax.axis("off")
    _finish(ax, "Only US-national forecasts are scored against US data",
            "The scope field holds out 43% of claims (foreign, regional, single-industry) "
            "that\nshould not be graded against national statistics -- kept as separate strata, "
            "not scored.")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_scope_gate.png", bbox_inches="tight")
    plt.close(fig)


# --- 5. Accuracy over time --------------------------------------------------
def fig_accuracy_over_time(sc):
    sc = sc.copy()
    sc["year"] = pd.to_datetime(sc["date"]).dt.year
    ep = (sc.groupby("episode")
            .agg(n=("hit", "size"), hit=("hit", "mean"),
                 year=("year", "median")).query("n >= 20").sort_values("year"))
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.plot(ep["year"], ep["hit"], color=BLUE, lw=1.5, zorder=2, alpha=.5)
    ax.scatter(ep["year"], ep["hit"], s=ep["n"] * 1.1, color=BLUE, zorder=3,
               edgecolor="white", linewidth=1.5)
    for _, r in ep.iterrows():
        lbl = r.name.replace(" Recession", "").replace(" Depression", "")\
               .replace(" Reconversion", "").replace(" Panic", "").replace(" Crash", "")
        dy = .045 if r["hit"] < 0.5 else -.06
        ax.annotate(f"{lbl}\n{r['hit']:.0%}", (r["year"], r["hit"]),
                    textcoords="offset points", xytext=(0, dy*260),
                    ha="center", fontsize=8, color=MUTED)
    ax.axhline(0.5, color="#bbbbbb", lw=1, ls="--", zorder=1)
    ax.text(1958.5, 0.5, "coin flip", fontsize=8, color=MUTED, va="center")
    ax.set_ylim(0, 0.8)
    ax.set_yticks([0, .25, .5, .75])
    ax.set_yticklabels(["0", "25%", "50%", "75%"])
    ax.set_xlim(1904, 1961)
    ax.set_xlabel("year of crisis")
    ax.set_ylabel("directional hit rate")
    _finish(ax, "Newspaper forecast accuracy across the crises, 1907-1958",
            "Each point is one crisis window (size = scorable claims). No secular "
            "trend -- accuracy\nswings with the crisis, not the era. A clean time "
            "series needs the continuous 1900-1963 corpus.")
    fig.tight_layout()
    fig.savefig(OUT / "fig5_accuracy_over_time.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    sc = pd.read_csv("claims_v2_scored.csv")
    sc = sc[(sc["scorable"] == True) & (sc["hit"].isin([0, 1]))].copy()
    all_claims = [json.loads(l) for l in open("claims_v2.jsonl", encoding="utf-8")]
    fig_extraction()
    fig_hit_by_episode(sc)
    fig_asymmetry(sc)
    fig_scope(all_claims)
    fig_accuracy_over_time(sc)
    print(f"4 figures -> {OUT}/")
    for f in sorted(OUT.glob("*.png")):
        print(f"  {f.name}")
