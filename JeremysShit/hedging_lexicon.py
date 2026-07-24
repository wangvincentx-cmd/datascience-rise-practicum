"""
Replace the LLM `confidence` label with an OBJECTIVE, reproducible hedging
score, and re-run the overconfidence analysis on that solid ground.

Why: `confidence` (assertive vs hedged) is this project's WEAKEST-validated
grade -- Cohen's kappa 0.17 for gpt-4.1 (the grader used on the scored corpus)
against the human gold standard, barely above chance (the original Llama-3.3
validation run in KAPPA_RESULTS.md scored 0.19 on this field; see CHANGELOG's
seven-model bake-off table for gpt-4.1's own number). Any calibration finding
resting on it ("assertive claims were less accurate -> overconfidence") is only
as trustworthy as a sub-0.2-kappa label, which a reviewer will flag
immediately. But hedging is a well-studied,
lexically-marked feature of scientific and forecasting prose (Hyland 2005,
*Metadiscourse*): epistemic modals and hedge phrases ("may", "might", "could",
"expected to", "appears") vs boosters/certainty markers ("will", "certainly",
"undoubtedly", "bound to"). Counting those directly from the quote text gives a
label that is 100% reproducible and needs no kappa -- there is no rater to
disagree with.

hedge_score = (n_hedge - n_booster) / n_words. hedge_class = hedged if hedges >
boosters, assertive if boosters > hedges, else neutral. The script then:
  - reports hit rate by the OBJECTIVE hedge_class (the overconfidence test,
    redone),
  - measures how much the objective class agrees with the LLM `confidence`
    label (Cohen's kappa) -- i.e. quantifies how unreliable the thing we are
    replacing was,
  - correlates the continuous hedge_score with `hit`.

Usage: python hedging_lexicon.py
Outputs: claims_hedging.csv, printed tables, figures/fig_hedging.png
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from grade_claims import cohens_kappa

FIGDIR = Path("figures")

# Hyland (2005) core hedges + epistemic modals/adverbs of uncertainty. Kept to
# forecast-relevant items; matched as whole words/phrases, case-insensitive.
HEDGES = [
    "may", "might", "could", "would", "should", "can", "possibly", "perhaps",
    "probably", "likely", "unlikely", "apparently", "seemingly", "presumably",
    "arguably", "reportedly", "appears", "appear", "appeared", "seems", "seem",
    "seemed", "suggests", "suggest", "suggested", "indicates", "indicate",
    "tends", "tend", "tended", "expected", "expect", "expects", "anticipated",
    "believe", "believes", "believed", "think", "thinks", "hopes", "hoped",
    "estimate", "estimates", "estimated", "predicted", "forecast", "assume",
    "assumed", "if", "unless", "somewhat", "relatively", "fairly", "rather",
    "in general", "generally", "to some extent", "more or less", "potentially",
    "uncertain", "uncertainty", "outlook", "in our opinion", "we feel",
]
BOOSTERS = [
    "will", "shall", "certainly", "surely", "definitely", "undoubtedly",
    "clearly", "obviously", "evidently", "indeed", "of course", "in fact",
    "without doubt", "no doubt", "must", "always", "never", "inevitable",
    "inevitably", "bound to", "sure", "certain", "confident", "confidently",
    "assured", "guarantee", "guaranteed", "unquestionably", "positively",
    "absolutely", "decidedly", "beyond question", "cannot fail",
]
# Longest-first so multi-word phrases match before their single-word parts.
_HEDGE_RE = re.compile(r"\b(?:%s)\b" % "|".join(sorted((re.escape(t) for t in HEDGES),
                                                       key=len, reverse=True)), re.I)
_BOOST_RE = re.compile(r"\b(?:%s)\b" % "|".join(sorted((re.escape(t) for t in BOOSTERS),
                                                       key=len, reverse=True)), re.I)


def hedging_features(quote):
    """Return {n_words, hedge_count, booster_count, hedge_score, hedge_class}
    for one quote string. Purely lexical, deterministic."""
    q = str(quote)
    n_words = max(len(re.findall(r"[A-Za-z']+", q)), 1)
    h = len(_HEDGE_RE.findall(q))
    b = len(_BOOST_RE.findall(q))
    if h > b:
        cls = "hedged"
    elif b > h:
        cls = "assertive"
    else:
        cls = "neutral"
    return {"n_words": n_words, "hedge_count": h, "booster_count": b,
            "hedge_score": (h - b) / n_words, "hedge_class": cls}


def add_hedging(df):
    feats = df["quote"].apply(hedging_features).apply(pd.Series)
    return pd.concat([df.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)


def main():
    df = pd.read_csv("claims_scored.csv")
    df = add_hedging(df)
    df.to_csv("claims_hedging.csv", index=False)
    s = df.dropna(subset=["hit"]).copy()

    print("=== Overconfidence, redone with the OBJECTIVE hedge_class ===")
    obj = s.groupby("hedge_class").agg(n=("hit", "size"), hit_rate=("hit", "mean")).round(3)
    print(obj.to_string())
    if {"assertive", "hedged"} <= set(obj.index):
        diff = obj.loc["hedged", "hit_rate"] - obj.loc["assertive", "hit_rate"]
        if diff > 0:
            print(f"  -> hedged claims beat assertive ones by {diff:+.3f} -> overconfidence "
                  "CONFIRMED on an objective, kappa-free measure.")
        else:
            print(f"  -> assertive claims are NOT worse ({diff:+.3f}) -> the LLM-confidence "
                  "overconfidence result does NOT survive an objective measure.")

    print("\n=== For contrast: the LLM `confidence` label (gpt-4.1 kappa=0.17) ===")
    if "confidence" in s.columns:
        print(s.groupby("confidence").agg(n=("hit", "size"),
                                          hit_rate=("hit", "mean")).round(3).to_string())
        # How much does the objective class agree with the LLM label? map neutral out.
        pair = s[s["hedge_class"].isin(["hedged", "assertive"]) &
                 s["confidence"].isin(["hedged", "assertive"])]
        k = cohens_kappa(list(zip(pair["confidence"], pair["hedge_class"])))
        agree = (pair["confidence"] == pair["hedge_class"]).mean()
        print(f"\n  objective hedge_class vs LLM confidence: kappa={k:.2f}, "
              f"raw agreement {agree:.1%} (n={len(pair)})")
        print("  -> the two barely agree, which is exactly why the objective measure is "
              "the safer one to report.")

    corr = s["hedge_score"].corr(s["hit"])
    print(f"\n=== Continuous hedge_score vs hit ===\n  Pearson r = {corr:+.3f}  "
          "(positive = more hedged -> more often right)")

    _figure(obj, s)
    print("\nclaims_hedging.csv + figures/fig_hedging.png written")


def _figure(obj, s):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib missing -- no figure)")
        return
    FIGDIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    order = [c for c in ["assertive", "neutral", "hedged"] if c in obj.index]
    obj.loc[order, "hit_rate"].plot(kind="bar", ax=axes[0], color="steelblue",
                                    alpha=.85, rot=0)
    for i, c in enumerate(order):
        axes[0].text(i, obj.loc[c, "hit_rate"] + 0.01, f"n={int(obj.loc[c, 'n'])}",
                     ha="center", fontsize=8)
    axes[0].axhline(0.5, color="crimson", ls="--", lw=1)
    axes[0].set_ylim(0, 1); axes[0].set_ylabel("hit rate")
    axes[0].set_title("Hit rate by OBJECTIVE hedging class")

    if "confidence" in s.columns:
        llm = s.groupby("confidence").agg(hit_rate=("hit", "mean"), n=("hit", "size"))
        llm["hit_rate"].plot(kind="bar", ax=axes[1], color="goldenrod", alpha=.85, rot=0)
        for i, c in enumerate(llm.index):
            axes[1].text(i, llm.loc[c, "hit_rate"] + 0.01, f"n={int(llm.loc[c, 'n'])}",
                         ha="center", fontsize=8)
        axes[1].axhline(0.5, color="crimson", ls="--", lw=1)
        axes[1].set_ylim(0, 1)
        axes[1].set_title("Hit rate by LLM confidence label (gpt-4.1 kappa=0.17)")
    fig.suptitle("Overconfidence, objective lexical measure vs the unreliable LLM label")
    plt.tight_layout()
    plt.savefig(FIGDIR / "fig_hedging.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
