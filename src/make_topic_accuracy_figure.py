"""Hit rate by what was forecast, and by who forecast it.

  figures/prelim_figures/fig11_topic_voice_accuracy.png
  data/scored/topic_voice_accuracy.csv   the same numbers, machine-readable

The question this answers: does accuracy depend on the SUBJECT of a forecast, on
the STANDING of whoever made it, or on both? Grouped bars put both dimensions on
one axis -- topic across the page, voice within each group -- so the answer can be
read as a shape rather than argued from a table. It is not close: the groups sit
at very different heights and the bars inside each group do not.

Sorted by topic hit rate, not alphabetically, so the ranking is the reading order.

Intervals are block bootstraps over 3-year eras, never binomial. Claims printed in
one era share wire copy and one macro reality, so the effective sample is ~21
blocks rather than 14,251 claims; a binomial interval on n = 4,391 would be about
five times too narrow and would manufacture significance everywhere.

Voices below ~300 claims (layperson, editorial, reporter, staff writer, unclear)
are dropped rather than drawn thin -- 5 editorial claims cannot carry a bar. They
stay in the CSV.

Run from the repo root:  python src/make_topic_accuracy_figure.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from make_index_figures import BLUE, VERM, GREEN, INK, MUTED, OUT

SCORED = "data/scored/monthly_scored.csv"
CSV = Path("data/scored/topic_voice_accuracy.csv")
# Fixed order, assigned by identity and never cycled. Validated for CVD
# separation against a white surface: worst adjacent pair deutan dE 11.0,
# normal-vision dE 25.8, all three above the 3:1 contrast floor.
VOICES = [("expert", BLUE), ("journalist", VERM), ("official", GREEN)]
MIN_N = 300          # a voice needs this many claims overall to get bars
BLOCK_YEARS = 3
N_BOOT = 4000
LABEL = {"general_business": "general business", "prices": "prices",
         "markets": "markets", "employment": "employment", "other": "other"}


def load():
    d = pd.read_csv(SCORED)
    d = d[(d["scorable"] == True) & (d["hit"].isin([0, 1]))].copy()
    d["year"] = pd.to_datetime(d["date"], errors="coerce").dt.year
    d = d[d["year"].notna()]
    d["block"] = (d["year"].astype(int) // BLOCK_YEARS) * BLOCK_YEARS
    return d


def boot_ci(d, cells):
    """95% CI per (topic, voice) cell, resampling 3-year blocks not claims.

    One resample serves every cell, so the intervals are mutually consistent --
    drawing a block puts all of its claims into every cell at once, which is the
    dependence a per-cell bootstrap would pretend away."""
    idx = {c: i for i, c in enumerate(cells)}
    key = list(zip(d["topic"], d["voice"]))
    d = d.assign(_cell=[idx.get(k, -1) for k in key])
    d = d[d["_cell"] >= 0]

    by_block = [(g["_cell"].to_numpy(), g["hit"].to_numpy(float))
                for _, g in d.groupby("block")]
    rng = np.random.default_rng(0)
    draws = np.empty((N_BOOT, len(cells)))
    for b in range(N_BOOT):
        pick = rng.integers(0, len(by_block), len(by_block))
        c = np.concatenate([by_block[i][0] for i in pick])
        h = np.concatenate([by_block[i][1] for i in pick])
        n = np.bincount(c, minlength=len(cells))
        s = np.bincount(c, weights=h, minlength=len(cells))
        draws[b] = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return (np.nanpercentile(draws, 2.5, axis=0),
            np.nanpercentile(draws, 97.5, axis=0))


def main():
    d = load()
    overall = d["hit"].mean()

    # full table first -- every voice, including the ones too small to draw
    full = (d.groupby(["topic", "voice"])["hit"].agg(n="size", hit_rate="mean")
            .reset_index())
    topics = (d.groupby("topic")["hit"].agg(n="size", hit_rate="mean")
              .sort_values("hit_rate", ascending=False))
    voices = [v for v, _ in VOICES]

    cells = [(t, v) for t in topics.index for v in voices]
    lo, hi = boot_ci(d, cells)
    ci = {c: (lo[i], hi[i]) for i, c in enumerate(cells)}

    full["ci_lo"] = [ci.get((t, v), (np.nan, np.nan))[0]
                     for t, v in zip(full["topic"], full["voice"])]
    full["ci_hi"] = [ci.get((t, v), (np.nan, np.nan))[1]
                     for t, v in zip(full["topic"], full["voice"])]
    full.sort_values(["topic", "n"], ascending=[True, False]).to_csv(CSV, index=False)

    rate = full.set_index(["topic", "voice"])["hit_rate"]
    nn = full.set_index(["topic", "voice"])["n"]

    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    xs = np.arange(len(topics))
    W = 0.26                       # 3 bars + a surface gap inside each group
    for k, (voice, colour) in enumerate(VOICES):
        pos = xs + (k - 1) * (W + 0.012)
        vals = [rate.get((t, voice), np.nan) for t in topics.index]
        los = [ci[(t, voice)][0] for t in topics.index]
        his = [ci[(t, voice)][1] for t in topics.index]
        ax.bar(pos, vals, W, color=colour, edgecolor="white", lw=1.2,
               zorder=3, label=voice)
        ax.vlines(pos, los, his, color=INK, lw=1.1, alpha=.55, zorder=4)
        for x, v, t in zip(pos, vals, topics.index):
            # Values in ink, not the bar colour: the bar carries identity, the
            # text carries the number.
            ax.text(x, v + .012, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=8, color=INK, zorder=5)
            ax.text(x, .012, f"{nn.get((t, voice), 0):,}", ha="center",
                    va="bottom", fontsize=6.5, color="white", zorder=5)

    ax.axhline(overall, color="#999999", lw=1, ls="--", zorder=2)
    ax.text(len(topics) - 0.52, overall + .008,
            f"all claims ({overall:.2f})", fontsize=8, color=MUTED, ha="right")

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{LABEL[t]}\n{topics.loc[t, 'n']:,} claims"
                        for t in topics.index])
    ax.set_ylim(0, 0.72)
    ax.set_ylabel("directional hit rate")
    ax.set_xlabel("")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper right", ncol=3, fontsize=9)
    ax.set_title("What a forecast was about decided whether it came true; "
                 "who said it did not",
                 fontsize=13, fontweight="bold", loc="left", pad=30)
    ax.text(0, 1.015,
            "Share of scorable US-national forecasts that came true, by subject "
            "and by the standing of the speaker. Whiskers are 95% block "
            "bootstraps\nover 3-year eras. Small numbers inside the bars are "
            "claim counts. Voices under "
            f"{MIN_N} claims are omitted; the dashed line is all "
            f"{len(d):,} scorable claims.",
            transform=ax.transAxes, fontsize=9, color=MUTED, va="bottom")

    fig.tight_layout()
    p = OUT / "fig11_topic_voice_accuracy.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)

    print(f"n = {len(d):,} scorable · overall hit {overall:.3f}")
    print(topics.to_string())
    print()
    print(full[full["voice"].isin(voices)]
          .sort_values(["topic", "voice"])
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"-> {p}\n-> {CSV}")


if __name__ == "__main__":
    main()
