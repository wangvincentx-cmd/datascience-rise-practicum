"""
The ladder bar chart, refitted with an L1 (lasso) penalty instead of L2.

Same rungs, same features, same leave-one-3-year-period-out CV as figK/figQ --
the ONLY thing that changes is the penalty on the logistic regression's
coefficients: L2 (ridge, shrinks every weight) becomes L1 (lasso, zeroes most
of them). The gradient-boosting rung carries no such penalty and is refitted
unchanged, so it is the same estimator in both versions and stays comparable.

The dotted line is a block-permutation null: `hit` shuffled WITHIN each 3-year
period and the full model refitted, which is what "this model has no signal"
actually looks like under this CV. It sits below 0.5 rather than on it because
leave-one-period-out drops a whole macro regime per fold.

AUCs are cached in data/models/ladder_l1.json so the figure can be restyled
without a 30-minute refit. Delete that file (or pass --refit) to recompute.

Usage (from the repo root):
    python src/make_l1_ladder_figure.py [--perm 40] [--refit]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hit_predictor as HP
from macro_context import FACTORS
from model_variants import clf_for

BLUE, RED, GRAY = "#0072B2", "#C0392B", "#9AA0A6"
INK, MUTED = "#1a1a1a", "#6b6b6b"
OUT = Path("figures/prelim_figures"); OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path("data/models/ladder_l1.json")
C_L1 = 0.5   # matches the L2 model's C=0.5, so only the penalty differs

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.size": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#cccccc", "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": "#f0f0f0", "font.family": "DejaVu Sans"})

LABEL = {"2. claim features only": "How the forecast is written",
         "3. economy only": "The economy at print time",
         "4. claim + economy (additive)": "Both, judged separately",
         "6. gradient boosting (same inputs)": "Gradient boosting, same inputs",
         "5. + direction x economy": "Both, with economy x direction labels"}


def build():
    """X / y / groups exactly as hit_predictor.run() builds them."""
    d = pd.read_csv("data/scored/macro_context.csv", low_memory=False)
    d = d[d["hit"].isin([0, 1])].reset_index(drop=True)
    y = d["hit"].astype(int).values
    year = pd.to_datetime(d["date"]).dt.year
    groups = ((year // HP.BLOCK_YEARS) * HP.BLOCK_YEARS).values
    X = pd.concat([HP.claim_features(d), HP.macro_block(d[FACTORS]),
                   HP.interaction_block(d, d[FACTORS])], axis=1)
    return X, y, groups


def compute(perm):
    X, y, groups = build()
    n_blocks = len(set(groups))
    print(f"forecasts {len(y):,}   blocks {n_blocks}   features {X.shape[1]}")
    l1 = lambda: clf_for("l1", C_L1)

    rungs = {}
    full_cat = full_num = None
    for name, cat, num in HP.LADDER:
        if not cat and not num:
            continue
        t0 = time.time()
        oof = HP.oof_predict(X, y, groups, cat, num, clf=l1())
        auc = HP.metrics(y, oof)["auc"]
        rungs[name] = auc
        print(f"  {name:<34}{auc:>8.3f}   ({time.time() - t0:.0f}s)")
        full_cat, full_num = cat, num

    # Same boosted rung as the L2 figure: no penalty to swap, so it is the
    # constant against which the penalty change is read.
    from sklearn.ensemble import HistGradientBoostingClassifier
    oof_gb = HP.oof_predict(X, y, groups, full_cat, full_num,
                            clf=HistGradientBoostingClassifier(
                                max_depth=3, max_iter=200, learning_rate=.06,
                                random_state=HP.SEED))
    key_gb = "6. gradient boosting (same inputs)"
    rungs[key_gb] = HP.metrics(y, oof_gb)["auc"]
    print(f"  {key_gb:<34}{rungs[key_gb]:>8.3f}")

    null_mean = None
    if perm:
        rng = np.random.default_rng(HP.SEED)
        null = []
        for i in range(perm):
            yp = y.copy()
            for g in set(groups):
                idx = np.where(groups == g)[0]
                yp[idx] = rng.permutation(yp[idx])
            m = HP.metrics(yp, HP.oof_predict(X, yp, groups, full_cat,
                                              full_num, clf=l1()))
            if m:
                null.append(m["auc"])
            print(f"  perm {i + 1}/{perm}: {null[-1]:.3f}", flush=True)
        null_mean = float(np.mean(null))
        best = max(rungs.values())
        p = (1 + sum(a >= best for a in null)) / (1 + len(null))
        print(f"  permutation null mean {null_mean:.3f}   p = {p:.3f}")

    res = {"penalty": "l1", "C": C_L1, "n": int(len(y)), "n_blocks": n_blocks,
           "ladder": rungs, "perm_null": null_mean, "perm_reps": perm}
    CACHE.write_text(json.dumps(res, indent=2))
    print(f"-> {CACHE}")
    return res


def draw(res):
    # The two "Both" rungs sit adjacent so the one comparison this figure
    # exists to make is read off neighbouring bars; boosting drops to the
    # bottom because it is the estimator control, not a rung of the ladder.
    order = ["5. + direction x economy", "4. claim + economy (additive)",
             "2. claim features only", "3. economy only",
             "6. gradient boosting (same inputs)"]
    rows = [(LABEL[k], res["ladder"][k]) for k in order if k in res["ladder"]]

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    y = np.arange(len(rows))[::-1]
    pos = {}
    for (name, auc), yy in zip(rows, y):
        top = name == LABEL["5. + direction x economy"]
        ax.barh(yy, auc - 0.40, left=0.40, height=.62, zorder=3,
                color=RED if top else BLUE)
        ax.text(auc + 0.004, yy, f"{auc:.3f}", va="center", fontsize=13.5,
                fontweight="bold", color=INK)
        pos[name] = (yy, auc)

    # What the interaction terms are worth: a drop from the red bar's tip to
    # the additive bar's level, with the gap the additive model never closes.
    hi, lo = LABEL["5. + direction x economy"], LABEL["4. claim + economy (additive)"]
    if hi in pos and lo in pos:
        (y_hi, a_hi), (y_lo, a_lo) = pos[hi], pos[lo]
        edge = y_lo + .34   # top edge of the lower bar; clears its value label
        ax.plot([a_lo, a_hi], [edge, edge], color=GRAY, lw=1.0, ls=":",
                zorder=4)
        ax.annotate("", xy=(a_hi, edge), xytext=(a_hi, y_hi - .34),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4,
                                    shrinkA=0, shrinkB=0), zorder=5)
        ax.text(a_hi - .008, (y_hi - .34 + edge) / 2, f"-{a_hi - a_lo:.3f}",
                fontsize=12, fontweight="bold", color=INK, ha="right",
                va="center", zorder=5)

    ax.axvline(.5, color="#444444", lw=1.4, ls="--", zorder=4)
    ax.text(.503, len(rows) - .55, "chance", fontsize=11.5, color=INK,
            ha="left", va="bottom")
    null = res.get("perm_null")
    if null:
        ax.axvline(null, color="#aaaaaa", lw=1.2, ls=":", zorder=4)
        ax.text(null - .003, len(rows) - .55, f"permutation\nnull  {null:.3f}",
                fontsize=10.5, color=MUTED, ha="right", va="bottom")

    ax.set_yticks(y); ax.set_yticklabels([n for n, _ in rows], fontsize=13)
    ax.set_xlim(.40, .70); ax.set_ylim(-.6, len(rows) - .1)
    ax.set_xticks([.40, .45, .50, .55, .60, .65, .70])
    ax.set_xticklabels(["0.40", "0.45", "0.50", "0.55", "0.60", "0.65", "0.70"],
                       fontsize=11.5)
    # Two lines: at this width the single-line version runs off the canvas.
    ax.set_xlabel("out-of-fold ROC-AUC\n(leave-one-3-year-period-out, "
                  f"{res['n_blocks']} blocks)", fontsize=11.5)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    p = OUT / "figQ2_ladder_l1.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--perm", type=int, default=0,
                    help="block-permutation shuffles for the null line")
    ap.add_argument("--refit", action="store_true",
                    help="recompute even if the cache exists")
    a = ap.parse_args()
    if CACHE.exists() and not a.refit:
        draw(json.loads(CACHE.read_text()))
    else:
        draw(compute(a.perm))
