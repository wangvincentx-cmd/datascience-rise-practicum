"""
Presentation figure for the economy->politics ablation.
Two panels that together show the economy does not help predict bill passage:
  A) held-out PR-AUC is no higher (in fact lower) when macro features are added
  B) even when given the macro features, the model barely relies on them

Reads /tmp/ablation_figdata.json (from _ablation_figdata.py).
Writes figures/fig_macro_ablation.png.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

D = json.load(open("/tmp/ablation_figdata.json"))

# --- design-system parameters (validated palette) ---
STRUCT, ECON = "#2a78d6", "#eb6834"          # blue = structural, orange = economy
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8a86"
SURFACE, GRID = "#fcfcfb", "#e7e7e3"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.linewidth": 1,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
})


def rounded_bar(ax, x, y, w, color, horizontal=False, r=0.018):
    """Approximate a thin bar with a rounded data-end, anchored to baseline."""
    if horizontal:
        ax.add_patch(FancyBboxPatch((0, x - w / 2), max(y, 1e-9), w,
            boxstyle=f"round,pad=0,rounding_size={r*0.4}", mutation_aspect=1,
            fc=color, ec=SURFACE, lw=1.5, clip_on=False))
    else:
        ax.add_patch(FancyBboxPatch((x - w / 2, 0), w, max(y, 1e-9),
            boxstyle=f"round,pad=0,rounding_size={r}", mutation_aspect=0.6,
            fc=color, ec=SURFACE, lw=1.5, clip_on=False))


fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 5.4),
                               gridspec_kw={"width_ratios": [1, 1.15]})
fig.subplots_adjust(left=0.07, right=0.98, top=0.80, bottom=0.13, wspace=0.28)

# ---------- Panel A: PR-AUC with vs without economy ----------
models = [("logistic_regression", "Logistic\nregression"),
          ("gradient_boosting", "Gradient\nboosting")]
gap, bw = 0.28, 0.24
for i, (key, label) in enumerate(models):
    s = D["pr_auc"][key]["structural"]
    m = D["pr_auc"][key]["macro"]
    rounded_bar(axA, i - gap / 2, s, bw, STRUCT)
    rounded_bar(axA, i + gap / 2, m, bw, ECON)
    axA.text(i - gap / 2, s + 0.008, f"{s:.2f}", ha="center", va="bottom",
             fontsize=10.5, color=INK, fontweight="bold")
    axA.text(i + gap / 2, m + 0.008, f"{m:.2f}", ha="center", va="bottom",
             fontsize=10.5, color=INK, fontweight="bold")
    # delta annotation
    axA.annotate(f"{m-s:+.2f}", (i + gap / 2, m), xytext=(i + gap/2 + 0.02, m + 0.05),
                 fontsize=9.5, color=ECON, fontweight="bold")

base = D["pr_auc"]["baseline"]
axA.axhline(base, ls=(0, (4, 3)), lw=1.3, color=MUTED, zorder=1)
axA.text(-0.46, base + 0.012, f"“always-dies” baseline ({base:.02f})",
         ha="left", va="bottom", fontsize=8.5, color=MUTED)

axA.set_xticks(range(len(models)))
axA.set_xticklabels([l for _, l in models])
axA.set_ylim(0, 0.38); axA.set_xlim(-0.5, 1.5)
axA.set_ylabel("PR-AUC on held-out Congresses  (higher = better)", fontsize=10, color=INK2)
for sp in ("top", "right"):
    axA.spines[sp].set_visible(False)
axA.set_axisbelow(True); axA.yaxis.grid(True, color=GRID, lw=1)
axA.set_title("Accuracy does not improve with the economy", fontsize=12,
              color=INK, fontweight="bold", pad=10, loc="left")

# legend (2 series, always present)
from matplotlib.patches import Patch
axA.legend(handles=[Patch(fc=STRUCT, label="Structural / political features only"),
                    Patch(fc=ECON, label="+ 6 economic features")],
           loc="upper left", frameon=False, fontsize=9.5, handlelength=1.1,
           bbox_to_anchor=(0.0, 1.0))

# ---------- Panel B: feature importance ----------
top = D["top_features"][::-1]   # smallest at bottom for barh
names = [t["name"] for t in top]
vals = [t["importance"] * 100 for t in top]
cols = [ECON if t["is_economy"] else STRUCT for t in top]
for i, (v, c) in enumerate(zip(vals, cols)):
    rounded_bar(axB, i, v, 0.62, c, horizontal=True)
    axB.text(v + 0.4, i, f"{v:.0f}%", va="center", ha="left",
             fontsize=9.5, color=INK2)
axB.set_yticks(range(len(names)))
axB.set_yticklabels(names, fontsize=9.5, color=INK)
axB.set_xlim(0, max(vals) * 1.18); axB.set_ylim(-0.7, len(names) - 0.3)
axB.set_xlabel("Share of what the model uses  (%)", fontsize=10, color=INK2)
for sp in ("top", "right", "left"):
    axB.spines[sp].set_visible(False)
axB.tick_params(axis="y", length=0)
axB.set_axisbelow(True); axB.xaxis.grid(True, color=GRID, lw=1)
share = D["macro_importance_share"] * 100
n_econ_shown = sum(1 for t in top if t["is_economy"])
axB.set_title(f"The model barely uses the economy  ({share:.0f}% of total)",
              fontsize=12, color=INK, fontweight="bold", pad=10, loc="left")
axB.text(max(vals) * 1.0, 0.15,
         f"only {n_econ_shown} of 6 economic features\nmake the top 12 — the rest ≈ 0%",
         ha="right", va="bottom", fontsize=8.5, color=ECON, style="italic")

fig.suptitle("Economic conditions don’t help predict whether a bill becomes law",
             x=0.07, y=0.955, ha="left", fontsize=15.5, fontweight="bold", color=INK)
fig.text(0.07, 0.895,
         "U.S. bills, 108th–118th Congress · trained on 108–116, tested on 117–118 "
         "(31,796 bills, 639 became law) · orange = economy, blue = politics/structure",
         ha="left", fontsize=9, color=INK2)

fig.savefig("figures/fig_macro_ablation.png", dpi=200, facecolor=SURFACE)
print("wrote figures/fig_macro_ablation.png")
