"""
Time-level economy->politics figure. Reads /tmp/timelevel_figdata.json.
Panel A: consumer sentiment vs quarterly passage rate (the clearest single link).
Panel B: detrended correlation of each economic indicator with passage rate,
         showing a modest but directionally-consistent signal.
Writes figures/fig_timelevel_economy.png.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = json.load(open("/tmp/timelevel_figdata.json"))

POS, NEG = "#2a78d6", "#eb6834"          # blue = positive r, orange = negative r
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8a86"
SURFACE, GRID = "#fcfcfb", "#e7e7e3"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2})

q = D["quarters"]
sent = np.array([r["consumer_sentiment"] for r in q])
pr = np.array([r["passage_rate"] * 100 for r in q])

fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 5.3),
                               gridspec_kw={"width_ratios": [1, 1.1]})
fig.subplots_adjust(left=0.07, right=0.98, top=0.80, bottom=0.14, wspace=0.30)

# ---- Panel A: scatter + fit ----
axA.scatter(sent, pr, s=34, color=POS, alpha=0.65, edgecolor=SURFACE, linewidth=0.6, zorder=3)
b = np.polyfit(sent, pr, 1)
xs = np.linspace(sent.min(), sent.max(), 50)
axA.plot(xs, np.polyval(b, xs), color=INK, lw=2, zorder=4)
sent_corr = next(c for c in D["correlations"] if c["col"] == "consumer_sentiment")
axA.text(0.04, 0.94, f"raw r = {sent_corr['r_raw']:+.2f}\ndetrended r = {sent_corr['r_detrended']:+.2f}",
         transform=axA.transAxes, va="top", fontsize=10, color=INK,
         bbox=dict(boxstyle="round,pad=0.4", fc=SURFACE, ec=GRID))
axA.set_xlabel("Consumer sentiment (quarter mean)", fontsize=10, color=INK2)
axA.set_ylabel("Bills passed that quarter  (%)", fontsize=10, color=INK2)
for sp in ("top", "right"):
    axA.spines[sp].set_visible(False)
axA.set_axisbelow(True); axA.grid(True, color=GRID, lw=1)
axA.set_title("Better economy → modestly more bills pass", fontsize=12,
              color=INK, fontweight="bold", pad=10, loc="left")

# ---- Panel B: detrended correlations ----
cors = sorted(D["correlations"], key=lambda c: c["r_detrended"])
labels = [c["indicator"] for c in cors]
rvals = [c["r_detrended"] for c in cors]
cols = [POS if v >= 0 else NEG for v in rvals]
ys = range(len(cors))
axB.axvline(0, color=MUTED, lw=1.2, zorder=1)
axB.barh(list(ys), rvals, color=cols, height=0.62, edgecolor=SURFACE, linewidth=1.5, zorder=2)
for i, c in enumerate(cors):
    star = "*" if c["p_detrended"] < 0.05 else ""
    off = 0.012 if c["r_detrended"] >= 0 else -0.012
    axB.text(c["r_detrended"] + off, i, f"{c['r_detrended']:+.2f}{star}",
             va="center", ha="left" if c["r_detrended"] >= 0 else "right",
             fontsize=9.5, color=INK2)
axB.set_yticks(list(ys)); axB.set_yticklabels(labels, fontsize=10, color=INK)
axB.set_xlim(-0.45, 0.45)
axB.set_xlabel("Correlation with passage rate  (detrended, * = p<0.05)", fontsize=10, color=INK2)
for sp in ("top", "right", "left"):
    axB.spines[sp].set_visible(False)
axB.tick_params(axis="y", length=0)
axB.set_axisbelow(True); axB.xaxis.grid(True, color=GRID, lw=1)
axB.set_title("Modest, directionally-consistent signal", fontsize=12,
              color=INK, fontweight="bold", pad=10, loc="left")

recp = D["recession_passage"] * 100; expp = D["expansion_passage"] * 100
fig.suptitle("At the TIME level a soft economy signal appears — unlike the bill level",
             x=0.07, y=0.955, ha="left", fontsize=15, fontweight="bold", color=INK)
fig.text(0.07, 0.895,
         f"88 quarters, 2003–2024, 128,777 bills · but the cleanest cut is null: "
         f"recession quarters {recp:.1f}% vs expansion {expp:.1f}% passage (p=0.61) · "
         f"exploratory (autocorrelated, ~11 political regimes)",
         ha="left", fontsize=8.6, color=INK2)

fig.savefig("figures/fig_timelevel_economy.png", dpi=200, facecolor=SURFACE)
print("wrote figures/fig_timelevel_economy.png")
