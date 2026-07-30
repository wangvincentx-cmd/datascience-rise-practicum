"""Schematic of the hit model: three feature blocks -> z -> sigmoid -> P(hit).

Not a chart of data -- a picture of the specification that poster_models/m3_dml.py
and src/model_hit.py estimate. It exists because the poster's central claim is
about WHICH BLOCK of features carries the signal, and that argument is much
easier to follow with the three blocks drawn side by side than with a formula:

    P(hit) = sigma( b_news' x_news  +  b_econ' x_econ  +  b_int' (x_econ * dir) )

The third block is the one people miss. docs/RESULTS_MACRO.md found that the
pooled macro effect is a CANCELLATION -- optimists and pessimists are right
under opposite conditions -- so the economy only enters usefully once it is
multiplied by the forecast's own direction. Drawing it as its own box is the
point of the figure.

Feature names are the real ones: claim block from poster_models/_common.py
claim_design, macro block from src/model_hit.py macro_features, interaction
block from m3_dml.interaction_design.

    python src/make_model_diagram.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = "figures/poster_v2"

# House palette (see src/make_poster_v2_figures.py; validated in
# references/palette.md). Blue and orange are the categorical pair; red is the
# diverging warm pole, used here for the interaction block because it is a
# signed quantity rather than a third category.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
RED = "#e34948"

W, H = 13.4, 7.0
BASE, LABEL, NOTE = 17, 22, 15

BOXES = [
    dict(
        color=BLUE,
        title="Newspaper labels",
        sub="what the article said, at print time",
        items=["direction  improve / worsen / no change",
               "topic  business, prices, markets, other",
               "voice  official, journalist, layperson",
               "hedged, quoted forecaster, named speaker",
               "contains a number,  length,  horizon"],
        sym=r"$x_{\mathrm{news}}$"),
    dict(
        color=ORANGE,
        title="Economic data",
        sub="what the economy had done, public by that date",
        items=["industrial production  6m, 12m, acceleration",
               "CPI year over year",
               "unemployment  level, 6m change",
               "stocks  6m return, 6m volatility",
               "coverage flags  (series exists yet?)"],
        sym=r"$x_{\mathrm{econ}}$"),
    dict(
        color=RED,
        title="Economic data " + r"$\times$" + " direction",
        sub="the same macro state, signed by the forecast",
        items=["(+1 improve / " + r"$-$" + "1 worsen)  " + r"$\times$" +
               "  each macro term",
               "lets the payoff to optimism depend",
               "on the state of the economy",
               "without it, optimists and pessimists",
               "cancel and the economy looks irrelevant"],
        sym=r"$x_{\mathrm{econ}}\!\cdot\!d$"),
]


def box(ax, x0, y0, w, h, spec):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), w, h, boxstyle="round,pad=0,rounding_size=1.4",
        linewidth=1.6, edgecolor=spec["color"], facecolor=spec["color"] + "12",
        zorder=2))
    # colored spine on the left edge: identity without tinting the text
    ax.add_patch(FancyBboxPatch(
        (x0, y0), 0.9, h, boxstyle="round,pad=0,rounding_size=1.4",
        linewidth=0, facecolor=spec["color"], zorder=3))

    ax.text(x0 + 2.8, y0 + h - 4.2, spec["title"], fontsize=LABEL,
            color=INK, fontweight="bold", va="center", zorder=4)
    ax.text(x0 + w - 2.4, y0 + h - 4.2, spec["sym"], fontsize=LABEL,
            color=spec["color"], va="center", ha="right", zorder=4)
    ax.text(x0 + 2.8, y0 + h - 8.8, spec["sub"], fontsize=NOTE, color=MUTED,
            va="center", zorder=4)
    for i, it in enumerate(spec["items"]):
        ax.text(x0 + 2.8, y0 + h - 13.6 - i * 3.7, it, fontsize=BASE,
                color=INK2, va="center", zorder=4)


def arrow(ax, xy_from, xy_to, color, rad=0.0, lw=1.8, alpha=1.0):
    ax.add_patch(FancyArrowPatch(
        xy_from, xy_to, connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>,head_length=6,head_width=3.2",
        linewidth=lw, color=color, alpha=alpha,
        shrinkA=0, shrinkB=0, zorder=5))


def main():
    os.makedirs(OUT, exist_ok=True)
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "font.family": ["DejaVu Sans"], "text.color": INK,
    })

    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.985, bottom=0.015)

    # --- the three feature blocks -------------------------------------------
    bw, bh = 58.0, 31.5
    ys = [67.0, 34.0, 1.0]  # top to bottom
    for spec, y0 in zip(BOXES, ys):
        box(ax, 0.0, y0, bw, bh, spec)

    # --- sigma(z): the node the three blocks feed ---------------------------
    # The axes are 0-100 in both directions but not square, so a Circle would
    # come out as an ellipse. Stretch the y radius by the inch-per-unit ratio.
    ax_w_in = W * (0.99 - 0.01)
    ax_h_in = H * (0.985 - 0.015)
    aspect = (ax_w_in / 100.0) / (ax_h_in / 100.0)
    zx, zy, zr = 80.0, 60.0, 10.5
    ax.add_patch(matplotlib.patches.Ellipse(
        (zx, zy), 2 * zr, 2 * zr * aspect, facecolor=SURFACE, edgecolor=INK,
        linewidth=2.4, zorder=6))
    ax.text(zx, zy, r"$\sigma(z)$", fontsize=38, color=INK, ha="center",
            va="center", zorder=7)

    # arrows: each block -> sigma(z), curved so the three do not overlap
    for spec, y0, rad in zip(BOXES, ys, (0.14, 0.0, -0.18)):
        y = y0 + bh / 2
        arrow(ax, (bw + 1.5, y), (zx - zr - 1.5, zy + (y - zy) * 0.16),
              spec["color"], rad=rad, lw=2.4)

    # what z is, spelled out under the node
    for i, line in enumerate([
            r"$z=\beta_{0}+\beta_{\mathrm{news}}'x_{\mathrm{news}}$",
            r"$+\;\beta_{\mathrm{econ}}'x_{\mathrm{econ}}$",
            r"$+\;\beta_{\mathrm{int}}'(x_{\mathrm{econ}}\!\cdot\!d)$"]):
        ax.text(zx, 27.5 - i * 7.5, line, fontsize=BASE + 2, color=INK2,
                ha="center", va="center")

    fig.savefig(os.path.join(OUT, "v2_fig7_model_diagram.png"), dpi=200,
                facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {OUT}/v2_fig7_model_diagram.png")


if __name__ == "__main__":
    main()
