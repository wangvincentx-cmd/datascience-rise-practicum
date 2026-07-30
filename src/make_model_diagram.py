"""The three feature blocks the hit model is built from, as one figure.

Not a chart of data -- a picture of what goes INTO the model that
src/hit_predictor.py and poster_models/m3_dml.py estimate. It exists because
the poster's central claim is about WHICH BLOCK carries the signal, and that
argument is easier to follow with the three blocks drawn side by side than
with a formula.

The third block is the one people miss. docs/RESULTS_MACRO.md found that the
pooled macro effect is a CANCELLATION -- optimists and pessimists are right
under opposite conditions -- so the economy only enters usefully once it is
multiplied by the forecast's own direction. Drawing it as its own box is the
point of the figure.

Feature names are the real ones: claim block from poster_models/_common.py
claim_design, macro block from src/model_hit.py macro_features, interaction
block from m3_dml.interaction_design. NOTE: the 0.647 model in RESULTS_MACRO
is src/hit_predictor.py, whose macro block also carries EPU and whose
interaction block is four named terms rather than every macro term -- if that
is the model on the poster, the lists below need updating to match.

    python src/make_model_diagram.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = "figures/poster_v2"

# House palette (see src/make_poster_v2_figures.py; validated in
# references/palette.md). Blue and orange are the categorical pair; red is the
# diverging warm pole, used here for the interaction block because it is a
# signed quantity rather than a third category.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
RED = "#e34948"

W, H = 11.0, 11.5
BASE, LABEL, NOTE = 23, 31, 20

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
        linewidth=1.8, edgecolor=spec["color"], facecolor=spec["color"] + "12",
        zorder=2))
    # colored spine on the left edge: identity without tinting the text
    ax.add_patch(FancyBboxPatch(
        (x0, y0), 0.9, h, boxstyle="round,pad=0,rounding_size=1.4",
        linewidth=0, facecolor=spec["color"], zorder=3))

    ax.text(x0 + 2.6, y0 + h - 4.0, spec["title"], fontsize=LABEL,
            color=INK, fontweight="bold", va="center", zorder=4)
    ax.text(x0 + w - 2.4, y0 + h - 4.0, spec["sym"], fontsize=LABEL,
            color=spec["color"], va="center", ha="right", zorder=4)
    ax.text(x0 + 2.6, y0 + h - 8.3, spec["sub"], fontsize=NOTE, color=MUTED,
            va="center", zorder=4)
    for i, it in enumerate(spec["items"]):
        ax.text(x0 + 2.6, y0 + h - 12.8 - i * 3.9, it, fontsize=BASE,
                color=INK2, va="center", zorder=4)


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
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

    bw, bh = 100.0, 30.5
    for spec, y0 in zip(BOXES, [69.0, 35.0, 1.0]):  # top to bottom
        box(ax, 0.0, y0, bw, bh, spec)

    fig.savefig(os.path.join(OUT, "v2_fig7_model_diagram.png"), dpi=200,
                facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {OUT}/v2_fig7_model_diagram.png")


if __name__ == "__main__":
    main()
