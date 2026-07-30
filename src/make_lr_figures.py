"""
Three candidate figures for the logistic regression itself.

figK/figM already cover the ladder and the "what is a logistic regression"
explainer. These three answer questions those two do not:

  figO_interaction   HOW the direction x economy block is built, and the slope
                     flip it buys. Left panel is the column construction; right
                     panel is the fitted model's own predicted probability, not
                     a raw correlation (figI_macro_scissors already shows those).
  figP_coefficients  WHAT the model learned -- standardized coefficients with
                     block-bootstrap CIs, interaction terms marked.
  figQ_ladder_lr     the ladder with gradient boosting removed, so every rung on
                     it is the same logistic regression with more columns.
  figR_lr_graph      ONE real forecast walked through the model: its inputs, what
                     each one did to it (value x weight, not the bare weight),
                     and the verdict against what the economy actually did. The
                     model is refitted without that forecast's own 3-year block,
                     so the probability printed is one it produced blind.

Outputs land in figures/prelim_figures/, not poster_figures/, because these are
candidates -- promote whichever survives.

Palette and rcParams match make_poster_figures.py (Okabe-Ito, CVD-safe). Blue is
optimistic and vermillion pessimistic throughout, which is a POLARITY pair with
gray "flat" at the midpoint, not three arbitrary categories.

Usage (from the repo root):
    python src/make_lr_figures.py [--context data/models/../macro_context.csv]
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import hit_predictor as hp
from macro_context import FACTORS, PRETTY

BLUE, VERM, GREEN, GRAY = "#0072B2", "#D55E00", "#009E73", "#9AA0A6"
INK, MUTED = "#1a1a1a", "#6b6b6b"
OUT = Path("figures/prelim_figures"); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.size": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#cccccc", "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": "#f0f0f0", "font.family": "DejaVu Sans"})

RESULTS = Path("data/models/hit_predictor_results.json")


def load(context):
    """The same frame hit_predictor.run() builds, so the figures describe the
    model that was actually reported rather than a re-specified cousin."""
    d = pd.read_csv(context, low_memory=False)
    d = d[d["hit"].isin([0, 1])].reset_index(drop=True)
    y = d["hit"].astype(int).values
    year = pd.to_datetime(d["date"]).dt.year
    groups = ((year // hp.BLOCK_YEARS) * hp.BLOCK_YEARS).values
    X = pd.concat([hp.claim_features(d), hp.macro_block(d[FACTORS]),
                   hp.interaction_block(d, d[FACTORS])], axis=1)
    return d, X, y, groups


def full_spec():
    """Rung 5's columns -- the only rung with the interaction block."""
    return hp.CLAIM_CAT, hp.CLAIM_NUM + hp.MACRO_NUM + hp.INTER_NUM


# --- O. the interaction, built and then fitted ------------------------------
def _typical_row(X, cat, num):
    """One representative claim: median numeric, modal categorical.

    The right panel is a partial-dependence slice, so everything not on the
    x-axis has to be pinned somewhere. Median/mode keeps the slice inside the
    data rather than at a corner no forecast ever occupied."""
    row = {}
    for c in cat:
        row[c] = X[c].mode().iloc[0]
    for c in num:
        row[c] = float(X[c].median())
    return row


def fig_interaction(d, X, y, groups, factor="epu"):
    cat, num = full_spec()
    est = hp.make_pipe(cat, num).fit(X, y)

    fig = plt.figure(figsize=(15.5, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.02, 1.0], wspace=0.22)

    # -- left: how the three column kinds are built for one month -------------
    ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    epu_hi = float(np.nanpercentile(d[factor], 90))
    ax.text(0, 0.97, f"one month, {PRETTY.get(factor, factor)} = {epu_hi:.0f}",
            fontsize=13.5, fontweight="bold", va="top")
    ax.text(0, 0.90, "three forecasts printed in it", fontsize=12,
            color=MUTED, va="top")

    cols = [0.015, 0.535, 0.685, 0.865]
    hdr = ["the forecast", f"m_{factor}", "sign", f"x_dir_{factor}"]
    ax.text(cols[0], 0.79, hdr[0], fontsize=11.5, color=MUTED, va="top")
    for x, h in zip(cols[1:], hdr[1:]):
        ax.text(x, 0.79, h, fontsize=11.5, color=MUTED, va="top",
                ha="center", family="monospace")
    ax.plot([0.015, 0.985], [0.755, 0.755], color="#dddddd", lw=1.2)

    rows = [('"business will improve"', "+1", f"+{epu_hi:.0f}", BLUE),
            ('"business will worsen"', "-1", f"-{epu_hi:.0f}", VERM),
            ('"will hold steady"', "0", "0", GRAY)]
    ys = [0.660, 0.545, 0.430]
    for (label, sign, prod, colr), yy in zip(rows, ys):
        ax.add_patch(plt.Rectangle((0.015, yy - 0.042), 0.018, 0.084,
                                   color=colr, zorder=3))
        ax.text(0.052, yy, label, fontsize=12, va="center")
        ax.text(cols[1], yy, f"{epu_hi:.0f}", fontsize=12.5, va="center",
                ha="center", family="monospace", color=MUTED)
        ax.text(cols[2], yy, sign, fontsize=12.5, va="center", ha="center",
                family="monospace", color=MUTED)
        ax.text(cols[3], yy, prod, fontsize=13, va="center", ha="center",
                family="monospace", fontweight="bold", color=colr)

    ax.plot([cols[1] - 0.05, cols[1] + 0.05], [0.360, 0.360],
            color="#cccccc", lw=1.2)
    ax.plot([cols[3] - 0.08, cols[3] + 0.08], [0.360, 0.360],
            color="#cccccc", lw=1.2)
    ax.text(cols[1], 0.325, "identical\nfor all three", fontsize=10.5,
            ha="center", va="top", color=MUTED, linespacing=1.4)
    ax.text(cols[3], 0.325, "the only column that\ntells them apart",
            fontsize=10.5, ha="center", va="top", color=INK,
            fontweight="bold", linespacing=1.4)

    ax.text(0, 0.15,
            "The economy column gets ONE weight, so it moves all three the same\n"
            "way. Only the product column can push optimists and pessimists in\n"
            "opposite directions — which is what the data says actually happened.",
            fontsize=11.5, va="top", color=MUTED, linespacing=1.55)

    # -- right: the fitted model's own slopes ---------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    lo, hi = np.nanpercentile(d[factor], [5, 95])
    grid = np.linspace(lo, hi, 60)
    base = _typical_row(X, cat, num)
    others = [f for f in hp.INTERACT_WITH if f != factor]

    for label, sign, colr, name in [
            ("improve", 1.0, BLUE, "optimistic"),
            ("flat", 0.0, GRAY, "flat"),
            ("worsen", -1.0, VERM, "pessimistic")]:
        frame = pd.DataFrame([base] * len(grid))
        frame["c_direction"] = label
        frame[f"m_{factor}"] = grid
        frame["x_dir_sign"] = sign
        frame[f"x_dir_{factor}"] = sign * grid
        for f in others:                      # the rest of the block flips too
            frame[f"x_dir_{f}"] = sign * base[f"m_{f}"]
        p = est.predict_proba(frame[X.columns])[:, 1]
        ax2.plot(grid, p, color=colr, lw=2.6, zorder=3,
                 solid_capstyle="round")
        ax2.text(grid[-1] + (hi - lo) * 0.02, p[-1], name, color=colr,
                 fontsize=12.5, fontweight="bold", va="center")

    # Label sits BELOW the line: the flat curve runs along the base rate (which
    # is the point -- a non-directional forecast is a coin flip), so anything
    # printed above the dashes lands on top of it.
    ax2.axhline(y.mean(), color="#bbbbbb", lw=1.4, ls="--", zorder=1)
    ax2.text(lo, y.mean() - 0.022, f"base rate {y.mean():.0%}", fontsize=10.5,
             color=MUTED, va="top")
    ax2.set_xlim(lo, hi + (hi - lo) * 0.30)
    ax2.set_xlabel(f"{PRETTY.get(factor, factor)} at print time  →")
    ax2.set_ylabel("model's predicted chance the forecast came true")
    ax2.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")

    fig.text(0.505, -0.02,
             "Right panel: the fitted logistic regression, every other input "
             "held at its median. The lines cross because β and γ combine to "
             "β+γ for optimists and β−γ for pessimists.",
             fontsize=10.5, color=MUTED, ha="center", va="top")
    fig.savefig(OUT / "figO_interaction.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {OUT / 'figO_interaction.png'}")


# --- P. what the model learned ----------------------------------------------
def _coef_frame(est, cat, num):
    """Fitted coefficients keyed by the transformer's own output names."""
    names = est.named_steps["pre"].get_feature_names_out()
    names = [n.split("__", 1)[-1] for n in names]
    return pd.Series(est.named_steps["clf"].coef_[0], index=names)


def fig_coefficients(X, y, groups, reps=300, top=16, seed=0):
    """Standardized coefficients with a 3-year-block bootstrap CI.

    Blocks, not rows: claims inside one period share one economy, so an i.i.d.
    bootstrap would report intervals about sqrt(claims-per-block) too narrow --
    the same reason hit_predictor bootstraps its AUCs by block."""
    cat, num = full_spec()
    est = hp.make_pipe(cat, num).fit(X, y)
    obs = _coef_frame(est, cat, num)

    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    by_block = {b: np.where(groups == b)[0] for b in uniq}
    draws = []
    for i in range(reps):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([by_block[b] for b in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        e = hp.make_pipe(cat, num).fit(X.iloc[idx], y[idx])
        draws.append(_coef_frame(e, cat, num))
    B = pd.DataFrame(draws)
    print(f"  [coefficient bootstrap: {len(B)} usable draws of {reps}]")

    keep = obs.reindex(B.columns).abs().sort_values(ascending=False).head(top)
    order = keep.sort_values().index
    lo = B[order].quantile(0.025)
    hi = B[order].quantile(0.975)
    mid = obs[order]

    def label(n):
        """One-hot names are `c_topic_employment`, so the prefix has to be
        stripped positionally -- a plain .replace("c_", "") also eats the c_ in
        the middle of `topi(c_)employment`."""
        if n.startswith("x_dir_"):
            base = n[len("x_dir_"):]
            if base == "sign":
                return "direction itself"
            return f"direction × {PRETTY.get(base, base)}"
        if n.startswith("m_"):
            return PRETTY.get(n[2:], n[2:])
        if n.startswith("has_"):
            return f"{PRETTY.get(n[4:], n[4:])} (series exists)"
        for c in hp.CLAIM_CAT:                    # c_topic_employment -> topic = employment
            if n.startswith(c + "_"):
                return f"{c[2:]} = {n[len(c) + 1:].replace('_', ' ')}"
        return n[2:].replace("_", " ") if n.startswith("c_") else n

    fig, ax = plt.subplots(figsize=(11.2, 0.46 * len(order) + 1.9))
    ypos = np.arange(len(order))
    for i, n in enumerate(order):
        inter = n.startswith("x_dir_")
        colr = BLUE if inter else GRAY
        crosses = lo[n] <= 0 <= hi[n]
        ax.plot([lo[n], hi[n]], [i, i], color=colr, lw=2.4,
                alpha=0.35 if crosses else 1.0, solid_capstyle="round",
                zorder=2)
        ax.plot([mid[n]], [i], "o", ms=9, color=colr, zorder=3,
                mec="white", mew=1.6, alpha=0.45 if crosses else 1.0)

    ax.axvline(0, color="#999999", lw=1.3, zorder=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([label(n) for n in order], fontsize=11.5)
    for tick, n in zip(ax.get_yticklabels(), order):
        if n.startswith("x_dir_"):
            tick.set_color(BLUE); tick.set_fontweight("bold")
    ax.set_xlabel("coefficient on the standardized input   "
                  "(log-odds of the forecast coming true)")
    ax.set_ylim(-0.8, len(order) - 0.2)
    ax.grid(axis="y", visible=False)

    ax.plot([], [], "o-", color=BLUE, lw=2.4, ms=9,
            label="direction × economy")
    ax.plot([], [], "o-", color=GRAY, lw=2.4, ms=9, label="everything else")
    ax.legend(frameon=False, loc="lower right", fontsize=11)

    fig.text(0.02, -0.015,
             f"Top {top} by absolute size. Bars are 95% intervals from "
             f"{len(B)} bootstraps over 3-year blocks; faded bars cross zero.",
             fontsize=10.5, color=MUTED, va="top")
    fig.savefig(OUT / "figP_coefficients.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {OUT / 'figP_coefficients.png'}")


# --- Q. the ladder, logistic regression only --------------------------------
def fig_ladder_lr():
    """figK's left panel with the boosted rung removed.

    Every bar is then the SAME estimator and the only thing changing is which
    columns it may use -- which is the point the ladder is making."""
    j = json.loads(RESULTS.read_text())
    lab = {"2. claim features only": "how the forecast is written",
           "3. economy only": "the economy at print time",
           "4. claim + economy (additive)": "both, added together",
           "5. + direction x economy": "both, allowed to INTERACT"}
    rungs = [(lab[k], j["ladder"][k]["auc"]) for k in lab]
    ci = j.get("auc_ci", {})

    fig, ax = plt.subplots(figsize=(10.4, 4.4))
    ypos = np.arange(len(rungs))[::-1]
    for i, ((name, auc), yy) in enumerate(zip(rungs, ypos)):
        last = i == len(rungs) - 1
        colr = BLUE if last else GRAY
        ax.plot([0.5, auc], [yy, yy], color=colr, lw=3.4 if last else 2.6,
                solid_capstyle="round", zorder=2)
        ax.plot([auc], [yy], "o", ms=13 if last else 10, color=colr, zorder=3,
                mec="white", mew=1.6)
        ax.text(auc + 0.004, yy, f"{auc:.3f}", va="center", fontsize=13,
                fontweight="bold" if last else "normal",
                color=INK if last else MUTED)

    if ci:
        ax.plot([ci["lo"], ci["hi"]], [ypos[-1], ypos[-1]], color=BLUE, lw=1.6,
                alpha=0.45, zorder=1, solid_capstyle="round")
        ax.text((ci["lo"] + ci["hi"]) / 2, ypos[-1] - 0.34,
                f"95% CI {ci['lo']:.3f}–{ci['hi']:.3f}", fontsize=10.5,
                color=MUTED, ha="center", va="top")

    ax.axvline(0.5, color="#999999", lw=1.3, ls="--", zorder=1)
    ax.text(0.5, len(rungs) - 0.45, "chance", fontsize=11, color=MUTED,
            ha="center", va="bottom")
    ax.set_yticks(ypos)
    ax.set_yticklabels([n for n, _ in rungs], fontsize=12.5)
    ax.set_xlim(0.49, 0.72)
    ax.set_ylim(-0.75, len(rungs) - 0.25)
    ax.set_xlabel("out-of-fold ROC-AUC  (leave-one-3-year-period-out)")
    ax.grid(axis="y", visible=False)

    fig.text(0.02, -0.02,
             "Every rung is the same logistic regression; only the columns it "
             "may use change. Interaction lift "
             f"{j['interaction_lift']['delta']:+.3f} "
             f"({j['interaction_lift']['lo']:+.3f} to "
             f"{j['interaction_lift']['hi']:+.3f}, p = "
             f"{j['interaction_lift']['p']:.3f}).",
             fontsize=10.5, color=MUTED, va="top")
    fig.savefig(OUT / "figQ_ladder_lr.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {OUT / 'figQ_ladder_lr.png'}")


# --- R. one forecast, run through the model ---------------------------------
# The three groups partition every column the model has: the `has_` flags sit
# with the economy they qualify. Leaving them out of all three (they were, once)
# makes the group subtotals fail to add up to the log-odds printed underneath.
GROUPS = [
    ("what the forecast said", hp.CLAIM_CAT + hp.CLAIM_NUM, GRAY),
    ("economic status",
     [f"m_{f}" for f in FACTORS] + [f"has_{f}" for f in FACTORS], GRAY),
    ("economy × direction", hp.INTER_NUM, BLUE),
]

# The one claim the figure walks through. Matched on fields rather than a row
# index: the frame is rebuilt from macro_context.csv every run and a bare
# integer would silently drift onto a different forecast. This quote ran as wire
# copy in three Washington papers, hence the publisher key.
EXAMPLE = dict(date="1929-11-01", quote_prefix="Every indication",
               publisher_prefix="evening star",
               occasion="three days after Black Tuesday")

# Everything but EPU is a percentage; EPU is an index level.
BASIS_PRETTY = {"INDPRO": "industrial output", "CPIAUCNS": "consumer prices",
                "UNRATE": "unemployment", "SP500": "the stock market",
                "NBER": "the business cycle"}


def _signed(v, nd=2):
    """A signed number with a typographic minus, not a hyphen."""
    return f"{v:+.{nd}f}".replace("-", "−")


def _units(factor, v):
    """A factor's value as a reader would say it out loud.

    NaN is not 'zero': macro_block fills missing series with 0 and flags them,
    so the model sees a filled value here and the label has to say so rather
    than printing an invented number."""
    if pd.isna(v):
        return "not published yet"
    if factor == "epu":
        s = f"{v:.0f}"
    elif factor in ("unrate", "stock_vol6"):
        s = f"{v:.0f}%"
    else:
        s = f"{v:+.0f}%"
    return s.replace("-", "−")           # typographic minus, not a hyphen


def _find_example(d, spec):
    """Row index of the walked-through claim, or None if it is not in the frame
    (an ablation or a replication corpus may not contain it)."""
    m = ((d["date"].astype(str) == spec["date"])
         & d["quote"].astype(str).str.startswith(spec["quote_prefix"])
         & d["publisher"].astype(str).str.startswith(spec["publisher_prefix"]))
    idx = d.index[m]
    return int(idx[0]) if len(idx) else None


def _fit_excluding(X, y, groups, block):
    """Fit on every 3-year block EXCEPT the example's own.

    The figure quotes a probability for a specific forecast, so that
    probability has to be one the model produced without having seen the
    forecast -- or its era, which is the stronger requirement here, since
    forecasts inside one period share an economy and its answers."""
    tr = groups != block
    est = hp.make_pipe(*full_spec()).fit(X[tr], y[tr])
    return est, int(tr.sum()), len(set(groups[tr]))


def _contributions(est, X, i):
    """value x weight for one row, in the transformer's own column space.

    This is what the model actually did to THIS forecast: a big coefficient on
    a column whose value happens to sit at its mean moved nothing, and drawing
    it as a thick edge would be a lie about this claim."""
    pre = est.named_steps["pre"]
    names = [n.split("__", 1)[-1] for n in pre.get_feature_names_out()]
    z = pre.transform(X.iloc[[i]])
    z = z.toarray()[0] if hasattr(z, "toarray") else z[0]
    beta = est.named_steps["clf"].coef_[0]
    return pd.DataFrame({"z": z, "beta": beta, "contrib": z * beta}, index=names)


def _tidy_publisher(p):
    return re.sub(r"\s*\d{4}\s*-\s*\d{4}\s*$", "", str(p)).title()


def fig_graph(d, X, y, groups, per_group=3, spec=EXAMPLE):
    """One real forecast -> its inputs -> the logistic regression -> hit or no.

    Edge width is |value x weight| for this claim and edge colour its sign, so
    the picture is what the model did to this forecast rather than a generic
    schematic. Only the largest few inputs per group are drawn -- all 56 would
    be a hairball -- and the rest are summed into the header line."""
    i = _find_example(d, spec)
    if i is None:
        print(f"  [skipping figR: no claim matching {spec['date']} "
              f"'{spec['quote_prefix']}...' in this frame]")
        return
    row = d.loc[i]
    est, n_train, n_blocks = _fit_excluding(X, y, groups, groups[i])
    C = _contributions(est, X, i)
    p = float(est.predict_proba(X.iloc[[i]])[0, 1])
    live = C[C["z"] != 0]           # an input at its mean moved nothing

    def value(n):
        """This claim's own value for the column, in its natural units."""
        if n.startswith("x_dir_"):
            b = n[len("x_dir_"):]
            sign = hp.DIR_SIGN.get(str(row["predicted_norm"]), 0.0)
            s = f"{sign:+.0f}"
            if b == "sign":
                return s
            return f"{s} × {_units(b, row[b])}"
        if n.startswith("m_"):
            return _units(n[2:], row[n[2:]])
        if n.startswith("has_"):
            return "yes" if X.at[i, n] else "no"
        if n == "c_len":
            return f"{int(X.at[i, n])} words"
        if n == "c_horizon":
            return f"{int(X.at[i, n])} months"
        if n in ("c_quoted", "c_named", "c_has_number"):
            return "yes" if X.at[i, n] else "no"
        return None                  # one-hots: the label already says it

    def nice(n):
        if n.startswith("x_dir_"):
            b = n[len("x_dir_"):]
            base = ("direction itself" if b == "sign"
                    else f"direction × {PRETTY.get(b, b)}")
        elif n.startswith("m_"):
            base = PRETTY.get(n[2:], n[2:])
        elif n.startswith("has_"):
            base = f"{PRETTY.get(n[4:], n[4:])} (series exists)"
        else:
            base = None
            for c in hp.CLAIM_CAT:
                if n.startswith(c + "_"):
                    base = f"{c[2:]} = {n[len(c) + 1:].replace('_', ' ')}"
            if base is None:
                base = n[2:].replace("_", " ") if n.startswith("c_") else n
        v = value(n)
        return base if v is None else f"{base} = {v}"

    def members(prefixes):
        return [c for c in live.index
                if any(c == p or c.startswith(p + "_") for p in prefixes)]

    # Lay the rows out with a GAP between groups and put each group's header in
    # its own gap. Rotated labels beside a bracket collided with each other and
    # with the "N more" counts; horizontal headers in whitespace cannot.
    ROW, GAP, TOP = 0.077, 0.062, 0.905
    chosen, headers, yy_cursor = [], [], TOP
    for name, prefixes, colr in GROUPS:
        cols = members(prefixes)
        sel = list(live.loc[cols, "contrib"].abs()
                   .sort_values(ascending=False).head(per_group).index)
        rest = [c for c in cols if c not in sel]
        headers.append((name, colr, yy_cursor + 0.052, len(rest),
                        float(live.loc[rest, "contrib"].sum())))
        for c in sel:
            chosen.append((c, colr, yy_cursor))
            yy_cursor -= ROW
        yy_cursor -= GAP

    # Wider than tall on purpose: the input labels are the longest text on the
    # figure, and inches of width are what let them be set large. The top limit
    # sits just above the quote box -- bbox_inches="tight" keeps the whole axes
    # patch, so any headroom past the box printed as a band of white above it.
    # The top limit sits just above the quote box -- bbox_inches="tight" keeps
    # the whole axes patch, so any headroom past the box printed as a band of
    # white above it. Heights other than ~9in need the quote block's own
    # geometry (in data units) rescaled, or its leading stretches with the page.
    fig, ax = plt.subplots(figsize=(17.0, 9.0))
    ax.set_xlim(0, 1); ax.set_ylim(yy_cursor + GAP - 0.055, 1.232)
    ax.axis("off"); ax.grid(False)

    # -- the forecast itself, quoted across the top ---------------------------
    ax.add_patch(plt.Rectangle((0.008, 1.010), 0.984, 0.210, facecolor="#f7f7f7",
                               edgecolor="#e3e3e3", lw=1.0, zorder=0))
    ax.plot([0.008, 0.008], [1.010, 1.220], color=BLUE, lw=4.0, zorder=1,
            solid_capstyle="butt")
    ax.text(0.030, 1.185, f"“{row['quote'].strip()}”", fontsize=17,
            style="italic", va="top", color=INK)
    speaker = str(row.get("speaker_name", "na"))
    who = "" if speaker.lower() == "na" else f"{speaker}, quoted in "
    ax.text(0.030, 1.075,
            f"{who}{_tidy_publisher(row['publisher'])}, "
            f"{pd.to_datetime(row['date']):%-d %B %Y} — {spec['occasion']}",
            fontsize=12, color=MUTED, va="top")

    ys = [t[2] for t in chosen]
    chosen = [(c, colr) for c, colr, _ in chosen]
    # Everything right of the inputs is packed as tightly as it will go, because
    # every hundredth of width saved here is width the labels can spend on type.
    # x_node is set by measurement, not taste: the longest label -- "direction x
    # stock market vs its 2-year peak = ..." -- is 0.613 wide at 20pt, so 0.640
    # lands its left edge on the margin. Enlarging the type means moving this.
    x_node = 0.640                      # the input column
    x_conv = x_node + 0.130             # where the edges meet
    x_sig = x_conv + 0.030              # sigma(z), set flush against the fan
    x_out = 0.920                       # the verdict
    y_mid = (ys[0] + ys[-1]) / 2
    biggest = max(abs(live.at[c, "contrib"]) for c, _ in chosen)

    # edges first, so the nodes sit on top of them
    for (c, colr), yy in zip(chosen, ys):
        w = live.at[c, "contrib"]
        ax.plot([x_node + 0.012, x_conv], [yy, y_mid],
                color=BLUE if w > 0 else VERM, lw=0.7 + 4.3 * abs(w) / biggest,
                alpha=0.75, zorder=2, solid_capstyle="round")

    for (c, colr), yy in zip(chosen, ys):
        ax.plot([x_node], [yy], "o", ms=11, color=colr, mec="white", mew=1.6,
                zorder=4)
        ax.text(x_node - 0.022, yy, nice(c), fontsize=20, ha="right",
                va="center", color=INK)
        # White bbox: the edge leaves the node at a steep angle and passes
        # straight through where the number would otherwise sit.
        ax.text(x_node + 0.030, yy + 0.028, f"{live.at[c, 'contrib']:+.2f}",
                fontsize=15, ha="left", va="center", color=MUTED,
                family="monospace", zorder=6,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.2))

    # group headers, each sitting in the whitespace above its own rows
    for name, colr, yy, rest, rest_sum in headers:
        ax.text(0.012, yy, name.upper(), fontsize=15, fontweight="bold",
                va="bottom", ha="left", color=colr)
        ax.text(x_node + 0.012, yy,
                f"+ {rest} more, together {_signed(rest_sum)}", fontsize=11.5,
                color=MUTED, va="bottom", ha="right")
        ax.plot([0.012, x_node + 0.012], [yy - 0.020, yy - 0.020],
                color="#e6e6e6", lw=1.0, zorder=1)

    # The step that turns the summed log-odds into a probability. No box: the
    # edges already converge on it, and the frame only competed with them.
    # Left-aligned at the convergence point, so it reads as the thing the fan
    # runs into rather than a label floating past it.
    ax.text(x_sig, y_mid + 0.004, "σ(z)", fontsize=21, ha="left",
            va="center", zorder=6, fontweight="bold")
    # Three short lines, not two long ones: a wide caption here either slid back
    # under the fan on its left or into the verdict on its right.
    ax.text(x_sig + 0.030, y_mid - 0.050,
            f"weights fitted on\n{n_train:,} forecasts from\n"
            f"the other {n_blocks} periods",
            fontsize=10.5, ha="center", va="top", color=MUTED, linespacing=1.4)

    # what the model said about THIS forecast, and what happened
    said = "HIT" if p >= 0.5 else "NO HIT"
    truth = "HIT" if row["hit"] == 1 else "NO HIT"
    what = BASIS_PRETTY.get(str(row["basis"]), str(row["basis"]))
    verb = {"improve": "rose", "up": "rose", "worsen": "fell",
            "down": "fell"}.get(str(row["realized"]), "moved the other way")

    ax.annotate("", xy=(x_out - 0.014, y_mid), xytext=(x_sig + 0.062, y_mid),
                arrowprops=dict(arrowstyle="-|>", color="#999999", lw=1.4),
                zorder=4)
    ax.text(x_out, y_mid, said.lower(), fontsize=21, va="center",
            fontweight="bold", color=VERM if said == "NO HIT" else BLUE)
    ax.text(x_out, y_mid - 0.056,
            "✓  the model called it" if said == truth
            else "✗  the model missed it",
            fontsize=13.5, va="top", fontweight="bold",
            color=GREEN if said == truth else VERM)
    ax.text(x_out, y_mid - 0.128,
            f"what happened: {truth}\n{what} {verb} over\n"
            f"the next {int(row['horizon_used'])} months",
            fontsize=12, va="top", color=MUTED, linespacing=1.5)

    # Anchored in DATA coords: the top-right corner of the axes is empty, but it
    # sits above the quoted forecast's box, so an axes-fraction anchor would put
    # the legend inside the quote.
    ax.plot([], [], color=BLUE, lw=3, label="pushed toward HIT")
    ax.plot([], [], color=VERM, lw=3, label="pushed toward NO HIT")
    ax.legend(frameon=False, loc="upper right", fontsize=13,
              bbox_to_anchor=(0.995, 0.985), bbox_transform=ax.transData)

    fig.savefig(OUT / "figR_lr_graph.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {OUT / 'figR_lr_graph.png'}  (example: {row['date']}, "
          f"p={p:.2f}, hit={int(row['hit'])})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", default="data/models/macro_context.csv")
    ap.add_argument("--factor", default="epu",
                    help="which factor the interaction panel illustrates")
    ap.add_argument("--reps", type=int, default=300)
    args = ap.parse_args()

    ctx = Path(args.context)
    if not ctx.exists():
        for alt in ["data/scored/macro_context.csv", "macro_context.csv",
                    "data/models/macro_context.csv"]:
            if Path(alt).exists():
                ctx = Path(alt); break
        else:
            raise SystemExit(f"{args.context} not found -- run "
                             f"`python src/macro_context.py` first.")
    print(f"context: {ctx}")
    d, X, y, groups = load(ctx)
    print(f"forecasts: {len(d):,}   hit rate {y.mean():.3f}   "
          f"blocks: {len(set(groups))}")

    fig_ladder_lr()
    fig_interaction(d, X, y, groups, factor=args.factor)
    fig_graph(d, X, y, groups)
    fig_coefficients(X, y, groups, reps=args.reps)


if __name__ == "__main__":
    main()
