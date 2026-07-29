"""
Forecaster-disagreement features (economy arm).

Existing forecaster-disagreement-as-leading-indicator research is built on
the Survey of Professional Forecasters, which only exists back to 1968 --
no existing study can test whether "disagreement predicts trouble" (or its
opposite, "false consensus predicts trouble") holds across genuinely
different monetary regimes. This module builds the same kind of disagreement
measure from claims_scored.csv (1905-2009), which can.

Two things live here:
  - `add_disagreement_features`: a per-claim, backward-looking, leakage-safe
    LOCAL disagreement feature for model.py (Part 1 -- does contemporaneous
    disagreement make an individual claim harder to call correctly).
  - `episode_disagreement_rate`: the same minority-share logic aggregated to
    one number per episode, for disagreement_severity.py (Part 2 -- does an
    episode's overall disagreement level correlate with how bad it got).

Both use the same DIRECTION-DISAGREEMENT definition so the per-claim and
per-episode stories are directly comparable: among improve/worsen claims
(no_change excluded -- it is rare, see claims_scored.csv, and doesn't fit a
two-sided "disagreement" framing), minority_share = min(n_improve, n_worsen)
/ (n_improve + n_worsen). 0 = everyone agreed. 0.5 = perfectly split.
"""

import pandas as pd

DIRECTIONS = ("improve", "worsen")


def _minority_share(directions):
    """directions: iterable of 'improve'/'worsen' (already filtered to those
    two). Returns None if empty (nothing to compute a share from)."""
    n_improve = sum(1 for d in directions if d == "improve")
    n_worsen = sum(1 for d in directions if d == "worsen")
    total = n_improve + n_worsen
    if total == 0:
        return None
    return min(n_improve, n_worsen) / total


def episode_disagreement_rate(df):
    """One minority-share number per episode (Series indexed by episode),
    using ALL improve/worsen claims in that episode -- no date windowing,
    since this is a whole-episode summary for Part 2, not a per-claim
    leakage-safe feature."""
    rates = {}
    for ep, group in df.groupby("episode"):
        rates[ep] = _minority_share(group.loc[group["direction"].isin(DIRECTIONS), "direction"])
    return pd.Series(rates, name="episode_disagreement")


def add_disagreement_features(df, window_months=3):
    """Adds `local_disagreement` to df (returned copy). For each claim,
    computed from OTHER claims in the SAME episode dated in
    [this claim's date - window_months, this claim's date] -- backward-
    looking only, matching this project's leakage rule everywhere else: a
    real forecaster at the time could only have seen what had already been
    said, not what would be said later.

    A claim with no other claims in its backward window (e.g. the first
    claim of an episode) gets its episode's overall disagreement rate
    (`episode_disagreement_rate`) instead of NaN -- documented assumption,
    same pattern as score_claims.resolve_horizon()'s default-12-month
    fallback for claims with no other basis to infer from.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    episode_rate = episode_disagreement_rate(df)

    local = pd.Series(index=df.index, dtype=float)
    window = pd.Timedelta(days=30.44 * window_months)
    for ep, group in df.groupby("episode"):
        group = group.sort_values("date")
        dates = group["date"].to_numpy()
        directions = group["direction"].to_numpy()
        is_scorable_dir = pd.Series(directions).isin(DIRECTIONS).to_numpy()
        for pos, idx in enumerate(group.index):
            this_date = dates[pos]
            in_window = (dates <= this_date) & (dates >= this_date - window) & is_scorable_dir
            in_window[pos] = False  # exclude the claim itself
            share = _minority_share(directions[in_window])
            local[idx] = share if share is not None else episode_rate[ep]

    df["local_disagreement"] = local
    return df
