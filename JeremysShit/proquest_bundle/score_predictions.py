"""
Score extracted predictions against real economic outcomes.

Reads the extractor's JSONL (extract_llm.py output), and for every claim asks:
did the direction it predicted match what the relevant series actually did over
its horizon? The answer is computed entirely by rule against truth_data.py --
NO language model decides correctness, here or anywhere downstream.

Each claim comes out with:
    predicted_norm   the prediction, normalized to the outcome vocabulary
    realized         what actually happened (truth_data)
    basis            which series scored it (INDPRO / CPI / UNRATE / NBER)
    scorable         True/False
    unscorable_reason  why not, when False
    hit              1 / 0 / None(=unscorable)

CRITICAL: `scorable` is honest. A claim is scored ONLY if it has a mappable
topic, a real direction, a resolvable horizon window, a national scope, and a
date inside the relevant series' coverage. Everything else is marked unscorable
with a reason and LEFT UNSCORED -- never guessed. The scored fraction is a
reported number, not something to maximise.

Horizon resolution: numeric horizons from the extractor are authoritative;
otherwise the verbatim time language in the quote is parsed (SHORT/LONG regex),
and only if that finds nothing does a neutral default apply. The basis for each
claim's window is recorded so "how many rest on the default" is always visible.

Usage (from JeremysShit/):
    python score_predictions.py --claims claims_v2.jsonl --out scored_v2.csv
    python score_predictions.py --claims claims_v2.jsonl --scorable-only
    python score_predictions.py --claims claims_v2.jsonl --horizon-scale 2.0  # sensitivity
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from truth_data import DEFAULT_BANDS, TruthData

# --- horizon resolution ----------------------------------------------------
# Numeric horizons are trusted; vague ones are read from the quote's own time
# language. Kept here (not in truth_data) because it operates on the CLAIM, not
# on the economy.
SHORT_HORIZON = re.compile(
    r"\b(soon|shortly|immediat\w+|at once|right away|near future|coming months|"
    r"next few months|before long|within (?:a few )?months|"
    r"(?:a few )?weeks?(?: (?:or|cr) months?)?|months? to come|from now on|"
    r"(?:by|in|for|next) the (?:spring|summer|fall|autumn|winter)|"
    r"this (?:spring|summer|fall|autumn|winter)|by (?:spring|summer|fall|winter))\b",
    re.I)
LONG_HORIZON = re.compile(
    r"\b(long[- ]?run|long[- ]?term|for years|coming years|years to come|"
    r"eventually|ultimately|in (?:the )?time|decade|for some time|"
    r"permanent\w*|lasting)\b", re.I)
H_SHORT, H_DEFAULT, H_LONG = 6, 12, 24


def resolve_horizon(claim, scale=1.0):
    """(months, basis). basis is 'stated' / 'inferred_short' / 'inferred_long'
    / 'default' so the reliance on the default is always countable."""
    h = str(claim.get("horizon_months", "")).strip()
    if h in ("6", "12", "24"):
        return int(h), "stated"
    q = str(claim.get("quote", ""))
    if LONG_HORIZON.search(q):
        m, basis = H_LONG, "inferred_long"
    elif SHORT_HORIZON.search(q):
        m, basis = H_SHORT, "inferred_short"
    else:
        m, basis = H_DEFAULT, "default"
    return max(1, round(m * scale)), basis


# --- predicted-direction normalization -------------------------------------
# Map what the extractor said the claim predicted into the SAME vocabulary
# truth_data reports outcomes in, per topic. Returns None when the prediction
# has no scorable direction (unclear / na / missing), which makes the claim
# unscorable rather than a coin flip.
def predicted_norm(claim):
    topic = claim.get("topic")
    if topic == "prices":
        pd_ = str(claim.get("price_direction", "")).lower()
        return {"up": "up", "down": "down", "stable": "flat"}.get(pd_)
    if topic == "employment":
        ud = str(claim.get("unemployment_direction", "")).lower()
        # An employment forecast can be phrased either as unemployment direction
        # or as general improve/worsen; prefer the explicit unemployment field,
        # fall back to mapping improve->down (jobs better = unemployment down).
        m = {"up": "up", "down": "down", "stable": "flat"}.get(ud)
        if m:
            return m
        d = str(claim.get("direction", "")).lower()
        return {"improve": "down", "worsen": "up", "no_change": "flat"}.get(d)
    # general_business / markets / other
    d = str(claim.get("direction", "")).lower()
    return {"improve": "improve", "worsen": "worsen", "no_change": "flat"}.get(d)


def score_claim(claim, truth, scale=1.0, bands=None):
    """Score one claim. Pure function of the claim + real data."""
    out = dict(claim)
    out["predicted_norm"] = predicted_norm(claim)
    out["realized"] = None
    out["basis"] = None
    out["hit"] = None
    out["scorable"] = False
    out["unscorable_reason"] = None

    # Scope gate: only US-national (and generic) claims may be graded against US
    # series. Foreign/regional/industry are real forecasts but not about the
    # national economy the series measure -- flag, do not score.
    scope = str(claim.get("scope", "national")).lower()
    if scope in ("foreign", "regional", "industry"):
        out["unscorable_reason"] = f"scope={scope} (not US national)"
        return out

    date = claim.get("date")
    if not date or pd.isna(pd.to_datetime(date, errors="coerce")):
        out["unscorable_reason"] = "no usable date"
        return out

    pred = out["predicted_norm"]
    if pred is None:
        out["unscorable_reason"] = "prediction has no scorable direction"
        return out

    months, hbasis = resolve_horizon(claim, scale)
    out["horizon_used"] = months
    out["horizon_basis"] = hbasis

    realized, ok, basis = truth.realized_direction(
        claim.get("topic"), pd.to_datetime(date), months, bands)
    out["realized"] = realized
    out["basis"] = basis
    if not ok:
        out["unscorable_reason"] = basis
        return out

    out["scorable"] = True
    out["hit"] = int(pred == realized)
    return out


def score_file(claims_path, truth=None, scale=1.0, bands=None):
    truth = truth or TruthData()
    claims = [json.loads(l) for l in open(claims_path, encoding="utf-8") if l.strip()]
    rows = [score_claim(c, truth, scale, bands) for c in claims]
    return pd.DataFrame(rows)


def summarize(df):
    n = len(df)
    scorable = df[df["scorable"]]
    print(f"\n=== SCORING SUMMARY ===")
    print(f"  claims total          {n}")
    print(f"  scorable              {len(scorable)} ({len(scorable)/n:.0%})")
    print(f"  hit rate (scorable)   {scorable['hit'].mean():.3f}"
          if len(scorable) else "  hit rate              n/a")
    print("\n  unscorable breakdown:")
    for reason, k in (df[~df["scorable"]]["unscorable_reason"]
                      .value_counts().items()):
        print(f"    {k:>5}  {reason}")
    if len(scorable):
        print("\n  scored by series:")
        for basis, k in scorable["basis"].value_counts().items():
            hr = scorable[scorable["basis"] == basis]["hit"].mean()
            print(f"    {k:>5}  {basis:<8} hit rate {hr:.3f}")
        print("\n  horizon basis (scorable claims):")
        for hb, k in scorable["horizon_basis"].value_counts().items():
            print(f"    {k:>5}  {hb}")
        rigid = scorable[scorable["horizon_basis"] != "default"]
        print(f"\n  RIGID subset (real horizon, not defaulted): {len(rigid)} "
              f"({len(rigid)/n:.0%} of all claims), hit rate "
              f"{rigid['hit'].mean():.3f}" if len(rigid) else "")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--claims", required=True, help="extractor JSONL")
    ap.add_argument("--out", default=None, help="scored CSV")
    ap.add_argument("--scorable-only", action="store_true")
    ap.add_argument("--horizon-scale", type=float, default=1.0,
                    help="multiply inferred horizons (sensitivity test)")
    args = ap.parse_args()

    df = score_file(args.claims, scale=args.horizon_scale)
    summarize(df)
    if args.scorable_only:
        df = df[df["scorable"]]
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\n-> {args.out}  ({len(df)} rows)")
