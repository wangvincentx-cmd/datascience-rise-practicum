"""
Proof that the scorer is correct.

The scorer is deterministic, so unlike the LLM extractor it can be proven right
rather than merely measured. Every test below builds a TINY synthetic economy
whose behaviour I know exactly, feeds a claim through the real scoring code, and
asserts the hit/miss/unscorable verdict that the maths must produce. If the
scoring rule is wrong, one of these fails.

Run:  python test_scoring.py
"""

import pandas as pd

from truth_data import TruthData
from score_predictions import (score_claim, predicted_norm, resolve_horizon)


def series(pairs):
    """Build a Period-indexed monthly series from (yyyy-mm, value) pairs."""
    idx = pd.PeriodIndex([p for p, _ in pairs], freq="M")
    return pd.Series([v for _, v in pairs], index=idx).sort_index()


PASS = FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [ok] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")


# A synthetic world, each regime in its own well-separated span so a 12-month
# window from January lands cleanly on the following January with no collisions:
#   INDPRO doubles over 1925 (expansion), halves over 1930 (crash),
#          +1% over 1935 (flat)
#   CPI +10% over 1925 (up), +0.2% over 1940 (flat)
#   UNRATE +2pt over 1949 (up), +0.05pt over 1955 (flat)
INDPRO = series([("1925-01", 100), ("1926-01", 200),
                 ("1930-01", 100), ("1931-01", 50),
                 ("1935-01", 100), ("1936-01", 101)])
CPI = series([("1925-01", 100), ("1926-01", 110),
              ("1940-01", 110), ("1941-01", 110.2)])
UNRATE = series([("1949-01", 4.0), ("1950-01", 6.0),
                 ("1955-01", 5.0), ("1956-01", 5.05)])
#   STOCKS +40% over 1925 (bull), -30% over 1930 (bear), +2% over 1935 (flat)
STOCKS = series([("1925-01", 100), ("1926-01", 140),
                 ("1930-01", 100), ("1931-01", 70),
                 ("1935-01", 100), ("1936-01", 102)])
RECESSIONS = set(pd.period_range("1907-05", "1908-06", freq="M"))

T = TruthData(indpro=INDPRO, cpi=CPI, unrate=UNRATE, recessions=RECESSIONS,
              stocks=STOCKS)


print("truth_data: realized outcomes over known 12-month windows")
check("INDPRO doubling -> improve",
      T.realized_direction("general_business", pd.Timestamp("1925-01-01"), 12)[0] == "improve")
check("INDPRO halving -> worsen",
      T.realized_direction("general_business", pd.Timestamp("1930-01-01"), 12)[0] == "worsen")
check("INDPRO ~flat (+1%) -> flat",
      T.realized_direction("general_business", pd.Timestamp("1935-01-01"), 12)[0] == "flat")
check("CPI +10% -> up",
      T.realized_direction("prices", pd.Timestamp("1925-01-01"), 12)[0] == "up")
check("CPI +0.2% -> flat",
      T.realized_direction("prices", pd.Timestamp("1940-01-01"), 12)[0] == "flat")
check("UNRATE +2pt -> up (unemployment rose)",
      T.realized_direction("employment", pd.Timestamp("1949-01-01"), 12)[0] == "up")
check("UNRATE +0.05pt -> flat",
      T.realized_direction("employment", pd.Timestamp("1955-01-01"), 12)[0] == "flat")

print("\ntruth_data: markets scored against STOCKS, not industrial output")
check("stocks +40% -> improve",
      T.realized_direction("markets", pd.Timestamp("1925-01-01"), 12) == ("improve", True, "STOCK"))
check("stocks -30% -> worsen",
      T.realized_direction("markets", pd.Timestamp("1930-01-01"), 12) == ("worsen", True, "STOCK"))
check("stocks +2% -> flat",
      T.realized_direction("markets", pd.Timestamp("1935-01-01"), 12)[0] == "flat")
check("markets uses the STOCK series, not INDPRO",
      T.realized_direction("markets", pd.Timestamp("1925-01-01"), 12)[2] == "STOCK")
check("markets before 1914 -> NBER fallback",
      T.realized_direction("markets", pd.Timestamp("1907-06-01"), 11)[2] == "NBER")

print("\ntruth_data: coverage boundaries -> unscorable, never guessed")
check("prices before 1913 -> unscorable",
      T.realized_direction("prices", pd.Timestamp("1905-01-01"), 12)[1] is False)
check("employment before 1948 -> unscorable",
      T.realized_direction("employment", pd.Timestamp("1930-01-01"), 12)[1] is False)
check("business 1907 (no INDPRO) falls back to NBER -> worsen",
      T.realized_direction("general_business", pd.Timestamp("1907-06-01"), 11)
      == ("worsen", True, "NBER"))
check("business 1901 (no INDPRO, no recession) -> improve via NBER",
      T.realized_direction("general_business", pd.Timestamp("1901-01-01"), 6)[0] == "improve")

print("\npredicted_norm: claim direction -> outcome vocabulary")
check("business improve -> improve",
      predicted_norm({"topic": "general_business", "direction": "improve"}) == "improve")
check("prices up -> up",
      predicted_norm({"topic": "prices", "price_direction": "up"}) == "up")
check("employment 'improve' -> unemployment down",
      predicted_norm({"topic": "employment", "direction": "improve",
                      "unemployment_direction": "na"}) == "down")
check("unclear direction -> None (unscorable)",
      predicted_norm({"topic": "general_business", "direction": "unclear"}) is None)

print("\nresolve_horizon: numeric authoritative, else parse quote, else default")
check("numeric 6 -> stated",
      resolve_horizon({"horizon_months": 6, "quote": "x"}) == (6, "stated"))
check("'in the spring' -> inferred_short",
      resolve_horizon({"horizon_months": "vague", "quote": "recovery in the spring"})[1] == "inferred_short")
check("'years to come' -> inferred_long",
      resolve_horizon({"horizon_months": "vague", "quote": "hard for years to come"})[1] == "inferred_long")
check("no time language -> default 12",
      resolve_horizon({"horizon_months": "vague", "quote": "business will improve"}) == (12, "default"))

print("\nscore_claim: end-to-end hit / miss / unscorable")
# Correct bullish call in 1925 (INDPRO doubled)
c_hit = {"topic": "general_business", "direction": "improve", "scope": "national",
         "date": "1925-01-15", "horizon_months": 12, "quote": "prosperity ahead"}
r = score_claim(c_hit, T)
check("correct improve in expansion -> hit=1", r["hit"] == 1 and r["scorable"])

# Wrong bullish call in 1930 (INDPRO halved) -- the archetype optimistic error
c_miss = {"topic": "general_business", "direction": "improve", "scope": "national",
          "date": "1930-01-15", "horizon_months": 12, "quote": "recovery is near"}
r = score_claim(c_miss, T)
check("optimistic call into a crash -> hit=0", r["hit"] == 0 and r["scorable"])

# Correct bearish call in 1930
c_bear = {"topic": "general_business", "direction": "worsen", "scope": "national",
          "date": "1930-01-15", "horizon_months": 12, "quote": "hard times coming"}
check("correct worsen into a crash -> hit=1", score_claim(c_bear, T)["hit"] == 1)

print("\nscore_claim: the scope gate (foreign forecasts never graded vs US data)")
c_foreign = {"topic": "general_business", "direction": "improve", "scope": "foreign",
             "date": "1925-01-15", "horizon_months": 12, "quote": "Mexican trade will grow"}
r = score_claim(c_foreign, T)
check("foreign scope -> unscorable, not scored against US INDPRO",
      r["scorable"] is False and "foreign" in r["unscorable_reason"])

print("\nscore_claim: no-hindsight guards")
c_nodir = {"topic": "general_business", "direction": "unclear", "scope": "national",
           "date": "1925-01-15", "horizon_months": 12, "quote": "who can say"}
check("no scorable direction -> unscorable",
      score_claim(c_nodir, T)["scorable"] is False)
c_nodate = {"topic": "general_business", "direction": "improve", "scope": "national",
            "date": "", "horizon_months": 12, "quote": "x"}
check("no date -> unscorable", score_claim(c_nodate, T)["scorable"] is False)

print("\nscore_claim: bands are real (a tiny move is not a hit)")
# 1926 INDPRO rose only ~1% -> realized flat -> an 'improve' call is a MISS
c_band = {"topic": "general_business", "direction": "improve", "scope": "national",
          "date": "1935-01-15", "horizon_months": 12, "quote": "improvement ahead"}
r = score_claim(c_band, T)
check("improve when reality is flat (<band) -> hit=0", r["hit"] == 0)
# but a 'no_change' call in the same flat window is a HIT
c_flat = {"topic": "general_business", "direction": "no_change", "scope": "national",
          "date": "1935-01-15", "horizon_months": 12, "quote": "steady as she goes"}
check("no_change when reality is flat -> hit=1", score_claim(c_flat, T)["hit"] == 1)

print("\nscore_claim: band width is adjustable (sensitivity)")
# With a huge 50% band, the 1930 halving (-50%) sits exactly at the edge; widen
# past it and the outcome becomes 'flat', flipping the bearish hit to a miss.
r = score_claim(c_bear, T, bands={"INDPRO": 60.0, "CPI": 1.5, "UNRATE": 0.5})
check("widening the band past the move -> outcome flat -> bearish call misses",
      r["realized"] == "flat" and r["hit"] == 0)

print(f"\n{'='*40}\n{PASS} passed, {FAIL} failed")
if FAIL:
    raise SystemExit(1)
