"""
Offline test harness for the economy arm (design adopted from Vincent's pipeline).

Feeds mock loc.gov responses through the REAL parsing/scoring functions — no
network, no API keys. Run any time something changes:

    python test_offline.py
"""

import json

import numpy as np
import pandas as pd

import newspaper_scraper as ns
import score_claims as sc
from grade_claims import cohens_kappa

PASS = 0


def check(name, cond):
    global PASS
    status = "ok" if cond else "FAIL"
    print(f"  [{status}] {name}")
    if not cond:
        raise SystemExit(f"TEST FAILED: {name}")
    PASS += 1


# ---------- extract_claims ----------
print("extract_claims:")
text = ("Local notes and weather. The financial panic will pass before spring, "
        "bankers here agree. Corn prices were steady last week. "
        "Great sale prices reduced call on us today for bargains on financial panic insurance. "
        "Short. " + "x" * 700 + ". The panic of last year is now history.")
claims = ns.extract_claims(text, "financial panic")
check("finds future-marker sentence near phrase",
      any("will pass before spring" in c for c in claims))
check("drops ad/junk sentences", not any("Great sale" in c for c in claims))
check("drops too-short and too-long sentences",
      not any(c == "Short." or len(c) > 600 for c in claims))
check("ignores retrospective sentence without future marker",
      not any("now history" in c for c in claims))

# ---------- search_pages pagination (mocked _get) ----------
print("search_pages pagination:")


def fake_get_factory(total, per_page, fail_past_end=True):
    def fake_get(url):
        sp = int(url.split("sp=")[1].split("&")[0]) if "sp=" in url else 1
        start = (sp - 1) * per_page
        if start >= total:
            if fail_past_end:
                raise RuntimeError(f"failed after retries: {url} (HTTP Error 404: Not Found)")
            return json.dumps({"results": [], "pagination": {"of": total}}).encode()
        n = min(per_page, total - start)
        return json.dumps({"results": [{"id": f"http://page/{start + i}"} for i in range(n)],
                           "pagination": {"of": total}}).encode()
    return fake_get


class DummyLog:
    def writerow(self, row):
        pass


orig_get = ns._get
ns._get = fake_get_factory(total=14, per_page=30)
hits = list(ns.search_pages("business slump", "1948-07-01", "1949-06-30", 30, DummyLog(), "t"))
check("stops at reported total instead of paging past the end (the 404 bug)", len(hits) == 14)

ns._get = fake_get_factory(total=50, per_page=30)
hits = list(ns.search_pages("recession", "1957-08-01", "1958-06-30", 30, DummyLog(), "t"))
check("respects max_pages cap when results exceed it", len(hits) == 30)

ns._get = fake_get_factory(total=0, per_page=30)
hits = list(ns.search_pages("nothing", "1905-01-01", "1905-12-31", 30, DummyLog(), "t"))
check("empty result set yields nothing", hits == [])

# ---------- fetch_full_text (mocked _get) ----------
print("fetch_full_text:")


def fake_get_resource(url):
    if "fo=json" in url:
        return json.dumps({"resource": {"fulltext_file": "https://tile.loc.gov/xyz"}}).encode()
    return "THE FULL OCR TEXT".encode()


ns._get = fake_get_resource
check("pulls text via resource.fulltext_file",
      ns.fetch_full_text("http://www.loc.gov/resource/sn1/1907-11-23/ed-1/?sp=4") == "THE FULL OCR TEXT")


def fake_get_no_fulltext(url):
    return json.dumps({"resource": {}}).encode()


ns._get = fake_get_no_fulltext
check("missing fulltext_file returns empty string, not crash",
      ns.fetch_full_text("http://www.loc.gov/resource/sn1/x/?sp=1") == "")
ns._get = orig_get

# ---------- scoring rules ----------
print("scoring rules:")
idx = pd.period_range("1900-01", "1965-12", freq="M")
flat = pd.Series(100.0, index=idx)
cpi_up = flat.copy()
cpi_up.loc[idx >= pd.Period("1930-01", "M")] = 110.0  # +10% step

lab, ok, basis = sc.realized_direction("prices", "", "", pd.Timestamp("1929-12-15"), 6,
                                       cpi_up, flat, flat)
check("prices: CPI step up -> 'up'", (lab, ok, basis) == ("up", True, "CPI"))

lab, ok, basis = sc.realized_direction("prices", "", "", pd.Timestamp("1907-10-25"), 12,
                                       flat[idx >= pd.Period("1913-01", "M")], flat, flat)
check("prices before 1913 unscorable", not ok)

ip_crash = flat.copy()
ip_crash.loc[idx >= pd.Period("1930-06", "M")] = 70.0
lab, ok, basis = sc.realized_direction("general_business", "", "", pd.Timestamp("1930-01-15"), 12,
                                       flat, ip_crash, flat)
check("general business: INDPRO crash -> 'worsen'", (lab, basis) == ("worsen", "INDPRO"))

empty = pd.Series(dtype=float, index=pd.PeriodIndex([], freq="M"))
lab, ok, basis = sc.realized_direction("general_business", "", "", pd.Timestamp("1907-10-25"), 6,
                                       empty, empty, empty)
check("pre-1919 falls back to NBER; May 1908 in recession -> 'worsen'",
      (lab, basis) == ("worsen", "NBER"))

lab, ok, basis = sc.realized_direction("general_business", "", "", pd.Timestamp("1907-10-25"), 12,
                                       empty, empty, empty)
check("NBER: Oct 1908 after trough -> 'improve'", (lab, basis) == ("improve", "NBER"))

check("predicted_label maps employment 'improve' to unemployment 'down'",
      sc.predicted_label({"topic": "employment", "unemployment_direction": "na",
                          "direction": "improve"}) == "down")
check("predicted_label passes general direction through",
      sc.predicted_label({"topic": "general_business", "direction": "worsen"}) == "worsen")

# ---------- heuristic grading ----------
print("heuristic grading:")
h = sc.heuristic_grade(pd.DataFrame({"quote": [
    "Prosperity and recovery will come soon.",
    "A panic and depression will engulf us.",
    "The weather was fine on Tuesday."]}))
check("optimistic keywords -> improve", h.loc[0, "direction"] == "improve")
check("pessimistic keywords -> worsen", h.loc[1, "direction"] == "worsen")
check("no signal -> not a prediction", h.loc[2, "is_prediction"] == "no")

# ---------- tier 2: geography mapping ----------
print("tier2 geography:")
from tier2_analysis import STATE_TO_REGION, FIN_CENTERS

check("all four census regions present",
      set(STATE_TO_REGION.values()) == {"northeast", "midwest", "south", "west"})
check("covers 50 states + DC", len(STATE_TO_REGION) == 51)
check("sample mappings correct",
      STATE_TO_REGION["ohio"] == "midwest" and STATE_TO_REGION["alaska"] == "west"
      and STATE_TO_REGION["district of columbia"] == "south")
check("financial centers are a subset of known states",
      FIN_CENTERS <= set(STATE_TO_REGION))

# ---------- Cohen's kappa ----------
print("cohens_kappa:")
check("perfect agreement -> 1.0",
      abs(cohens_kappa([("a", "a"), ("b", "b"), ("a", "a")]) - 1.0) < 1e-9)
rng = np.random.default_rng(0)
rand_pairs = [(str(rng.integers(2)), str(rng.integers(2))) for _ in range(2000)]
check("random labels -> kappa near 0", abs(cohens_kappa(rand_pairs)) < 0.08)

# ---------- disagreement.py ----------
print("disagreement:")
from disagreement import add_disagreement_features, episode_disagreement_rate

da_df = pd.DataFrame([
    {"episode": "test_ep", "date": "2000-01-01", "direction": "improve", "claim_id": "A"},
    {"episode": "test_ep", "date": "2000-01-15", "direction": "improve", "claim_id": "B"},
    {"episode": "test_ep", "date": "2000-01-20", "direction": "no_change", "claim_id": "F"},
    {"episode": "test_ep", "date": "2000-02-01", "direction": "worsen", "claim_id": "C"},
    {"episode": "test_ep", "date": "2000-02-15", "direction": "worsen", "claim_id": "D"},
    {"episode": "test_ep", "date": "2000-06-01", "direction": "improve", "claim_id": "E"},
])
da_rate = episode_disagreement_rate(da_df)
check("episode_disagreement_rate: 3 improve/2 worsen -> minority share 0.4",
      abs(da_rate["test_ep"] - 0.4) < 1e-9)

da_out = add_disagreement_features(da_df, window_months=3).set_index("claim_id")
check("first claim in episode (no prior claims) imputed to episode rate",
      abs(da_out.loc["A", "local_disagreement"] - 0.4) < 1e-9)
check("second claim sees only claim A (improve) -> full agreement, 0.0",
      abs(da_out.loc["B", "local_disagreement"] - 0.0) < 1e-9)
check("no_change claim still gets a local_disagreement computed from context "
     "(A, B both improve, no worsen -> 0.0), it's just excluded from COUNTING",
      abs(da_out.loc["F", "local_disagreement"] - 0.0) < 1e-9)
check("claim C sees A, B (both improve) -> full agreement, 0.0",
      abs(da_out.loc["C", "local_disagreement"] - 0.0) < 1e-9)
check("claim D sees A, B (improve) + C (worsen) -> minority share 1/3",
      abs(da_out.loc["D", "local_disagreement"] - 1 / 3) < 1e-9)
check("claim E is >3 months after D (backward window empty) -> imputed to "
     "episode rate, not leaking from claims before the window OR the empty "
     "window silently becoming 0",
      abs(da_out.loc["E", "local_disagreement"] - 0.4) < 1e-9)

da_future = pd.DataFrame([
    {"episode": "fwd_ep", "date": "2000-01-01", "direction": "improve", "claim_id": "X"},
    {"episode": "fwd_ep", "date": "2000-01-02", "direction": "worsen", "claim_id": "Y"},
])
da_future_out = add_disagreement_features(da_future, window_months=3).set_index("claim_id")
check("backward-only window: claim X must NOT see claim Y, which comes after it "
     "-- X has no prior claims so it's imputed to the episode rate (0.5 here), "
     "not contaminated by Y's later, opposite-direction claim",
      abs(da_future_out.loc["X", "local_disagreement"] - 0.5) < 1e-9)
check("claim Y correctly sees X (the one claim strictly before it, improve) "
     "-> full agreement, 0.0",
      abs(da_future_out.loc["Y", "local_disagreement"] - 0.0) < 1e-9)

# ---------- optimism_timeline.py ----------
print("optimism_timeline:")
import optimism_timeline as ot

ot_df = pd.DataFrame([
    # test_ep, Jan 2000: 3 improve, 1 worsen -> net = (3-1)/4 = 0.5
    {"episode": "e", "kind": "crisis", "date": "2000-01-05", "predicted_label": "improve"},
    {"episode": "e", "kind": "crisis", "date": "2000-01-10", "predicted_label": "improve"},
    {"episode": "e", "kind": "crisis", "date": "2000-01-20", "predicted_label": "improve"},
    {"episode": "e", "kind": "crisis", "date": "2000-01-25", "predicted_label": "worsen"},
    # Feb 2000: 1 improve, 1 worsen -> net 0; plus a price 'up' that must be ignored
    {"episode": "e", "kind": "crisis", "date": "2000-02-05", "predicted_label": "improve"},
    {"episode": "e", "kind": "crisis", "date": "2000-02-10", "predicted_label": "worsen"},
    {"episode": "e", "kind": "crisis", "date": "2000-02-15", "predicted_label": "up"},
])
oi = ot.optimism_index(ot_df).set_index("period")
check("optimism_index: Jan net optimism (3 improve,1 worsen) = 0.5",
      abs(oi.loc[pd.Period("2000-01", "M"), "net_optimism"] - 0.5) < 1e-9)
check("optimism_index: Feb net optimism (1,1) = 0.0",
      abs(oi.loc[pd.Period("2000-02", "M"), "net_optimism"] - 0.0) < 1e-9)
check("optimism_index: price 'up' claim excluded from the count (Feb n=2, not 3)",
      int(oi.loc[pd.Period("2000-02", "M"), "n"]) == 2)

ip = pd.Series([100.0, 105.0, 103.0, 90.0],
               index=pd.period_range("2000-01", "2000-04", freq="M"))
peak, basis = ot.episode_peak_month(pd.Timestamp("2000-01-01"), pd.Timestamp("2000-04-30"), ip)
check("episode_peak_month: INDPRO argmax picks Feb (105) as the peak",
      (peak, basis) == (pd.Period("2000-02", "M"), "INDPRO"))
empty_ip = pd.Series(dtype=float, index=pd.PeriodIndex([], freq="M"))
peak2, basis2 = ot.episode_peak_month(pd.Timestamp("1907-06-01"),
                                      pd.Timestamp("1908-06-30"), empty_ip)
check("episode_peak_month: pre-INDPRO span falls back to the NBER peak (May 1907)",
      (peak2, basis2) == (pd.Period("1907-05", "M"), "NBER"))
check("months_to_peak: 3 months before the peak is -3",
      ot.months_to_peak(pd.Period("2000-02", "M"), pd.Period("2000-05", "M")) == -3)
check("weighted_slope: perfectly rising points give a positive slope",
      ot.weighted_slope([-3, -2, -1, 0], [-0.3, -0.2, -0.1, 0.0], [1, 1, 1, 1]) > 0)

# ---------- regret_scoring.py ----------
print("regret_scoring:")
import regret_scoring as rs

check("classify_error: improve predicted, improve realized -> hit",
      rs.classify_error("improve", "improve") == "hit")
check("classify_error: improve predicted, worsen realized -> optimistic_error",
      rs.classify_error("improve", "worsen") == "optimistic_error")
check("classify_error: worsen predicted, improve realized -> pessimistic_error",
      rs.classify_error("worsen", "improve") == "pessimistic_error")
check("classify_error: improve predicted, no_change realized is still an optimistic error",
      rs.classify_error("improve", "no_change") == "optimistic_error")
check("classify_error: price up/down labels are out of scope -> na",
      rs.classify_error("up", "down") == "na")
check("regret: a hit costs nothing regardless of weights/severity",
      rs.regret("hit", 1.0, 3.0, 1.0) == 0.0)
check("regret: optimistic error scaled by w_opt and severity (3*0.5=1.5)",
      abs(rs.regret("optimistic_error", 0.5, 3.0, 1.0) - 1.5) < 1e-9)
check("regret: pessimistic error uses the smaller w_pess (1*0.5=0.5)",
      abs(rs.regret("pessimistic_error", 0.5, 3.0, 1.0) - 0.5) < 1e-9)

# ---------- hedging_lexicon.py ----------
print("hedging_lexicon:")
import hedging_lexicon as hl

f_hedge = hl.hedging_features("Prosperity may perhaps return, though it seems uncertain.")
check("hedging_features: hedge-heavy sentence classed 'hedged'", f_hedge["hedge_class"] == "hedged")
check("hedging_features: counts multiple hedge terms (may, perhaps, seems, uncertain)",
      f_hedge["hedge_count"] >= 4)
f_boost = hl.hedging_features("Recovery will certainly come; a boom is inevitable and assured.")
check("hedging_features: booster-heavy sentence classed 'assertive'",
      f_boost["hedge_class"] == "assertive")
check("hedging_features: assertive sentence has negative hedge_score",
      f_boost["hedge_score"] < 0)
f_neutral = hl.hedging_features("Corn was harvested across the county last week.")
check("hedging_features: no markers -> neutral, zero score",
      f_neutral["hedge_class"] == "neutral" and f_neutral["hedge_score"] == 0)

# ---------- spf_benchmark.py ----------
print("spf_benchmark:")
import spf_benchmark as spf

check("direction_label: strong growth -> improve", spf.direction_label(3.0, 1.0) == "improve")
check("direction_label: contraction -> worsen", spf.direction_label(-2.0, 1.0) == "worsen")
check("direction_label: within band -> no_change", spf.direction_label(0.5, 1.0) == "no_change")
check("direction_label: NaN forecast -> empty (unscorable)",
      spf.direction_label(float("nan"), 1.0) == "")
check("survey_date: 1970 Q2 starts in April 1970",
      spf.survey_date(1970, 2) == pd.Timestamp("1970-04-01"))

# score_spf with an injected realized fn (no FRED needed): forecast columns
# average to +4.0 -> 'improve'; stub says reality also improved -> hit.
spf_df = pd.DataFrame([{"YEAR": 1975, "QUARTER": 1,
                        "DRGDP3": 3.0, "DRGDP4": 4.0, "DRGDP5": 5.0, "DRGDP6": 4.0},
                       {"YEAR": 1980, "QUARTER": 1,
                        "DRGDP3": -3.0, "DRGDP4": -2.0, "DRGDP5": -2.0, "DRGDP6": -1.0}])
sc_spf = spf.score_spf(spf_df, realized_dir_fn=lambda d: "improve", band=1.0)
check("score_spf: mean forecast +4 -> predicted 'improve'",
      sc_spf.loc[0, "predicted_label"] == "improve")
check("score_spf: predicted improve vs realized improve -> hit=1",
      sc_spf.loc[0, "hit"] == 1)
check("score_spf: mean forecast -1 -> predicted 'worsen', realized improve -> hit=0",
      sc_spf.loc[1, "predicted_label"] == "worsen" and sc_spf.loc[1, "hit"] == 0)

# ---------- narratives.py ----------
print("narratives:")
import narratives as nr

check("classify_narrative: 'permanently high plateau' -> new_era",
      nr.classify_narrative("Stocks have reached a permanently high plateau.") == "new_era")
check("classify_narrative: 'fundamentally sound' -> sound_fundamentals",
      nr.classify_narrative("Business is fundamentally sound, bankers say.") == "sound_fundamentals")
check("classify_narrative: 'temporary readjustment' -> temporary_setback",
      nr.classify_narrative("This is only a temporary readjustment.") == "temporary_setback")
check("classify_narrative: 'panic and depression' -> panic_fear",
      nr.classify_narrative("A panic and depression will engulf the nation.") == "panic_fear")
check("classify_narrative: 'recovery is underway' -> recovery_normalcy",
      nr.classify_narrative("Recovery is underway and revival is near.") == "recovery_normalcy")
check("classify_narrative: no economic story -> none",
      nr.classify_narrative("The county fair opens on Tuesday.") == "none")

nr_df = pd.DataFrame({"quote": ["Business is fundamentally sound.",
                                "A crash is coming.",
                                "The weather was fine."]})
nr_out = nr.add_narratives(nr_df)
check("add_narratives: complacent flag true for sound_fundamentals",
      bool(nr_out.loc[0, "complacent"]) is True)
check("add_narratives: complacent flag false for panic_fear",
      bool(nr_out.loc[1, "complacent"]) is False)

print(f"\nALL {PASS} CHECKS PASSED")
