"""Offline verification for the whole bill-passage pipeline. Runs the ACTUAL
pipeline functions against mock responses shaped like the real Congress.gov
and NYT APIs, and a fake LLM client, no network or API key needed.

Run before spending any API budget:  python test_offline.py
"""
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd

import build_features
import coverage_report
import download_bills
import extract_press
import link_coverage

FAIL = []


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        FAIL.append(name)


# ---------------------------------------------------------------------------
print("\n[1] download_bills.py: enactment detection")

check("laws array -> enacted",
      download_bills.is_enacted({"laws": [{"type": "Public Law", "number": "118-5"}],
                                 "latestAction": {"text": "Referred"}}))
check("latestAction 'Became Public Law' -> enacted",
      download_bills.is_enacted({"laws": [],
                                 "latestAction": {"text": "Became Public Law No: 118-5."}}))
check("latestAction 'Became Private Law' -> enacted",
      download_bills.is_enacted({"laws": [],
                                 "latestAction": {"text": "Became Private Law No: 118-1."}}))
check("dead bill (referred, no laws) -> not enacted",
      not download_bills.is_enacted({"laws": [],
                                     "latestAction": {"text": "Referred to the Committee on..."}}))
check("passed one chamber but not law -> not enacted",
      not download_bills.is_enacted({"laws": [],
                                     "latestAction": {"text": "Passed House, referred to Senate"}}))

# ---------------------------------------------------------------------------
print("\n[2] download_bills.py: list pagination + build_record via mocked HTTP")

ALL_BILLS = [{"number": str(815 + i), "type": "HR"} for i in range(3)]


def fake_list_get_json(url, api_key, params=None, sleep=0.4):
    offset, page_size = params["offset"], params["limit"]
    page = ALL_BILLS[offset:offset + page_size]
    return {"bills": page, "pagination": {"count": len(ALL_BILLS)}}


with patch.object(download_bills, "get_json", side_effect=fake_list_get_json), \
     patch.object(download_bills.time, "sleep", lambda s: None):
    stubs = list(download_bills.iter_bill_list(118, "hr", "KEY", limit=None, sleep=0))
    check("iter_bill_list paginates across pages until empty", len(stubs) == 3)

    stubs_capped = list(download_bills.iter_bill_list(118, "hr", "KEY", limit=2, sleep=0))
    check("iter_bill_list respects limit", len(stubs_capped) == 2)

DETAIL_RESP = {"bill": {
    "congress": 118, "type": "HR", "number": "815",
    "introducedDate": "2023-02-02", "title": "A bill to do a healthy thing.",
    "policyArea": {"name": "Health"},
    "sponsors": [{"party": "D", "state": "CA", "bioguideId": "X000001",
                 "lastName": "Smith", "fullName": "Rep. Smith, Jane [D-CA-5]"}],
    "latestAction": {"text": "Became Public Law No: 118-5.", "actionDate": "2023-06-01"},
    "laws": [{"type": "Public Law", "number": "118-5"}],
}}
COSPONSOR_RESP = {"cosponsors": [
    {"party": "D", "state": "CA", "isOriginalCosponsor": True},
    {"party": "R", "state": "TX", "isOriginalCosponsor": True},
    {"party": "D", "state": "NY", "isOriginalCosponsor": False},  # added later, must be excluded
]}
COMMITTEE_RESP = {"committees": [{"name": "House Energy and Commerce Committee"}]}
RELATED_RESP = {"relatedBills": [{"congress": 118, "type": "S", "number": "400"}]}


def fake_record_get_json(url, api_key, params=None, sleep=0.4):
    if url.endswith("/bill/118/hr/815"):
        return DETAIL_RESP
    if url.endswith("/cosponsors"):
        return COSPONSOR_RESP
    if url.endswith("/committees"):
        return COMMITTEE_RESP
    if url.endswith("/relatedbills"):
        return RELATED_RESP
    return None


with patch.object(download_bills, "get_json", side_effect=fake_record_get_json):
    rec = download_bills.build_record(118, "hr", "815", "KEY", sleep=0, fetch_text=False)
    check("build_record picks up sponsor party/state",
          rec["sponsor_party"] == "D" and rec["sponsor_state"] == "CA")
    check("build_record marks became_law True", rec["became_law"] is True)
    check("build_record keeps only original cosponsors (2, not 3)",
          rec["n_original_cosponsors"] == 2)
    check("build_record excludes the later-added cosponsor (no NY state)",
          not any(c.get("state") == "NY" for c in rec["original_cosponsors"]))
    check("build_record captures sponsor last name",
          rec["sponsor_last_name"] == "Smith")
    check("build_record captures primary committee",
          rec["primary_committee"] == "House Energy and Commerce Committee")
    check("build_record captures has_companion_bill True",
          rec["has_companion_bill"] is True)
    check("build_record has no introduced_text when fetch_text=False",
          "introduced_text" not in rec)

    missing = download_bills.build_record(118, "hr", "999", "KEY", sleep=0, fetch_text=False)
    check("build_record returns None for a bill detail that doesn't resolve",
          missing is None)

# ---------------------------------------------------------------------------
print("\n[3] download_bills.py: resume-safety (load_done)")
import json
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as d:
    p = Path(d) / "118.jsonl"
    with open(p, "w") as f:
        f.write(json.dumps({"bill_type": "hr", "number": "815"}) + "\n")
        f.write(json.dumps({"bill_type": "s", "number": "10"}) + "\n")
    done = download_bills.load_done(p)
    check("load_done reads existing (type, number) pairs",
          ("hr", "815") in done and ("s", "10") in done and len(done) == 2)
    empty_done = download_bills.load_done(Path(d) / "nonexistent.jsonl")
    check("load_done returns empty set for missing file", empty_done == set())

# ---------------------------------------------------------------------------
print("\n[4] build_features.py: per-bill feature derivation")

MAJORITY_TABLE = {
    118: {"start_year": 2023, "end_year": 2024, "House": "R", "Senate": "D"},
}

check("intro_month_in_session: first month of congress is 1",
      build_features.intro_month_in_session("2023-01-15", 2023) == 1)
check("intro_month_in_session: Dec of second year is 24",
      build_features.intro_month_in_session("2024-12-01", 2023) == 24)
check("intro_month_in_session: Feb 2023 is month 2",
      build_features.intro_month_in_session("2023-02-02", 2023) == 2)
check("intro_month_in_session: None for missing date",
      build_features.intro_month_in_session(None, 2023) is None)

check("bipartisan True when both D and R present",
      build_features.bipartisan([{"party": "D"}, {"party": "R"}]))
check("bipartisan False when only D present",
      not build_features.bipartisan([{"party": "D"}, {"party": "D"}]))
check("bipartisan False for empty cosponsor list",
      not build_features.bipartisan([]))

check("frac_cosponsors_majority computes correctly",
      build_features.frac_cosponsors_majority(
          [{"party": "R"}, {"party": "R"}, {"party": "D"}], "R") == 2 / 3)
check("frac_cosponsors_majority is 0.0 for empty cosponsor list",
      build_features.frac_cosponsors_majority([], "R") == 0.0)

sample_record = {
    "congress": 118, "bill_type": "hr", "number": "815",
    "introduced_date": "2023-02-02", "title": "A bill to do a healthy thing.",
    "policy_area": "Health", "sponsor_party": "D", "sponsor_state": "CA",
    "latest_action_text": "Became Public Law No: 118-5.",
    "original_cosponsors": [{"party": "D", "state": "CA"}, {"party": "R", "state": "TX"}],
    "n_original_cosponsors": 2, "primary_committee": "House Energy and Commerce Committee",
    "has_companion_bill": True, "became_law": True,
}
row = build_features.row_from_record(sample_record, MAJORITY_TABLE)
check("row_from_record: chamber derived from bill_type",
      row["chamber"] == "House")
check("row_from_record: sponsor_in_majority False (sponsor D, House majority R)",
      row["sponsor_in_majority"] is False)
check("row_from_record: bipartisan True",
      row["bipartisan"] is True)
check("row_from_record: title_length word count",
      row["title_length"] == 7)
check("row_from_record: became_law carried through",
      row["became_law"] is True)

sample_record_r = dict(sample_record, sponsor_party="R")
row_r = build_features.row_from_record(sample_record_r, MAJORITY_TABLE)
check("row_from_record: sponsor_in_majority True (sponsor R, House majority R)",
      row_r["sponsor_in_majority"] is True)

no_congress_record = dict(sample_record, congress=999)
row_missing = build_features.row_from_record(no_congress_record, MAJORITY_TABLE)
check("row_from_record: unknown congress -> sponsor_in_majority None, no crash",
      row_missing["sponsor_in_majority"] is None)

print("\n[5] build_features.py: end-to-end feature table from a temp jsonl")
with tempfile.TemporaryDirectory() as d:
    bills_path = Path(d) / "118.jsonl"
    with open(bills_path, "w") as f:
        f.write(json.dumps(sample_record) + "\n")
        f.write(json.dumps(dict(sample_record, number="816", became_law=False,
                                original_cosponsors=[])) + "\n")
    maj_path = Path(d) / "majority.csv"
    pd.DataFrame([{"congress": 118, "start_year": 2023, "end_year": 2024,
                  "house_majority_party": "R", "senate_majority_party": "D"}]).to_csv(
        maj_path, index=False)
    df = build_features.build_features([bills_path], maj_path)
    check("build_features returns one row per bill", len(df) == 2)
    check("build_features base rate matches input (1 of 2 became law)",
          abs(df["became_law"].mean() - 0.5) < 1e-9)
    check("build_features has all Section 7 columns",
          {"congress", "chamber", "bill_type", "sponsor_party", "sponsor_state",
           "sponsor_in_majority", "n_original_cosponsors", "bipartisan",
           "frac_cosponsors_majority", "policy_area", "primary_committee",
           "intro_month_in_session", "title_length", "has_companion_bill",
           "title_text", "introduced_text", "became_law"}.issubset(df.columns))

print("\n[6] link_coverage.py: query building, search window, pagination")

bill_for_query = {"title": "A bill to provide health care access for veterans in rural areas",
                  "sponsor_last_name": "Smith", "policy_area": "Health"}
queries = link_coverage.build_queries(bill_for_query, include_broad=True)
check("build_queries includes the full title first",
      queries[0] == bill_for_query["title"])
check("build_queries includes a stopword-stripped key-noun query",
      queries[1] != queries[0]
      and "veterans" in queries[1].lower().split()
      and "provide" not in queries[1].lower().split()
      and "for" not in queries[1].lower().split())
check("build_queries includes sponsor+policy when include_broad=True",
      "Smith Health" in queries)
check("build_queries dedupes",
      len(queries) == len(set(queries)))

no_broad = link_coverage.build_queries(bill_for_query, include_broad=False)
check("build_queries omits sponsor+policy when include_broad=False",
      "Smith Health" not in no_broad)

empty_title_bill = {"title": "", "sponsor_last_name": None, "policy_area": None}
check("build_queries returns [] for a bill with no usable fields",
      link_coverage.build_queries(empty_title_bill) == [])

begin, end = link_coverage.search_window("2023-02-02")
check("search_window: begin is 7 days before introduction",
      begin == "20230126")
check("search_window: end is 180 days after introduction with no final action",
      end == "20230801")

begin2, end2 = link_coverage.search_window("2023-02-02", latest_action_date="2023-03-01")
check("search_window: end is capped at latest_action_date if sooner",
      end2 == "20230301")

begin3, end3 = link_coverage.search_window("2023-02-02", latest_action_date="2023-01-15")
check("search_window: end never falls before begin even with an early final action",
      end3 >= begin3)

LC_DOC = {"web_url": "https://nyt.com/2023/a", "headline": {"main": "Bill Advances"},
         "abstract": "It advances.", "lead_paragraph": "The bill advances in committee.",
         "snippet": "It advances.", "pub_date": "2023-02-10T00:00:00Z",
         "section_name": "Politics", "type_of_material": "News"}
LC_PAGE0 = {"response": {"meta": {"hits": 1}, "docs": [LC_DOC]}}
LC_EMPTY = {"response": {"meta": {"hits": 1}, "docs": []}}
_lc_calls = {"n": 0}


def fake_lc_get_json(params):
    _lc_calls["n"] += 1
    return LC_PAGE0 if _lc_calls["n"] == 1 else LC_EMPTY


with patch.object(link_coverage, "get_json", side_effect=fake_lc_get_json), \
     patch.object(link_coverage.time, "sleep", lambda s: None):
    docs = list(link_coverage.search_query("KEY", "health care veterans",
                                           "20230126", "20230801", max_pages=3))
    check("search_query yields docs and stops when a page is empty", len(docs) == 1)

text = link_coverage.combine_text(LC_DOC)
check("link_coverage.combine_text merges headline+lead, dedupes",
      "Bill Advances" in text and text.count("It advances.") == 1)

print("\n[7] link_coverage.py: search log resume-safety")
with tempfile.TemporaryDirectory() as d:
    log_path = Path(d) / "press_search_log.csv"
    with patch.object(link_coverage, "Path", side_effect=lambda p: (
        log_path if p == "data/press_search_log.csv" else Path(p))):
        link_coverage.log_search(118, "hr", "815", "veterans health", 3)
        link_coverage.log_search(118, "hr", "815", "veterans health", 3)  # simulate rerun
    searched = link_coverage.load_searched(log_path)
    check("load_searched reads back a logged (congress, bill_type, number, query)",
          (118, "hr", "815", "veterans health") in searched)

# ---------------------------------------------------------------------------
print("\n[8] extract_press.py: LLM extraction with a fake LLM client")


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeCompletionResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeLLMClient:
    """Stands in for extract_press.LLMPool -- exposes .create() directly."""
    def __init__(self, text):
        self._text = text

    def create(self, **kw):
        return FakeCompletionResponse(self._text)


ep_bill = {"congress": 118, "bill_type": "hr", "number": "815",
          "title": "A bill to provide health care access for veterans",
          "sponsor_party": "D", "sponsor_state": "CA", "policy_area": "Health",
          "introduced_date": "2023-02-02"}
ep_article = {"headline": "Veterans Health Bill Set to Pass", "pub_date": "2023-02-10",
             "section": "Politics",
             "snippet_text": "The bill is expected to become law by summer."}

match_reply = ('{"about_this_bill": true, "prediction": "pass", "confidence": "firm"}')
result = extract_press.extract_from_article(FakeLLMClient(match_reply), ep_bill, ep_article)
check("extract_from_article parses a clean match",
      result == {"about_this_bill": True, "prediction": "pass", "confidence": "firm"})

fenced_reply = ('```json\n{"about_this_bill": false, "prediction": null, '
               '"confidence": null}\n```')
result2 = extract_press.extract_from_article(FakeLLMClient(fenced_reply), ep_bill, ep_article)
check("extract_from_article strips markdown fences",
      result2 == {"about_this_bill": False, "prediction": None, "confidence": None})

bad_reply = ('{"about_this_bill": true, "prediction": "definitely-yes", "confidence": "firm"}')
result3 = extract_press.extract_from_article(FakeLLMClient(bad_reply), ep_bill, ep_article)
check("extract_from_article rejects an out-of-schema prediction value",
      result3["prediction"] is None)

malformed = extract_press.extract_from_article(FakeLLMClient("not json"), ep_bill, ep_article)
check("extract_from_article returns None for a malformed reply", malformed is None)

# note: if the LLM incorrectly says about_this_bill=false but still fills in a
# prediction, extract_from_article must not leak it through as a signal.
leaky_reply = ('{"about_this_bill": false, "prediction": "pass", "confidence": "firm"}')
result4 = extract_press.extract_from_article(FakeLLMClient(leaky_reply), ep_bill, ep_article)
check("extract_from_article nulls out prediction/confidence when about_this_bill is false",
      result4["prediction"] is None and result4["confidence"] is None)

print("\n[9] extract_press.py: bill-level aggregation")

agg_bills = [
    {"congress": 118, "bill_type": "hr", "number": "1", "introduced_date": "2023-01-10"},
    {"congress": 118, "bill_type": "hr", "number": "2", "introduced_date": "2023-01-10"},
    {"congress": 118, "bill_type": "hr", "number": "3", "introduced_date": "2023-01-10"},
]
labeled = [
    {"bill_type": "hr", "number": "1", "about_this_bill": True, "prediction": "pass",
     "confidence": "firm", "pub_date": "2023-01-20"},
    {"bill_type": "hr", "number": "1", "about_this_bill": True, "prediction": "pass",
     "confidence": "hedged", "pub_date": "2023-01-15"},
    {"bill_type": "hr", "number": "2", "about_this_bill": True, "prediction": "neutral",
     "confidence": None, "pub_date": "2023-01-25"},
    {"bill_type": "hr", "number": "2", "about_this_bill": False, "prediction": "pass",
     "confidence": "firm", "pub_date": "2023-01-25"},  # must be excluded: not about the bill
]
agg = extract_press.aggregate_bill_features(labeled, agg_bills)
row1 = agg[agg.number == "1"].iloc[0]
row2 = agg[agg.number == "2"].iloc[0]
row3 = agg[agg.number == "3"].iloc[0]

check("aggregate: bill 1 has coverage with 2 articles",
      row1.has_national_coverage and row1.n_articles == 2)
check("aggregate: bill 1 press_predicts_pass is 'pass' (2-0 vote)",
      row1.press_predicts_pass == "pass")
check("aggregate: bill 1 press_confidence prefers firm over hedged",
      row1.press_confidence == "firm")
check("aggregate: bill 1 days_intro_to_first_coverage uses the EARLIEST article",
      row1.days_intro_to_first_coverage == 5)
check("aggregate: bill 2 excludes the article where about_this_bill is False",
      row2.n_articles == 1 and row2.press_predicts_pass == "neutral")
check("aggregate: bill 3 with zero coverage gets 'none' and no coverage flag",
      not row3.has_national_coverage and row3.press_predicts_pass == "none"
      and pd.isna(row3.days_intro_to_first_coverage))

mixed_labeled = [
    {"bill_type": "hr", "number": "1", "about_this_bill": True, "prediction": "pass",
     "confidence": "firm", "pub_date": "2023-01-20"},
    {"bill_type": "hr", "number": "1", "about_this_bill": True, "prediction": "fail",
     "confidence": "firm", "pub_date": "2023-01-21"},
]
mixed_agg = extract_press.aggregate_bill_features(mixed_labeled, agg_bills[:1])
check("aggregate: a 1-1 pass/fail split is reported as 'mixed', not silently picked",
      mixed_agg.iloc[0].press_predicts_pass == "mixed")

# ---------------------------------------------------------------------------
print("\n[10] coverage_report.py: decision gate math")

gate_df = pd.DataFrame([
    {"has_national_coverage": True, "press_predicts_pass": "pass"},
    {"has_national_coverage": True, "press_predicts_pass": "fail"},
    {"has_national_coverage": True, "press_predicts_pass": "neutral"},
    {"has_national_coverage": False, "press_predicts_pass": "none"},
])
counts = coverage_report.compute_counts(gate_df)
check("compute_counts: total bills counted",
      counts["total"] == 4)
check("compute_counts: covered bills counted (has_national_coverage True)",
      counts["n_covered"] == 3)
check("compute_counts: non-neutral (pass/fail) subset counted",
      counts["n_non_neutral"] == 2)
check("compute_counts: gate fails below the underpowered threshold",
      counts["gate_pass"] is False)

big_df = pd.concat([gate_df[gate_df.press_predicts_pass.isin(["pass", "fail"])]] * 200,
                   ignore_index=True)
big_counts = coverage_report.compute_counts(big_df)
check("compute_counts: gate passes at/above the underpowered threshold",
      big_counts["gate_pass"] is True and big_counts["n_non_neutral"] >= 300)

# ---------------------------------------------------------------------------
print("\n[11] join_dataset.py: structural + press merge")
import join_dataset

with tempfile.TemporaryDirectory() as d:
    prev_cwd = Path.cwd()
    import os
    os.chdir(d)
    try:
        Path("data").mkdir()
        pd.DataFrame([
            {"congress": 117, "bill_type": "hr", "number": "1", "became_law": False},
            {"congress": 118, "bill_type": "hr", "number": "1", "became_law": True},
            {"congress": 118, "bill_type": "hr", "number": "2", "became_law": False},
        ]).to_csv("data/features.csv", index=False)
        pd.DataFrame([
            {"congress": 118, "bill_type": "hr", "number": "1",
             "has_national_coverage": True, "n_articles": 2,
             "press_predicts_pass": "pass", "press_confidence": "firm",
             "days_intro_to_first_coverage": 3},
            {"congress": 118, "bill_type": "hr", "number": "2",
             "has_national_coverage": False, "n_articles": 0,
             "press_predicts_pass": "none", "press_confidence": None,
             "days_intro_to_first_coverage": None},
        ]).to_csv("data/press_features_118.csv", index=False)

        merged, joined_congresses = join_dataset.join("data/features.csv", [118])
        check("join_dataset: only the congress with a press file is included",
              joined_congresses == [118] and set(merged["congress"]) == {118})
        check("join_dataset: keeps uncovered bills too (does not drop them)",
              len(merged) == 2)
        check("join_dataset: press columns attached to the right bill",
              merged.loc[merged.number.astype(str) == "1", "press_predicts_pass"].iloc[0]
              == "pass")

        # dtype-mismatch regression: if features.csv's "number" column infers as
        # int64 but the press file (e.g. via an unrelated NaN elsewhere) infers
        # as float64, a naive merge silently drops every row instead of erroring.
        pd.DataFrame([
            {"congress": 118, "bill_type": "hr", "number": 1.0,  # float64, not int
             "has_national_coverage": True, "n_articles": 1,
             "press_predicts_pass": "pass", "press_confidence": "firm",
             "days_intro_to_first_coverage": 3},
        ]).to_csv("data/press_features_118.csv", index=False)
        merged2, _ = join_dataset.join("data/features.csv", [118])
        check("join_dataset: survives int-vs-float dtype mismatch on the merge key "
             "without silently dropping rows",
              len(merged2) == 1 and merged2.iloc[0]["press_predicts_pass"] == "pass")
    finally:
        os.chdir(prev_cwd)

# ---------------------------------------------------------------------------
print("\n[12] model.py: research question 3 hit rate + bootstrap delta + small-sample fallback")
import model

q3_df = pd.DataFrame([
    {"press_predicts_pass": "pass", "y": 1},   # hit
    {"press_predicts_pass": "pass", "y": 0},   # miss
    {"press_predicts_pass": "fail", "y": 0},   # hit
    {"press_predicts_pass": "fail", "y": 1},   # miss
    {"press_predicts_pass": "neutral", "y": 1},  # excluded
])
q3 = model.research_q3_hit_rate(q3_df)
check("research_q3_hit_rate excludes neutral calls from n",
      q3["n"] == 4)
check("research_q3_hit_rate: 2 of 4 non-neutral calls were correct",
      abs(q3["hit_rate"] - 0.5) < 1e-9)

empty_q3 = model.research_q3_hit_rate(pd.DataFrame([{"press_predicts_pass": "neutral", "y": 1}]))
check("research_q3_hit_rate returns n=0 and nan when nothing is non-neutral",
      empty_q3["n"] == 0 and np.isnan(empty_q3["hit_rate"]))

y_true = np.array([0] * 90 + [1] * 10)
proba_bad = np.full(100, 0.1)          # uninformative
proba_good = np.where(y_true == 1, 0.9, 0.05)  # near-perfect
boot = model.bootstrap_pr_auc_delta(y_true, proba_bad, proba_good, n_boot=300, seed=0)
check("bootstrap_pr_auc_delta: a clearly better model shows a positive delta",
      boot["mean_delta"] > 0.3)
check("bootstrap_pr_auc_delta: CI excludes zero for a large, real effect",
      boot["ci_low"] > 0)

no_signal = model.bootstrap_pr_auc_delta(y_true, proba_bad, proba_bad, n_boot=200, seed=0)
check("bootstrap_pr_auc_delta: identical models give ~zero delta",
      abs(no_signal["mean_delta"]) < 1e-9)

from sklearn.linear_model import LogisticRegression as _LR

# This is the exact crash caught during smoke-testing: CalibratedClassifierCV
# with cv=3 raises when a training set has fewer than 3 positive examples,
# which is the normal case for the covered-subset press experiment
# (Section 8.1's small-sample regime). fit_calibrated must fall back instead
# of propagating that crash.
tiny_train = pd.DataFrame({"x": [0.1, 0.2, 0.3, 0.4, 0.5], "y": [0, 0, 0, 0, 1]})
with patch("builtins.print"):
    result = model.fit_calibrated(_LR(), tiny_train, max_cv=3)
check("fit_calibrated: 1-positive training set does not raise, "
     "falls back to an uncalibrated estimator",
      not hasattr(result, "calibrated_classifiers_"))

plenty_train = pd.DataFrame({"x": list(range(20)),
                            "y": [0] * 10 + [1] * 10})
result_plenty = model.fit_calibrated(_LR(), plenty_train, max_cv=3)
check("fit_calibrated: enough positives uses real CalibratedClassifierCV",
      hasattr(result_plenty, "calibrated_classifiers_"))

# ---------------------------------------------------------------------------
print("\n[13] download_bills_bulk.py: BILLSTATUS XML parsing")
import io
import zipfile

import download_bills_bulk

BILLSTATUS_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<billStatus>
  <bill>
    <number>3746</number>
    <updateDate>2024-01-05T15:23:11Z</updateDate>
    <type>HR</type>
    <introducedDate>2023-05-29</introducedDate>
    <congress>118</congress>
    <committees>
      <item>
        <systemCode>hswm00</systemCode>
        <name>Ways and Means Committee</name>
        <chamber>House</chamber>
        <activities>
          <item><name>Discharged from</name><date>2023-05-31T00:00:00Z</date></item>
        </activities>
      </item>
      <item>
        <systemCode>hsbu00</systemCode>
        <name>Budget Committee</name>
        <chamber>House</chamber>
        <activities>
          <item><name>Referred to</name><date>2023-05-29T00:00:00Z</date></item>
        </activities>
      </item>
    </committees>
    <relatedBills>
      <item><number>1234</number><type>S</type><congress>118</congress></item>
    </relatedBills>
    <actions>
      <item><actionDate>2023-06-03</actionDate><text>Became Public Law No: 118-5.</text></item>
      <item><actionDate>2023-05-29</actionDate><text>Introduced in House</text></item>
    </actions>
    <sponsors>
      <item>
        <bioguideId>M001156</bioguideId>
        <fullName>Rep. McCarthy, Kevin [R-CA-20]</fullName>
        <firstName>Kevin</firstName>
        <lastName>McCarthy</lastName>
        <party>R</party>
        <state>CA</state>
      </item>
    </sponsors>
    <cosponsors>
      <item>
        <bioguideId>A000001</bioguideId>
        <party>R</party>
        <state>TX</state>
        <isOriginalCosponsor>True</isOriginalCosponsor>
      </item>
      <item>
        <bioguideId>B000002</bioguideId>
        <party>D</party>
        <state>NY</state>
        <isOriginalCosponsor>False</isOriginalCosponsor>
      </item>
    </cosponsors>
    <policyArea>
      <name>Economics and Public Finance</name>
    </policyArea>
    <title>Fiscal Responsibility Act of 2023</title>
    <latestAction>
      <actionDate>2023-06-03</actionDate>
      <text>Became Public Law No: 118-5.</text>
    </latestAction>
    <laws>
      <item><type>Public Law</type><number>118-5</number></item>
    </laws>
  </bill>
</billStatus>"""

rec = download_bills_bulk.parse_billstatus_xml(BILLSTATUS_XML)
check("bulk parse: congress/type/number",
      rec["congress"] == 118 and rec["bill_type"] == "hr" and rec["number"] == "3746")
check("bulk parse: introduced date and title",
      rec["introduced_date"] == "2023-05-29"
      and rec["title"] == "Fiscal Responsibility Act of 2023")
check("bulk parse: sponsor fields",
      rec["sponsor_party"] == "R" and rec["sponsor_state"] == "CA"
      and rec["sponsor_last_name"] == "McCarthy")
check("bulk parse: policy area",
      rec["policy_area"] == "Economics and Public Finance")
check("bulk parse: keeps only original cosponsors (1 of 2)",
      rec["n_original_cosponsors"] == 1
      and rec["original_cosponsors"] == [{"party": "R", "state": "TX"}])
check("bulk parse: primary committee is the 'Referred to' one, not list order",
      rec["primary_committee"] == "Budget Committee")
check("bulk parse: latest action from latestAction element",
      rec["latest_action_text"] == "Became Public Law No: 118-5."
      and rec["latest_action_date"] == "2023-06-03")
check("bulk parse: laws array parsed",
      rec["laws"] == [{"type": "Public Law", "number": "118-5"}])
check("bulk parse: became_law True via shared is_enacted",
      rec["became_law"] is True)
check("bulk parse: has_companion_bill from relatedBills",
      rec["has_companion_bill"] is True)

# a dead bill: no laws, no latestAction element (older-schema fallback path)
DEAD_XML = BILLSTATUS_XML.replace(
    b"""    <latestAction>
      <actionDate>2023-06-03</actionDate>
      <text>Became Public Law No: 118-5.</text>
    </latestAction>
    <laws>
      <item><type>Public Law</type><number>118-5</number></item>
    </laws>
""", b"").replace(
    b"<actionDate>2023-06-03</actionDate><text>Became Public Law No: 118-5.</text>",
    b"<actionDate>2023-06-03</actionDate><text>Referred to the Committee on the Budget.</text>")
dead = download_bills_bulk.parse_billstatus_xml(DEAD_XML)
check("bulk parse: dead bill without latestAction element -> newest action via fallback",
      dead["latest_action_text"] == "Referred to the Committee on the Budget."
      and dead["latest_action_date"] == "2023-06-03")
check("bulk parse: dead bill -> became_law False",
      dead["became_law"] is False)

check("bulk parse: document without a <bill> element returns None",
      download_bills_bulk.parse_billstatus_xml(b"<billStatus></billStatus>") is None)

# parse_zip: real zipfile in memory through the real function
zip_buf = io.BytesIO()
with zipfile.ZipFile(zip_buf, "w") as zf:
    zf.writestr("BILLSTATUS-118hr3746.xml", BILLSTATUS_XML)
    zf.writestr("BILLSTATUS-118hr9999.xml", b"not xml at all")
    zf.writestr("readme.txt", b"ignore me")
with tempfile.TemporaryDirectory() as d:
    zp = Path(d) / "test.zip"
    zp.write_bytes(zip_buf.getvalue())
    with patch("builtins.print"):
        zrecs = list(download_bills_bulk.parse_zip(zp, "hr"))
    check("parse_zip: yields the good record, skips non-XML member and bad XML",
          len(zrecs) == 1 and zrecs[0]["number"] == "3746")
    with patch("builtins.print"):
        wrong_type = list(download_bills_bulk.parse_zip(zp, "s"))
    check("parse_zip: skips records whose bill_type doesn't match the zip's type",
          wrong_type == [])

print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))
sys.exit(1 if FAIL else 0)
