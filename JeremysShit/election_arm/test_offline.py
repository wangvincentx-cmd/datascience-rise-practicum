"""Offline verification for the merged project. Runs the ACTUAL pipeline
functions against mock responses shaped like the real loc.gov and NYT
outputs, plus the NBER scoring logic. No network or API keys needed.

Run before spending any API budget:  python test_offline.py
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import download_loc
import download_nyt
import extract_predictions
from analyze_economy import load_recessions, state_at

FAIL = []
def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        FAIL.append(name)

# ---------------------------------------------------------------------------
print("\n[1] download_loc.py")
SEARCH_RESP = {
    "results": [
        {"id": "http://www.loc.gov/resource/sn92070146/1948-10-20/ed-1/?sp=6",
         "original_format": ["newspaper"]},
        {"id": "http://www.loc.gov/resource/sn83045462/1948-10-25/ed-1/?sp=3",
         "original_format": ["newspaper"]},
    ],
    "pagination": {"next": None, "of": 2},
}
RESOURCE_RESP = {
    "item": {"number_lccn": ["sn92070146"],
             "newspaper_title": ["imperial valley press (el centro, calif.) 1907-current"],
             "date": ["1948-10-20"], "location_state": ["california"],
             "location_city": ["el centro"]},
    "pagination": {"current": 6},
    "resource": {"full_text": "TRUMAN WILL BE ELECTED say local Democrats ..."},
}

def fake_get_json(url, params=None):
    return SEARCH_RESP if "collections/chronicling-america" in url else RESOURCE_RESP

with patch.object(download_loc, "get_json", side_effect=fake_get_json):
    results = list(download_loc.search_pages("elections", "1948",
                                             "will be elected",
                                             "1948-09-01", "1948-11-02"))
    check("search_pages returns 2 results", len(results) == 2)
    rec = download_loc.fetch_page_detail(results[0])
    check("lccn unwrapped from list", rec["lccn"] == "sn92070146")
    check("full_text found via recursion", "TRUMAN WILL BE ELECTED" in rec["ocr_text"])
    check("rejects non-loc id",
          download_loc.fetch_page_detail({"id": "http://example.com/x"}) is None)

log = Path("data/search_log.csv")
check("search_log.csv written with hits", log.exists() and "will be elected" in log.read_text())

econ_windows = download_loc.load_economy_windows()
check("economy windows load, pre-1963 only",
      "crash_1929" in econ_windows and "gfc_2008" not in econ_windows)

# ---------------------------------------------------------------------------
print("\n[2] download_nyt.py")
NYT_PAGE0 = {"response": {"meta": {"hits": 2}, "docs": [
    {"web_url": "https://nyt.com/2008/a",
     "headline": {"main": "Economists See Deep Recession Ahead"},
     "abstract": "Forecasters expect the downturn to worsen.",
     "lead_paragraph": "Forecasters expect the downturn to worsen into 2009.",
     "snippet": "Forecasters expect the downturn to worsen.",
     "pub_date": "2008-10-15T00:00:00Z", "type_of_material": "News",
     "section_name": "Business"},
    {"web_url": "https://nyt.com/2008/b", "headline": {"main": ""},
     "abstract": "", "lead_paragraph": "", "snippet": "",
     "pub_date": "2008-10-16T00:00:00Z"},
]}}
NYT_EMPTY = {"response": {"meta": {"hits": 2}, "docs": []}}
_calls = {"n": 0}
def fake_nyt_get(params):
    _calls["n"] += 1
    return NYT_PAGE0 if _calls["n"] == 1 else NYT_EMPTY

with patch.object(download_nyt, "get_json", side_effect=fake_nyt_get), \
     patch.object(download_nyt.time, "sleep", lambda s: None):
    docs = list(download_nyt.search_phrase("KEY", "economy", "gfc_2008",
                                           '"recession likely"',
                                           "20080901", "20090331"))
    check("NYT search yields 2 docs", len(docs) == 2)
    text = download_nyt.combine_text(docs[0])
    check("combine_text merges headline+lead",
          "Deep Recession Ahead" in text and "into 2009" in text)
    check("combine_text dedupes abstract/snippet",
          text.count("Forecasters expect the downturn to worsen.") == 1)
    check("combine_text empty for empty doc", download_nyt.combine_text(docs[1]) == "")
    check("NYT economy windows include post-1963",
          "gfc_2008" in download_nyt.load_economy_windows())

# ---------------------------------------------------------------------------
print("\n[3] extract_predictions.py, both arms")
class FakeBlock:
    def __init__(self, t): self.text = t
class FakeMsg:
    def __init__(self, t): self.content = [FakeBlock(t)]
class FakeClient:
    def __init__(self, t): self._t = t; self.messages = self
    def create(self, **kw): return FakeMsg(self._t)

erec = {"page_id": "p1", "source": "loc", "window": "1948", "cycle": 1948,
        "newspaper_title": "Test Gazette", "date": "1948-10-20",
        "state": "california", "ocr_text": "Truman will be elected."}
ereply = ('[{"claim_text":"Truman will be elected","predicted_winner":"Truman",'
          '"scope":"national","state":null,"source_type":"editorial_opinion",'
          '"hedged":false,"attributed_to":null}]')
eclaims = extract_predictions.extract_from_page(FakeClient(ereply), erec, "elections")
check("elections claim parsed + metadata merged",
      len(eclaims) == 1 and eclaims[0]["arm"] == "elections"
      and eclaims[0]["source"] == "loc")

crec = {"page_id": "p2", "source": "nyt", "window": "gfc_2008",
        "window_kind": "crisis", "newspaper_title": "The New York Times",
        "date": "2008-10-15", "state": None,
        "ocr_text": "Economists see deep recession ahead."}
creply = ("```json\n"
          '[{"claim_text":"Deep recession ahead","predicted_direction":"worsen",'
          '"predicted_state_at_horizon":"recession","horizon_months":6,'
          '"voice":"quoted_banker_or_economist","hedged":false,'
          '"attributed_to":"economists"}]\n```')
cclaims = extract_predictions.extract_from_page(FakeClient(creply), crec, "economy")
check("economy claim parsed from fenced JSON",
      len(cclaims) == 1 and cclaims[0]["predicted_state_at_horizon"] == "recession"
      and cclaims[0]["window_kind"] == "crisis")
check("malformed reply yields []",
      extract_predictions.extract_from_page(FakeClient("not json"), crec, "economy") == [])

# ---------------------------------------------------------------------------
print("\n[4] NBER scoring (analyze_economy.py)")
rec = load_recessions()
check("Oct 1929 claim +6m lands in recession",
      state_at(pd.Period("1930-04", freq="M"), rec) == "recession")
check("Oct 1945 postwar: +6m is expansion (the scare that never came)",
      state_at(pd.Period("1946-04", freq="M"), rec) == "expansion")
check("NBER convention: peak month itself is expansion",
      state_at(pd.Period("1929-08", freq="M"), rec) == "expansion")
check("month after peak is recession",
      state_at(pd.Period("1929-09", freq="M"), rec) == "recession")
check("trough month is still recession",
      state_at(pd.Period("1933-03", freq="M"), rec) == "recession")
check("month after trough is expansion",
      state_at(pd.Period("1933-04", freq="M"), rec) == "expansion")
check("1987 crash +12m: no recession (the negative case)",
      state_at(pd.Period("1988-10", freq="M"), rec) == "expansion")

print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))
sys.exit(1 if FAIL else 0)
