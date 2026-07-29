"""Offline verification for the merged project. Runs the ACTUAL pipeline
functions against mock responses shaped like the real loc.gov and NYT
outputs, plus the NBER scoring logic. No network or API keys needed.

Run before spending any API budget:  python test_offline.py
"""
import json
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

# The pipeline scripts resolve data/ relative to the working directory, so the
# suite has to run from election_arm/. Do it here instead of requiring the
# caller to cd, matching how the suites on main behave: runnable from anywhere.
os.chdir(Path(__file__).resolve().parent)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import download_loc
import download_nyt
import extract_gpt
import extract_predictions
import strip_for_export
from analyze_economy import load_recessions, state_at

REAL_DIR = Path(__file__).resolve().parent
# Snapshot of the committed provenance log, to prove the suite leaves it alone.
# Its real entries include the phrases the fixtures below reuse, so only a
# before/after comparison can tell a test write from a genuine one.
SEARCH_LOG = REAL_DIR / "data/search_log.csv"
SEARCH_LOG_BEFORE = SEARCH_LOG.read_text() if SEARCH_LOG.exists() else None


@contextmanager
def sandbox_cwd():
    """Run a block in a throwaway working directory.

    The downloaders append to data/search_log.csv, which is a COMMITTED
    provenance artifact ("we sampled, we didn't cherry-pick"). A test run must
    not write fake 1948/2008 searches into it. The read-only CSVs the scripts
    look up are copied in so they still resolve."""
    with TemporaryDirectory() as d:
        (Path(d) / "data").mkdir()
        for name in ("windows_economy.csv", "nber_recessions.csv"):
            src = REAL_DIR / "data" / name
            if src.exists():
                shutil.copy(src, Path(d) / "data" / name)
        os.chdir(d)
        try:
            yield Path(d)
        finally:
            os.chdir(REAL_DIR)


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

with sandbox_cwd():
    with patch.object(download_loc, "get_json", side_effect=fake_get_json):
        results = list(download_loc.search_pages("elections", "1948",
                                                 "will be elected",
                                                 "1948-09-01", "1948-11-02"))
        check("search_pages returns 2 results", len(results) == 2)
        rec = download_loc.fetch_page_detail(results[0])
        check("lccn unwrapped from list", rec["lccn"] == "sn92070146")
        check("full_text found via recursion",
              "TRUMAN WILL BE ELECTED" in rec["ocr_text"])
        check("rejects non-loc id",
              download_loc.fetch_page_detail({"id": "http://example.com/x"}) is None)
    log = Path("data/search_log.csv")
    check("search_log.csv written with hits",
          log.exists() and "will be elected" in log.read_text())


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

with sandbox_cwd(), \
     patch.object(download_nyt, "get_json", side_effect=fake_nyt_get), \
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

# ---------------------------------------------------------------------------
# The ProQuest extractor emits main's label schema so main's scorer, adapter and
# gold-standard eval can read its output. These checks pin the properties that
# docs/SCORING.md makes non-negotiable -- they are the difference between a
# measurement and a laundered opinion, and none of them fail loudly on their own.
print("\n[5] extract_gpt.py (ProQuest arm, schema v2)")

PAGE = {"page_id": "pq1", "source": "proquest", "window": "crash_1929",
        "window_kind": "crisis", "newspaper_title": "The New York Times",
        "date": "1929-12-16", "state": None,
        "ocr_text": ("Business will improve before long, said Mr. Mitchell. "
                     "Steel output is climbing today. "
                     "Wheat prices may fall for years to come.")}

# --- no hindsight reaches the prompt ---
prompt = extract_gpt.page_prompts(PAGE)[0]
check("window id is NOT in the prompt (outcome leakage)",
      "crash_1929" not in prompt and "crisis" not in prompt)
check("newspaper and date ARE in the prompt (knowable at print time)",
      "The New York Times" in prompt and "1929-12-16" in prompt)
check("prompt never asks the model for recession/expansion",
      "recession\" or" not in prompt and "predicted_state_at_horizon" not in prompt)
check("prompt forbids guessing a horizon",
      '"vague"' in prompt and "use 6 if unstated" not in prompt)

REPLY = json.dumps([
    {"quote": "Business will improve before long", "topic": "general_business",
     "direction": "improve", "price_direction": "na",
     "unemployment_direction": "na", "horizon_months": "vague",
     "confidence": "assertive", "voice": "expert",
     "speaker_name": "Mr. Mitchell", "scope": "national",
     "is_quoted_forecaster": True, "current_state": "na"},
    {"quote": "Wheat prices may fall for years to come", "topic": "prices",
     "direction": "worsen", "price_direction": "down",
     "unemployment_direction": "na", "horizon_months": "vague",
     "confidence": "hedged", "voice": "journalist", "speaker_name": "na",
     "scope": "industry", "is_quoted_forecaster": False, "current_state": "na"},
    # Fluent, plausible, and nowhere on the page. Must not survive.
    {"quote": "The Reserve Board will abolish every tariff before the autumn",
     "topic": "other", "direction": "improve", "price_direction": "na",
     "unemployment_direction": "na", "horizon_months": 6,
     "confidence": "assertive", "voice": "official", "speaker_name": "na",
     "scope": "national", "is_quoted_forecaster": True, "current_state": "na"},
])

claims, dropped = extract_gpt.assemble(PAGE, [REPLY])
check("grounded claims kept, invented quote dropped",
      len(claims) == 2 and dropped == 1)
check("same claim from two overlapping chunks is deduplicated",
      len(extract_gpt.assemble(PAGE, [REPLY, REPLY])[0]) == 2)
check("fenced JSON still parses",
      len(extract_gpt.parse_claims("```json\n" + REPLY + "\n```")) == 3)
check("malformed reply yields []", extract_gpt.parse_claims("not json") == [])

# --- horizon honesty: the RIGID stratum must survive the export ---
check("horizon read from the quote's own time language",
      claims[0]["horizon_hint"] == "inferred_short"      # "before long"
      and claims[1]["horizon_hint"] == "inferred_long")  # "years to come"
check("a stated horizon is marked stated",
      extract_gpt.horizon_hint({"horizon_months": 6, "quote": "x"}) == "stated")
check("no time language at all falls back to default, not a guess",
      extract_gpt.horizon_hint(
          {"horizon_months": "vague", "quote": "business will improve"}) == "default")

# --- quote-derived MODEL features must survive the export ---
# main's claim_features() builds c_len and c_has_number straight off the quote
# text. ProQuest rows lose the quote, so unless these are precomputed in-VM the
# two features come out 0/0 -- not missing, but wrong in a way that is perfectly
# correlated with the source, i.e. a free "this row is ProQuest" signal.
# Replicating main's arithmetic exactly is the whole point, so pin it.
qf = extract_gpt.quote_features({"quote": "Trade will rise 3 per cent by spring"})
check("quote word count matches main's q.str.split().apply(len)",
      qf["quote_n_words"] == len("Trade will rise 3 per cent by spring".split()))
check("quote digit flag matches main's q.str.contains(r'\\d')",
      qf["quote_has_number"] == 1
      and extract_gpt.quote_features({"quote": "Trade will rise"})[
          "quote_has_number"] == 0)
check("word count is exported UNCLIPPED (main applies the 80 clip itself)",
      extract_gpt.quote_features({"quote": "w " * 100})["quote_n_words"] == 100)
check("quote features are attached to every claim",
      all("quote_n_words" in c and "quote_has_number" in c for c in claims))
check("quote features survive the strip, the quote does not",
      not (strip_for_export.DROP_FIELDS
           & {"quote_n_words", "quote_has_number", "horizon_hint"}))

# --- metadata rides on the record, never through the model ---
check("window metadata attached to the output record",
      claims[0]["window"] == "crash_1929" and claims[0]["window_kind"] == "crisis")
check("records are stamped with the schema version",
      all(c["schema_version"] == extract_gpt.SCHEMA_VERSION for c in claims))
check("model is not asked to state the outcome vocabulary",
      "predicted_state_at_horizon" not in claims[0])

# --- a truncated reply is a FAILURE, never a silent empty page ---
# A JSON array cut off mid-object parses to nothing. Banking that as "no
# predictions on this page" deflates the claim count on exactly the densest
# pages, and nothing downstream can tell it apart from a genuinely empty one.
class _Truncating:
    """Fake proxy client: truncates until max_tokens reaches `unlock`."""

    def __init__(self, unlock):
        self.unlock = unlock
        self.budgets = []

    @property
    def chat(self):
        return SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, model, max_tokens, temperature, messages):
        self.budgets.append(max_tokens)
        cut = max_tokens < self.unlock
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content='[{"quote":"Business will' if cut
                                    else REPLY),
            finish_reason="length" if cut else "stop")])

extract_gpt._ADAPT.clear()
recov = _Truncating(unlock=3200)
got = extract_gpt.call_model(recov, "fake-a", "prompt")
check("truncated reply retries with a bigger token budget",
      got is not None and recov.budgets == [1600, 3200])

extract_gpt._ADAPT.clear()
stuck = _Truncating(unlock=10 ** 9)
check("still truncated at the cap -> None (a failure), not parseable-to-empty",
      extract_gpt.call_model(stuck, "fake-b", "prompt") is None
      and stuck.budgets[-1] == extract_gpt.MAX_TOKENS_CAP)

extract_gpt._ADAPT.clear()
with TemporaryDirectory() as d:
    outp = Path(d) / "o.jsonl"
    extract_gpt.run([dict(PAGE)], _Truncating(unlock=10 ** 9), "fake-c",
                    outp, set())
    check("a persistently truncated page is left unmarked, NOT recorded empty",
          outp.read_text().strip() == "")

    # An unparseable source record should not cost an API call at all.
    outp2 = Path(d) / "o2.jsonl"
    never = _Truncating(unlock=0)          # would succeed if called
    extract_gpt.run([{**PAGE, "ocr_text": "   "}], never, "fake-d", outp2, set())
    rec = json.loads(outp2.read_text().strip())
    check("a page with no source text is recorded without an API call",
          not never.budgets and rec["no_predictions"] and rec["empty_source_text"])

# --- schema-mixing guard ---
with patch.object(Path, "exists", return_value=True), \
     patch("builtins.open", create=True) as mock_open:
    mock_open.return_value.__enter__.return_value = [
        json.dumps({"page_id": "old", "claim_text": "t"}),          # v1
        json.dumps({"page_id": "new", "schema_version": 2}),        # v2
    ]
    done, stale = extract_gpt.load_done_ids(Path("fake.jsonl"))
check("v1 records are not counted as done; v2 records are",
      done == {"new"} and stale == 1)
try:
    extract_gpt.check_schema(Path("fake.jsonl"), stale=1, allow_mixed=False)
    mixed_refused = False
except SystemExit:
    mixed_refused = True
check("refuses to append v2 records to a file holding v1", mixed_refused)

# ---------------------------------------------------------------------------
# ProQuest forbids exporting article text. A leak here is a licence breach, not
# a crash, so it has to be caught by a test rather than by noticing.
print("\n[6] strip_for_export.py (nothing text-bearing leaves the VM)")

check("the verbatim quote is dropped", "quote" in strip_for_export.DROP_FIELDS)
check("v1's claim_text is still dropped too",
      "claim_text" in strip_for_export.DROP_FIELDS)
check("labels the scorers need are NOT dropped",
      not (strip_for_export.DROP_FIELDS & {
          "topic", "direction", "price_direction", "unemployment_direction",
          "scope", "horizon_months", "horizon_hint", "confidence", "voice",
          "date", "window", "window_kind", "source"}))

stripped = {k: v for k, v in claims[0].items()
            if k not in strip_for_export.DROP_FIELDS}
check("horizon_hint survives the strip (main's scorer can't re-derive it)",
      stripped.get("horizon_hint") == "inferred_short")
try:
    strip_for_export.check_no_prose(stripped, 1)
    clean = True
except SystemExit:
    clean = False
check("a real stripped record passes the prose tripwire", clean)
try:
    strip_for_export.check_no_prose({**stripped, "rationale": "word " * 100}, 1)
    caught = False
except SystemExit:
    caught = True
check("an unlisted prose field trips the tripwire", caught)

# Last, so it covers every block above: the suite must be hermetic.
check("the committed search_log.csv is not touched by a test run",
      SEARCH_LOG_BEFORE is None or SEARCH_LOG.read_text() == SEARCH_LOG_BEFORE)

print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))
sys.exit(1 if FAIL else 0)
