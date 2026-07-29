"""
LLM extraction of economy-arm predictions via TDM Studio's built-in GPT proxy.

Runs INSIDE the ProQuest TDM Studio workbench. The VM has no general internet,
but ProQuest ships a proxied OpenAI endpoint (see their sample notebook
'GPT_Batch_Processing.ipynb'): the standard `openai` SDK pointed at a ProQuest
`base_url`, authenticated with a key file ProQuest drops in the workspace.

To avoid transcribing ProQuest's long base_url / key path by hand, this script
auto-discovers them from the sample notebook you already exported to
gpt_sample.txt:

    jupyter nbconvert --to script --stdout \
      ".../ProQuest TDM Studio Samples/GPT_Batch_Processing.ipynb" > gpt_sample.txt

Override with --base-url / --key-file / --model if discovery misses.


WHAT THIS EMITS (schema v2) -- and why it changed
-------------------------------------------------
v1 of this script had its own private label vocabulary
(`claim_text` / `predicted_direction` / `predicted_state_at_horizon` / `hedged`).
Nothing on main can read that. This version emits **main's extraction schema**
-- the one `src/extract_llm.py` produces and the one everything on main is built
to consume -- so ProQuest claims can be checked by main's machinery instead of
by a parallel set of rules:

    src/score_predictions.py           reads topic, direction, price_direction,
                                       unemployment_direction, scope,
                                       horizon_months, date
    src/adapt_proquest_claims.py       reads scope, direction, confidence and
                                       DERIVES predicted_state_at_horizon +
                                       hedged for analyze_economy.py
    validation/gold_extraction/        scores topic, direction, horizon_months,
      eval_extraction.py               confidence, voice against the gold pages

That last one is the point of the exercise: with a shared vocabulary, the same
gold standard that measured gemini/gpt-4.1 can measure gpt-4o-mini (`--pages`
mode below), so this arm's numbers slot into main's table instead of being
reported as an unmeasured different thing.

Four things v1 did that docs/SCORING.md forbids, fixed here:

1. **Outcome leakage.** v1 put `Window: {window}` in the model's context.
   Window ids state the OUTCOME ("crash_1929", "calm_1955"), so the model was
   told what happened before it labelled which way a forecast pointed. Only
   newspaper and date are passed now -- both genuinely known to whoever set the
   page. Window is still attached to the OUTPUT record as metadata (the scorer
   needs it); it never reaches the prompt. This is the exact bug main's
   extract_llm.py header calls out.
2. **The model was asked for the outcome vocabulary.** v1 asked it for
   `predicted_state_at_horizon: recession | expansion` -- the LLM stating a
   business-cycle verdict. SCORING.md's one rule is that the LLM says what was
   predicted and real data says whether it happened. The model now reports only
   a direction; adapt_proquest_claims.py maps direction -> recession/expansion
   by a fixed table, deterministically.
3. **Manufactured horizons.** v1's prompt said "use 6 if unstated", which turns
   every vague forecast into a falsely precise one and inflates the RIGID
   stratum (the subset whose window comes from the claim rather than a default
   -- the only honest basis for the accuracy model). The schema is now
   6 | 12 | "vague", as on main.
4. **No hallucination guard.** v1 asked for an "OCR-corrected" claim_text, so a
   returned span could not be checked against the page at all. The schema now
   asks for a VERBATIM quote and `quote_is_grounded()` drops any quote whose
   tokens are not really on the page. This matters more here than on main: the
   in-VM proxy only offers gpt-4o-mini, the weakest extractor in the bake-off.

Also: pages are chunked with overlap instead of truncated at 12k chars (v1
silently discarded the tail of every long ProQuest full-text document).


EXPORT SAFETY -- read before changing the schema
------------------------------------------------
`quote` is VERBATIM ARTICLE TEXT and must never leave the VM. strip_for_export.py
drops it (DROP_FIELDS), and that coupling is load-bearing: if you add another
text-bearing field here, add it there too, in the same commit.

Because the quote is stripped at export, main's score_predictions.py cannot do
its own horizon inference on our records -- it reads the quote's time language
("soon", "for years") to decide short/long, and by then the quote is gone. So we
run that inference HERE, where the text still exists, and export the derived
label as `horizon_hint`. It is a one-token verdict, not text.


Usage (in the workbench):
  python extract_gpt.py --source proquest --window gfc_2008 --limit 10  # TEST FIRST
  python extract_gpt.py --source proquest --window gfc_2008

  # measure this model against main's gold standard (16 calls, see docstring)
  python extract_gpt.py --pages gold_pages.jsonl --out pred_gpt4omini_gold.jsonl

Start with --limit 10 and check ProQuest's usage page: they enforce daily and
per-minute LLM limits. Resume-safe: processed page_ids are skipped on rerun.
"""

import argparse
import json
import re
import time
from pathlib import Path

# NOTE: `openai` is imported lazily inside make_client(). The SDK exists in the
# TDM Studio VM but not on the Mac, and the Mac is where test_offline.py checks
# the pure functions (prompt building, parsing, grounding, horizon inference).
# A top-level import would make this whole module unimportable there.

# Bumped when the label vocabulary changes. Stamped on every record so a file
# can never silently mix vocabularies -- see load_done_ids().
SCHEMA_VERSION = 2

# A ProQuest full-text document can run well past 12k chars. Chunk with overlap
# so a forecast straddling a boundary still appears whole in one window;
# cross-chunk duplicates are removed by normalized quote text in assemble().
CHUNK_CHARS = 8000
CHUNK_OVERLAP = 500
MAX_TOKENS = 1600
# Ceiling for the adaptive bump in call_model() when a reply comes back
# truncated. Per-model and sticky, so a corpus of dense pages pays the discovery
# cost once.
MAX_TOKENS_CAP = 6000
_ADAPT = {}

MODEL_FALLBACK = "gpt-4o-mini"
REQUEST_DELAY = 0.5     # be gentle with ProQuest's shared proxy
MAX_RETRIES = 4

# Fraction of a returned quote's tokens that must appear in the source text for
# it to be accepted. Below 1.0 because models silently repair OCR
# ("mille"->"mills"), which is desirable; far enough above chance that an
# invented sentence fails. Same threshold as main's extract_llm.py.
QUOTE_MATCH_THRESHOLD = 0.65
TOKEN_RE = re.compile(r"[a-z0-9]+")


# The schema block and every label-definition rule below are kept WORD-FOR-WORD
# from main's src/extract_llm.py EXTRACTION_PROMPT. That is deliberate and is
# the whole point of this file: labels produced here are only comparable to
# main's gold standard (and poolable with main's corpus) if the definitions the
# model was given are the same ones. If main's prompt changes, re-copy it.
# Only the corpus framing differs -- ProQuest spans 1905-2020 and mixes clean
# NYT headline+lead text with OCR'd full pages, where main's is LOC OCR
# 1900-1963.
EXTRACTION_PROMPT = """You extract economic PREDICTIONS from American newspaper text printed between 1905 and 2020.

The text is either clean digitised article text (headline and opening paragraphs) or OCR of a full newspaper page. OCR text is noisy: columns interleave, words break across lines, letters are wrong ("tho" for "the", "busi ness" for "business"). Read through the noise. A full page carries many unrelated stories at once -- news, advertisements, fiction, sports, social notes -- and most of it is not a prediction.

Find every sentence that makes a FALSIFIABLE CLAIM ABOUT FUTURE ECONOMIC CONDITIONS: business conditions, prices, employment, markets, prosperity, recession or panic.

INCLUDE:
- Forecasts quoted from a named person (banker, economist, official, executive). The paper is the vehicle; the forecaster is the source.
- A real forecast printed next to advertising or under an unrelated headline.
- A headline that itself states a forecast.
- A forecast recoverable through OCR damage, when the words can be reconstructed with confidence.
- Forecasts about a region, an industry, or a foreign country, not only the national economy. Label them with `scope`; do not discard them.

EXCLUDE, even when economic words appear:
- ADVERTISEMENTS and promotional copy. Ad copy constantly uses future tense and economic vocabulary -- price claims, New Year "prosperity" greetings, "you will not buy them for less next year". None of it counts.
- FICTION. Serialized novels and humour sketches run on these pages and their dialogue can be full of debt, bankruptcy, wages and hard times. Invented speech by a character is never a prediction.
- REPRINTED items from "Twenty Years Ago" / "From Our Files" columns. They are future-tense but were written decades earlier, so they do not belong to this page's date.
- DESCRIPTIONS OF THE PRESENT OR PAST: "business is good today", "steel production is climbing", "the panic ruined us". This is the most common error. A report of current conditions is not a claim about future ones.
- RETROSPECTIVES ON FORECASTS THAT ALREADY FAILED. If the passage's point is that earlier predictions were WRONG ("how the prophets were mistaken"), it is not making that prediction. A forecast that is still live and merely doubted by the writer DOES count.
- EXPLICIT REFUSALS to forecast: "it is too early to say", "no one can know", "anyone who makes a positive statement must be a fool".
- CONDITIONALS with no committed direction: "if volume slips, that business is in trouble", "the bill would create jobs" (of a bill that has not passed).
- ANNOUNCEMENTS of a speech, meeting or report ABOUT the outlook. The outlook is the event's topic, not a forecast the sentence makes.
- SCHEDULES: a store opening, a construction timetable, a contract letting date.
- POLICY ADVOCACY: "the Reserve Board should", "Congress must".
- NON-ECONOMIC futures: elections, legislation, weather, sport, a person's health.
- STOCK TIPS about a single company, and investment arithmetic.
- Text too mangled by OCR to reconstruct.

Return ONLY a JSON array. No markdown fences, no commentary. Each element:
{{
  "quote": "the prediction, copied VERBATIM from the text including any OCR errors, max 60 words",
  "topic": "general_business" | "prices" | "employment" | "markets" | "other",
  "direction": "improve" | "worsen" | "no_change" | "unclear",
  "price_direction": "up" | "down" | "stable" | "na",
  "unemployment_direction": "up" | "down" | "stable" | "na",
  "horizon_months": 6 | 12 | "vague",
  "confidence": "assertive" | "hedged",
  "voice": "journalist" | "expert" | "official" | "layperson" | "unclear",
  "speaker_name": "personal name of the forecaster if stated or clearly implied, else \\"na\\"",
  "scope": "national" | "regional" | "foreign" | "industry",
  "is_quoted_forecaster": true or false,
  "current_state": "how the claim characterizes conditions NOW: \\"good\\" | \\"bad\\" | \\"mixed\\" | \\"na\\""
}}

Rules:
- "quote" MUST be copied verbatim from the text given to you. Do not clean up the OCR, do not paraphrase, do not merge separated sentences. A quote that does not appear in the text is discarded.
- direction: reassurance that conditions are sound or that fears are unfounded ("nothing in the outlook to cause uneasiness") is "improve", NOT "no_change". Use "no_change" only when the sentence explicitly says conditions hold flat. Use "unclear" only when it is genuinely a forecast whose direction cannot be read -- never as a default.
- For price claims, ask what the sentence implies for conditions OVERALL, using its own framing.
- horizon_months: 6 or 12 ONLY when the claim itself points at that horizon ("by spring", "next year"). If the claim does not say when, answer "vague". Do NOT guess a number -- a vague forecast labelled 6 becomes a falsely precise one downstream.
- confidence: judge the words, not the speaker's authority. "will", "is certain", "undoubtedly" = assertive; "may", "likely", "is expected", "we do not think" = hedged.
- current_state: how the claim characterizes the CURRENT economy it is forecasting from -- "good" (booming, sound), "bad" (depressed, in panic), "mixed", or "na" if it says nothing about present conditions. "Business is bad now but will recover" -> current_state "bad", direction "improve". This is the base the forecast starts from, not the forecast itself.
- voice: judge WHO is speaking, not what it is about.
- scope: WHICH economy the claim is about. "national" = the US economy overall. "regional" = a US state, city or region. "foreign" = a non-US economy (a forecast about Mexican exports or the Brazilian currency printed in a US paper is "foreign"). "industry" = one US industry or sector (steel, autos, farming, railroads). If a claim is about a foreign industry, use "foreign". This matters because only national claims can fairly be compared against national economic statistics.
- is_quoted_forecaster: true when the forecast is attributed to someone other than the newspaper itself -- a banker, economist, official, company, trade body or government report. false when it is the paper's own editorial voice or its reporter's own assessment. When in doubt, false.
- Two predictions in one sentence get two entries ONLY if they point in different directions or concern different topics.
- If the text contains no predictions, return []. Many pages genuinely contain none; returning [] is a correct answer, not a failure.

Newspaper: {newspaper}
Date: {date}

Text:
{text}"""

# NOTE: main's prompt also carries `conditional_on` and `reasoning`. They are
# omitted here on purpose -- both are free-text summaries that can come back as
# copied article phrasing, which cannot leave the VM, and neither is read by any
# scorer. main's own prompt flags their recall impact as UNVERIFIED. Nothing
# downstream misses them.


# --- horizon inference (must run in-VM, while the quote still exists) --------
# Copied verbatim from main's src/score_predictions.py. main runs these against
# the claim's quote at scoring time; our quotes are stripped before export, so
# we run them here and ship the verdict as `horizon_hint`. Keep in sync with
# main -- if these regexes drift, our RIGID stratum stops meaning what main's
# means.
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


# Same regex main's model code uses for its `c_has_number` feature.
NUM_RE = re.compile(r"\d")


def quote_features(claim):
    """The quote-derived MODEL features, computed while the quote still exists.

    main's claim_features() (src/model_hit.py and src/hit_predictor.py) builds
    two features straight off the quote text:

        c_len         = quote.split() word count, clipped to 80
        c_has_number  = whether the quote contains a digit

    Our quotes are stripped before export, so on a ProQuest row those would come
    out 0 and 0 -- not missing, but silently WRONG, and wrong in a way that is
    perfectly correlated with the source. Pooled with LOC claims, `c_len == 0`
    becomes a free "this row is ProQuest" indicator and the model learns the
    data source instead of the forecast. So both are computed here and exported
    as plain numbers (a word count and a flag carry no text).

    The word count is exported UNCLIPPED; the clip is main's modelling choice and
    stays on main, applied identically to both sources."""
    q = str(claim.get("quote", "") or "")
    return {"quote_n_words": len(q.split()),
            "quote_has_number": int(bool(NUM_RE.search(q)))}


def horizon_hint(claim):
    """main's horizon_basis, computed while the quote is still available.

    'stated' -- the claim itself named the horizon; 'inferred_short'/'_long' --
    read from the quote's own time language; 'default' -- nothing to go on, the
    scorer will apply its neutral default. Only non-'default' claims belong in
    the RIGID subset, so this is what keeps that subset honest after the text is
    stripped for export."""
    if str(claim.get("horizon_months", "")).strip() in ("6", "12", "24"):
        return "stated"
    q = str(claim.get("quote", ""))
    if LONG_HORIZON.search(q):
        return "inferred_long"
    if SHORT_HORIZON.search(q):
        return "inferred_short"
    return "default"


def discover_config(sample_path):
    """Pull base_url, key-file path, and model out of ProQuest's GPT sample export."""
    text = Path(sample_path).read_text(errors="ignore") if Path(sample_path).exists() else ""
    base_url = _search(text, r'base_url\s*=\s*["\']([^"\']+)["\']')
    model = _search(text, r'model\s*=\s*["\']([^"\']+)["\']')
    # Prefer an open() whose path looks like a key/token/credential file.
    opens = re.findall(r'open\(\s*["\']([^"\']+)["\']', text)
    key_path = next((p for p in opens
                     if re.search(r'key|token|cred|secret', p, re.I)), None)
    if key_path is None and opens:
        key_path = opens[0]
    return base_url, key_path, model


def _search(text, pattern):
    m = re.search(pattern, text)
    return m.group(1) if m else None


def make_client(args):
    base_url, key_path, model = discover_config(args.sample)
    base_url = args.base_url or base_url
    key_path = args.key_file or key_path
    model = args.model or model or MODEL_FALLBACK
    if not base_url:
        raise SystemExit(
            "Could not find base_url. Re-export the sample:\n"
            '  jupyter nbconvert --to script --stdout '
            '".../GPT_Batch_Processing.ipynb" > gpt_sample.txt\n'
            "or pass --base-url and --key-file explicitly.")
    if not key_path or not Path(key_path).exists():
        raise SystemExit(f"Key file not found (discovered: {key_path!r}). "
                         f"Pass --key-file with the path the sample opens.")
    api_key = Path(key_path).read_text().strip()
    print(f"using proxy base_url={base_url}\n  key_file={key_path}\n  model={model}")
    from openai import OpenAI          # lazy: see note at the imports
    return OpenAI(api_key=api_key, base_url=base_url), model


class RateLimitReached(Exception):
    """The proxy's daily/rate quota is exhausted. Retrying won't help until it
    resets, so we stop cleanly and resume on the next run instead of grinding."""


TPM_WAIT = 65   # per-minute token/request window resets each minute; wait it out


def _is_daily_cap(err):
    """True ONLY for the per-DAY cost cap (permanent until the daily reset).
    A per-minute token/request limit must NOT match here -- it is transient and
    should be waited out and retried, not misread as end-of-day. The daily cap
    surfaces as 'Application cost/day rate exceeded'."""
    s = str(err).lower()
    return any(k in s for k in ("day rate", "cost/day", "per day", "per-day",
                                "daily"))


def _is_transient_rate_limit(err):
    """True for a transient throttle (per-minute token/request cap) that clears
    on its own -- wait ~60s and retry rather than giving up on the article."""
    s = str(err).lower()
    return any(k in s for k in ("token/minute", "per minute", "per-minute",
                                "requests per", "rate limit", "ratelimit",
                                "too many requests", "429"))


def chunks(text, size=CHUNK_CHARS, overlap=CHUNK_OVERLAP):
    if len(text) <= size:
        return [text]
    out, start = [], 0
    while start < len(text):
        out.append(text[start:start + size])
        if start + size >= len(text):
            break
        start += size - overlap
    return out


def page_prompts(record):
    """One prompt per text window.

    NOTE the context block: newspaper and date ONLY. `window` is deliberately
    absent -- window ids name the outcome, and passing one would tell the model
    what happened before it labels the forecast. See docs/SCORING.md.
    """
    newspaper = (record.get("publisher") or record.get("newspaper_title")
                 or "unknown")
    return [EXTRACTION_PROMPT.format(newspaper=newspaper,
                                     date=record.get("date") or "unknown",
                                     text=part)
            for part in chunks(record.get("ocr_text") or "")]


def call_model(client, model, prompt):
    """Return the model's raw text, or None on a non-fatal failure after retries.
    Raises RateLimitReached only on the DAILY cost cap (caller should stop). A
    per-minute limit is waited out in-place so the run keeps going."""
    adapt = _ADAPT.setdefault(model, {"token_budget": MAX_TOKENS})
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model, max_tokens=adapt["token_budget"], temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            choice = resp.choices[0]
            content = (choice.message.content or "").strip()
            if choice.finish_reason == "length":
                # A JSON array cut off mid-object parses to NOTHING, and the
                # caller would then bank that as a genuine "no predictions on
                # this page". That false empty is the same class of bug as
                # recording a failed call as an empty -- it silently deflates
                # the claim count on exactly the densest pages. So: raise the
                # budget and retry, and if it is STILL truncated, report failure
                # rather than emptiness. The budget is per-model and sticks, so
                # the cost is paid once, not once per page.
                budget = adapt["token_budget"]
                if budget < MAX_TOKENS_CAP:
                    adapt["token_budget"] = min(budget * 2, MAX_TOKENS_CAP)
                    print(f"  truncated at {budget} tokens -- raising budget to "
                          f"{adapt['token_budget']} and retrying")
                    continue
                print(f"  STILL truncated at the {budget}-token cap; treating as "
                      f"a failure, NOT as an empty page")
                return None
            return content
        except Exception as e:
            # Broad on purpose: the proxy surfaces failures as OpenAIError, as
            # bare httpx transport errors, and occasionally as neither. The
            # classification below is on the message text, so it works for all
            # of them -- and an unclassified error just retries with backoff.
            if _is_daily_cap(e):
                raise RateLimitReached(str(e))   # permanent until reset; stop now
            if _is_transient_rate_limit(e):
                # Transient per-minute cap: wait out the window and retry. Bounded
                # by MAX_RETRIES so a (surprising) persistent limit can't loop
                # forever; an exhausted article is left unmarked for the next run.
                print(f"  per-minute rate limit; waiting {TPM_WAIT}s and retrying")
                time.sleep(TPM_WAIT)
                continue
            wait = 2 ** attempt * 5
            print(f"  OpenAI error ({e}); retry in {wait}s")
            time.sleep(wait)
    return None


def parse_claims(raw):
    """Strip markdown fences and parse the JSON array. Models wrap output in
    ```json despite instructions often enough that handling it is cheaper than
    retrying."""
    raw = (raw or "").strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    if not raw:
        return []
    try:
        claims = json.loads(raw)
    except json.JSONDecodeError:
        # Salvage the first well-formed array if the model added commentary.
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if not m:
            return []
        try:
            claims = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    if isinstance(claims, dict):
        claims = claims.get("claims", [])
    return claims if isinstance(claims, list) else []


def quote_is_grounded(quote, text_tokens, threshold=QUOTE_MATCH_THRESHOLD):
    """Fraction of the quote's tokens that appear in the source text.

    The guard that matters: without it, a model under instruction to return
    verbatim spans will still occasionally produce a fluent, plausible, entirely
    invented forecast -- indistinguishable from a real one downstream, and it
    would be scored against real NBER data as if a newspaper had printed it."""
    q = TOKEN_RE.findall((quote or "").lower())
    if len(q) < 4:
        return False
    return sum(1 for t in q if t in text_tokens) / len(q) >= threshold


def assemble(record, raw_responses):
    """Raw model output -> deduplicated, grounded, metadata-stamped claims."""
    text_tokens = set(TOKEN_RE.findall((record.get("ocr_text") or "").lower()))
    seen, claims, dropped = set(), [], 0
    for raw in raw_responses:
        for c in parse_claims(raw):
            if not isinstance(c, dict) or not c.get("quote"):
                continue
            key = " ".join(TOKEN_RE.findall(c["quote"].lower()))
            if not key or key in seen:
                continue
            if not quote_is_grounded(c["quote"], text_tokens):
                dropped += 1
                continue
            seen.add(key)
            # Everything derived from the quote must be computed HERE -- the
            # text does not survive strip_for_export.py.
            c["horizon_hint"] = horizon_hint(c)
            c.update(quote_features(c))
            # Window/window_kind ride along as RECORD metadata because
            # analyze_economy.py groups by them. They were never shown to the
            # model -- see page_prompts().
            c.update({
                "schema_version": SCHEMA_VERSION,
                "page_id": record["page_id"],
                "source": record.get("source"),
                "arm": "economy",
                "window": record.get("window"),
                "window_kind": record.get("window_kind"),
                "cycle": record.get("cycle"),
                "publisher": (record.get("publisher")
                              or record.get("newspaper_title")),
                "newspaper_title": record.get("newspaper_title"),
                "lccn": record.get("lccn"),
                "date": record.get("date"),
                "publisher_state": record.get("state"),
                "source_text_type": record.get("type_of_material") or "article",
            })
            claims.append(c)
    return claims, dropped


def load_done_ids(out_path):
    """page_ids already extracted AT THE CURRENT SCHEMA VERSION.

    Records written by an older schema do not count as done: their labels use a
    vocabulary nothing downstream can read, so a rerun must redo those pages.
    Returns (done_ids, n_stale)."""
    done, stale = set(), 0
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "page_id" not in r:
                    continue
                if r.get("schema_version") == SCHEMA_VERSION:
                    done.add(r["page_id"])
                else:
                    stale += 1
    return done, stale


def check_schema(out_path, stale, allow_mixed):
    """Refuse to append v2 records to a file that still holds v1 ones.

    Mixing is the quiet failure mode: analyze_economy.py globs every
    pred_*_economy_*.jsonl and would silently average two different label
    vocabularies. Re-extracting costs daily quota, which is this arm's binding
    constraint, so the call is the user's -- we stop and explain rather than
    deleting or auto-redoing thousands of records."""
    if not stale or allow_mixed:
        if stale and allow_mixed:
            print(f"  WARNING: {stale} v1 records kept in {out_path.name} "
                  f"(--allow-mixed). They are NOT readable by main's scorer.")
        return
    raise SystemExit(
        f"\n*** {out_path} holds {stale} records from an older schema.\n"
        f"*** This script now emits schema v{SCHEMA_VERSION} (main's vocabulary);\n"
        f"*** appending would mix two label sets in one file that\n"
        f"*** analyze_economy.py globs together.\n\n"
        f"Pick one:\n"
        f"  1. Re-extract this window cleanly (costs quota):\n"
        f"       mv {out_path} {out_path}.v1.bak\n"
        f"     then re-run this command.\n"
        f"  2. Keep the v1 records and add v2 alongside (NOT recommended --\n"
        f"     only the v2 lines are scorable):  --allow-mixed\n")


def run(records, client, model, out_path, done, limit=None):
    processed = with_pred = total_claims = total_dropped = 0
    with open(out_path, "a") as out:
        for record in records:
            if record["page_id"] in done:
                continue
            if not (record.get("ocr_text") or "").strip():
                # A parse that yielded no text. Calling the model on "" would
                # spend quota to be told there are no predictions in nothing.
                out.write(json.dumps({"page_id": record["page_id"],
                                      "schema_version": SCHEMA_VERSION,
                                      "window": record.get("window"),
                                      "no_predictions": True,
                                      "empty_source_text": True}) + "\n")
                done.add(record["page_id"])
                continue
            raws, failed = [], False
            for prompt in page_prompts(record):
                try:
                    raw = call_model(client, model, prompt)
                except RateLimitReached as e:
                    print(f"\n*** DAILY/RATE LIMIT reached: {e}")
                    print(f"*** Stopping cleanly at {processed} pages this run "
                          f"({with_pred} with predictions, {total_claims} claims).")
                    print("*** This page is NOT marked done -- just re-run after "
                          "the quota resets and it resumes exactly here.")
                    raise SystemExit(2)   # exit 2 signals the batch runner to stop
                if raw is None:
                    failed = True
                    break
                raws.append(raw)
                time.sleep(REQUEST_DELAY)
            if failed:
                # A non-quota failure that exhausted retries. Do NOT mark done and
                # do NOT record a fake 'no_predictions' -- leave it for a rerun to
                # retry, so real articles aren't lost as false empties. A page
                # whose later chunk failed is redone whole rather than saved
                # half-extracted, which would understate its claim count.
                print(f"  giving up on {record['page_id']} after retries "
                      f"(left unmarked; will retry next run)")
                continue
            claims, dropped = assemble(record, raws)
            total_dropped += dropped
            if not claims:
                # The call succeeded, so [] is a genuine empty page. Recorded so
                # the denominator stays honest (pages read, not just pages that
                # produced something).
                out.write(json.dumps({"page_id": record["page_id"],
                                      "schema_version": SCHEMA_VERSION,
                                      "window": record.get("window"),
                                      "no_predictions": True}) + "\n")
            else:
                with_pred += 1
                total_claims += len(claims)
                for c in claims:
                    out.write(json.dumps(c) + "\n")
            out.flush()
            done.add(record["page_id"])
            processed += 1
            if processed % 20 == 0:
                print(f"  processed {processed}")
            if limit and processed >= limit:
                break
    return processed, with_pred, total_claims, total_dropped


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["loc", "nyt", "proquest"],
                    help="corpus tag; with --window picks data/raw/<source>_economy_<window>.jsonl")
    ap.add_argument("--window", help="economy window_id, e.g. gfc_2008")
    ap.add_argument("--pages", help="extract an arbitrary pages JSONL instead "
                                    "(e.g. main's gold_pages.jsonl)")
    ap.add_argument("--out", help="output path (required with --pages)")
    ap.add_argument("--limit", type=int, default=None,
                    help="max pages (test with 10 first)")
    ap.add_argument("--allow-mixed", action="store_true",
                    help="append to a file that still holds older-schema records")
    ap.add_argument("--sample", default="gpt_sample.txt",
                    help="ProQuest GPT sample export to read base_url/key from")
    ap.add_argument("--base-url", help="override the discovered proxy base_url")
    ap.add_argument("--key-file", help="override the discovered API key file path")
    ap.add_argument("--model", help="override the discovered model name")
    args = ap.parse_args()

    if args.pages:
        if not args.out:
            raise SystemExit("--pages requires --out")
        in_path, out_path = Path(args.pages), Path(args.out)
    else:
        if not (args.source and args.window):
            raise SystemExit("give either --source and --window, or --pages and --out")
        in_path = Path(f"data/raw/{args.source}_economy_{args.window}.jsonl")
        out_path = Path("data/predictions") / \
            f"pred_{args.source}_economy_{args.window}.jsonl"
    if not in_path.exists():
        raise SystemExit(f"No input at {in_path}."
                         + ("" if args.pages else " Run tdm_parse.py first."))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done, stale = load_done_ids(out_path)
    check_schema(out_path, stale, args.allow_mixed)
    client, model = make_client(args)

    with open(in_path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    processed, with_pred, total_claims, dropped = run(
        records, client, model, out_path, done, args.limit)

    print(f"done: {processed} pages, {with_pred} with >=1 prediction, "
          f"{total_claims} claims -> {out_path}")
    if dropped:
        print(f"  {dropped} quotes dropped as ungrounded (not really in the text)")
    if not args.pages:
        print(f"  NEXT: python strip_for_export.py {out_path}   "
              f"(removes `quote` -- required before Export)")


if __name__ == "__main__":
    main()
