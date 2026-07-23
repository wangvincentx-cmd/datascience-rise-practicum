"""
Whole-page LLM extraction of economic predictions -- the replacement for
newspaper_scraper.extract_claims.

The old path was: phrase-search a page, then keep sentences within +/-2
sentences of the phrase that also matched a future-tense regex. Measured against
the gold pages (gold_extraction/), that has recall 0.19 and precision 0.44 -- it
misses four claims in five, and nearly half of what it returns is not a
prediction. This reads the whole page instead and returns structured claims.

Ported from election_arm/extract_predictions.py (its call_llm param-adaptation
logic, its fence-stripping JSON parser, its ECONOMY_PROMPT as the starting
point), kept self-contained per the one-arm-one-codebase convention in
CLAUDE.md. Four things differ:

1. **No outcome leakage.** The old context block passed `Window: crash_1929`.
   Episode/window names state the OUTCOME, so passing them tells the model what
   happened before it labels which way a forecast pointed. Only date and
   newspaper are passed -- both genuinely known to whoever set the page.
2. **Chunked, not truncated.** MAX_OCR_CHARS=12000 silently discarded half of a
   typical 21k-char page. A sliding window with overlap covers all of it.
3. **One pass, full schema.** Returns the grading fields too, so the separate
   grade_claims.py call per claim disappears.
4. **Hallucination guard.** Every returned quote must fuzzy-match text actually
   on the page or it is dropped. Enforced in code -- a model that paraphrases,
   "corrects", or invents a quote fails the check.

Usage (from JeremysShit/):
    # DeepInfra (OpenAI-compatible)
    export DEEPINFRA_API_KEY=...
    python extract_llm.py --pages gold_extraction/gold_pages.jsonl \\
        --out gold_extraction/pred_llama70b.jsonl \\
        --model meta-llama/Llama-3.3-70B-Instruct

    # local vLLM on the cluster -- same code path, just a different base URL
    python extract_llm.py --base-url http://gpu-node:8000/v1 --api-key-env NONE \\
        --model meta-llama/Llama-3.3-70B-Instruct --pages data/pages.jsonl

Resumes from an existing --out file; pass --overwrite to start clean.
"""

import argparse
import json
import os
import re
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path

USER_AGENT = "BU-RISE-student-research/0.2 (economic prediction accuracy study)"

DEEPINFRA_BASE = "https://api.deepinfra.com/v1/openai"

# A page is ~21k chars. 8k-char windows with 500 of overlap mean a claim
# straddling a boundary still appears whole in one window; cross-window
# duplicates are removed afterwards by normalized text.
CHUNK_CHARS = 8000
CHUNK_OVERLAP = 500
MAX_TOKENS = 1600
MAX_TOKENS_REASONING_CAP = 6000

# Fraction of a returned quote's tokens that must appear on the page for it to
# be accepted. Below 1.0 because models silently repair OCR ("mille"->"mills"),
# which is desirable; far enough above chance that an invented sentence fails.
QUOTE_MATCH_THRESHOLD = 0.65

_PARAM_ADAPTATIONS = {}
REASONING_EFFORT = None
TOKEN_RE = re.compile(r"[a-z0-9]+")

def _ssl_context():
    """Build a verifying SSL context that survives a TLS-intercepting proxy.

    Machines behind corporate TLS inspection (and some antivirus products)
    present a re-signed certificate chain whose root lives in the OS trust
    store and NOT in certifi's bundle, so certifi-based verification fails
    every HTTPS host with CERTIFICATE_VERIFY_FAILED -- FRED and loc.gov
    included, not just this API. `truststore` delegates verification to the OS
    store, which fixes it while STILL VERIFYING. Never fall back to an
    unverified context here: this request carries an API key."""
    try:
        import truststore
        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except ImportError:
        pass
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


SSL_CONTEXT = _ssl_context()


# The examples below are INVENTED, not drawn from any page in the corpus. The
# prompt-tuning handoff (handgrade_newspapers/HANDOFF_prompt_tuning.md) records
# that two illustrative examples once came from the validation set and inflated
# the reported kappa. Anything concrete here must stay out-of-corpus.
EXTRACTION_PROMPT = """You extract economic PREDICTIONS from a page of an American newspaper printed between 1900 and 1963.

The text is OCR of a full newspaper page, so it is noisy: columns interleave, words break across lines, letters are wrong ("tho" for "the", "busi ness" for "business"). Read through the noise. The page carries many unrelated stories at once -- news, advertisements, fiction, sports, social notes -- and most of it is not a prediction.

Find every sentence that makes a FALSIFIABLE CLAIM ABOUT FUTURE ECONOMIC CONDITIONS: business conditions, prices, employment, markets, prosperity, recession or panic.

INCLUDE:
- Forecasts quoted from a named person (banker, economist, official, executive). The paper is the vehicle; the forecaster is the source.
- A real forecast printed next to advertising or under an unrelated headline.
- A headline that itself states a forecast.
- A forecast recoverable through OCR damage, when the words can be reconstructed with confidence.
- Forecasts about a region, an industry, or a foreign country, not only the national economy.

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
  "quote": "the prediction, copied VERBATIM from the page text including its OCR errors, max 60 words",
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
  "conditional_on": "the stated condition in 10 words or fewer, else \\"na\\""
}}

Rules:
- "quote" MUST be copied verbatim from the text given to you. Do not clean up the OCR, do not paraphrase, do not merge separated sentences. A quote that does not appear on the page is discarded.
- direction: reassurance that conditions are sound or that fears are unfounded ("nothing in the outlook to cause uneasiness") is "improve", NOT "no_change". Use "no_change" only when the sentence explicitly says conditions hold flat. Use "unclear" only when it is genuinely a forecast whose direction cannot be read -- never as a default.
- For price claims, ask what the sentence implies for conditions OVERALL, using its own framing.
- confidence: judge the words, not the speaker's authority. "will", "is certain", "undoubtedly" = assertive; "may", "likely", "is expected", "we do not think" = hedged.
- voice: judge WHO is speaking, not what it is about.
- scope: WHICH economy the claim is about. "national" = the US economy overall.
  "regional" = a US state, city or region. "foreign" = a non-US economy (a
  forecast about Mexican exports or the Brazilian currency printed in a US paper
  is "foreign"). "industry" = one US industry or sector (steel, autos, farming,
  railroads). If a claim is about a foreign industry, use "foreign". This
  matters because only national claims can fairly be compared against national
  economic statistics.
- is_quoted_forecaster: true when the forecast is attributed to someone other
  than the newspaper itself -- a banker, economist, official, company, trade
  body or government report. false when it is the paper's own editorial voice or
  its reporter's own assessment. When in doubt, false.
- conditional_on: if the forecast openly rests on something ("barring a strike",
  "if the crop holds", "assuming the tariff passes"), state that condition
  briefly. Otherwise "na". A forecast whose condition is unresolved AND which
  commits to no direction should not be extracted at all -- see the exclusions
  above; this field is for committed forecasts that carry a caveat.
- Two predictions in one sentence get two entries ONLY if they point in different directions or concern different topics.
- If the page contains no predictions, return []. Many pages genuinely contain none; returning [] is a correct answer, not a failure.

Newspaper: {newspaper}
Date: {date}

Page text:
{text}"""


def call_llm(prompt, model, base_url, api_key, max_retries=5):
    adapt = _PARAM_ADAPTATIONS.setdefault(model, {})
    for attempt in range(max_retries):
        params = {"model": model, "messages": [{"role": "user", "content": prompt}]}
        if REASONING_EFFORT:
            # Reasoning models bill (and, on local hardware, SPEND WALL-CLOCK on)
            # invisible thinking tokens. gpt-oss-120b emits ~3,600 output
            # tokens/page here against Gemini's ~350, and decode is the
            # bottleneck on a GPU, so this knob moves runtime roughly 10x.
            params["reasoning_effort"] = REASONING_EFFORT
        if not adapt.get("no_temperature"):
            params["temperature"] = 0.0
        token_key = "max_completion_tokens" if adapt.get("no_max_tokens") else "max_tokens"
        params[token_key] = adapt.get("token_budget", MAX_TOKENS)
        headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
                                     data=json.dumps(params).encode(), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=180, context=SSL_CONTEXT) as resp:
                out = json.load(resp)
            choice = out["choices"][0]
            content = choice["message"].get("content")
            if choice.get("finish_reason") == "length":
                # ANY truncation triggers the bump, not just an empty answer.
                # Thinking models bill invisible reasoning against the same
                # budget as the visible JSON: gemini-3.5-flash returned
                # completion_tokens=63 out of ~1,600 consumed, i.e. a JSON array
                # cut off mid-object. That parses to nothing, so the model
                # scored 2 claims across 16 pages and looked broken when it was
                # only starved. Non-empty truncated output is just as unusable
                # as empty truncated output.
                budget = adapt.get("token_budget", MAX_TOKENS)
                if budget < MAX_TOKENS_REASONING_CAP and attempt < max_retries - 1:
                    adapt["token_budget"] = min(budget * 3, MAX_TOKENS_REASONING_CAP)
                    print(f"    truncated ({len(content or '')} chars of visible "
                          f"content) -- raising token budget to "
                          f"{adapt['token_budget']}", flush=True)
                    continue
                print(f"    STILL truncated at budget {budget} -- this page's "
                      f"claims may be incomplete", flush=True)
            return content, out.get("usage", {})
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:300]
            if e.code == 429:
                # Not every 429 is a throttle. Providers also return 429 for
                # "credits depleted" and for daily quota exhaustion, and both
                # look exactly like a rate limit if the body is discarded --
                # which wastes the whole retry budget silently. Print the body
                # and stop immediately when it is clearly not transient.
                if re.search(r"credit|quota|billing|exhaust|depleted", body, re.I):
                    raise SystemExit(f"provider rejected the request (HTTP 429, "
                                     f"not a throttle): {body}")
                wait = float(e.headers.get("Retry-After") or 0) or 15.0
                print(f"    rate limited ({body[:120]}), waiting {wait:.0f}s",
                      flush=True)
                time.sleep(wait)
                continue
            if e.code == 400 and "max_tokens" in body and not adapt.get("no_max_tokens"):
                adapt["no_max_tokens"] = True
                continue
            if e.code == 400 and "temperature" in body and not adapt.get("no_temperature"):
                adapt["no_temperature"] = True
                continue
            print(f"    HTTP {e.code}: {body}", flush=True)
            if attempt == max_retries - 1:
                raise
            time.sleep(5 * (attempt + 1))
        except Exception as e:
            print(f"    {type(e).__name__}: {e}", flush=True)
            if attempt == max_retries - 1:
                raise
            time.sleep(5 * (attempt + 1))
    return None, {}


def chunks(text, size=None, overlap=CHUNK_OVERLAP):
    # Larger windows are cheaper, not just faster: the ~1,100-token prompt is
    # re-sent with EVERY chunk, so a 21k-char page split three ways pays for the
    # instructions three times. One window per page removes two thirds of that
    # overhead. Only worth narrowing for models with a small context window.
    size = size or CHUNK_CHARS
    if len(text) <= size:
        return [text]
    out, start = [], 0
    while start < len(text):
        out.append(text[start:start + size])
        if start + size >= len(text):
            break
        start += size - overlap
    return out


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


def quote_is_grounded(quote, page_tokens, threshold=QUOTE_MATCH_THRESHOLD):
    """Fraction of the quote's tokens that appear on the page.

    The guard that matters: without it, a model under instruction to return
    verbatim spans will still occasionally produce a fluent, plausible,
    entirely invented forecast -- which is indistinguishable from a real one
    downstream and would be scored against real macro data."""
    q = TOKEN_RE.findall((quote or "").lower())
    if len(q) < 4:
        return False
    return sum(1 for t in q if t in page_tokens) / len(q) >= threshold


def page_prompts(page, chunk_chars=None):
    """The prompts for one page, one per text window."""
    return [EXTRACTION_PROMPT.format(
                newspaper=page.get("publisher") or page.get("newspaper_title")
                or "unknown",
                date=page.get("date") or "unknown", text=part)
            for part in chunks(page["ocr_text"], chunk_chars)]


def assemble_claims(page, raw_responses):
    """Turn a page's raw model outputs into deduplicated, grounded claims.

    Shared by the synchronous and batch paths on purpose. The two differ only in
    how the raw text was obtained; if they assembled claims separately they
    would drift, and a batch corpus would quietly stop being comparable to the
    gold numbers measured on the sync path."""
    page_tokens = set(TOKEN_RE.findall(page["ocr_text"].lower()))
    seen, claims, dropped = set(), [], 0
    for raw in raw_responses:
        for c in parse_claims(raw):
            if not isinstance(c, dict) or not c.get("quote"):
                continue
            key = " ".join(TOKEN_RE.findall(c["quote"].lower()))
            if not key or key in seen:
                continue
            if not quote_is_grounded(c["quote"], page_tokens):
                dropped += 1
                continue
            seen.add(key)
            c.update({"page_id": page["page_id"], "date": page.get("date"),
                      "publisher": page.get("publisher"),
                      "source_text_type": "full_page"})
            claims.append(c)
    return claims, dropped


def extract_page(page, model, base_url, api_key, sleep=0.0, chunk_chars=None):
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0}
    raws = []
    for prompt in page_prompts(page, chunk_chars):
        raw, usage = call_llm(prompt, model, base_url, api_key)
        for k in usage_total:
            usage_total[k] += usage.get(k, 0) or 0
        raws.append(raw)
        if sleep:
            time.sleep(sleep)
    claims, dropped = assemble_claims(page, raws)
    return claims, usage_total, dropped


# --- Gemini Batch API -------------------------------------------------------
# 50% off input AND output, target turnaround 24h (usually far less). For the
# 2,192-page cached corpus that is the difference between ~$28 and ~$14.
#
# Uses Gemini's NATIVE batch endpoint with inline requests, not the OpenAI
# compatibility layer: that layer exposes /batches but returns 404 on /files, so
# there is no way to upload a request file through it. Inline requests cap at
# 20MB per batch, so prompts are packed into several batches under a byte
# budget and the results stitched back together.
GEMINI_NATIVE = "https://generativelanguage.googleapis.com/v1beta"
INLINE_BYTE_BUDGET = 15_000_000  # under the 20MB cap, with headroom
BATCH_POLL_SECONDS = 60


def _requests_session():
    import requests
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass
    return requests


def batch_submit(prompts_with_keys, model, api_key, display_name):
    """Submit one inline batch. prompts_with_keys is [(key, prompt), ...]."""
    requests = _requests_session()
    reqs = [{"request": {"contents": [{"parts": [{"text": p}]}],
                         "generationConfig": {"temperature": 0.0,
                                              "maxOutputTokens": MAX_TOKENS_REASONING_CAP}},
             "metadata": {"key": k}}
            for k, p in prompts_with_keys]
    r = requests.post(
        f"{GEMINI_NATIVE}/models/{model}:batchGenerateContent?key={api_key}",
        json={"batch": {"display_name": display_name,
                        "input_config": {"requests": {"requests": reqs}}}},
        timeout=300)
    r.raise_for_status()
    return r.json()["name"]


def batch_wait(name, api_key, poll=BATCH_POLL_SECONDS):
    requests = _requests_session()
    while True:
        r = requests.get(f"{GEMINI_NATIVE}/{name}?key={api_key}", timeout=120)
        r.raise_for_status()
        data = r.json()
        meta = data.get("metadata") or {}
        state = meta.get("state", "?")
        stats = meta.get("batchStats") or {}
        print(f"    {name.split('/')[-1]}: {state}  "
              f"({stats.get('successfulRequestCount', 0)}/"
              f"{stats.get('requestCount', 0)} ok, "
              f"{stats.get('failedRequestCount', 0)} failed)", flush=True)
        if state in ("BATCH_STATE_SUCCEEDED", "BATCH_STATE_FAILED",
                     "BATCH_STATE_CANCELLED", "BATCH_STATE_EXPIRED"):
            return data
        time.sleep(poll)


def batch_results(data):
    """{key: text} from a finished batch, plus summed usage."""
    meta = data.get("metadata") or {}
    inlined = (((meta.get("output") or {}).get("inlinedResponses") or {})
               .get("inlinedResponses") or [])
    out, usage = {}, {"prompt_tokens": 0, "completion_tokens": 0}
    for item in inlined:
        key = (item.get("metadata") or {}).get("key")
        resp = item.get("response") or {}
        if key is None:
            continue
        cands = resp.get("candidates") or []
        text = ""
        if cands:
            parts = ((cands[0].get("content") or {}).get("parts") or [])
            text = "".join(p.get("text", "") for p in parts)
        out[key] = text
        um = resp.get("usageMetadata") or {}
        usage["prompt_tokens"] += um.get("promptTokenCount", 0) or 0
        usage["completion_tokens"] += um.get("candidatesTokenCount", 0) or 0
    return out, usage


def run_batch(args, pages, done):
    """Extract every page through the Gemini Batch API."""
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise SystemExit(f"set {args.api_key_env}")
    model = args.model.replace("models/", "")

    # (page_index, chunk_index) -> prompt, keyed so results map back exactly.
    todo = [p for p in pages if p["page_id"] not in done]
    items, prompts_by_key = [], {}
    for pi, page in enumerate(todo):
        for ci, prompt in enumerate(page_prompts(page, args.chunk_chars)):
            key = f"{pi}:{ci}"
            items.append((key, prompt))
            prompts_by_key[key] = prompt
    if not items:
        print("nothing to do")
        return
    print(f"{len(todo)} pages -> {len(items)} requests")

    # Pack into batches under the inline byte budget.
    batches, cur, cur_bytes = [], [], 0
    for key, prompt in items:
        size = len(prompt.encode("utf-8")) + 200
        if cur and cur_bytes + size > INLINE_BYTE_BUDGET:
            batches.append(cur)
            cur, cur_bytes = [], 0
        cur.append((key, prompt))
        cur_bytes += size
    if cur:
        batches.append(cur)
    print(f"packed into {len(batches)} batch job(s) under "
          f"{INLINE_BYTE_BUDGET/1e6:.0f}MB each")

    names = []
    for i, b in enumerate(batches, 1):
        name = batch_submit(b, model, api_key, f"{args.batch_label}-{i}")
        names.append(name)
        print(f"  submitted [{i}/{len(batches)}] {name} ({len(b)} requests)",
              flush=True)

    raw_by_key, totals = {}, {"prompt_tokens": 0, "completion_tokens": 0}
    for i, name in enumerate(names, 1):
        print(f"  waiting on batch {i}/{len(names)}...", flush=True)
        data = batch_wait(name, api_key)
        res, usage = batch_results(data)
        raw_by_key.update(res)
        for k in totals:
            totals[k] += usage[k]

    mode = "a" if (done and not args.overwrite) else "w"
    n_claims = n_dropped = n_missing = 0
    with open(args.out, mode, encoding="utf-8") as fh:
        for pi, page in enumerate(todo):
            n_chunks = len(page_prompts(page, args.chunk_chars))
            raws = []
            for ci in range(n_chunks):
                key = f"{pi}:{ci}"
                if key not in raw_by_key:
                    n_missing += 1
                raws.append(raw_by_key.get(key, ""))
            claims, dropped = assemble_claims(page, raws)
            for c in claims:
                fh.write(json.dumps(c, ensure_ascii=False) + "\n")
            n_claims += len(claims)
            n_dropped += dropped

    print(f"\n{n_claims} claims -> {args.out}")
    if n_dropped:
        print(f"{n_dropped} dropped by the hallucination guard")
    if n_missing:
        print(f"WARNING: {n_missing} requests returned nothing -- rerun to "
              f"fill the gaps (already-written pages are skipped)")
    print(f"tokens: {totals['prompt_tokens']:,} in / "
          f"{totals['completion_tokens']:,} out  (billed at 50% batch rate)")


def run(args):
    pages = [json.loads(l) for l in open(args.pages, encoding="utf-8") if l.strip()]
    if args.limit:
        pages = pages[:args.limit]
    out_path = Path(args.out)
    done = set()
    if out_path.exists() and not args.overwrite:
        for line in open(out_path, encoding="utf-8"):
            try:
                done.add(json.loads(line)["page_id"])
            except (json.JSONDecodeError, KeyError):
                pass
        print(f"resuming: {len(done)} pages already extracted")

    api_key = "" if args.api_key_env == "NONE" else os.environ.get(args.api_key_env, "")
    if not api_key and args.api_key_env != "NONE":
        raise SystemExit(f"set {args.api_key_env} (or pass --api-key-env NONE for a "
                         f"local server that needs no key)")

    mode = "a" if (done and not args.overwrite) else "w"
    totals = {"prompt_tokens": 0, "completion_tokens": 0}
    n_claims = n_dropped = 0
    with open(out_path, mode, encoding="utf-8") as fh:
        for i, page in enumerate(pages, 1):
            if page["page_id"] in done:
                continue
            claims, usage, dropped = extract_page(page, args.model, args.base_url,
                                                  api_key, args.sleep,
                                                  args.chunk_chars)
            for c in claims:
                fh.write(json.dumps(c, ensure_ascii=False) + "\n")
            fh.flush()
            for k in totals:
                totals[k] += usage.get(k, 0)
            n_claims += len(claims)
            n_dropped += dropped
            print(f"  [{i}/{len(pages)}] {page.get('date')}  "
                  f"{len(claims)} claims"
                  + (f"  ({dropped} failed the quote check)" if dropped else ""),
                  flush=True)

    print(f"\n{n_claims} claims -> {out_path}")
    if n_dropped:
        print(f"{n_dropped} claims dropped by the hallucination guard "
              f"({n_dropped / max(1, n_claims + n_dropped):.1%} of returned)")
    print(f"tokens: {totals['prompt_tokens']:,} in / "
          f"{totals['completion_tokens']:,} out")
    if pages:
        per = (totals["prompt_tokens"] + totals["completion_tokens"]) / len(pages)
        print(f"~{per:,.0f} tokens/page -> ~{per * 2192 / 1e6:.1f}M tokens for the "
              f"2,192-page cached corpus")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pages", default="data/pages.jsonl")
    ap.add_argument("--out", default="claims_v2.jsonl")
    ap.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct")
    ap.add_argument("--base-url", default=DEEPINFRA_BASE)
    ap.add_argument("--api-key-env", default="DEEPINFRA_API_KEY",
                    help="env var holding the key; NONE for a keyless local server")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--batch", action="store_true",
                    help="submit via the Gemini Batch API: 50%% off input and "
                         "output, target turnaround 24h (usually far less). "
                         "Gemini models only.")
    ap.add_argument("--batch-label", default="rise-extract",
                    help="display name prefix for submitted batch jobs")
    ap.add_argument("--reasoning-effort", default=None,
                    choices=["low", "medium", "high"],
                    help="for reasoning models (gpt-oss, o-series): trades "
                         "quality against output tokens, i.e. against GPU time")
    ap.add_argument("--chunk-chars", type=int, default=None,
                    help=f"page window size (default {CHUNK_CHARS}); raise it to "
                         f"stop re-sending the prompt per chunk")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    REASONING_EFFORT = args.reasoning_effort
    if args.batch:
        pages_all = [json.loads(l) for l in open(args.pages, encoding="utf-8")
                     if l.strip()]
        if args.limit:
            pages_all = pages_all[:args.limit]
        done_ids = set()
        if Path(args.out).exists() and not args.overwrite:
            for line in open(args.out, encoding="utf-8"):
                try:
                    done_ids.add(json.loads(line)["page_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
            if done_ids:
                print(f"resuming: {len(done_ids)} pages already extracted")
        run_batch(args, pages_all, done_ids)
    else:
        run(args)
