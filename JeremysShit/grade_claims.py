"""
LLM grading of scraped newspaper claims (DeepSeek or any OpenAI-compatible API).

Reads claims_raw.csv, asks the model to grade each claim against the project
rubric, writes claims_graded.csv. Also exports a random 20% sample for HUMAN
double-coding (validation_sample.csv) and can compute Cohen's kappa between the
humans and the model once the sample is filled in.

Setup (one of):
    set DEEPSEEK_API_KEY=sk-...                       (PowerShell: $env:DEEPSEEK_API_KEY="sk-...")
    set OPENAI_API_KEY=... and --base-url/--model overrides for another provider

Any OpenAI-compatible endpoint works. Free tiers, with the --sleep each needs to
stay under its requests-per-minute cap:
    Groq              --base-url https://api.groq.com/openai/v1                  --sleep 2.5
    Google AI Studio  --base-url https://generativelanguage.googleapis.com/v1beta/openai --sleep 4.5
    OpenRouter        --base-url https://openrouter.ai/api/v1                    --sleep 3
    (OpenRouter's free tier caps at ~50 requests/day without $10 of credit —
     too slow for the full 1,324-claim corpus.)

OpenAI, paid tier 1 (verified 2026-07-16): 500 requests/min AND 200,000
tokens/min -- the TOKEN cap binds first here, not the request cap, because
RUBRIC_PROMPT is ~960 input tokens/call. With --model gpt-5.6-luna (a
reasoning model -- see MAX_TOKENS_REASONING_CAP below) mean usage is
~960 in + ~250 out =~1,200 tokens/call, so 200k/min sustains only ~166
calls/min, not 500. --sleep 0.45 keeps ~30% headroom under the token cap;
--sleep 0.12 (i.e. respecting only the request cap) WILL throttle.

Usage:
    python grade_claims.py                             # grade claims_raw.csv
    python grade_claims.py --limit 20                  # cheap test run first!
    python grade_claims.py --kappa validation_sample_filled.csv   # after humans grade

Reruns resume from an existing claims_graded.csv rather than regrading from
scratch; pass --overwrite to force a clean run.

Cost: DeepSeek prices are ~$0.3/1M input tokens; a 500-claim corpus is a few cents.
"""

import argparse
import csv
import json
import os
import random
import re
import ssl
import time
import urllib.error
import urllib.request

import requests

USER_AGENT = "BU-RISE-student-research/0.2 (economic prediction accuracy study)"

# A 429 whose Retry-After exceeds this is the DAILY request cap (Groq free tier:
# 1,000 req/day on the 70b, on a ~2.5h rolling reset), not a transient throttle.
# Sleeping a daily cap off would hang the process for hours; bail out instead and
# resume tomorrow — graded rows are already on disk. Must sit well above any
# transient throttle, which comes back as minutes, not hours.
DAILY_CAP_SECONDS = 1800

# Providers reserve a request's MAXIMUM possible output against the
# tokens-per-minute budget, not what it actually uses. Left unset, Groq reserves
# its default (thousands of tokens) per call against a 12k/min ceiling and
# throttles us hard. The graded JSON is ~100 tokens; 300 is ample -- for a
# NON-reasoning model. Do not raise this default: it would needlessly halve
# Groq's sustainable throughput for every model, reasoning or not. Reasoning
# models get bumped per-model, adaptively, in call_llm() instead.
MAX_TOKENS = 300
MAX_TOKENS_REASONING_CAP = 2000  # ceiling for the adaptive bump, see call_llm


class DailyCapReached(Exception):
    def __init__(self, wait):
        self.wait = wait
        super().__init__(f"daily request cap hit (provider says retry in {wait/3600:.1f}h)")


class AllKeysExhausted(Exception):
    pass


def load_labeled_keys(path, label_pattern):
    """Parse a plaintext 'Label: sk-...' style notes file (bill_arm/.env is
    NOT valid shell syntax -- it's human notes with one key per line) and
    return the values whose label matches label_pattern, in file order."""
    keys = []
    if not os.path.exists(path):
        return keys
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = re.match(label_pattern, line.strip())
            if m:
                keys.append(m.group(1))
    return keys


class KeyRotator:
    """Wraps call_llm with automatic rotation across multiple API keys for the
    SAME provider (e.g. Groq's 5 free-tier keys) -- when one key hits its
    daily request cap (DailyCapReached), move to the next key and keep going
    instead of stopping the whole run. Only raises AllKeysExhausted once
    every key in the list is capped."""

    def __init__(self, keys):
        if not keys:
            raise ValueError("KeyRotator needs at least one key")
        self.keys = keys
        self.i = 0

    def call(self, prompt, model, base_url, min_tokens=None):
        while True:
            try:
                return call_llm(prompt, model, base_url, self.keys[self.i], min_tokens=min_tokens)
            except DailyCapReached as e:
                print(f"    key {self.i + 1}/{len(self.keys)} capped ({e})", flush=True)
                self.i += 1
                if self.i >= len(self.keys):
                    raise AllKeysExhausted(f"all {len(self.keys)} keys exhausted")
                print(f"    rotating to key {self.i + 1}/{len(self.keys)}", flush=True)


try:
    # macOS python.org builds ship without a populated system trust store, so
    # urllib fails TLS verification against every https endpoint.
    import certifi
    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CONTEXT = ssl.create_default_context()

RUBRIC_PROMPT = """You are grading sentences scraped (with OCR errors) from American
newspapers published between 1900 and 1963 for a research project on the accuracy of
economic predictions. Grade ONLY what the sentence itself asserts.

Return strict JSON with exactly these fields:
- is_prediction: "yes" if the sentence itself makes a falsifiable claim about FUTURE
  economic conditions (business, prices, employment, markets, prosperity,
  recession/panic). Two errors are equally costly: a false "yes" injects fake signal,
  and a false "no" discards a real forecast. Weigh both — do not reflexively answer
  "no".
  Answer "no" for the following, even when economic words appear:
    * advertisements or pure promotional copy (price lists, product pitches, sales
      boosterism like "best in shape to serve our customers")
    * text so mangled by OCR that you cannot reconstruct what is being claimed
    * sentences that explicitly DECLINE to forecast ("it is too early yet to say
      when the recession will be over")
    * retrospectives or descriptions of the PRESENT/PAST ("prohibition has been
      beneficial", "business is good today", "the panic ruined us")
    * conditional or hypothetical arithmetic with no committed direction ("if our
      reduction goes as scheduled, this adds up to $2,000,000,000")
    * metaphor, poetry, or aphorism ("prosperity will bloom in the spring")
    * announcements of a meeting, speech, or publication ABOUT the outlook (the
      outlook is the topic of an event, not a forecast the sentence makes)
    * non-economic content, or a person's medical/personal future
  BUT answer "yes" when a genuine forecast is present even if wrapped in noise —
  grade the forecast, not the packaging:
    * a clear, reconstructable prediction survives messy OCR: a named forecaster
      giving a dated directional call (e.g. "Prof. Vane expects fac tory output to
      climb through the spring of 1912") is "yes", not "no"
    * a real forecast printed beside ad copy or under a headline still counts
      (e.g. a banker quoted as saying "trade will slacken next year" in a column
      that also carries a department-store advertisement is still "yes")
- topic: one of "general_business", "prices", "employment", "markets", "other"
- direction: the predicted direction of ECONOMIC CONDITIONS: "improve", "worsen",
  "no_change", or "unclear". Reassurance that conditions are sound or fears are
  unfounded ("nothing in the outlook to cause uneasiness") is "improve", NOT
  "no_change". Use "no_change" ONLY when the sentence explicitly says conditions
  hold flat or level. Use "unclear" when it is a real forecast but you genuinely
  cannot read the direction — never as a default.
- price_direction: only if topic is "prices": "up", "down", "stable", else "na"
- unemployment_direction: only if topic is "employment": "up", "down", "stable", else "na"
- horizon_months: best estimate of the prediction horizon: 6, 12, or "vague"
- confidence: "assertive" (will, is certain, undoubtedly) or "hedged" (may, might,
  likely, is expected, if)
- voice: WHO is making the prediction —
    "journalist" (the paper's own editorial line or a reporter's own words),
    "expert"     (economist, banker, businessman, financial analyst, professor),
    "official"   (a government officeholder: president, senator, cabinet, Fed),
    "layperson"  (an ordinary citizen, consumer, "man on the street"),
    "unclear"    (can't tell who is speaking).
  Judge the SPEAKER, not the topic.
- speaker_name: the personal name of whoever makes the prediction, if one is stated
  or clearly implied in the sentence (e.g. "Roger Babson", "Secretary Mellon"),
  else "na". Names only — not organizations or newspapers.

The sentence was printed on {date} during the "{episode}" period. Sentence:
\"\"\"{quote}\"\"\"
"""

GRADE_FIELDS = ["is_prediction", "topic", "direction", "price_direction",
                "unemployment_direction", "horizon_months", "confidence", "voice",
                "speaker_name"]


# Newer "reasoning" models (OpenAI o1/o3/gpt-5.x family) reject the classic
# max_tokens + temperature=0 request shape. Rather than hardcode model-name
# patterns that will go stale, learn it once from the API's own error and
# remember it for the rest of the run -- works for this family and any future
# one with the same restriction, with no per-call rediscovery cost.
_PARAM_ADAPTATIONS = {}  # model -> {"no_max_tokens": bool, "no_temperature": bool}


def call_llm(prompt, model, base_url, api_key, max_retries=5, min_tokens=None):
    adapt = _PARAM_ADAPTATIONS.setdefault(model, {})
    if min_tokens and "token_budget" not in adapt:
        # Skip the wasted first-attempt-at-300 retry tax when we already know
        # (from a prior survey) that this model needs more -- every failed
        # attempt still bills tokens even with an empty answer, so on a heavy
        # reasoner this isn't just slower, it's real wasted money at scale.
        adapt["token_budget"] = min_tokens
    for attempt in range(max_retries):
        params = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        }
        if not adapt.get("no_temperature"):
            params["temperature"] = 0.0
        token_key = "max_completion_tokens" if adapt.get("no_max_tokens") else "max_tokens"
        params[token_key] = adapt.get("token_budget", MAX_TOKENS)
        req = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions", data=json.dumps(params).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {api_key}",
                     # Groq's edge rejects the default Python-urllib agent (CF 1010).
                     "User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=120, context=SSL_CONTEXT) as resp:
                out = json.load(resp)
            choice = out["choices"][0]
            content = choice["message"]["content"]
            if not content and choice.get("finish_reason") == "length":
                # Reasoning models (o1/o3/gpt-5.x) bill invisible "thinking"
                # tokens against the SAME budget as the visible JSON answer.
                # Empty content + finish_reason=length means reasoning ate the
                # whole budget before writing anything -- not a real failure,
                # just needs headroom. Bump once, remember it for this model
                # for the rest of the run (cheap models never hit this path).
                budget = adapt.get("token_budget", MAX_TOKENS)
                if budget < MAX_TOKENS_REASONING_CAP and attempt < max_retries - 1:
                    adapt["token_budget"] = min(budget * 3, MAX_TOKENS_REASONING_CAP)
                    print(f"    reasoning ate the token budget (empty answer) -- "
                          f"raising to {adapt['token_budget']} for this model", flush=True)
                    continue
            return json.loads(content)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = float(e.headers.get("Retry-After") or 0) or 15.0
                if wait > DAILY_CAP_SECONDS:
                    raise DailyCapReached(wait)
                if attempt < max_retries - 1:
                    # The tokens/min window resets in about a second, so a short
                    # wait clears it. Exponential backoff here just stalls the run.
                    print(f"    rate limited, waiting {wait:.0f}s", flush=True)
                    time.sleep(wait)
                    continue
            if e.code == 400:
                try:
                    err = json.loads(e.read())["error"]
                except Exception:
                    err = {}
                param = err.get("param")
                if param == "max_tokens" and "max_tokens" in params:
                    adapt["no_max_tokens"] = True
                    params["max_completion_tokens"] = params.pop("max_tokens")
                    print("    adapting: max_tokens -> max_completion_tokens "
                          "(remembered for rest of run)", flush=True)
                    continue
                if param == "temperature" and "temperature" in params:
                    adapt["no_temperature"] = True
                    del params["temperature"]
                    print("    adapting: dropping unsupported temperature param "
                          "(remembered for rest of run)", flush=True)
                    continue
            if attempt == max_retries - 1:
                raise
            print(f"    retry after HTTP {e.code}", flush=True)
            time.sleep(5 * (attempt + 1))
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"    retry after error: {e}")
            time.sleep(5 * (attempt + 1))


def build_rotator(args):
    """Resolve which key(s) to grade with, in priority order: an explicit
    --groq-keys-file (multi-key rotation), else the single DEEPSEEK_API_KEY /
    OPENAI_API_KEY env var."""
    if args.groq_keys_file:
        keys = load_labeled_keys(args.groq_keys_file, r"^Groq key \d+:\s*(gsk_\S+)")
        if not keys:
            raise SystemExit(f"No 'Groq key N: gsk_...' lines found in {args.groq_keys_file}")
        print(f"Loaded {len(keys)} Groq keys from {args.groq_keys_file} for rotation")
        return KeyRotator(keys)
    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set DEEPSEEK_API_KEY (or OPENAI_API_KEY) first.")
    return KeyRotator([api_key])


def grade(args, rotator=None):
    """Returns True if every todo claim was graded, False if the run stopped
    early with keys/quota exhausted (caller can decide whether to fall back
    to another provider for the remainder -- claims_graded.csv already has
    everything graded so far, safe to resume or hand off)."""
    rotator = rotator or build_rotator(args)
    with open(args.claims, encoding="utf-8") as f:
        claims = list(csv.DictReader(f))
    if args.limit:
        claims = claims[: args.limit]
    out_fields = list(claims[0].keys()) + GRADE_FIELDS

    # Resume: a long run on a rate-limited free tier gets interrupted. Keep the
    # rows already graded and only spend calls on what is missing.
    graded, done_ids = [], set()
    if os.path.exists(args.out) and not args.overwrite:
        with open(args.out, encoding="utf-8") as f:
            graded = [r for r in csv.DictReader(f) if r.get("is_prediction", "").strip()]
        done_ids = {r["claim_id"] for r in graded}
        if done_ids:
            print(f"Resuming: {len(done_ids)} claims already graded in {args.out}")
    todo = [c for c in claims if c["claim_id"] not in done_ids]
    print(f"Grading {len(todo)} claims with {args.model} at {args.base_url}")

    complete = True
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        for row in graded:
            writer.writerow({k: row.get(k, "") for k in out_fields})
        for i, row in enumerate(todo, 1):
            prompt = RUBRIC_PROMPT.format(date=row["date"], episode=row["episode"],
                                          quote=row["quote"])
            try:
                g = rotator.call(prompt, args.model, args.base_url, min_tokens=args.min_tokens)
            except AllKeysExhausted as e:
                print(f"\n=== {e} ===")
                print(f"Stopping cleanly with {len(graded)} claims graded and saved.")
                print(f"Rerun the same command once a cap resets (or with fresh keys); it "
                      f"will resume from claim {i} of {len(todo)} remaining.", flush=True)
                complete = False
                break
            except Exception as e:
                print(f"  claim {row['claim_id']}: FAILED ({e}) — marked ungraded", flush=True)
                g = {}
            for k in GRADE_FIELDS:
                row[k] = str(g.get(k, ""))
            writer.writerow(row)
            graded.append(row)
            f.flush()
            if i % 25 == 0:
                print(f"  {i}/{len(todo)} graded", flush=True)
            if args.sleep and i < len(todo):
                time.sleep(args.sleep)
    n_pred = sum(1 for r in graded if r["is_prediction"] == "yes")
    print(f"\nDone: {len(graded)} graded -> {args.out}  ({n_pred} judged real predictions)")

    # Export the human validation sample (only rows the model calls predictions,
    # since those are the ones that reach the scoring stage).
    preds = [r for r in graded if r["is_prediction"] == "yes"]
    if not preds:
        print("No graded predictions — leaving any existing validation_sample.csv alone.")
        return complete
    random.seed(42)
    sample = random.sample(preds, min(len(preds), max(5, int(len(preds) * args.sample_frac))))
    with open("validation_sample.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields + [f"human_{k}" for k in GRADE_FIELDS])
        w.writeheader()
        for r in sample:
            w.writerow({**r, **{f"human_{k}": "" for k in GRADE_FIELDS}})
    print(f"Human validation sample ({len(sample)} rows) -> validation_sample.csv")
    print("Two graders: fill the human_* columns independently, then run --kappa on each.")
    return complete


# --- OpenAI Batch API: submit the whole job as one file, poll, retrieve.
# 50% cheaper than synchronous chat/completions calls, which matters on a
# fixed prepaid balance -- see run_batch(). Batch-specific, not a generic
# provider option like --base-url (unlike the synchronous path, the Batch
# API's request/response envelope isn't something every OpenAI-compatible
# provider implements the same way, so this targets OpenAI only).
OPENAI_BATCH_BASE = "https://api.openai.com/v1"
GROQ_BASE = "https://api.groq.com/openai/v1"


def _batch_request_lines(todo, model, min_tokens=None):
    for row in todo:
        prompt = RUBRIC_PROMPT.format(date=row["date"], episode=row["episode"], quote=row["quote"])
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
            "max_tokens": min_tokens or MAX_TOKENS,
        }
        yield json.dumps({"custom_id": row["claim_id"], "method": "POST",
                          "url": "/v1/chat/completions", "body": body})


def submit_batch(api_key, todo, model, min_tokens=None):
    jsonl = "\n".join(_batch_request_lines(todo, model, min_tokens))
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.post(f"{OPENAI_BATCH_BASE}/files", headers=headers,
                         files={"file": ("batch_input.jsonl", jsonl.encode("utf-8"),
                                        "application/jsonl")},
                         data={"purpose": "batch"}, timeout=120)
    resp.raise_for_status()
    file_id = resp.json()["id"]
    resp = requests.post(f"{OPENAI_BATCH_BASE}/batches",
                         headers={**headers, "Content-Type": "application/json"},
                         json={"input_file_id": file_id, "endpoint": "/v1/chat/completions",
                              "completion_window": "24h"}, timeout=60)
    resp.raise_for_status()
    return resp.json()


def poll_batch(api_key, batch_id, interval=30):
    headers = {"Authorization": f"Bearer {api_key}"}
    while True:
        resp = requests.get(f"{OPENAI_BATCH_BASE}/batches/{batch_id}", headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        counts = data.get("request_counts", {}) or {}
        print(f"  batch {batch_id}: {data['status']}  (completed "
             f"{counts.get('completed', 0)}/{counts.get('total', 0)}, "
             f"failed {counts.get('failed', 0)})", flush=True)
        if data["status"] in ("completed", "failed", "expired", "cancelled"):
            return data
        time.sleep(interval)


def fetch_file(api_key, file_id):
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(f"{OPENAI_BATCH_BASE}/files/{file_id}/content", headers=headers, timeout=120)
    resp.raise_for_status()
    return resp.text


def run_batch(args, api_key):
    """Grade via the OpenAI Batch API instead of one call per claim -- half
    the per-token price, and one job instead of hundreds of requests. Trades
    latency (can take up to 24h, usually far less) for cost; not
    interactive, meant to be backgrounded.

    Returns True if the whole todo list came back graded, False if the job
    failed/expired or requests errored out (e.g. the account ran out of
    balance) -- callers (see auto_grade) can fall back to another provider
    for whatever's left, since claims_graded.csv already has everything
    that succeeded and the resume-by-claim_id logic skips it next time."""
    with open(args.claims, encoding="utf-8") as f:
        claims = list(csv.DictReader(f))
    if args.limit:
        claims = claims[: args.limit]
    out_fields = list(claims[0].keys()) + GRADE_FIELDS

    graded, done_ids = [], set()
    if os.path.exists(args.out) and not args.overwrite:
        with open(args.out, encoding="utf-8") as f:
            graded = [r for r in csv.DictReader(f) if r.get("is_prediction", "").strip()]
        done_ids = {r["claim_id"] for r in graded}
        if done_ids:
            print(f"Resuming: {len(done_ids)} claims already graded in {args.out}")
    todo = [c for c in claims if c["claim_id"] not in done_ids]
    if not todo:
        print("Nothing left to grade.")
        return True

    def save():
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=out_fields)
            writer.writeheader()
            for row in graded:
                writer.writerow({k: row.get(k, "") for k in out_fields})

    # OpenAI's Batch API caps ENQUEUED tokens per model per org (hit this for
    # real 2026-07-18: 2,415 claims at once tripped "Enqueued token limit
    # reached ... Limit: 900,000 enqueued tokens" and the whole job failed
    # with 0 graded -- the cap counts every batch still in_progress, not just
    # this submission, so chunks must run one-at-a-time, not concurrently.
    # ~960 prompt tokens + up to MAX_TOKENS output reserved per claim (see
    # module docstring's per-call estimate); chunk_size defaults conservative
    # enough to leave real headroom under 900k even if a future model raises
    # the per-call estimate.
    chunks = [todo[i:i + args.batch_chunk_size] for i in range(0, len(todo), args.batch_chunk_size)]
    print(f"Submitting {len(todo)} claims to the OpenAI Batch API ({args.openai_model}) in "
         f"{len(chunks)} chunk(s) of up to {args.batch_chunk_size}...")

    for ci, chunk in enumerate(chunks, 1):
        print(f"--- chunk {ci}/{len(chunks)}: {len(chunk)} claims ---", flush=True)
        for attempt in range(3):
            try:
                batch = submit_batch(api_key, chunk, args.openai_model, args.min_tokens)
                break
            except requests.exceptions.HTTPError as e:
                body = e.response.text[:500] if e.response is not None else str(e)
                print(f"Batch submission failed: {e}\n{body}")
                save()
                return False
        print(f"  batch id {batch['id']}, status {batch['status']}")
        result = poll_batch(api_key, batch["id"], interval=args.poll_interval)

        # A chunk-level "token_limit_exceeded" means a still-in_progress batch
        # (ours or another job in the same org) hasn't cleared yet -- worth one
        # retry after a short wait before concluding the account is out of
        # money and handing off to the Groq fallback.
        err_codes = {e.get("code") for e in (result.get("errors") or {}).get("data", [])}
        if result["status"] == "failed" and err_codes == {"token_limit_exceeded"} and attempt == 0:
            print("  enqueued-token limit hit; waiting 60s for other in_progress "
                 "batches to clear, then retrying this chunk once", flush=True)
            time.sleep(60)
            try:
                batch = submit_batch(api_key, chunk, args.openai_model, args.min_tokens)
                result = poll_batch(api_key, batch["id"], interval=args.poll_interval)
            except requests.exceptions.HTTPError as e:
                print(f"  retry failed: {e}")

        by_id = {row["claim_id"]: row for row in chunk}
        n_ok = 0
        if result.get("output_file_id"):
            for line in fetch_file(api_key, result["output_file_id"]).splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                row = by_id.get(rec["custom_id"])
                if row is None:
                    continue
                body = (rec.get("response") or {}).get("body") or {}
                try:
                    g = json.loads(body["choices"][0]["message"]["content"])
                except Exception:
                    g = {}
                for k in GRADE_FIELDS:
                    row[k] = str(g.get(k, ""))
                graded.append(row)
                n_ok += 1
        if result.get("error_file_id"):
            err_text = fetch_file(api_key, result["error_file_id"])
            err_lines = [l for l in err_text.splitlines() if l.strip()]
            print(f"  {len(err_lines)} request(s) errored in this chunk (often an "
                 f"insufficient-balance signal) -- first: "
                 f"{err_lines[0][:300] if err_lines else ''}")
        if result.get("errors") and not result.get("output_file_id"):
            print(f"  batch-level error: {result['errors']}")
        save()

        chunk_complete = result["status"] == "completed" and n_ok >= len(chunk)
        print(f"  chunk {ci}/{len(chunks)} {result['status']}: {n_ok}/{len(chunk)} graded "
             f"({len(graded)} total so far -> {args.out})")
        if not chunk_complete:
            print("Chunk did not fully complete -- stopping the OpenAI phase here; "
                 "remaining claims stay ungraded for a fallback provider (--auto) or a rerun.")
            return False

    print(f"\nAll {len(chunks)} chunk(s) complete -> {args.out} ({len(graded)} total graded)")
    return True


def auto_grade(args):
    """User's explicit cost plan: grade with the OpenAI key (batched, 50%
    cheaper) until the account balance runs out, then automatically switch
    to rotating across the Groq free-tier keys for whatever's left."""
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        raise SystemExit("Set OPENAI_API_KEY first (or pass --groq-keys-file with --batch off "
                         "to skip straight to Groq).")
    print(f"=== Phase 1: OpenAI Batch API ({args.openai_model}) until the account runs out ===")
    if run_batch(args, openai_key):
        print("\nAll claims graded via OpenAI batch -- done.")
        return
    print("\n=== Phase 2: OpenAI batch didn't finish everything -- rotating Groq keys for "
         "the remainder ===")
    keys = load_labeled_keys(args.groq_keys_file, r"^Groq key \d+:\s*(gsk_\S+)")
    if not keys:
        raise SystemExit(f"No 'Groq key N: gsk_...' lines found in {args.groq_keys_file} -- "
                         f"can't run the Groq fallback phase.")
    print(f"Loaded {len(keys)} Groq keys from {args.groq_keys_file} for rotation")
    groq_args = argparse.Namespace(**vars(args))
    groq_args.model = args.groq_model
    groq_args.base_url = GROQ_BASE
    groq_args.sleep = args.groq_sleep
    grade(groq_args, rotator=KeyRotator(keys))


def cohens_kappa(pairs):
    """pairs = list of (label_a, label_b); returns kappa."""
    labels = sorted({a for a, _ in pairs} | {b for _, b in pairs})
    n = len(pairs)
    p_obs = sum(1 for a, b in pairs if a == b) / n
    p_exp = sum((sum(1 for a, _ in pairs if a == L) / n) *
                (sum(1 for _, b in pairs if b == L) / n) for L in labels)
    return (p_obs - p_exp) / (1 - p_exp) if p_exp < 1 else 1.0


def kappa_report(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Model vs. human agreement on {len(rows)} claims ({path}):")
    for k in ["is_prediction", "topic", "direction", "confidence"]:
        pairs = [(r[k].strip().lower(), r[f"human_{k}"].strip().lower())
                 for r in rows if r.get(f"human_{k}", "").strip()]
        if pairs:
            print(f"  {k:15s} kappa = {cohens_kappa(pairs):+.2f}  (n={len(pairs)})")
    print("Target: kappa >= 0.7 on direction. Below that, tighten the rubric and regrade.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--claims", default="claims_raw.csv")
    ap.add_argument("--out", default="claims_graded.csv")
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--base-url", default="https://api.deepseek.com")
    ap.add_argument("--limit", type=int, default=0, help="grade only the first N (test runs)")
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="seconds between calls; free tiers rate-limit by requests/minute "
                         "(Groq: 2.5, Google AI Studio: 4.5)")
    ap.add_argument("--min-tokens", type=int, default=None,
                    help="seed the reasoning-model token budget above the 300 "
                         "default -- skips the wasted first-attempt retry tax "
                         "when a survey already showed this model needs more")
    ap.add_argument("--overwrite", action="store_true",
                    help="ignore an existing --out file instead of resuming from it")
    ap.add_argument("--sample-frac", type=float, default=0.2)
    ap.add_argument("--kappa", metavar="FILLED_CSV",
                    help="compute model-vs-human Cohen's kappa from a filled validation sample")
    ap.add_argument("--batch", action="store_true",
                    help="grade via the OpenAI Batch API (needs OPENAI_API_KEY) instead of "
                         "one call per claim -- 50%% cheaper, async, meant to be backgrounded")
    ap.add_argument("--auto", action="store_true",
                    help="OpenAI batch first (needs OPENAI_API_KEY), then automatically "
                         "rotate --groq-keys-file's keys for whatever's left once the OpenAI "
                         "account runs out")
    ap.add_argument("--openai-model", default="gpt-4.1",
                    help="model for --batch/--auto's OpenAI phase -- gpt-4.1 is this project's "
                         "already-validated grader choice (see CHANGELOG bake-off)")
    ap.add_argument("--groq-model", default="llama-3.3-70b-versatile",
                    help="model for --auto's Groq fallback phase")
    ap.add_argument("--groq-sleep", type=float, default=2.5,
                    help="seconds between calls in --auto's Groq phase (free-tier rpm cap)")
    ap.add_argument("--groq-keys-file",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "..", "bill_arm", ".env"),
                    help="notes file with 'Groq key N: gsk_...' lines, for multi-key rotation "
                         "via --groq-keys-file directly or --auto's fallback phase")
    ap.add_argument("--poll-interval", type=float, default=30,
                    help="seconds between batch status checks (--batch/--auto)")
    ap.add_argument("--batch-chunk-size", type=int, default=600,
                    help="claims per OpenAI Batch API submission (--batch/--auto) -- org-level "
                         "enqueued-token caps (e.g. 900k for gpt-4.1) mean one giant batch can "
                         "fail outright; chunks run sequentially so only one is in_progress at "
                         "a time")
    args = ap.parse_args()
    if args.kappa:
        kappa_report(args.kappa)
    elif args.auto:
        auto_grade(args)
    elif args.batch:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("Set OPENAI_API_KEY first for --batch.")
        run_batch(args, api_key)
    else:
        grade(args)
