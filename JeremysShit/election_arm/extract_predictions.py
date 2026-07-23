"""
Extract structured predictions from newspaper text using gpt-4.1 (OpenAI,
non-reasoning). Two arms.

Elections arm: forecasts of the presidential election outcome.
Economy arm: forecasts of economic direction (recession/recovery) with a
horizon, a voice (whose prediction it is), and a hedged flag.

Design: cleanup and extraction in ONE call per page. The model reads noisy
OCR fine; each claim comes back with its own corrected text, so a separate
cleanup pass doubles cost for no gain.

gpt-4.1 was the JeremysShit grading bake-off's winner (0.89/0.90 on
is_prediction/direction, cheapest reliable option, no reasoning-token
overhead) -- see JeremysShit/grade_claims.py and CHANGELOG.md. The call_llm
below keeps the same self-adapting param logic grade_claims.py uses
(max_completion_tokens fallback, dropping unsupported temperature, bumping
the token budget on a truncated empty response) so it degrades gracefully
if pointed at a reasoning model again, but none of that should actually
trigger for gpt-4.1 at MAX_TOKENS below.

Input:  data/raw/{source}_{arm}_{window}.jsonl
Output: data/predictions/pred_{source}_{arm}_{window}.jsonl

Requires: pip install requests ; export OPENAI_API_KEY=...

Usage:
  python extract_predictions.py --source loc --arm economy --window crash_1929 --limit 20
  python extract_predictions.py --source nyt --arm elections --window 1980
"""

import argparse
import json
import os
import time
from pathlib import Path

import requests

MODEL = "gpt-4.1"
API_URL = "https://api.openai.com/v1/chat/completions"
MAX_OCR_CHARS = 12000
MAX_TOKENS = 900
MAX_TOKENS_REASONING_CAP = 4000

# Model -> {"no_max_tokens": bool, "no_temperature": bool, "token_budget": int}.
# Discovered once from the API's own error/finish_reason signals, then reused
# for the rest of the run (same self-adapting approach as grade_claims.py).
_PARAM_ADAPTATIONS = {}


def call_llm(prompt, context_text, api_key, max_retries=5):
    adapt = _PARAM_ADAPTATIONS.setdefault(MODEL, {})
    for attempt in range(max_retries):
        params = {
            "model": MODEL,
            "messages": [{"role": "system", "content": prompt},
                         {"role": "user", "content": context_text}],
        }
        if not adapt.get("no_temperature"):
            params["temperature"] = 0.0
        token_key = "max_completion_tokens" if adapt.get("no_max_tokens") else "max_tokens"
        params[token_key] = adapt.get("token_budget", MAX_TOKENS)
        try:
            resp = requests.post(
                API_URL, json=params, timeout=120,
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json"})
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                raise
            print(f"    retry after error: {e}")
            time.sleep(5 * (attempt + 1))
            continue

        if resp.status_code == 429:
            wait = float(resp.headers.get("Retry-After") or 15.0)
            if attempt < max_retries - 1:
                print(f"    rate limited, waiting {wait:.0f}s", flush=True)
                time.sleep(wait)
                continue
        if resp.status_code == 400:
            err = (resp.json().get("error") or {})
            param = err.get("param")
            if param == "max_tokens" and "max_tokens" in params:
                adapt["no_max_tokens"] = True
                print("    adapting: max_tokens -> max_completion_tokens "
                      "(remembered for rest of run)", flush=True)
                continue
            if param == "temperature" and "temperature" in params:
                adapt["no_temperature"] = True
                print("    adapting: dropping unsupported temperature param "
                      "(remembered for rest of run)", flush=True)
                continue
        if not resp.ok:
            if attempt == max_retries - 1:
                resp.raise_for_status()
            print(f"    retry after HTTP {resp.status_code}", flush=True)
            time.sleep(5 * (attempt + 1))
            continue

        out = resp.json()
        choice = out["choices"][0]
        content = choice["message"]["content"]
        if not content and choice.get("finish_reason") == "length":
            # Reasoning models (o1/o3/gpt-5.x) bill invisible "thinking" tokens
            # against the SAME budget as the visible JSON answer -- empty
            # content here means reasoning ate the whole budget, not a real
            # failure. Bump once, remember it for this model for the rest of
            # the run.
            budget = adapt.get("token_budget", MAX_TOKENS)
            if budget < MAX_TOKENS_REASONING_CAP and attempt < max_retries - 1:
                adapt["token_budget"] = min(budget * 3, MAX_TOKENS_REASONING_CAP)
                print(f"    reasoning ate the token budget (empty answer) -- "
                      f"raising to {adapt['token_budget']} for this model", flush=True)
                continue
        return content or ""
    return ""

ELECTIONS_PROMPT = """You extract election predictions from historical newspaper text.

The text is from American newspapers covering presidential elections 1900-2008.
It comes from noisy OCR (pre-1964) or clean New York Times headline+lead text.
Mentally correct OCR errors while reading.

Find every statement predicting the outcome of the upcoming presidential
election: who will win nationally, who will carry a specific state, margin
forecasts, poll results framed as forecasts, betting odds framed as forecasts.

Return ONLY a JSON array. No markdown fences, no commentary. Each element:
{
  "claim_text": "the prediction, OCR-corrected, max 60 words",
  "predicted_winner": "candidate or party, normalized (e.g. 'Truman', 'Republican')",
  "scope": "national" or "state",
  "state": "state name if scope is state, else null",
  "source_type": "editorial_opinion", "reported_poll", "betting_odds", or "correspondent_analysis",
  "hedged": true or false,
  "attributed_to": "who made the prediction if named, else null"
}

Rules:
- Only forward-looking predictions of the election result. Skip vote tallies
  after the election, campaign schedules, generic praise.
- If no predictions, return [].
- hedged=true when qualified (likely, probably, expected).
- Do not invent candidates. If the winner is unclear, skip the claim."""

ECONOMY_PROMPT = """You extract economic predictions from historical newspaper text.

The text is from American newspapers, 1905-2010. It comes from noisy OCR
(pre-1964) or clean New York Times headline+lead text. Mentally correct OCR
errors while reading.

Find every statement predicting the direction of the US economy: recession or
depression coming, recovery expected, prosperity returning, hard times ahead,
panic over, business improving or worsening. Include predictions about output,
employment, or general business conditions. EXCLUDE stock tips, predictions
about single companies, and pure descriptions of current conditions with no
forward-looking element.

Return ONLY a JSON array. No markdown fences, no commentary. Each element:
{
  "claim_text": "the prediction, OCR-corrected, max 60 words",
  "predicted_direction": "worsen", "improve", or "stable",
  "predicted_state_at_horizon": "recession" or "expansion",
  "horizon_months": integer months ahead the claim refers to; use 6 if unstated,
  "voice": "editorial", "quoted_banker_or_economist", "politician",
           "government_official", "reported_survey", or "other",
  "hedged": true or false,
  "attributed_to": "who made the prediction if named, else null"
}

Rules:
- predicted_state_at_horizon is what the claim implies the economy will be at
  that horizon: "hard times ahead" -> recession; "worst is over",
  "recovery expected", "prosperity will return" -> expansion.
- Only forward-looking claims. If no predictions, return [].
- hedged=true when qualified (may, could, likely, some fear).
- If direction is genuinely unclear, skip the claim."""


def _prompt_and_context(record, arm):
    prompt = ELECTIONS_PROMPT if arm == "elections" else ECONOMY_PROMPT
    text = record["ocr_text"][:MAX_OCR_CHARS]
    # The window id is deliberately withheld from the model: ids like
    # "crash_1929" / "calm_1965" name the OUTCOME, so passing them told the
    # extractor what happened before it labeled the prediction's direction --
    # hindsight leakage into the label. Date and newspaper are legitimate (both
    # were known to whoever wrote the page). The election cycle is kept for the
    # elections arm because it names the contest, not the result.
    context = (f"Newspaper: {record.get('newspaper_title')}\n"
               f"Date: {record.get('date')}\n")
    if arm == "elections":
        context += f"Election cycle: {record.get('cycle')}\n"
    return prompt, f"{context}\nText:\n{text}"


def _parse_claims(raw, record, arm):
    """Shared by the synchronous path (extract_from_page) and the batch
    retrieval path (run_batch) -- same JSON-array parsing + metadata merge
    either way, only how `raw` was obtained (one call vs. a batch result
    file) differs."""
    raw = (raw or "").strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        claims = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(claims, list):
        return []
    out = []
    for c in claims:
        if not isinstance(c, dict) or not c.get("claim_text"):
            continue
        c.update({
            "page_id": record["page_id"],
            "source": record.get("source"),
            "arm": arm,
            "window": record.get("window"),
            "window_kind": record.get("window_kind"),
            "cycle": record.get("cycle"),
            "newspaper_title": record.get("newspaper_title"),
            "lccn": record.get("lccn"),
            "date": record.get("date"),
            "publisher_state": record.get("state"),
        })
        out.append(c)
    return out


def extract_from_page(api_key, record, arm):
    prompt, user_content = _prompt_and_context(record, arm)
    raw = call_llm(prompt, user_content, api_key)
    return _parse_claims(raw, record, arm)


# --- OpenAI Batch API: submit the whole window as one job, poll, retrieve.
# 50% cheaper than synchronous chat/completions calls and one job instead of
# hundreds/thousands of requests -- same rationale and envelope as
# JeremysShit/grade_claims.py's run_batch (ported here since extraction's
# per-page prompt returns a JSON ARRAY of claims rather than one graded
# field, so the request-building and retrieval differ from grade_claims.py's
# version even though the submit/poll/fetch scaffolding is identical).
OPENAI_BATCH_BASE = "https://api.openai.com/v1"


def _batch_request_lines(records, arm, model):
    for record in records:
        prompt, user_content = _prompt_and_context(record, arm)
        body = {
            "model": model,
            "messages": [{"role": "system", "content": prompt},
                         {"role": "user", "content": user_content}],
            "temperature": 0.0,
            "max_tokens": MAX_TOKENS,
        }
        yield json.dumps({"custom_id": record["page_id"], "method": "POST",
                          "url": "/v1/chat/completions", "body": body})


def submit_batch(api_key, records, arm, model):
    jsonl = "\n".join(_batch_request_lines(records, arm, model))
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


def load_records(source, arm, window):
    """window == 'all' merges every raw file for this source/arm (e.g. every
    election cycle's nyt_elections_*.jsonl) so the whole corpus can go through
    the Batch API as one chunked job instead of one small job per window --
    each of those pays OpenAI's ~5-10 min validating/finalizing latency, which
    dominates wall-clock time when windows only have a few dozen pages each."""
    if window == "all":
        paths = sorted(Path("data/raw").glob(f"{source}_{arm}_*.jsonl"))
        if not paths:
            raise SystemExit(f"No data/raw/{source}_{arm}_*.jsonl files found.")
        records = []
        for p in paths:
            with open(p) as f:
                records.extend(json.loads(line) for line in f)
        return records
    in_path = Path(f"data/raw/{source}_{arm}_{window}.jsonl")
    if not in_path.exists():
        raise SystemExit(f"No raw file at {in_path}. Run the {source} "
                         f"downloader for that arm/window first.")
    with open(in_path) as f:
        return [json.loads(line) for line in f]


def run_batch(api_key, records, out_path, arm, model, chunk_size, poll_interval, limit=None):
    """Batch equivalent of the synchronous loop in main(): resume-safe by
    page_id, chunked (one chunk in_progress at a time -- grade_claims.py's
    run_batch hit an enqueued-token cap submitting too much at once), each
    chunk's results parsed with the same _parse_claims as the sync path."""
    done = load_done_ids(out_path)
    records = [r for r in records if r["page_id"] not in done]
    if limit:
        records = records[:limit]
    if not records:
        print("Nothing left to extract.")
        return
    by_id = {r["page_id"]: r for r in records}

    chunks = [records[i:i + chunk_size] for i in range(0, len(records), chunk_size)]
    print(f"Submitting {len(records)} pages to the OpenAI Batch API ({model}) in "
         f"{len(chunks)} chunk(s) of up to {chunk_size}...")

    with open(out_path, "a") as out:
        for ci, chunk in enumerate(chunks, 1):
            print(f"--- chunk {ci}/{len(chunks)}: {len(chunk)} pages ---", flush=True)
            try:
                batch = submit_batch(api_key, chunk, arm, model)
            except requests.exceptions.HTTPError as e:
                body = e.response.text[:500] if e.response is not None else str(e)
                print(f"Batch submission failed: {e}\n{body}")
                return
            print(f"  batch id {batch['id']}, status {batch['status']}")
            result = poll_batch(api_key, batch["id"], interval=poll_interval)

            err_codes = {e.get("code") for e in (result.get("errors") or {}).get("data", [])}
            if result["status"] == "failed" and err_codes == {"token_limit_exceeded"}:
                print("  enqueued-token limit hit; waiting 60s for other in_progress "
                     "batches to clear, then retrying this chunk once", flush=True)
                time.sleep(60)
                try:
                    batch = submit_batch(api_key, chunk, arm, model)
                    result = poll_batch(api_key, batch["id"], interval=poll_interval)
                except requests.exceptions.HTTPError as e:
                    print(f"  retry failed: {e}")

            n_ok = 0
            if result.get("output_file_id"):
                for line in fetch_file(api_key, result["output_file_id"]).splitlines():
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    record = by_id.get(rec["custom_id"])
                    if record is None:
                        continue
                    body = (rec.get("response") or {}).get("body") or {}
                    try:
                        content = body["choices"][0]["message"]["content"]
                    except Exception:
                        content = ""
                    claims = _parse_claims(content, record, arm)
                    if not claims:
                        out.write(json.dumps({"page_id": record["page_id"],
                                              "no_predictions": True}) + "\n")
                    for c in claims:
                        out.write(json.dumps(c) + "\n")
                    n_ok += 1
            out.flush()
            if result.get("error_file_id"):
                err_text = fetch_file(api_key, result["error_file_id"])
                err_lines = [l for l in err_text.splitlines() if l.strip()]
                print(f"  {len(err_lines)} request(s) errored in this chunk -- first: "
                     f"{err_lines[0][:300] if err_lines else ''}")
            print(f"  chunk {ci}/{len(chunks)} {result['status']}: {n_ok}/{len(chunk)} pages "
                 f"processed -> {out_path}")
            if result["status"] != "completed" or n_ok < len(chunk):
                print("Chunk did not fully complete -- stopping here; rerun the same "
                     "command to resume (page_id-based resume skips completed pages).")
                return
    print(f"\nAll {len(chunks)} chunk(s) complete -> {out_path}")


def load_done_ids(out_path):
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["page_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def main():
    global MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["loc", "nyt"], required=True)
    ap.add_argument("--arm", choices=["elections", "economy"], required=True)
    ap.add_argument("--window", required=True,
                    help="elections: the year (e.g. 1948) or 'all' for every "
                         "downloaded cycle in one (chunked) run; economy: window_id "
                         "(e.g. crash_1929) or 'all'")
    ap.add_argument("--limit", type=int, default=None, help="max pages to process")
    ap.add_argument("--sleep", type=float, default=0.35,
                    help="seconds between calls (matches grade_claims.py's "
                         "validated gpt-4.1 full-corpus run)")
    ap.add_argument("--model", default=MODEL,
                    help="chat-completions model for both the sync and --batch paths "
                         "-- default is gpt-4.1, this project's bake-off-validated "
                         "grader (see CHANGELOG); pass gpt-4.1-mini for a cheaper run "
                         "(re-run the Stage 3 kappa check before trusting its labels, "
                         "since the bake-off validated gpt-4.1, not mini)")
    ap.add_argument("--batch", action="store_true",
                    help="extract via the OpenAI Batch API instead of one call per "
                         "page -- 50%% cheaper, async, meant to be backgrounded")
    ap.add_argument("--batch-chunk-size", type=int, default=150,
                    help="pages per Batch API submission (--batch) -- kept small "
                         "since each page's prompt can run up to ~12k OCR chars, "
                         "much heavier per-request than grade_claims.py's short "
                         "claim-grading prompts, so the org enqueued-token cap bites "
                         "at a much lower page count")
    ap.add_argument("--poll-interval", type=float, default=30,
                    help="seconds between batch status checks (--batch)")
    args = ap.parse_args()
    MODEL = args.model

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY first.")

    records = load_records(args.source, args.arm, args.window)
    out_dir = Path("data/predictions")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pred_{args.source}_{args.arm}_{args.window}.jsonl"

    if args.batch:
        run_batch(api_key, records, out_path, args.arm, args.model,
                 args.batch_chunk_size, args.poll_interval, args.limit)
        return

    done = load_done_ids(out_path)
    processed = 0
    with open(out_path, "a") as out:
        for record in records:
            if record["page_id"] in done:
                continue
            try:
                claims = extract_from_page(api_key, record, args.arm)
            except requests.RequestException as e:
                print(f"API error on {record['page_id']}: {e}")
                continue
            if not claims:
                out.write(json.dumps({"page_id": record["page_id"],
                                      "no_predictions": True}) + "\n")
            for c in claims:
                out.write(json.dumps(c) + "\n")
            out.flush()
            time.sleep(args.sleep)
            done.add(record["page_id"])
            processed += 1
            if processed % 20 == 0:
                print(f"processed {processed} pages")
            if args.limit and processed >= args.limit:
                break
    print(f"done, processed {processed} pages -> {out_path}")


if __name__ == "__main__":
    main()
