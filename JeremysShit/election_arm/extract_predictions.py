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


def extract_from_page(api_key, record, arm):
    prompt = ELECTIONS_PROMPT if arm == "elections" else ECONOMY_PROMPT
    text = record["ocr_text"][:MAX_OCR_CHARS]
    context = (f"Newspaper: {record.get('newspaper_title')}\n"
               f"Date: {record.get('date')}\n"
               f"Window: {record.get('window')}\n")
    if arm == "elections":
        context += f"Election cycle: {record.get('cycle')}\n"
    raw = call_llm(prompt, f"{context}\nText:\n{text}", api_key).strip()
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["loc", "nyt"], required=True)
    ap.add_argument("--arm", choices=["elections", "economy"], required=True)
    ap.add_argument("--window", required=True,
                    help="elections: the year (e.g. 1948); economy: window_id (e.g. crash_1929)")
    ap.add_argument("--limit", type=int, default=None, help="max pages to process")
    ap.add_argument("--sleep", type=float, default=0.35,
                    help="seconds between calls (matches grade_claims.py's "
                         "validated gpt-4.1 full-corpus run)")
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY first.")

    in_path = Path(f"data/raw/{args.source}_{args.arm}_{args.window}.jsonl")
    if not in_path.exists():
        raise SystemExit(f"No raw file at {in_path}. Run the {args.source} "
                         f"downloader for that arm/window first.")
    out_dir = Path("data/predictions")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pred_{args.source}_{args.arm}_{args.window}.jsonl"
    done = load_done_ids(out_path)

    processed = 0
    with open(in_path) as f, open(out_path, "a") as out:
        for line in f:
            record = json.loads(line)
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
