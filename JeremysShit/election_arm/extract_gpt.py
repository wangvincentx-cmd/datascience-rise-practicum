"""
LLM extraction of economy-arm predictions via TDM Studio's built-in GPT proxy.

Runs INSIDE the ProQuest TDM Studio workbench. The VM has no general internet,
but ProQuest ships a proxied OpenAI endpoint (see their sample notebook
'GPT_Batch_Processing.ipynb'): the standard `openai` SDK pointed at a ProQuest
`base_url`, authenticated with a key file ProQuest drops in the workspace.

This is the LLM-quality replacement for extract_offline.py: it runs the SAME
economy prompt as extract_predictions.py and writes the SAME pred_*.jsonl the
scorer reads (analyze_economy.py globs pred_*_economy_*.jsonl).

To avoid transcribing ProQuest's long base_url / key path by hand, this script
auto-discovers them from the sample notebook you already exported to
gpt_sample.txt:

    jupyter nbconvert --to script --stdout \
      ".../ProQuest TDM Studio Samples/GPT_Batch_Processing.ipynb" > gpt_sample.txt

Override with --base-url / --key-file / --model if discovery misses.

The output has NO article text (labels only), so it is safe to Export.

Usage (in the workbench):
  python extract_gpt.py --source proquest --window gfc_2008 --limit 10   # TEST FIRST
  python extract_gpt.py --source proquest --window gfc_2008

Start with --limit 10 and check ProQuest's usage page: they enforce daily and
per-minute LLM limits. Resume-safe: processed page_ids are skipped on rerun.
"""

import argparse
import json
import re
import time
from pathlib import Path

from openai import OpenAI, OpenAIError

MAX_OCR_CHARS = 12000
MODEL_FALLBACK = "gpt-4o-mini"
REQUEST_DELAY = 0.5     # be gentle with ProQuest's shared proxy
MAX_RETRIES = 6     # ride out per-minute (TPM/RPM) limits, which reset in ~60s

# Identical to ECONOMY_PROMPT in extract_predictions.py -- keep them in sync.
ECONOMY_PROMPT = """You extract economic predictions from historical newspaper text.

The text is from American newspapers, 1905-2010. It comes from noisy OCR
(pre-1964) or clean New York Times headline+lead text. Mentally correct OCR
errors while reading.

Find every statement predicting the direction of the US economy: recession or
depression coming, recovery expected, prosperity returning, hard times ahead,
panic over, business improving or worsening. Include predictions about output,
employment, or general business conditions. EXCLUDE stock tips, predictions
about single companies, forecasts about a foreign economy (UK, Europe, Japan,
etc.), and pure descriptions of current conditions with no forward-looking
element.

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
- Only forward-looking claims about the US national economy; skip forecasts
  about other countries' economies. If no predictions, return [].
- hedged=true when qualified (may, could, likely, some fear).
- If direction is genuinely unclear, skip the claim."""


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
    return OpenAI(api_key=api_key, base_url=base_url), model


class RateLimitReached(Exception):
    """The proxy's per-DAY cost/quota cap is exhausted (needs ~a day to reset).
    Retrying won't help until then, so we stop cleanly and resume on the next
    run instead of grinding."""


def _is_daily_cap(err):
    """True ONLY for the per-DAY cost/quota cap. A per-minute (tokens- or
    requests-per-minute) 429 is NOT this -- it resets in ~60s, so it must fall
    through to the retry/backoff path instead of stopping the whole batch.
    ProQuest's daily error reads 'Application cost/day rate exceeded'."""
    s = str(err).lower()
    return any(k in s for k in ("day rate", "day-rate", "cost/day", "per day",
                                "per-day", "daily limit", "daily quota"))


def _retry_after(err):
    """Seconds to wait from a Retry-After header, if the error carries one."""
    try:
        ra = err.response.headers.get("retry-after")
        return float(ra) if ra else None
    except Exception:
        return None


def call_model(client, model, record):
    """Return the model's raw text, or None on a non-fatal failure after retries.
    Raises RateLimitReached on a quota error (caller should stop, not retry)."""
    text = (record.get("ocr_text") or "")[:MAX_OCR_CHARS]
    context = (f"Newspaper: {record.get('newspaper_title')}\n"
               f"Date: {record.get('date')}\nWindow: {record.get('window')}\n")
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model, max_tokens=2000,
                messages=[{"role": "system", "content": ECONOMY_PROMPT},
                          {"role": "user", "content": f"{context}\nText:\n{text}"}],
            )
            return resp.choices[0].message.content.strip()
        except OpenAIError as e:
            if _is_daily_cap(e):
                raise RateLimitReached(str(e))   # needs ~a day; stop the batch
            # Transient: per-minute 429, timeout, 5xx. Honor Retry-After if the
            # server sent one (per-minute limits reset fast), else back off.
            wait = _retry_after(e) or (2 ** attempt * 5)
            print(f"  transient error, retry in {wait:.0f}s: {e}", flush=True)
            time.sleep(wait)
    return None


def parse_claims(raw, record):
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
            "arm": "economy",
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
    ap.add_argument("--source", choices=["loc", "nyt", "proquest"], required=True)
    ap.add_argument("--window", required=True, help="economy window_id, e.g. gfc_2008")
    ap.add_argument("--limit", type=int, default=None, help="max pages (test with 10 first)")
    ap.add_argument("--sample", default="gpt_sample.txt",
                    help="ProQuest GPT sample export to read base_url/key from")
    ap.add_argument("--base-url", help="override the discovered proxy base_url")
    ap.add_argument("--key-file", help="override the discovered API key file path")
    ap.add_argument("--model", help="override the discovered model name")
    args = ap.parse_args()

    in_path = Path(f"data/raw/{args.source}_economy_{args.window}.jsonl")
    if not in_path.exists():
        raise SystemExit(f"No raw file at {in_path}. Run tdm_parse.py first.")
    client, model = make_client(args)

    out_dir = Path("data/predictions")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pred_{args.source}_economy_{args.window}.jsonl"
    done = load_done_ids(out_path)

    processed = with_pred = total_claims = 0
    with open(in_path) as f, open(out_path, "a") as out:
        for line in f:
            record = json.loads(line)
            if record["page_id"] in done:
                continue
            try:
                raw = call_model(client, model, record)
            except RateLimitReached as e:
                print(f"\n*** DAILY/RATE LIMIT reached: {e}")
                print(f"*** Stopping cleanly at {processed} pages this run "
                      f"({with_pred} with predictions, {total_claims} claims).")
                print("*** This page is NOT marked done -- just re-run after the "
                      "quota resets and it resumes exactly here.")
                raise SystemExit(2)   # exit 2 signals the batch runner to stop
            if raw is None:
                # A non-quota failure that exhausted retries. Do NOT mark done and
                # do NOT record a fake 'no_predictions' -- leave it for a rerun to
                # retry, so real articles aren't lost as false empties.
                print(f"  giving up on {record['page_id']} after retries "
                      f"(left unmarked; will retry next run)")
                continue
            claims = parse_claims(raw, record)   # call succeeded: [] is a genuine empty
            if not claims:
                out.write(json.dumps({"page_id": record["page_id"],
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
            time.sleep(REQUEST_DELAY)
            if args.limit and processed >= args.limit:
                break
    print(f"done: {processed} pages, {with_pred} with >=1 prediction, "
          f"{total_claims} claims -> {out_path}")


if __name__ == "__main__":
    main()
