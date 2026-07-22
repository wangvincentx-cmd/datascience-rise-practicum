"""
Extract structured predictions from newspaper text using Claude. Two arms.

Elections arm: forecasts of the presidential election outcome.
Economy arm: forecasts of economic direction (recession/recovery) with a
horizon, a voice (whose prediction it is), and a hedged flag.

Design: cleanup and extraction in ONE call per page. Claude reads noisy OCR
fine; each claim comes back with its own corrected text, so a separate
cleanup pass doubles cost for no gain.

Input:  data/raw/{source}_{arm}_{window}.jsonl
Output: data/predictions/pred_{source}_{arm}_{window}.jsonl

Requires: pip install anthropic ; export ANTHROPIC_API_KEY=...

Usage:
  python extract_predictions.py --source loc --arm economy --window crash_1929 --limit 20
  python extract_predictions.py --source nyt --arm elections --window 1980
"""

import argparse
import json
import os
from pathlib import Path

import anthropic

MODEL = "claude-haiku-4-5"   # bump to a sonnet model if quality lags on hard pages
MAX_OCR_CHARS = 12000

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


def extract_from_page(client, record, arm):
    prompt = ELECTIONS_PROMPT if arm == "elections" else ECONOMY_PROMPT
    text = record["ocr_text"][:MAX_OCR_CHARS]
    context = (f"Newspaper: {record.get('newspaper_title')}\n"
               f"Date: {record.get('date')}\n"
               f"Window: {record.get('window')}\n")
    if arm == "elections":
        context += f"Election cycle: {record.get('cycle')}\n"
    msg = client.messages.create(
        model=MODEL, max_tokens=2000, system=prompt,
        messages=[{"role": "user", "content": f"{context}\nText:\n{text}"}],
    )
    raw = msg.content[0].text.strip()
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
    ap.add_argument("--source", choices=["loc", "nyt", "proquest"], required=True)
    ap.add_argument("--arm", choices=["elections", "economy"], required=True)
    ap.add_argument("--window", required=True,
                    help="elections: the year (e.g. 1948); economy: window_id (e.g. crash_1929)")
    ap.add_argument("--limit", type=int, default=None, help="max pages to process")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY first.")

    client = anthropic.Anthropic()
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
                claims = extract_from_page(client, record, args.arm)
            except anthropic.APIError as e:
                print(f"API error on {record['page_id']}: {e}")
                continue
            if not claims:
                out.write(json.dumps({"page_id": record["page_id"],
                                      "no_predictions": True}) + "\n")
            for c in claims:
                out.write(json.dumps(c) + "\n")
            out.flush()
            done.add(record["page_id"])
            processed += 1
            if processed % 20 == 0:
                print(f"processed {processed} pages")
            if args.limit and processed >= args.limit:
                break
    print(f"done, processed {processed} pages -> {out_path}")


if __name__ == "__main__":
    main()
