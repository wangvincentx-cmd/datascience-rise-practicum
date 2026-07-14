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
import ssl
import time
import urllib.error
import urllib.request

USER_AGENT = "BU-RISE-student-research/0.2 (economic prediction accuracy study)"

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
- is_prediction: "yes" if the sentence makes a falsifiable claim about FUTURE economic
  conditions (business, prices, employment, markets, prosperity, recession/panic).
  "no" for retrospectives, ads, pure descriptions of the present, or OCR garbage.
- topic: one of "general_business", "prices", "employment", "markets", "other"
- direction: the predicted direction of ECONOMIC CONDITIONS: "improve", "worsen",
  "no_change", or "unclear"
- price_direction: only if topic is "prices": "up", "down", "stable", else "na"
- unemployment_direction: only if topic is "employment": "up", "down", "stable", else "na"
- horizon_months: best estimate of the prediction horizon: 6, 12, or "vague"
- confidence: "assertive" (will, is certain, undoubtedly) or "hedged" (may, might,
  likely, is expected, if)
- voice: "editorial" (the paper's own view), "quoted_expert" (banker, economist,
  businessman), "quoted_official" (president, senator, government), or "unclear"
- speaker_name: the personal name of whoever makes the prediction, if one is stated
  or clearly implied in the sentence (e.g. "Roger Babson", "Secretary Mellon"),
  else "na". Names only — not organizations or newspapers.

The sentence was printed on {date} during the "{episode}" period. Sentence:
\"\"\"{quote}\"\"\"
"""

GRADE_FIELDS = ["is_prediction", "topic", "direction", "price_direction",
                "unemployment_direction", "horizon_months", "confidence", "voice",
                "speaker_name"]


def call_llm(prompt, model, base_url, api_key, max_retries=5):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }).encode()
    for attempt in range(max_retries):
        req = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {api_key}",
                     # Groq's edge rejects the default Python-urllib agent (CF 1010).
                     "User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=120, context=SSL_CONTEXT) as resp:
                out = json.load(resp)
            return json.loads(out["choices"][0]["message"]["content"])
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < max_retries - 1:
                wait = float(e.headers.get("Retry-After") or 0) or 10 * (2 ** attempt)
                print(f"    rate limited, waiting {wait:.0f}s")
                time.sleep(wait)
                continue
            if attempt == max_retries - 1:
                raise
            print(f"    retry after HTTP {e.code}")
            time.sleep(5 * (attempt + 1))
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"    retry after error: {e}")
            time.sleep(5 * (attempt + 1))


def grade(args):
    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set DEEPSEEK_API_KEY (or OPENAI_API_KEY) first.")
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

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        for row in graded:
            writer.writerow({k: row.get(k, "") for k in out_fields})
        for i, row in enumerate(todo, 1):
            prompt = RUBRIC_PROMPT.format(date=row["date"], episode=row["episode"],
                                          quote=row["quote"])
            try:
                g = call_llm(prompt, args.model, args.base_url, api_key)
            except Exception as e:
                print(f"  claim {row['claim_id']}: FAILED ({e}) — marked ungraded")
                g = {}
            for k in GRADE_FIELDS:
                row[k] = str(g.get(k, ""))
            writer.writerow(row)
            graded.append(row)
            f.flush()
            if i % 25 == 0:
                print(f"  {i}/{len(todo)} graded")
            if args.sleep and i < len(todo):
                time.sleep(args.sleep)
    n_pred = sum(1 for r in graded if r["is_prediction"] == "yes")
    print(f"\nDone: {len(graded)} graded -> {args.out}  ({n_pred} judged real predictions)")

    # Export the human validation sample (only rows the model calls predictions,
    # since those are the ones that reach the scoring stage).
    preds = [r for r in graded if r["is_prediction"] == "yes"]
    if not preds:
        print("No graded predictions — leaving any existing validation_sample.csv alone.")
        return
    random.seed(42)
    sample = random.sample(preds, min(len(preds), max(5, int(len(preds) * args.sample_frac))))
    with open("validation_sample.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields + [f"human_{k}" for k in GRADE_FIELDS])
        w.writeheader()
        for r in sample:
            w.writerow({**r, **{f"human_{k}": "" for k in GRADE_FIELDS}})
    print(f"Human validation sample ({len(sample)} rows) -> validation_sample.csv")
    print("Two graders: fill the human_* columns independently, then run --kappa on each.")


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
    ap.add_argument("--overwrite", action="store_true",
                    help="ignore an existing --out file instead of resuming from it")
    ap.add_argument("--sample-frac", type=float, default=0.2)
    ap.add_argument("--kappa", metavar="FILLED_CSV",
                    help="compute model-vs-human Cohen's kappa from a filled validation sample")
    args = ap.parse_args()
    if args.kappa:
        kappa_report(args.kappa)
    else:
        grade(args)
