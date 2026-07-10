"""
LLM grading of scraped newspaper claims (DeepSeek or any OpenAI-compatible API).

Reads claims_raw.csv, asks the model to grade each claim against the project
rubric, writes claims_graded.csv. Also exports a random 20% sample for HUMAN
double-coding (validation_sample.csv) and can compute Cohen's kappa between the
humans and the model once the sample is filled in.

Setup (one of):
    set DEEPSEEK_API_KEY=sk-...                       (PowerShell: $env:DEEPSEEK_API_KEY="sk-...")
    set OPENAI_API_KEY=... and --base-url/--model overrides for another provider

Usage:
    python grade_claims.py                             # grade claims_raw.csv
    python grade_claims.py --limit 20                  # cheap test run first!
    python grade_claims.py --kappa validation_sample_filled.csv   # after humans grade

Cost: DeepSeek prices are ~$0.3/1M input tokens; a 500-claim corpus is a few cents.
"""

import argparse
import csv
import json
import os
import random
import time
import urllib.request

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


def call_llm(prompt, model, base_url, api_key, max_retries=3):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions", data=body,
        # Groq (and other Cloudflare-fronted providers) 403 the default
        # Python-urllib User-Agent; a browser-like UA is required.
        headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0",
                 "Authorization": f"Bearer {api_key}"})
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                out = json.load(resp)
            return json.loads(out["choices"][0]["message"]["content"])
        except urllib.error.HTTPError as e:
            # 429 = rate limit. Honor the provider's Retry-After (Groq sends it)
            # and don't spend the failure budget on it — free tiers throttle,
            # they don't fail, so keep waiting rather than dropping the claim.
            if e.code == 429:
                wait = int(e.headers.get("Retry-After", 15)) + 1
                print(f"    rate limited, sleeping {wait}s")
                time.sleep(wait)
                continue
            if attempt == max_retries - 1:
                raise
            print(f"    retry after error: {e}")
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

    # Resume-safety: reuse claims already graded in a prior run (is_prediction
    # filled), regrade only the missing/failed ones. Free-tier rate limits and
    # daily request caps mean a full 1,300-claim run spans several sessions;
    # this makes each rerun cheap and picks up stragglers left by rate limiting.
    done = {}
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("is_prediction", "").strip():
                    done[r["claim_id"]] = r
        print(f"Resuming: {len(done)} claims already graded in {args.out}")

    todo = [c for c in claims if c["claim_id"] not in done]
    print(f"Grading {len(todo)} claims with {args.model} at {args.base_url} "
          f"({len(claims) - len(todo)} already done)")

    graded = list(done.values())
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        for r in done.values():                       # preserve prior grades
            writer.writerow({k: r.get(k, "") for k in out_fields})
        f.flush()
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
            if g:                                      # only count real grades
                graded.append(row)
            if i % 25 == 0:
                print(f"  {i}/{len(todo)} graded ({len(graded)} total)")
                f.flush()
    n_pred = sum(1 for r in graded if r["is_prediction"] == "yes")
    print(f"\nDone: {len(graded)} graded -> {args.out}  ({n_pred} judged real predictions)")

    # Export the human validation sample (only rows the model calls predictions,
    # since those are the ones that reach the scoring stage).
    preds = [r for r in graded if r["is_prediction"] == "yes"]
    random.seed(42)
    n_sample = min(len(preds), max(5, int(len(preds) * args.sample_frac)))
    sample = random.sample(preds, n_sample) if preds else []
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
    ap.add_argument("--sample-frac", type=float, default=0.2)
    ap.add_argument("--kappa", metavar="FILLED_CSV",
                    help="compute model-vs-human Cohen's kappa from a filled validation sample")
    args = ap.parse_args()
    if args.kappa:
        kappa_report(args.kappa)
    else:
        grade(args)
