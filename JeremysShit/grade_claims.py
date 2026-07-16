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
import ssl
import time
import urllib.error
import urllib.request

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
                g = call_llm(prompt, args.model, args.base_url, api_key,
                            min_tokens=args.min_tokens)
            except DailyCapReached as e:
                print(f"\n=== {e} ===")
                print(f"Stopping cleanly with {len(graded)} claims graded and saved.")
                print(f"Rerun the same command once the cap resets; it will resume "
                      f"from claim {i} of {len(todo)} remaining.", flush=True)
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
    ap.add_argument("--min-tokens", type=int, default=None,
                    help="seed the reasoning-model token budget above the 300 "
                         "default -- skips the wasted first-attempt retry tax "
                         "when a survey already showed this model needs more")
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
