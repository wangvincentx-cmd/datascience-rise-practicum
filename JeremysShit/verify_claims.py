"""
Stage 2: filter extracted claims. Turns a high-recall extractor into a
high-recall AND high-precision pipeline, for almost no money.

The bake-off (gold_extraction/RESULTS.md) found that no single model is good at
both halves of this task, and that the failures are opposite:

    gpt-oss-120b            recall 0.71, precision 0.54   -- finds them, over-collects
    gemini-3.5-flash-lite   recall 0.33, precision 0.90   -- rarely wrong, misses most

Extraction is the expensive half, because it reads whole 21k-char pages.
Verification is cheap, because it only ever sees the candidate quote plus a few
hundred characters of surrounding context -- roughly 3% of the token volume. So
the sensible pipeline is: extract with the model that finds the most, then hand
the candidates to a judge that is good at saying no.

Judging is batched per page: one call carries all of a page's candidates with
their local context, which keeps the request count at one per page rather than
one per claim.

Usage (from JeremysShit/):
    python verify_claims.py --claims gold_extraction/pred_gptoss120b.jsonl \\
        --pages gold_extraction/gold_pages.jsonl \\
        --out gold_extraction/pred_gptoss120b_verified.jsonl \\
        --model gemini-3.5-flash-lite \\
        --base-url https://generativelanguage.googleapis.com/v1beta/openai \\
        --api-key-env GEMINI_API_KEY

Claims that the judge keeps are written through unchanged, plus `verify_reason`.
Dropped claims go to <out>.dropped.jsonl so the filter itself stays auditable --
a silent filter is untestable, and this one has to be scored against the gold
like everything else.
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

from extract_llm import (SSL_CONTEXT, TOKEN_RE, call_llm, parse_claims)  # noqa: F401

CONTEXT_CHARS = 400  # each side of the quote

VERIFY_PROMPT = """You are checking candidate economic PREDICTIONS pulled from a page of an American newspaper printed between 1900 and 1963. Another system proposed them; many are wrong. Your job is to say which are genuine.

KEEP a candidate only if it makes a FALSIFIABLE CLAIM ABOUT FUTURE ECONOMIC CONDITIONS -- business conditions, prices, employment, markets, prosperity, recession or panic.

DROP it if it is any of the following, however economic its vocabulary:
- ADVERTISING or promotional copy, including New Year "prosperity" greetings and sale-price claims.
- FICTION: dialogue from a serialized novel or a humour sketch.
- A REPRINT from a "Twenty Years Ago" / "From Our Files" column (written decades before this page's date).
- A DESCRIPTION OF THE PRESENT OR PAST rather than the future: "steel production is climbing", "business is good today", "failures were larger than last year".
- A POST-MORTEM ON A FORECAST THAT HAD ALREADY FAILED by the time this page was printed. If the passage's point is that earlier predictions were WRONG -- headlines like "Prophecies Gone Wrong", phrases like "how the prophets were mistaken" -- then the page is not making that forecast, it is burying it. DROP. But a forecast that is still LIVE and merely doubted or disputed by the writer is a real forecast: KEEP.
- A REFUSAL to forecast: "it is too early to say", "no one can know".
- A CONDITIONAL with no committed direction, or a claim about a bill that has not passed.
- An ANNOUNCEMENT of a speech, meeting or report ABOUT the outlook, or a schedule (store opening, construction timetable).
- POLICY ADVOCACY: "the Reserve Board should", "Congress must".
- NON-ECONOMIC: elections, legislation, weather, sport, health.
- A STOCK TIP about one company, or investment arithmetic.
- Text too mangled by OCR to interpret.

The newspaper is {newspaper}, dated {date}.

Candidates, each with the text surrounding it on the page:

{candidates}

Return ONLY a JSON array, one element per candidate, in the same order:
[{{"id": <the candidate's id>, "keep": true or false, "reason": "<8 words or fewer>"}}]

Judge each candidate on its own. Keeping and dropping are equally costly: dropping a real forecast loses evidence, keeping a false one injects fake signal into the results."""


def local_context(page_text, quote, width=CONTEXT_CHARS):
    """The quote plus surrounding text, so the judge can see whether it sits in
    an advertisement, a novel, or a reprint column. Falls back to a token-anchor
    search when the quote does not appear literally (extractors repair OCR)."""
    idx = page_text.find(quote[:60])
    if idx < 0:
        anchor = " ".join(TOKEN_RE.findall(quote.lower())[:5])
        if anchor:
            m = re.search(re.escape(anchor).replace(r"\ ", r"\W+"),
                          page_text, re.IGNORECASE)
            idx = m.start() if m else -1
    if idx < 0:
        return quote
    start = max(0, idx - width)
    end = min(len(page_text), idx + len(quote) + width)
    return page_text[start:end]


def verify_page(page, claims, model, base_url, api_key):
    blocks = []
    for i, c in enumerate(claims):
        ctx = local_context(page["ocr_text"], c.get("quote", ""))
        blocks.append(f"--- candidate {i} ---\n"
                      f"CANDIDATE QUOTE: {c.get('quote','')}\n"
                      f"SURROUNDING PAGE TEXT: ...{ctx}...")
    prompt = VERIFY_PROMPT.format(
        newspaper=page.get("publisher") or "unknown",
        date=page.get("date") or "unknown",
        candidates="\n\n".join(blocks))
    raw, usage = call_llm(prompt, model, base_url, api_key)
    verdicts = {}
    for v in parse_claims(raw):
        if isinstance(v, dict) and "id" in v:
            try:
                verdicts[int(v["id"])] = v
            except (TypeError, ValueError):
                continue
    return verdicts, usage


def run(args):
    pages = {p["page_id"]: p for p in
             (json.loads(l) for l in open(args.pages, encoding="utf-8") if l.strip())}
    claims = [json.loads(l) for l in open(args.claims, encoding="utf-8") if l.strip()]
    by_page = {}
    for c in claims:
        by_page.setdefault(c.get("page_id"), []).append(c)

    api_key = "" if args.api_key_env == "NONE" else os.environ.get(args.api_key_env, "")
    if not api_key and args.api_key_env != "NONE":
        raise SystemExit(f"set {args.api_key_env}")

    out_path = Path(args.out)
    dropped_path = out_path.with_suffix(out_path.suffix + ".dropped.jsonl")
    kept = dropped = unjudged = 0
    totals = {"prompt_tokens": 0, "completion_tokens": 0}

    with open(out_path, "w", encoding="utf-8") as fh, \
         open(dropped_path, "w", encoding="utf-8") as dh:
        for i, (page_id, page_claims) in enumerate(by_page.items(), 1):
            page = pages.get(page_id)
            if page is None:
                for c in page_claims:
                    fh.write(json.dumps(c, ensure_ascii=False) + "\n")
                    kept += 1
                continue
            verdicts, usage = verify_page(page, page_claims, args.model,
                                          args.base_url, api_key)
            for k in totals:
                totals[k] += usage.get(k, 0) or 0
            for j, c in enumerate(page_claims):
                v = verdicts.get(j)
                if v is None:
                    # No verdict returned. Keep it -- a judge that failed to
                    # answer must not silently delete evidence.
                    unjudged += 1
                    c["verify_reason"] = "unjudged (kept by default)"
                    fh.write(json.dumps(c, ensure_ascii=False) + "\n")
                    kept += 1
                elif v.get("keep"):
                    c["verify_reason"] = v.get("reason", "")
                    fh.write(json.dumps(c, ensure_ascii=False) + "\n")
                    kept += 1
                else:
                    c["verify_reason"] = v.get("reason", "")
                    dh.write(json.dumps(c, ensure_ascii=False) + "\n")
                    dropped += 1
            print(f"  [{i}/{len(by_page)}] {page.get('date')}  "
                  f"{len(page_claims)} candidates -> "
                  f"{sum(1 for j in range(len(page_claims)) if verdicts.get(j, {}).get('keep', True))} kept",
                  flush=True)
            if args.sleep:
                time.sleep(args.sleep)

    print(f"\nkept {kept}, dropped {dropped}"
          + (f", {unjudged} unjudged (kept)" if unjudged else ""))
    print(f"dropped claims -> {dropped_path}")
    print(f"tokens: {totals['prompt_tokens']:,} in / {totals['completion_tokens']:,} out")
    if by_page:
        per = (totals["prompt_tokens"] + totals["completion_tokens"]) / len(by_page)
        print(f"~{per:,.0f} tokens/page -> ~{per * 2192 / 1e6:.1f}M for the "
              f"2,192-page corpus")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--claims", required=True)
    ap.add_argument("--pages", default="data/pages.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--base-url", default="https://api.deepinfra.com/v1/openai")
    ap.add_argument("--api-key-env", default="DEEPINFRA_API_KEY")
    ap.add_argument("--sleep", type=float, default=0.0)
    args = ap.parse_args()
    run(args)
