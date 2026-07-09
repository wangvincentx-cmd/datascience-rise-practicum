"""
Classify each NYT candidate article (from link_coverage.py) as about-this-bill
or not, and if so, whether it predicts pass/fail/neutral. Then aggregate to
bill-level press features (Section 8).

Repoints the reused LLM structured-extraction pattern (see JeremysShit/
election_arm/extract_predictions.py: one call per item, fixed JSON schema,
fence stripping, malformed-reply handling) to DeepSeek via its OpenAI-
compatible API, and to the article-about-this-bill + passage-prediction
schema instead of the elections/economy claim schema.

Design: article-match and passage-prediction are decided in ONE call per
article, same rationale as the reused pattern -- a separate matching pass
would double cost for no gain, and the model can hold both judgments in
context at once.

Input:  data/press_raw/{congress}.jsonl   (from link_coverage.py)
        data/bills/{congress}.jsonl       (for bill context + introduced_date)
Output: data/press_labeled/{congress}.jsonl   (per-article classification)
        data/press_features_{congress}.csv    (bill-level aggregated features)

Requires: pip install openai ; export DEEPSEEK_API_KEY=...

Usage:
  python extract_press.py --congress 118 --limit 20
  python extract_press.py --congress 118
"""

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
from openai import OpenAI

MODEL = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MAX_ARTICLE_CHARS = 4000

PROMPT = """You are given metadata about a specific U.S. Congressional bill and the text
of a candidate newspaper article that MIGHT be about that bill. The article
text is headline + abstract + lead paragraph + snippet only (New York Times
Article Search API), never the full article.

Decide:
1. Is this article actually ABOUT this specific bill (not a different bill on
   a similar topic, not an earlier or later Congress's version of a similar
   idea, not just a general mention of the policy area)? Judge conservatively
   -- if genuinely unsure, say no.
2. If yes, does the article predict the bill will PASS, FAIL, or is its tone
   NEUTRAL / purely descriptive about the outcome? Is that prediction FIRM
   (confident language: "will become law", "is on track to pass") or HEDGED
   (uncertain language: "could", "may", "faces an uphill battle")?

Return ONLY a JSON object. No markdown fences, no commentary.
{
  "about_this_bill": true or false,
  "prediction": "pass", "fail", "neutral", or null,
  "confidence": "firm", "hedged", or null
}

Rules:
- If about_this_bill is false, prediction and confidence must both be null.
- prediction judges the article's view of the bill's ultimate fate (become
  law or not), not just its next procedural step (e.g. "passed committee" is
  not itself a fate prediction unless the article frames it as one).
- If about_this_bill is true but the article makes no forward-looking
  judgment, use prediction "neutral" and confidence null."""


def bill_context_str(bill):
    return (f"Congress: {bill.get('congress')}\n"
           f"Bill: {str(bill.get('bill_type')).upper()} {bill.get('number')}\n"
           f"Title: {bill.get('title')}\n"
           f"Sponsor party: {bill.get('sponsor_party')}\n"
           f"Sponsor state: {bill.get('sponsor_state')}\n"
           f"Policy area: {bill.get('policy_area')}\n"
           f"Introduced date: {bill.get('introduced_date')}\n")


def extract_from_article(client, bill, article, model=MODEL):
    context = bill_context_str(bill)
    article_text = (f"Headline: {article.get('headline')}\n"
                    f"Date: {article.get('pub_date')}\n"
                    f"Section: {article.get('section')}\n"
                    f"Text:\n{(article.get('snippet_text') or '')[:MAX_ARTICLE_CHARS]}")
    resp = client.chat.completions.create(
        model=model, temperature=0, max_tokens=300,
        messages=[{"role": "system", "content": PROMPT},
                 {"role": "user", "content": f"BILL:\n{context}\nARTICLE:\n{article_text}"}],
    )
    raw = resp.choices[0].message.content.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(result, dict) or "about_this_bill" not in result:
        return None
    about = bool(result.get("about_this_bill"))
    prediction = result.get("prediction") if about else None
    confidence = result.get("confidence") if about else None
    if prediction not in ("pass", "fail", "neutral", None):
        prediction = None
    if confidence not in ("firm", "hedged", None):
        confidence = None
    return {"about_this_bill": about, "prediction": prediction, "confidence": confidence}


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_done_keys(out_path):
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["bill_type"], r["number"], r["article_url"]))
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def label_articles(client, candidates, bills_by_key, out_path, limit=None):
    done = load_done_keys(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    with open(out_path, "a") as out:
        for c in candidates:
            key = (c["bill_type"], c["number"], c["article_url"])
            if key in done:
                continue
            bill = bills_by_key.get((c["bill_type"], c["number"]))
            if bill is None:
                continue
            result = extract_from_article(client, bill, c)
            record = dict(c)
            if result is None:
                record.update({"about_this_bill": False, "prediction": None,
                              "confidence": None, "parse_failed": True})
            else:
                record.update(result)
            out.write(json.dumps(record) + "\n")
            out.flush()
            processed += 1
            if processed % 20 == 0:
                print(f"  labeled {processed} articles")
            if limit and processed >= limit:
                break
    print(f"labeled {processed} new articles -> {out_path}")


def aggregate_bill_features(labeled_records, bills):
    by_bill = defaultdict(list)
    for r in labeled_records:
        if r.get("about_this_bill"):
            by_bill[(r["bill_type"], r["number"])].append(r)

    rows = []
    for bill in bills:
        key = (bill["bill_type"], bill["number"])
        articles = by_bill.get(key, [])
        n_articles = len(articles)
        has_coverage = n_articles > 0

        days_first = None
        if articles:
            dates = [a["pub_date"] for a in articles if a.get("pub_date")]
            intro = bill.get("introduced_date")
            if dates and intro:
                first = min(dates)
                days_first = (datetime.strptime(first, "%Y-%m-%d")
                             - datetime.strptime(intro, "%Y-%m-%d")).days

        non_neutral = [a for a in articles if a.get("prediction") in ("pass", "fail")]
        if non_neutral:
            passes = sum(1 for a in non_neutral if a["prediction"] == "pass")
            fails = sum(1 for a in non_neutral if a["prediction"] == "fail")
            if passes > fails:
                press_predicts_pass = "pass"
            elif fails > passes:
                press_predicts_pass = "fail"
            else:
                press_predicts_pass = "mixed"
            confidences = {a.get("confidence") for a in non_neutral if a.get("confidence")}
            press_confidence = "firm" if "firm" in confidences else (
                "hedged" if confidences else None)
        elif articles:
            press_predicts_pass, press_confidence = "neutral", None
        else:
            press_predicts_pass, press_confidence = "none", None

        rows.append({
            "congress": bill["congress"], "bill_type": bill["bill_type"],
            "number": bill["number"], "has_national_coverage": has_coverage,
            "n_articles": n_articles, "press_predicts_pass": press_predicts_pass,
            "press_confidence": press_confidence,
            "days_intro_to_first_coverage": days_first,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--congress", type=int, required=True)
    ap.add_argument("--press-raw-dir", default="data/press_raw")
    ap.add_argument("--bills-dir", default="data/bills")
    ap.add_argument("--labeled-dir", default="data/press_labeled")
    ap.add_argument("--features-out", default=None,
                    help="defaults to data/press_features_{congress}.csv")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of NEW articles labeled, for testing")
    args = ap.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("Set DEEPSEEK_API_KEY first.")

    raw_path = Path(args.press_raw_dir) / f"{args.congress}.jsonl"
    bills_path = Path(args.bills_dir) / f"{args.congress}.jsonl"
    if not raw_path.exists():
        raise SystemExit(f"No candidate articles at {raw_path}. Run link_coverage.py first.")
    if not bills_path.exists():
        raise SystemExit(f"No bills file at {bills_path}. Run download_bills.py first.")

    candidates = load_jsonl(raw_path)
    bills = load_jsonl(bills_path)
    bills_by_key = {(b["bill_type"], b["number"]): b for b in bills}

    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
    labeled_path = Path(args.labeled_dir) / f"{args.congress}.jsonl"
    label_articles(client, candidates, bills_by_key, labeled_path, args.limit)

    labeled_records = load_jsonl(labeled_path)
    features = aggregate_bill_features(labeled_records, bills)
    features_out = args.features_out or f"data/press_features_{args.congress}.csv"
    features.to_csv(features_out, index=False)
    print(f"{len(features)} bills -> {features_out}")
    print(f"has_national_coverage: {features['has_national_coverage'].sum()} "
         f"({features['has_national_coverage'].mean():.4%})")


if __name__ == "__main__":
    main()
