"""
Stage 2 for the ProQuest arm: filter extracted claims to raise precision.

The gold eval put gpt-4o-mini at precision 0.409 / recall 0.519 / F1 0.458 --
it finds forecasts (better recall than the LOC pipeline's 0.43) but over-collects
badly, and on the three gold pages that contain NO forecast at all it invented
seven. That asymmetry is the same one main's bake-off found across models, and
main's answer is a second pass: extract with the model that finds the most, then
hand the candidates to a judge that is good at saying no.

From validation/gold_extraction/RESULTS.md:

    gpt-oss-120b                      prec 0.54   F1 0.612
    gpt-oss-120b -> gpt-oss verify    prec 0.62   F1 0.636   (SELF-verify)
    gpt-oss-120b -> gem-flash verify  prec 0.93   F1 0.667   (stronger judge)

The VM proxy offers only gpt-4o-mini, so this is the self-verify row: a real but
modest gain, not the 0.93. Worth it because precision is our weak axis and
verification is cheap -- it sees a quote plus ~400 characters of context, not a
whole document, so it costs a small fraction of extraction per claim.

The prompt's KEEP/DROP list is kept WORD-FOR-WORD from main's src/verify_claims.py
(only the date range differs, 1905-2020 vs 1900-1963). That list is the calibrated
part; rewording it would make our precision number incomparable to main's table.

Judging is batched per page: one call carries all of a page's candidates, so the
request count is one per page rather than one per claim.

NOTHING IS DELETED. Kept claims are written through with a `verify_reason`;
dropped ones go to <out>.dropped.jsonl so the filter stays auditable and can be
scored against the gold like everything else. A judge that fails to answer keeps
the claim -- a silent filter that deletes evidence on an API hiccup is worse than
no filter.

Usage (in the VM, from election_arm/):
    # measure the gain on the gold pages first
    python verify_gpt.py --claims pred_gpt4omini_gold.jsonl \
        --pages gold_extraction/gold_pages.jsonl \
        --out pred_gpt4omini_gold_verified.jsonl

    # then a real window -- NOTE the output goes to data/verified/, NOT
    # data/predictions/ (see the guard in check_out_path below)
    python verify_gpt.py --claims data/predictions/pred_proquest_economy_gulf_1990.jsonl \
        --pages data/raw/proquest_economy_gulf_1990.jsonl \
        --out data/verified/pred_proquest_economy_gulf_1990.jsonl

Resume-safe: pages already judged are skipped on a rerun, and the daily cap
stops it cleanly with exit 2 like extract_gpt.py.
"""

import argparse
import json
import re
import time
from pathlib import Path

# Reuse the proxy discovery and the rate-limit handling rather than copying them:
# the daily-cap-vs-transient distinction and the truncation retry were both hard
# won, and two copies would drift.
from extract_gpt import (RateLimitReached, TOKEN_RE, call_model, make_client,
                         parse_claims)

CONTEXT_CHARS = 400  # each side of the quote
REQUEST_DELAY = 0.5

# KEEP/DROP list verbatim from main's src/verify_claims.py -- see module docstring.
VERIFY_PROMPT = """You are checking candidate economic PREDICTIONS pulled from an American newspaper printed between 1905 and 2020. Another system proposed them; many are wrong. Your job is to say which are genuine.

KEEP a candidate only if it makes a FALSIFIABLE CLAIM ABOUT FUTURE ECONOMIC CONDITIONS -- business conditions, prices, employment, markets, prosperity, recession or panic.

DROP it if it is any of the following, however economic its vocabulary:
- ADVERTISING or promotional copy, including New Year "prosperity" greetings and sale-price claims.
- FICTION: dialogue from a serialized novel or a humour sketch.
- A REPRINT from a "Twenty Years Ago" / "From Our Files" column (written decades before this page's date).
- A DESCRIPTION OF THE PRESENT OR PAST rather than the future: "steel production is climbing", "business is good today", "failures were larger than last year". This includes a body dating a downturn after the fact -- "the committee declared that a recession probably began in the summer" is a statement about the past, not a forecast.
- A POST-MORTEM ON A FORECAST THAT HAD ALREADY FAILED by the time this page was printed. If the passage's point is that earlier predictions were WRONG -- headlines like "Prophecies Gone Wrong", phrases like "how the prophets were mistaken" -- then the page is not making that forecast, it is burying it. DROP. But a forecast that is still LIVE and merely doubted or disputed by the writer is a real forecast: KEEP.
- A REFUSAL to forecast: "it is too early to say", "no one can know".
- A CONDITIONAL with no committed direction, or a claim about a bill that has not passed. Also DROP a statement of what WOULD BE REQUIRED rather than what will happen: "rates will need to fall a point to turn the economy around" is a condition, not a forecast.
- A GENERAL TRUTH about how markets or economies work, with no time attached: "the market must clear itself of surpluses before it can move again".
- An ANNOUNCEMENT of a speech, meeting or report ABOUT the outlook, or a schedule (store opening, construction timetable).
- POLICY ADVOCACY: "the Reserve Board should", "Congress must".
- NON-ECONOMIC: elections, legislation, weather, sport, health.
- A STOCK TIP about one company, or investment arithmetic.
- Text too mangled by OCR to interpret.
- A fragment too short or too contextless to carry a claim on its own: "I don't think that will be that far off".

The newspaper is {newspaper}, dated {date}.

Candidates, each with the text surrounding it:

{candidates}

Return ONLY a JSON array, one element per candidate, in the same order:
[{{"id": <the candidate's id>, "keep": true or false, "reason": "<8 words or fewer>"}}]

Judge each candidate on its own. Keeping and dropping are equally costly: dropping a real forecast loses evidence, keeping a false one injects fake signal into the results."""


def local_context(page_text, quote, width=CONTEXT_CHARS):
    """The quote plus surrounding text, so the judge can see whether it sits in
    an advertisement, a reprint column, or a retrospective. Falls back to a
    token-anchor search when the quote is not a literal substring -- the gold
    quotes are normalized and 0 of 52 appear verbatim, so this path matters."""
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


def build_prompt(page, claims):
    blocks = []
    for i, c in enumerate(claims):
        ctx = local_context(page.get("ocr_text", ""), c.get("quote", ""))
        blocks.append(f"--- candidate {i} ---\n"
                      f"CANDIDATE QUOTE: {c.get('quote', '')}\n"
                      f"SURROUNDING PAGE TEXT: ...{ctx}...")
    return VERIFY_PROMPT.format(
        newspaper=(page.get("publisher") or page.get("newspaper_title")
                   or "unknown"),
        date=page.get("date") or "unknown",
        candidates="\n\n".join(blocks))


def parse_verdicts(raw):
    out = {}
    for v in parse_claims(raw):
        if isinstance(v, dict) and "id" in v:
            try:
                out[int(v["id"])] = v
            except (TypeError, ValueError):
                continue
    return out


def check_out_path(out_path):
    """Refuse to write inside data/predictions/.

    analyze_economy.py globs `data/predictions/pred_*_economy_*.jsonl` with no
    exclusions, so a verified file written there is loaded ALONGSIDE the
    unverified original -- and `<out>.dropped.jsonl` matches that glob too. The
    result is every claim counted twice and the rejects silently readmitted,
    with nothing to notice: the run succeeds and the totals just quietly inflate.

    Verified output belongs outside that directory. A one-line mistake here
    would corrupt every downstream table, so it is a hard stop rather than a
    warning."""
    parts = {p.lower() for p in out_path.parts}
    if "predictions" in parts:
        raise SystemExit(
            f"\n*** Refusing to write to {out_path}\n"
            f"*** data/predictions/ is globbed by analyze_economy.py as\n"
            f"***   pred_*_economy_*.jsonl\n"
            f"*** which would match BOTH this file and its .dropped.jsonl,\n"
            f"*** double-counting every claim and re-admitting the rejects.\n\n"
            f"Write to data/verified/ instead:\n"
            f"  --out data/verified/{out_path.name.replace('.verified', '')}\n")


def judged_pages(out_path, dropped_path):
    """page_ids already judged, from BOTH output files -- a page whose claims
    were all dropped appears only in the dropped file, and would otherwise be
    re-judged (and re-billed) on every rerun."""
    done = set()
    for p in (out_path, dropped_path):
        if p.exists():
            with open(p) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if r.get("verify_reason") is not None and r.get("page_id"):
                        done.add(r["page_id"])
    return done


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--claims", required=True, help="extractor JSONL to filter")
    ap.add_argument("--pages", required=True,
                    help="the pages those claims came from (for context)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="max pages this run")
    ap.add_argument("--sample", default="gpt_sample.txt")
    ap.add_argument("--base-url")
    ap.add_argument("--key-file")
    ap.add_argument("--model")
    args = ap.parse_args()

    pages = {}
    with open(args.pages) as f:
        for line in f:
            if line.strip():
                p = json.loads(line)
                pages[p["page_id"]] = p

    claims, empties = [], []
    with open(args.claims) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            (empties if r.get("no_predictions") else claims).append(r)

    by_page = {}
    for c in claims:
        by_page.setdefault(c.get("page_id"), []).append(c)

    out_path = Path(args.out)
    check_out_path(out_path)
    dropped_path = out_path.with_suffix(out_path.suffix + ".dropped.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = judged_pages(out_path, dropped_path)
    if done:
        print(f"resuming: {len(done)} page(s) already judged")

    client, model = make_client(args)
    kept = dropped = unjudged = processed = 0

    with open(out_path, "a") as fh, open(dropped_path, "a") as dh:
        # Carry the genuine empties through once, so the denominator (pages
        # read, not just pages that produced something) survives the filter.
        if not done:
            for e in empties:
                fh.write(json.dumps(e) + "\n")

        for page_id, page_claims in by_page.items():
            if page_id in done:
                continue
            page = pages.get(page_id)
            if page is None:
                # No page text to judge against: keep, do not guess.
                for c in page_claims:
                    c["verify_reason"] = "no page text (kept by default)"
                    fh.write(json.dumps(c) + "\n")
                    kept += 1
                continue
            try:
                raw = call_model(client, model, build_prompt(page, page_claims))
            except RateLimitReached as e:
                print(f"\n*** DAILY/RATE LIMIT reached: {e}")
                print(f"*** Stopping cleanly after {processed} page(s) this run.")
                print("*** Re-run after the quota resets; it resumes here.")
                raise SystemExit(2)
            verdicts = parse_verdicts(raw) if raw else {}
            for j, c in enumerate(page_claims):
                v = verdicts.get(j)
                if v is None:
                    # No verdict. KEEP -- a judge that failed to answer must
                    # never silently delete evidence.
                    c["verify_reason"] = "unjudged (kept by default)"
                    fh.write(json.dumps(c) + "\n")
                    kept += 1
                    unjudged += 1
                elif v.get("keep"):
                    c["verify_reason"] = v.get("reason", "")
                    fh.write(json.dumps(c) + "\n")
                    kept += 1
                else:
                    c["verify_reason"] = v.get("reason", "")
                    dh.write(json.dumps(c) + "\n")
                    dropped += 1
            fh.flush()
            dh.flush()
            processed += 1
            n_kept = sum(1 for j in range(len(page_claims))
                         if verdicts.get(j, {}).get("keep", True))
            print(f"  [{processed}/{len(by_page) - len(done)}] "
                  f"{page.get('date')}  {len(page_claims)} candidates "
                  f"-> {n_kept} kept")
            time.sleep(REQUEST_DELAY)
            if args.limit and processed >= args.limit:
                break

    total = kept + dropped
    print(f"\nkept {kept}, dropped {dropped}"
          + (f" ({dropped / total:.0%} of candidates)" if total else "")
          + (f", {unjudged} unjudged (kept)" if unjudged else ""))
    print(f"  kept    -> {out_path}")
    print(f"  dropped -> {dropped_path}   (read these; the filter must be auditable)")
    print("\nScore the gain the same way the baseline was scored:")
    print(f"  python gold_extraction/eval_extraction.py \\")
    print(f"      --gold gold_extraction/gold_claims.jsonl \\")
    print(f"      --pred {out_path} --name gpt-4o-mini+verify")


if __name__ == "__main__":
    main()
