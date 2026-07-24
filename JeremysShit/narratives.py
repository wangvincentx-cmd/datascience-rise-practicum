"""
Narrative Economics (Shiller 2019): which recurring economic STORY does each
newspaper claim tell, and do complacent narratives ("new era", "fundamentally
sound") crowd out caution right before the worst crises?

Shiller argues a handful of perennial economic narratives recur across a
century and drive booms and busts. This corpus -- 1905-2009, machine-graded,
kappa-validated -- is a rare chance to measure their prevalence directly, over
a far longer span than the Survey-of-Professional-Forecasters era most
narrative work is confined to. This connects the project's optimism-gap
result to a named literature: an optimistic DIRECTION is one thing, but WHICH
optimistic story ("this time is different" vs "just a temporary readjustment")
carries the signal is the narrative question.

Taxonomy (six perennial narratives, mapped to this corpus):
  new_era            permanent prosperity, a new plateau, the old rules are gone
  sound_fundamentals reassurance; fears are unfounded, business is basically sound
  temporary_setback  current trouble is a passing correction, will pass soon
  panic_fear         impending collapse / hard times / disaster ahead
  recovery_normalcy  recovery underway, return to normal, turning the corner
  none               no clear economic narrative

TWO passes, deliberately in this order (same "objective/reproducible first,
then LLM" discipline as hedging_lexicon.py -> grade_claims.py):
  1. `classify_narrative` -- a transparent, deterministic LEXICAL screen that
     runs today with no API. Crude (keyword argmax), but reproducible and
     offline-tested; good enough to see the shape of the result.
  2. `grade_narratives_llm` (--llm) -- the AUTHORITATIVE pass: gpt-4.1 via the
     existing grade_claims.call_llm plumbing, one narrative per claim. NOT run
     by default (needs OPENAI_API_KEY + a small spend); its output should get
     the same ~40-80-claim human kappa check every other LLM field here got
     before it goes on the poster. The lexical pass is a preview, not the
     finding.

Usage:
    python narratives.py                 # lexical screen + analysis (no API)
    python narratives.py --llm --limit 40   # smoke-test the gpt-4.1 pass
Outputs: claims_narratives.csv, printed prevalence + accuracy tables,
         figures/fig_narratives.png
"""

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

FIGDIR = Path("figures")

# The "this time is fine" family -- optimistic/dismissive stories whose spike
# before a crash is the hypothesis of interest.
COMPLACENT = ("new_era", "sound_fundamentals", "temporary_setback")

# Lexical markers per narrative. Deterministic screen only; the LLM pass is
# authoritative. Phrases matched as whole words, case-insensitive.
NARRATIVE_TERMS = {
    "new_era": [
        "new era", "new age", "permanent prosperity", "permanently high",
        "high plateau", "boundless", "never been better", "unprecedented prosperity",
        "here to stay", "old rules", "new economy", "limitless", "no end in sight",
    ],
    "sound_fundamentals": [
        "fundamentally sound", "basically sound", "sound basis", "solid foundation",
        "no cause for alarm", "no reason for alarm", "unfounded", "exaggerated",
        "confidence", "well founded", "healthy", "firm footing", "reassur",
        "nothing to fear", "conditions are sound",
    ],
    "temporary_setback": [
        "temporary", "readjustment", "correction", "passing", "will pass",
        "short-lived", "brief", "turn the corner", "around the corner",
        "setback", "pause", "breathing spell", "transitory", "momentary",
    ],
    "panic_fear": [
        "panic", "crash", "collapse", "depression", "hard times", "disaster",
        "crisis", "ruin", "catastrophe", "calamity", "slump", "breakdown",
        "wave of", "storm", "dark", "gloom",
    ],
    "recovery_normalcy": [
        "recovery", "recover", "revival", "rebound", "upturn", "return to normal",
        "back to normal", "improvement", "on the mend", "pickup", "pick up",
        "resumption", "restored", "reviving", "comeback",
    ],
}
_NARRATIVE_RE = {k: re.compile(r"\b(?:%s)" % "|".join(re.escape(t) for t in v), re.I)
                 for k, v in NARRATIVE_TERMS.items()}


def classify_narrative(quote):
    """Deterministic lexical narrative label for one quote. Argmax of per-
    narrative keyword hits; ties or zero hits -> 'none'."""
    q = str(quote)
    counts = {k: len(rx.findall(q)) for k, rx in _NARRATIVE_RE.items()}
    best = max(counts.values())
    if best == 0:
        return "none"
    winners = [k for k, c in counts.items() if c == best]
    return winners[0] if len(winners) == 1 else "none"


def add_narratives(df, col="narrative"):
    df = df.copy()
    df[col] = df["quote"].apply(classify_narrative)
    df["complacent"] = df[col].isin(COMPLACENT)
    return df


# ---- LLM pass (authoritative; not run by default) -------------------------
NARRATIVE_PROMPT = """You are labeling the ECONOMIC NARRATIVE in a sentence from an
American newspaper (1905-2009; expect OCR noise). Choose exactly ONE label that
best describes the STORY the sentence tells about the economy:

- new_era: prosperity is permanent / a new plateau reached / old economic rules no longer apply
- sound_fundamentals: reassurance -- fears are unfounded, business is fundamentally sound
- temporary_setback: current trouble is a passing correction that will soon pass
- panic_fear: impending collapse, hard times, disaster, or crisis ahead
- recovery_normalcy: recovery is underway / a return to normal / turning the corner
- none: no clear economic narrative

Return ONLY JSON: {{"narrative": "<one label>"}}

Date: {date}   Episode: {episode}
Sentence: {quote}"""


def grade_narratives_llm(df, api_key, model, base_url, limit=None):
    """Authoritative gpt-4.1 narrative pass, reusing grade_claims.call_llm.
    Returns df with an `narrative_llm` column. Kept thin on purpose -- for the
    full corpus use --batch (below), which reuses grade_claims.py's generic
    OpenAI Batch API plumbing (poll_batch/fetch_file); this synchronous path
    is for a --limit smoke test and the kappa-validation sample only."""
    from grade_claims import call_llm
    rows = df.head(limit) if limit else df
    labels = []
    for _, r in rows.iterrows():
        prompt = NARRATIVE_PROMPT.format(date=r.get("date", ""),
                                         episode=r.get("episode", ""), quote=r["quote"])
        try:
            labels.append(call_llm(prompt, model, base_url, api_key).get("narrative", "none"))
        except Exception as e:
            labels.append(f"ERROR:{type(e).__name__}")
    out = rows.copy()
    out["narrative_llm"] = labels
    return out


# ---- Full-corpus batch pass (--batch) --------------------------------------
# Reuses grade_claims.py's poll_batch/fetch_file (generic HTTP plumbing, no
# dependency on its RUBRIC_PROMPT/GRADE_FIELDS) but builds its own request
# lines, since the narrative prompt/response shape (one label field, keyed by
# claim_id) differs from grade_claims.py's multi-field grading rubric.
NARRATIVE_BATCH_MAX_TOKENS = 40


def _narrative_batch_lines(rows, model):
    for r in rows:
        prompt = NARRATIVE_PROMPT.format(date=r.get("date", ""),
                                         episode=r.get("episode", ""), quote=r["quote"])
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
            "max_tokens": NARRATIVE_BATCH_MAX_TOKENS,
        }
        yield json.dumps({"custom_id": str(r["claim_id"]), "method": "POST",
                          "url": "/v1/chat/completions", "body": body})


def _submit_narrative_batch(api_key, rows, model):
    import requests
    from grade_claims import OPENAI_BATCH_BASE
    jsonl = "\n".join(_narrative_batch_lines(rows, model))
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


def run_narrative_batch(df, api_key, model, chunk_size=800, poll_interval=30,
                        out_path="claims_narratives_llm_full.csv"):
    """Batch-grade every claim's narrative label over the full corpus (1,628
    claims as of 2026-07-22 -- one or two chunks at NARRATIVE_BATCH_MAX_TOKENS
    output, comfortably under the org enqueued-token cap that bit
    grade_claims.py at 2,415 claims/900k tokens with a much heavier prompt).
    Resume-safe by claim_id via an existing out_path."""
    from grade_claims import fetch_file, poll_batch

    done_ids = set()
    if os.path.exists(out_path):
        done_ids = set(pd.read_csv(out_path)["claim_id"].astype(str))
        if done_ids:
            print(f"Resuming: {len(done_ids)} claims already labeled in {out_path}")

    todo = [r for _, r in df.iterrows() if str(r["claim_id"]) not in done_ids]
    if not todo:
        print("Nothing left to label.")
        return pd.read_csv(out_path)

    chunks = [todo[i:i + chunk_size] for i in range(0, len(todo), chunk_size)]
    print(f"Submitting {len(todo)} claims to the OpenAI Batch API ({model}) in "
         f"{len(chunks)} chunk(s) of up to {chunk_size}...")

    header_written = os.path.exists(out_path)
    for ci, chunk in enumerate(chunks, 1):
        print(f"--- chunk {ci}/{len(chunks)}: {len(chunk)} claims ---", flush=True)
        batch = _submit_narrative_batch(api_key, chunk, model)
        print(f"  batch id {batch['id']}, status {batch['status']}")
        result = poll_batch(api_key, batch["id"], interval=poll_interval)

        labels = {}
        if result.get("output_file_id"):
            for line in fetch_file(api_key, result["output_file_id"]).splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                body = (rec.get("response") or {}).get("body") or {}
                try:
                    content = body["choices"][0]["message"]["content"]
                    label = json.loads(content).get("narrative", "none")
                except Exception:
                    label = "ERROR:parse"
                labels[rec["custom_id"]] = label
        if result.get("error_file_id"):
            err_text = fetch_file(api_key, result["error_file_id"])
            err_lines = [l for l in err_text.splitlines() if l.strip()]
            print(f"  {len(err_lines)} request(s) errored -- first: "
                 f"{err_lines[0][:300] if err_lines else ''}")

        chunk_rows = pd.DataFrame(chunk)
        chunk_rows["narrative_llm"] = chunk_rows["claim_id"].astype(str).map(labels).fillna("ERROR:missing")
        chunk_rows.to_csv(out_path, mode="a", index=False, header=not header_written)
        header_written = True
        n_ok = sum(1 for v in labels.values() if not str(v).startswith("ERROR"))
        print(f"  chunk {ci}/{len(chunks)} {result['status']}: {n_ok}/{len(chunk)} labeled "
             f"-> {out_path}")

    print(f"\nAll {len(chunks)} chunk(s) complete -> {out_path}")
    return pd.read_csv(out_path)


def main(args):
    df = pd.read_csv("claims_scored.csv")
    df = add_narratives(df)

    if args.batch:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("Set OPENAI_API_KEY for --batch.")
        graded = run_narrative_batch(df, api_key, args.model,
                                     chunk_size=args.batch_chunk_size,
                                     poll_interval=args.poll_interval,
                                     out_path="claims_narratives_llm_full.csv")
        graded["complacent_llm"] = graded["narrative_llm"].isin(COMPLACENT)
        agree = (graded["narrative"] == graded["narrative_llm"]).mean()
        print(f"\nFull-corpus LLM pass (n={len(graded)}): lexical-vs-LLM raw "
              f"agreement {agree:.1%} -> claims_narratives_llm_full.csv")

        print("\n=== Narrative prevalence (AUTHORITATIVE gpt-4.1-class LLM pass) ===")
        print(graded["narrative_llm"].value_counts(normalize=True).round(3).to_string())

        s = graded.dropna(subset=["hit"]).copy()
        print("\n=== Accuracy by narrative (LLM labels) ===")
        acc = (s.groupby("narrative_llm").agg(n=("hit", "size"), hit_rate=("hit", "mean"))
               .sort_values("hit_rate").round(3))
        print(acc.to_string())
        if "new_era" in acc.index or "sound_fundamentals" in acc.index:
            comp_hit = s[s["complacent_llm"]]["hit"].mean()
            other_hit = s[~s["complacent_llm"]]["hit"].mean()
            print(f"\n  complacent narratives hit {comp_hit:.1%} vs everything else "
                  f"{other_hit:.1%}  (delta {comp_hit - other_hit:+.3f})")

        comp = (graded.groupby("episode")
                .agg(n=("complacent_llm", "size"), complacent_share=("complacent_llm", "mean"),
                     kind=("kind", "first"))
                .sort_values("complacent_share", ascending=False).round(3))
        print("\n=== Complacent-narrative share by episode (LLM labels) ===")
        print(comp.to_string())
        _figure(comp, path=FIGDIR / "fig_narratives_llm.png",
               title="Complacent narratives by episode (red = crisis, green = calm control)\n"
                     "AUTHORITATIVE gpt-4.1-class LLM pass")
        print("\nclaims_narratives_llm_full.csv + figures/fig_narratives_llm.png written")
        print("NEXT: pull a 40-80-claim human kappa sample (see narratives_kappa_sample.py) "
              "before reporting these on the poster.")
        return

    if args.llm:
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise SystemExit("Set OPENAI_API_KEY for the --llm pass.")
        graded = grade_narratives_llm(df, api_key, args.model, args.base_url, args.limit)
        graded.to_csv("claims_narratives_llm.csv", index=False)
        agree = (graded["narrative"] == graded["narrative_llm"]).mean()
        print(f"LLM narrative smoke test (n={len(graded)}): lexical-vs-LLM raw "
              f"agreement {agree:.1%} -> claims_narratives_llm.csv")
        print("NEXT: run --batch over the full corpus, then a "
              "~40-80-claim human kappa check before trusting these.")
        return

    df.to_csv("claims_narratives.csv", index=False)
    s = df.dropna(subset=["hit"]).copy()

    print("=== Narrative prevalence (lexical screen -- preview, LLM pass is authoritative) ===")
    print(df["narrative"].value_counts(normalize=True).round(3).to_string())

    print("\n=== Complacent-narrative share by episode (new_era + sound_fundamentals "
          "+ temporary_setback) ===")
    comp = (df.groupby("episode")
            .agg(n=("complacent", "size"), complacent_share=("complacent", "mean"),
                 kind=("kind", "first"))
            .sort_values("complacent_share", ascending=False).round(3))
    print(comp.to_string())

    print("\n=== Accuracy by narrative (does a story predict being wrong?) ===")
    acc = (s.groupby("narrative").agg(n=("hit", "size"), hit_rate=("hit", "mean"))
           .sort_values("hit_rate").round(3))
    print(acc.to_string())
    if "new_era" in acc.index or "sound_fundamentals" in acc.index:
        comp_hit = s[s["narrative"].isin(COMPLACENT)]["hit"].mean()
        other_hit = s[~s["narrative"].isin(COMPLACENT)]["hit"].mean()
        print(f"\n  complacent narratives hit {comp_hit:.1%} vs everything else "
              f"{other_hit:.1%}  (delta {comp_hit - other_hit:+.3f})")
        print("  (preview only -- confirm on the LLM pass + a permutation test "
              "before claiming it)")

    _figure(comp)
    print("\nclaims_narratives.csv + figures/fig_narratives.png written")
    print("NOTE: lexical screen. The authoritative result needs the --llm pass "
          "(gpt-4.1) + a human kappa check, not yet run.")


def _figure(comp, path=None, title=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib missing -- no figure)")
        return
    FIGDIR.mkdir(exist_ok=True)
    d = comp.sort_values("complacent_share")
    colors = ["crimson" if k == "crisis" else "seagreen" for k in d["kind"]]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(range(len(d)), d["complacent_share"], color=colors, alpha=.85)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d.index, fontsize=8)
    ax.set_xlabel("share of claims telling a complacent story "
                  "(new era / sound fundamentals / temporary)")
    ax.set_title(title or "Complacent narratives by episode (red = crisis, green = calm control)\n"
                 "lexical screen -- preview of the Narrative Economics angle")
    plt.tight_layout()
    plt.savefig(path or (FIGDIR / "fig_narratives.png"), dpi=200)
    plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--llm", action="store_true", help="run the gpt-4.1 narrative pass (needs OPENAI_API_KEY)")
    ap.add_argument("--batch", action="store_true",
                    help="run the authoritative LLM narrative pass over the FULL corpus via "
                         "the OpenAI Batch API (needs OPENAI_API_KEY) -- 50%% cheaper, "
                         "meant to be backgrounded")
    ap.add_argument("--limit", type=int, default=None, help="limit rows for the --llm smoke test")
    ap.add_argument("--model", default="gpt-4.1",
                    help="gpt-4.1 is this project's bake-off-validated grader; pass "
                         "gpt-4.1-mini for a cheaper run (re-check kappa before trusting it)")
    ap.add_argument("--base-url", default="https://api.openai.com/v1")
    ap.add_argument("--batch-chunk-size", type=int, default=800,
                    help="claims per Batch API submission (--batch)")
    ap.add_argument("--poll-interval", type=float, default=30,
                    help="seconds between batch status checks (--batch)")
    main(ap.parse_args())
