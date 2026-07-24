# Using the labeler + scorer inside ProQuest TDM Studio

ProQuest's walls block two things: **you cannot export the full text**, and
**inside you only have GPT-4o**, with a **15 MB** transfer cap. The pipeline
splits cleanly along exactly that line, so both halves run *inside* and only tiny
results come out.

The key fact: **the scorer needs no text and no network.** It grades a forecast
from derived labels alone (date, topic, direction, horizon, scope) against public
FRED/NBER data. So everything runs in the sandbox; you export only summary
numbers.

```
   INSIDE ProQuest (GPT-4o, no export needed)          OUTSIDE (optional)
   ┌─────────────────────────────────────────┐
   │ article full text                        │
   │   └─ EXTRACTION_PROMPT via GPT-4o  ──►    │
   │        structured claim labels           │
   │           └─ score_predictions.py  ──►   │        just the summary
   │                scored table (hit/miss) ───────►  CSV of rates (<< 15 MB)
   └─────────────────────────────────────────┘
```

## 1. Bring the scorer IN (112 KB — fits the 15 MB cap 130× over)

Upload the `proquest_bundle/` folder:

```
proquest_bundle/
    truth_data.py            the ground-truth engine (no network needed)
    score_predictions.py     the deterministic scorer
    fred/
        fred_INDPRO.csv          industrial production 1919-
        fred_CPIAUCNS.csv        prices 1913-
        fred_UNRATE.csv          unemployment 1948-
        fred_M1109BUSM293NNBR.csv  stock index 1914-1968
```

At the top of your notebook, point the scorer at the bundled data so it never
touches the network:

```python
import os
os.environ["RISE_FRED_DIR"] = "proquest_bundle/fred"
```

## 2. Extract inside, with GPT-4o

Copy `EXTRACTION_PROMPT` from `extract_llm.py` into a notebook cell (it is
model-agnostic). For each article, call ProQuest's GPT-4o with that prompt as the
system/instruction and the article text as the input, then parse the JSON array
it returns. `extract_llm.py`'s `parse_claims()` and `quote_is_grounded()` can be
pasted in as-is for robust parsing and the hallucination guard.

You keep the `quote` **inside** for the hallucination check, then **drop it**
before anything leaves — it is the only copyrighted field. Everything else is a
derived label.

## 3. Score inside — no text, no network

```python
import json
from truth_data import TruthData
from score_predictions import score_claim

truth = TruthData()                      # loads the bundled CSVs, offline
scored = [score_claim(c, truth) for c in claims]   # claims = the label dicts
```

Each row comes back with `hit` (1/0/None), `realized`, `basis`, `scorable`, and
`unscorable_reason`. Aggregate to hit rates, by-year, by-feature — whatever the
poster needs.

## 4. Export only the summary

A table of rates and counts is kilobytes. Even the full label table (no quotes)
is ~150 bytes/claim, so 15 MB holds ~100,000 claims — you will not hit the cap.

## What GPT-4o changes (be honest about this)

- **GPT-4o is not in our validated bake-off.** We validated Gemini-3.5-flash
  (F1 0.79); v1 validated gpt-4.1 (κ 0.89), same family. The prompt should
  transfer, but **spot-check ~20 articles against `gold_extraction/PROTOCOL.md`
  by hand** before trusting the labels — that is the one validation we cannot do
  for you inside the walls.
- **Have GPT-4o emit `horizon_months` explicitly.** The scorer normally infers a
  vague horizon from the quote's time-language; with the quote dropped it falls
  back to a 12-month default. A model-stated horizon is a derived label, so
  keeping it loses no information and keeps the copyrighted text out.
- **No batch discount** — ProQuest's GPT-4o is synchronous. Cost sits on the
  institutional access, not a key you manage.

## What you gain over the LOC corpus

FRED runs to the present, so **post-1963 ProQuest content is scorable** — the LOC
corpus stops at 1963 because that is where LOC full text ends. And ProQuest gives
clean per-article text rather than whole-page scans, which the extractor handles
at least as well (less ad/fiction noise to reject).

## The one thing that must stay outside

Do not try to bring the LLM extractor's *cost/quality* comparison in — that was
run against Gemini. Inside ProQuest you are committing to GPT-4o, so the honest
methods line is "extraction by GPT-4o, hand-validated on N articles," not a
reference to our bake-off numbers.
