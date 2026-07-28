# Extraction bake-off — results

**Date:** 2026-07-22
**Gold:** 16 pages, 52 claims, 3 pages with zero claims (`gold_claims.jsonl`),
annotated against [PROTOCOL.md](PROTOCOL.md) before any model output existed.
**Hard cases:** 44 boundary decisions (`hard_cases.jsonl`), 32 traps to refuse
and 12 awkward real forecasts to find.
**Provenance caveat:** the gold is model-adjudicated, not human. See
[PROTOCOL.md](PROTOCOL.md); a two-human recode of ~40 claims is still required
before publication. Every number here is a development metric.

## Headline

Costs are from MEASURED token counts (`cost_table.py`), scaled by page count and
priced at each provider's list rate. Three corpus sizes: the 16 gold pages, the
2,192 LOC pages already in `cache/`, and the ~23,000-page continuous 1900-1963
scrape (P3).

```
pipeline                            prec   rec     F1  hard    16pg   2,192pg   23,000pg
----------------------------------------------------------------------------------------
gemini-3.5-flash 1-window           1.00  0.65  0.791  93%    0.20     27.67     290.36
gemini-3.5-flash 8k                 0.84  0.73  0.784  98%    0.28     38.47     403.70
gpt-oss-120b -> gem-lite verify     0.76  0.60  0.667  86%    0.03      4.15      43.52
gpt-oss-120b -> gem-flash verify    0.93  0.52  0.667  91%    0.07     10.05     105.43
gpt-oss-120b -> gpt-oss verify      0.62  0.65  0.636  82%    0.02      2.56      26.82
gpt-oss-120b                        0.54  0.71  0.612  70%    0.02      2.16      22.65
DeepSeek-V3.1-Terminus              0.55  0.52  0.535  82%    0.05      6.32      66.31
Llama-3.3-70B                       0.43  0.67  0.522  77%    0.04      5.31      55.76
gemini-3.5-flash-lite 1-window      0.65  0.38  0.482  77%    0.04      5.83      61.22
gemini-3.5-flash-lite 8k            0.89  0.33  0.479  80%    0.05      7.14      74.93
Qwen3-235B-A22B                     0.70  0.31  0.427  77%    0.02      2.11      22.13
Mistral-Small-3.1-24B               0.28  0.65  0.391  66%    0.01      1.29      13.54
regex (current pipeline)            0.61  0.27  0.373  77%    0.00      0.00       0.00
Qwen3-Next-80B-A3B                  0.26  0.58  0.361  77%    0.03      4.02      42.20
Llama-3.1-8B                        0.12  0.46  0.191  64%    0.00      0.68       7.09
```

All figures USD. **Gemini batch mode is 50% off input and output** -- halve any
Gemini row for a bulk run. DeepInfra publishes no equivalent discount.
Total actually spent running this entire bake-off: **~$0.86**.

Matching is by containment at 0.7 (`eval_extraction.py`), one-to-one and greedy.

> **Pricing correction.** An earlier version of this file costed
> `gemini-3.5-flash` at $0.30/$2.50. That is the **Flash-Lite** rate; Flash is
> **$1.50 in / $9.00 out**, 5x more. Every Gemini-Flash figure here is the
> corrected one, and the gap it opens between the quality leader and the value
> leader is what the recommendation below now turns on.

## Recommendation depends on which corpus you are building

**For the 2,192-page cached corpus: `gemini-3.5-flash`, 8k windows, $38.**
It is the only extractor that solves the h01/h40 pair, scores 32/32 on the
exclusion traps, and has the best recall of anything tested. At $38 (or $19
batched) for a corpus every downstream result depends on, quality wins outright.

**For the ~23,000-page continuous scrape, the same choice costs $404** ($202
batched), and the calculus changes:

| option | F1 | hard | 23,000 pages |
|---|---|---|---|
| gemini-3.5-flash 8k | 0.784 | 98% | $404 |
| gemini-3.5-flash 1-window | 0.791 | 93% | $290 |
| gpt-oss-120b -> gemini-flash-lite verify | 0.667 | 86% | **$44** |
| gpt-oss-120b -> gpt-oss-120b verify | 0.636 | 82% | **$27** |

Ten times the price for +0.12 F1 and +12 points of hard-case accuracy. That is
a real judgement call, not an obvious one — and it is **the first point in this
project where the H200 cluster earns its keep**: gpt-oss-120b or Llama-3.3-70B
served locally under vLLM costs nothing per token, which makes the high-recall
extraction stage free and leaves only the judging stage on a paid API.

## Where gemini-3.5-flash-lite lands

Flash-Lite is **the most precise extractor tested (0.895)** and has **the worst
recall of any capable model (0.327)** — it finds one forecast in three. It never
truncated, so unlike Flash its first score was already fair. Verification cannot
recover claims an extractor never proposed, so its recall rules it out as the
extraction stage no matter how careful it is.

It is, however, **an excellent judge**, which is where its precision pays: as the
verifier behind gpt-oss-120b it produces the best value in the whole table (F1
0.667, 86% hard, $44 at continuous scale).

**Widening the window helps Flash and hurts Flash-Lite.** Given a whole page in
one call instead of three 8k windows, Flash improved (F1 0.784 -> 0.791, and
cheaper). Flash-Lite got *worse*: precision collapsed 0.895 -> 0.645 and its
hard-case score fell 80% -> 77%. The smaller model appears to lose discrimination
as context grows, so window size is a per-model setting, not a global one.

## Two findings that changed the ranking

**A truncation bug made the best model look like the worst.** gemini-3.5-flash
first scored F1 0.074 -- 2 claims across 16 pages. It is a thinking model:
`prompt_tokens 4059, total 5655, completion_tokens 63`. Invisible reasoning
consumed the budget and the JSON array was cut off mid-object. `extract_llm.py`
only raised the budget when content came back *empty*, so truncated-but-non-empty
output silently parsed to nothing. Fixed to bump on any `finish_reason=length`.
Any thinking model evaluated before that fix was scored unfairly.

**Verification raises precision but costs recall, and the trade is not free.**
Judging gpt-oss-120b's 69 candidates with gemini-3.5-flash lifted precision
0.536 -> 0.931 but cut recall 0.712 -> 0.519: the judge dropped 40 candidates,
and not all of them were wrong. Verification cannot recover a claim the
extractor never found, so the extractor's recall is a hard ceiling on the
pipeline.

## Cost is not the binding constraint

For the **2,192-page cached corpus**, every viable option lands between $2 and
$38 — small enough that model choice should be made on the hard cases, not the
bill.

For the **~23,000-page continuous scrape** it stops being free: $27 to $404
depending on the pipeline, a 15x spread. That is where the H200 cluster finally
matters, because serving gpt-oss-120b or Llama-3.3-70B locally under vLLM makes
the expensive half — reading 23,000 whole pages — cost nothing, leaving only the
cheap judging stage on a paid API.

## What the bake-off says about the regex

**The regex's problem is recall, not precision.** At 0.61 its precision is
better than Llama-70B's and second only to Qwen. It finds 27% of the forecasts
on a page. Everything the corpus is missing, it is missing here.

**The recall gap is real and large.** Llama-70B finds 2.5x as many real claims
as the regex (35 vs 14 of 52). This confirms the CHANGELOG's n=1 spot-check at
n=16 pages: whole-page reading reaches forecasts that sit nowhere near a search
phrase — a relief-demand forecast inside a municipal-council story (h37), a
banker's aside in a market wrap (h33).

**Parameter count is not the axis.** Llama-3.1-8B returned 199 claims for 52
real ones, 39 of them on pages with no forecast at all. DeepSeek-V3.1 and
Llama-70B are close on F1 and far apart in style; Qwen is precise and timid.
The 8B result is decisive: this task needs a capable model, and cheapness at
the bottom of the range buys nothing.

(Recommendation is in the section above; DeepSeek-V3.1 led the first round only
because gemini-3.5-flash was being scored through the truncation bug.)

## Hard cases, by what they test

gemini-3.5-flash's category profile (32/32 traps refused, 11/12 awkward
forecasts found):

- **Every structural trap refused, 32/32.** 3/3 fiction (serialized westerns,
  satire), 2/2 "Twenty Years Ago" reprints, 4/4 advertisements including the New
  Year "return of prosperity" bank ad, 2/2 refusals to forecast, 1/1 refuted
  rumour, present-tense reports, conditionals, stock tips, event announcements.
  These are the categories a keyword approach cannot see at all.
- **One miss:** h34, a forward claim carried by a present-tense verb
  ("creates a machinery to avert panics").

**Only gemini-3.5-flash solves the h01/h40 pair.** Both cases are a paper
reporting a forecast it doubts; they differ only in whether that forecast had
already been falsified when the page went to press. h01 is the single most
damaging false positive in the corpus — it records a bullish October-1907
"return of prosperity" call from a passage headlined *Prophecies Gone Wrong*,
which then gets scored against the Panic and manufactures exactly the "nobody
saw it coming" signal the project exists to measure honestly. **The current
regex takes h01.** gpt-oss-120b, DeepSeek-V3.1, Llama-70B, Qwen3-235B and the
one-window gemini variant each get at least one side wrong. This was the reason
a targeted prompt rule was queued; the model handles it unaided, so no rule is
needed.

## Corpus-construction finding (independent of any model)

Page 5 (Brookings Register, 1920-12-30) contains the literal search phrase
`return of prosperity` — one of the 29 terms behind the v1 corpus, and the
source of 300 claims — **only inside a bank's New Year advertisement**, on a page
with zero real forecasts. Page 1 and page 6 carry the same pattern with other
ad copy. December/January advertising is saturated with "prosperity" boilerplate,
so a phrase search run over a winter window preferentially retrieves it.

This is a plausible mechanism for optimism bias in the v1 corpus that is
independent of what newspapers actually predicted, and it is one more reason the
phrase-agnostic continuous scrape (P3) matters.

## Cost

~10,000 tokens/page measured across all four models -> **~22M tokens for the
2,192-page cached corpus**, roughly **$5-10 on DeepInfra**. Cost is not a
constraint at this corpus size; the H200 cluster only becomes necessary at the
~23,000-page scale of the continuous 1900-1963 scrape.

The hallucination guard rejected 1.2% of Llama-70B's returned claims and 2.5% of
Llama-8B's — low, but non-zero, and those would have been fluent invented
forecasts scored against real macro data.

## Reproduce

```
python gold_extraction/sample_gold_pages.py                 # deterministic, seed 20260722
python gold_extraction/run_regex_baseline.py                # baseline, no network
python extract_llm.py --pages gold_extraction/gold_pages.jsonl \
    --out gold_extraction/pred_deepseekv3.jsonl \
    --model deepseek-ai/DeepSeek-V3.1-Terminus --overwrite
python gold_extraction/eval_extraction.py  --pred gold_extraction/pred_deepseekv3.jsonl
python gold_extraction/score_hard_cases.py --pred gold_extraction/pred_deepseekv3.jsonl
```

## Known limits of this evaluation

- **n = 52 claims, 44 hard cases, one annotator.** Enough to rank extractors and
  to reject the 8B outright; not enough to separate DeepSeek from Llama-70B with
  confidence (F1 0.535 vs 0.522 is well inside noise at this n).
- **Crisis-window pages only.** The cache holds no 1905/1925/1955 control pages,
  so nothing here tests behaviour when forecast density is low.
- **The gold's own boundary calls are contestable** and are documented per page
  in `gold_claims.jsonl`'s `excluded_notes` precisely so a human recode can
  overturn them.
- **Precision here (regex 0.61) is higher than the corpus-level figure** implied
  by v1's grading pass (~40% of 4,125 candidates accepted). The gold's inclusion
  boundary is not identical to `RUBRIC_PROMPT`'s, and n is small — treat the
  cross-comparison as indicative, not as a contradiction.

## Running gpt-oss-120b on the H200

**It fits, comfortably.** gpt-oss-120b ships MXFP4-quantised at ~63 GB against
the H200's 141 GB of HBM3e, so the whole model sits on ONE card with ~78 GB left
for KV cache — ample at our ~7-10k context. Do **not** use tensor parallelism
across cards: the model already fits, so TP would only add interconnect
overhead. Run one vLLM instance per GPU and shard the page list between them
(data parallel), which scales close to linearly.

```
vllm serve openai/gpt-oss-120b --port 8000 --max-model-len 16384
python extract_llm.py --base-url http://localhost:8000/v1 --api-key-env NONE \
    --model openai/gpt-oss-120b --pages data/pages.jsonl \
    --chunk-chars 40000 --reasoning-effort low --out claims_v2.jsonl
```

**Decode, not prefill, sets the wall-clock — and reasoning effort sets decode.**
Measured output volume per page:

| model | out tokens/page |
|---|---|
| gpt-oss-120b, default effort | 3,585 |
| gpt-oss-120b, **low** effort | 646 |
| gemini-3.5-flash | 349 |
| Llama-3.3-70B | 703 |

gpt-oss-120b at default effort writes 10x more tokens than Gemini because it is
emitting thinking traces. Prefill is parallel and cheap; decode is sequential
per sequence, so those traces are what you actually wait for.

**Estimated single-H200 runtime for the ~23,000-page continuous corpus**
(161M input tokens; output as above), assuming ~25k tok/s prefill and ~6k tok/s
aggregate batched decode:

| effort | output tokens | prefill | decode | **total** |
|---|---|---|---|---|
| default | 82M | ~1.8 h | ~3.8 h | **~5-8 h** |
| low | 15M | ~1.8 h | ~0.7 h | **~2-4 h** |

The 2,192-page cached corpus is ~1/10 of that: **20-45 minutes**. With 4 H200s
running data-parallel, divide by ~4.

These are estimates with wide error bars — throughput depends on vLLM version,
batch size, and `--max-num-seqs`. **Benchmark 50 pages first** and scale from the
measured rate rather than trusting the table.

### The quality cost of low effort

| config | precision | recall | F1 | hard |
|---|---|---|---|---|
| gpt-oss-120b default, 8k windows | 0.536 | 0.712 | 0.612 | 70% |
| gpt-oss-120b **low**, 1 window | 0.615 | 0.462 | 0.527 | 73% |

**Caveat: this comparison changes two variables at once** (reasoning effort AND
window size), so the recall drop cannot be attributed cleanly to effort. Re-run
with only the effort varied before treating it as settled. `high` effort was
also tried and is unusable as configured: it exhausts the 6,000-token budget on
thinking and returns zero claims per page.

Either way the trade is real — roughly 5.5x fewer output tokens for ~0.09 F1 —
and on local hardware, where tokens cost time rather than money, low effort plus
a verification pass is likely the better shape.

## Batch mode and the extended schema (added 2026-07-23)

**Gemini Batch API is wired in** (`extract_llm.py --batch`): 50% off input and
output, verified end-to-end on the gold pages. It uses Gemini's NATIVE batch
endpoint with inline requests, not the OpenAI compatibility layer -- that layer
exposes `/batches` but returns 404 on `/files`, so there is no way to upload a
request file through it. Prompts are packed into jobs under a 15MB budget
(the inline cap is 20MB) and stitched back together.

Batch reproduces the synchronous path almost exactly, as it must:

| | TP | FP | recall | F1 | hard |
|---|---|---|---|---|---|
| sync, 1 window | 34 | 0 | 0.654 | 0.791 | 93% |
| batch, 1 window | 34 | 1 | 0.654 | 0.782 | 93% |

The one extra false positive is ordinary sampling nondeterminism, not a defect
in the batch path. Both paths share `assemble_claims()` on purpose: if they
assembled claims separately they would drift, and a batch-built corpus would
quietly stop being comparable to the numbers measured here.

**Three fields added to the schema**, before any corpus run rather than after:

- `scope` -- `national` / `regional` / `foreign` / `industry`. On the gold pages
  this immediately caught **3 of 34 claims (9%) that are about foreign
  economies** — Mexican sugar exports from Sonora and Sinaloa, and a forecast of
  the Brazilian milreis — all printed in US papers. Those are genuine forecasts,
  but grading them against US INDPRO would be simply wrong, and nothing in the
  v1 schema distinguished them. The model independently flagged exactly the
  three claims that were hand-flagged as scope problems during gold annotation.
- `is_quoted_forecaster` -- true when the forecast is attributed to someone other
  than the paper (23 of 34 here), false for the paper's own editorial voice (11).
  Overlaps with `voice` but is a cleaner binary, and separates "what the press
  believed" from "what the press relayed".
- `conditional_on` -- the weakest of the three. Set on only 3 of 34 claims and
  the values are loose ("return of prosperity" is not a condition). Keep it, but
  do not build a result on it without validating it first.

No quality cost: precision 1.000, recall 0.654, F1 0.791, hard cases 93% —
identical to the pre-schema run. Prompt length grew ~5% (109k -> 115k input
tokens on 16 pages), which is noise against the batch discount.
