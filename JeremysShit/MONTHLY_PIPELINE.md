# Monthly index pipeline (option C) — cost-gated, run when the scrape finishes

The plan: extract the full 1900–1963 monthly corpus with **gpt-oss-120b** (cheap,
F1 ~0.61), then score → index → model. Everything below is built and rehearsed on
the cached corpus. **Do NOT start until the scrape's fetch stage has produced
`data/monthly/pages_monthly.jsonl`.**

Budget so far: **~$17.50 spent, ~$12.50 left under $30.** This whole stage is
designed to land at ~$5–11, and step 2 is a hard gate that refuses to overrun.

---

## Step 0 — finish the scrape (free, on the desktop)

Recopy the fixed `scrape_monthly.py`, resume, let both stages complete:
```
python scrape_monthly.py --stage both        # resumes; ~2h search + ~16-24h fetch
```
Bring `data/monthly/pages_monthly.jsonl` back to the repo (see PORTABLE_SCRAPE.md).
Confirm it exists and looks right:
```
python -c "import json; n=sum(1 for _ in open('data/monthly/pages_monthly.jsonl',encoding='utf-8')); print(n,'pages')"
```

## Step 1 — COST TEST on 100 pages first (~$0.10, non-negotiable)

Never launch the full run on an estimate. Measure the real per-page cost:
```
export DEEPINFRA_API_KEY=...   # from .env
python extract_llm.py --pages data/monthly/pages_monthly.jsonl \
    --out /tmp/cost_test.jsonl --model openai/gpt-oss-120b \
    --base-url https://api.deepinfra.com/v1/openai --api-key-env DEEPINFRA_API_KEY \
    --chunk-chars 40000 --reasoning-effort low --limit 100
```
It prints `tokens: <in> in / <out> out` at the end. Compute the projected full cost:
```
python -c "
IN, OUT = <in_from_run>, <out_from_run>          # paste the two numbers
pages_tested, pages_total = 100, <total_pages>   # total from Step 0
scale = pages_total / pages_tested
cost = (IN*scale/1e6)*0.04 + (OUT*scale/1e6)*0.17   # gpt-oss-120b rates
print(f'projected full extraction: \${cost:.2f}')
"
```

## Step 2 — the GATE

- Projected **≤ $11** → proceed to Step 3.
- Projected **> $11** → drop to `--reasoning-effort low` if not already (caps output
  tokens, ~halves cost, F1 ~0.53), OR cut `pages_per_month` by re-scraping fewer,
  OR stop and reconsider. **Do not run a full extraction projected over budget.**

`--reasoning-effort low` is the safety valve: it is the config that made the
100-page test cheap, and it holds output tokens down across the full corpus.

## Step 3 — full extraction (the metered spend)

Same command, drop `--limit`:
```
python extract_llm.py --pages data/monthly/pages_monthly.jsonl \
    --out claims_monthly.jsonl --model openai/gpt-oss-120b \
    --base-url https://api.deepinfra.com/v1/openai --api-key-env DEEPINFRA_API_KEY \
    --chunk-chars 40000 --reasoning-effort low
```
Resumable: if it stops, rerun the same command (already-done pages are skipped).
It reports final token totals — reconcile against the Step-1 projection.

## Step 4 — score, index, model (all built, all free)

```
# score every claim against real economic data (no LLM)
python score_predictions.py --claims claims_monthly.jsonl --out monthly_scored.csv

# THE HEADLINE: the monthly press-expectations index
python build_press_index.py --claims claims_monthly.jsonl \
    --pages data/monthly/pages_monthly.jsonl --out data/press_index.csv

# the accuracy model (now non-degenerate: continuous months, real macro variation)
python model_hit.py --scored monthly_scored.csv
```

## Step 5 — figures & external validity

- Plot each `press_index.csv` series 1900–1963 with NBER recession shading
  (extend `make_prelim_figures.py`).
- Correlate `hedge_rate` / `disagreement` against the historical EPU index
  (`cache/US_Historical_EPU_data.xlsx`, 1900–2014) — external validation.
- The **model_hit macro baseline should now work** (many months per regime,
  time-blocked CV) — the degenerate-baseline warning should NOT fire. If it does,
  the corpus is still too thin; report the descriptive index instead.

---

## Honest disclosures for the poster

- **Two extractors, two quality levels.** The crisis analysis uses the validated
  Gemini corpus (F1 0.79); this monthly index uses gpt-oss-120b (F1 ~0.61,
  ~$5–11). State it. It is defensible because the index is an *aggregate* (per-
  claim noise partly averages out over ~15–30 claims/month), but it is not the
  same instrument as the crisis corpus and must not be presented as such.
- **The index is preliminary, the methods result is not.** Lead the poster with
  the verified extraction contribution (keyword 27% recall vs LLM 73%, gold-
  validated, 33 passing scorer tests). The index and the accuracy model are
  "preliminary patterns from a validated pipeline," reported with the F1 caveat.
