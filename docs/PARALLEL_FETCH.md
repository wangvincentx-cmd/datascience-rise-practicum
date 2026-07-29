# Splitting the page fetch across 4 computers

The slow part (downloading OCR) is embarrassingly parallel: the search stage
already produced the full list of pages (`monthly_manifest.csv`, 15,744 rows), so
each machine just fetches a disjoint quarter of it. No machine talks to the
others; you combine the results at the end with automatic de-duplication.

**Time:** ~4× faster. What took ~24–36h on one machine finishes in ~6–9h across
four.

---

## What each machine needs (tiny)

Two files, copied to an empty folder:

```
scrape_monthly.py          the script
data/monthly/monthly_manifest.csv   the page list from the search stage
```

That's it — the manifest is a few MB. Each machine builds its own `cache/` and
downloads only its shard. **Do not copy anyone's `pages_monthly.*` files between
machines.**

Setup on each machine:
```
pip install truststore          # only if that machine is behind a proxy/AV
mkdir -p data/monthly            # put monthly_manifest.csv here
```

## Run — one command per machine, differing only in the shard number

**Machine 1:**
```
python scrape_monthly.py --stage fetch --shard 1/4
```
**Machine 2:**
```
python scrape_monthly.py --stage fetch --shard 2/4
```
**Machine 3:**
```
python scrape_monthly.py --stage fetch --shard 3/4
```
**Machine 4:**
```
python scrape_monthly.py --stage fetch --shard 4/4
```

Each handles exactly 3,936 pages (every 4th manifest row — balanced, so no
machine gets stuck with all the heavy months) and writes its own file:
`pages_monthly.shard1of4.jsonl`, `…shard2of4.jsonl`, etc. The names carry the
shard, so they never collide.

**Resumable per machine:** if one stops (reboot, LOC flakiness), rerun the exact
same command on that machine — it skips what it already has and continues. Only
that machine's shard is affected.

> Not limited to 4. `--shard 2/6` = machine 2 of 6, etc. Use however many
> computers you have; keep the `/N` the same across all of them.

## Combine — on ONE machine, after all four finish

Copy the four `pages_monthly.shard*of4.jsonl` files into one folder (any of the
machines, or a fresh one — they total ~300–400 MB, ~100 MB zipped), then:

```
python combine_shards.py --in-dir data/monthly --out data/monthly/pages_monthly.jsonl
```

It merges them, **drops any duplicate page (by page_id)**, and prints a health
line:

```
15744 unique pages -> data/monthly/pages_monthly.jsonl
median 24,000 chars/page (healthy is ~20k-26k; ~7-8k would mean truncation)
```

- **`unique pages` should be ~15,744** (a bit less if some pages had no OCR and
  were skipped — that's normal).
- **`median ~20k–26k chars`** confirms the pages are complete, not truncated.
- If it reports a large number of **duplicates skipped**, two machines ran the
  same `--shard` number — harmless (dupes were dropped) but check your shard
  assignments so you didn't *miss* a shard.

That combined `pages_monthly.jsonl` is the corpus the extraction step reads
(see MONTHLY_PIPELINE.md).

## Sanity check before combining (optional)

On each machine, confirm its shard looks healthy:
```
python -c "import json,statistics; r=[json.loads(l) for l in open('data/monthly/pages_monthly.shard1of4.jsonl',encoding='utf-8')]; print(len(r),'pages, median',statistics.median([x['n_chars'] for x in r]),'chars')"
```
(change the filename per machine). Median ~20k+ = good.

## Why there are no duplicates

Machine K takes manifest row `i` only when `i % N == K-1`. Four machines with
`N=4` partition the rows perfectly — verified: 4×3,936 = 15,744, every row
assigned exactly once, zero overlap. The de-dup in `combine_shards.py` is a
belt-and-suspenders check, not something the sharding relies on.
