# Election arm — FAILED / abandoned

**Nothing in the election arm reached the 2026 RISE poster. Do not treat any
number in here as a project result.**

Kept for the record, not for reuse. The economy side of this folder is the live
work; see [PROQUEST_TDM_GUIDE.md](PROQUEST_TDM_GUIDE.md).

## Why it failed

**The NYT API returns article titles only.** Headlines do not carry the forecast
text, so there is nothing for an extractor to read — the whole election arm was
built on a source that cannot supply the evidence it needs. This is the same
limitation listed on the poster. `download_nyt.py`, the 28
`data/raw/nyt_elections_*.jsonl` files, and `analyze_elections.py` all inherit
it.

Secondary blockers, never cleared: the full NYT pull needs an `NYT_API_KEY` and
runs multi-day at 500 requests/day, and the κ validation needed two human graders
that were never assigned to this arm.

## What is dead here

| file | status |
|---|---|
| `analyze_elections.py` | dead — election scoring, no usable corpus |
| `download_nyt.py`, `download_loc.py` | dead — the NYT path above |
| `data/raw/nyt_elections_*.jsonl` (28 files) | titles only, unusable |
| `data/ground_truth_elections.csv` | dead — nothing left to score against |
| `data/raw/nyt_economy_*.jsonl` (19 files) | titles only, superseded by `data/proquest/` |
| `README.md`, `VINCENT_README.md` | describe the two-arm plan that was dropped |

## What is still live

`extract_gpt.py`, `verify_gpt.py`, `qa_extraction.py`, `tdm_parse.py`,
`run_corpus_economy.sh`, `vm_doctor.py`, `strip_for_export.py` — the ProQuest
economy pipeline, which is what this branch exists for. Its output is committed
at `data/proquest/`.

The one election-arm result worth remembering is not an election result at all:
the Livingston benchmark in `data/scored_livingston.csv` — economists 83.9%
overall but 38.5% within six months of a business-cycle peak (n=13). That
finding survived into the main analysis.
