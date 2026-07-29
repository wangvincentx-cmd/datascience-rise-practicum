# Running the 1900–1963 scrape on another machine

`scrape_monthly.py` is a ~24 hour download job. It belongs on a machine that
stays on. The script imports **nothing** from this repo, so the entire bundle is
one file.

**No API key is needed.** loc.gov is public. Nothing here touches DeepInfra,
Gemini, or any paid service — the extraction step that costs money happens later,
back on the main machine.

---

## 1. Copy one file

Copy **`scrape_monthly.py`** to the desktop. That is the whole bundle. Put it in
its own folder — it will create `cache/` and `data/monthly/` beside itself.

```
D:\rise-scrape\
    scrape_monthly.py
```

## 2. Install

Python 3.9 or newer. Check:

```
python --version
```

The script runs on the **standard library alone**. One optional install, only if
the machine sits behind corporate TLS inspection or aggressive antivirus (this
laptop does — every HTTPS request failed with `CERTIFICATE_VERIFY_FAILED` until
it was added):

```
pip install truststore
```

If the desktop is on a normal home connection you can skip it. If you are not
sure, just install it — it is harmless either way, and without it a proxied
machine fails on the very first request.

## 3. Run

Open a terminal in that folder.

**Windows (PowerShell)** — survives the terminal closing, but not a reboot:

```powershell
Start-Process -WindowStyle Hidden python -ArgumentList "scrape_monthly.py --stage both" -RedirectStandardOutput scrape.log -RedirectStandardError scrape.err
```

Simpler, if you can leave a window open:

```powershell
python scrape_monthly.py --stage both *>&1 | Tee-Object scrape.log
```

**Linux / macOS:**

```bash
nohup python scrape_monthly.py --stage both > scrape.log 2>&1 &
```

Watch progress:

```
Get-Content scrape.log -Tail 20 -Wait      # PowerShell
tail -f scrape.log                          # bash
```

### Test it first

Do not start the 24-hour run blind. One month takes about a minute:

```
python scrape_monthly.py --stage both --start 1929-10 --end 1929-10 --pages-per-month 10
```

Expect roughly:

```
[1/1] 1929-10:   8 pages of 11,353 digitised
8 pages to fetch (of 8 in manifest)
8 pages -> data\monthly\pages_monthly.jsonl  (0 skipped)
```

If the digitised count comes back as **23,745,587**, the date filter is being
ignored and the denominators are worthless — stop and tell me. Any month should
be in the thousands or tens of thousands.

## 4. Interruptions are fine

Reboots, network drops, closing the window, Ctrl-C — all safe. **Rerun the exact
same command** and it picks up where it stopped:

- completed months are skipped via `monthly_manifest.csv`
- already-downloaded pages are skipped via `pages_monthly.jsonl`
- every HTTP response is cached in `cache/`

There is no "resume" flag. Just run it again.

## 5. What it produces

```
data/monthly/
    monthly_manifest.csv       ~23,000 rows: which pages were sampled
    monthly_denominators.csv   per month: total digitised pages + hits per term
    pages_monthly.jsonl        THE OUTPUT: full OCR text, ~500 MB
cache/                         raw API responses, ~3 GB -- DELETE, regenerable
```

Rough timings and sizes for the full 1900–1963 span at 30 pages/month:

| stage | requests | time | disk |
|---|---|---|---|
| search | ~4,600 | 4–6 h | small |
| fetch | ~46,000 | 16–24 h | ~3 GB cache + ~500 MB output |

If disk is tight, delete `cache/` once `pages_monthly.jsonl` looks complete —
`cache/` is a re-download optimisation, not data.

## 6. Bringing it back

Only two files matter. Compress them (the OCR text is very compressible,
~500 MB → ~150 MB):

**PowerShell:**
```powershell
Compress-Archive -Path data\monthly\pages_monthly.jsonl, data\monthly\monthly_manifest.csv, data\monthly\monthly_denominators.csv -DestinationPath monthly_corpus.zip
```

**bash:**
```bash
tar czf monthly_corpus.tar.gz data/monthly/pages_monthly.jsonl data/monthly/monthly_manifest.csv data/monthly/monthly_denominators.csv
```

Move it back by whatever is easiest — Drive, Dropbox, USB, `scp`. Then on the
repo machine, unpack into `JeremysShit/data/monthly/` so the paths line up:

```
JeremysShit/data/monthly/pages_monthly.jsonl
JeremysShit/data/monthly/monthly_manifest.csv
JeremysShit/data/monthly/monthly_denominators.csv
```

Sanity-check before deleting anything on the desktop:

```
python -c "import json; n=sum(1 for _ in open('data/monthly/pages_monthly.jsonl',encoding='utf-8')); print(n,'pages')"
```

These paths are gitignored (`cache/`, `JeremysShit/data/monthly/`), so nothing
here gets committed by accident. That rule had gone missing from `.gitignore`
and was restored on 2026-07-23 — if you ever see `cache/` or a `.jsonl` corpus
in `git status`, the rule has been lost again; do not commit them.

## 7. Then what

Back on the repo machine, extraction runs against that file:

```
python extract_llm.py --pages data/monthly/pages_monthly.jsonl \
    --out claims_monthly.jsonl --model gemini-3.5-flash --chunk-chars 40000 \
    --base-url https://generativelanguage.googleapis.com/v1beta/openai \
    --api-key-env GEMINI_API_KEY
```

That step costs money and is the one to budget for — see
`gold_extraction/RESULTS.md` for the price/quality table across models.

## Knobs

| flag | default | note |
|---|---|---|
| `--stage` | `both` | `search` and `fetch` can be run separately |
| `--start` / `--end` | `1900-01` / `1963-12` | LOC full text ends 1963 |
| `--pages-per-month` | `30` | split across the 5 neutral search terms |
| `--out-dir` | `data/monthly` | |
| `--overwrite` | off | **wipes progress** — do not use to resume |

Halving `--pages-per-month` roughly halves both the runtime and the later
extraction bill, at the cost of a noisier monthly index.
