# CC-News prototype (BU SCC / SGE)

One-month feasibility test for pulling newspaper articles from **CC-News**
(Common Crawl's news subset, Aug 2016 → present) on the SCC. Goal: measure
outlet coverage, text quality, and disk/time cost before committing to a full
2016→present pull.

Structural limit: CC-News only reaches **Aug 2016 onward** — it cannot cover
the historical crisis windows (GFC 2008 and earlier). Use it for the recent /
deployable-economy direction, not the historical retrospective.

## Setup (once)

```bash
module load python3/3.10.12          # match `module avail python3`
pip install --user warcio trafilatura
mkdir -p logs
```

## Run a small test — first 20 WARCs of Jan 2022

```bash
YEAR=2022 MONTH=01 qsub -t 1-20 run_ccnews.sh
```

A full month is ~600–1000+ WARC files; check the count after the listing is
cached:  `wc -l ccnews_out/2022-01/warc.paths`. To do a whole month, set
`-t 1-<count>`.

## Collect results

```bash
cat ccnews_out/2022-01/articles.*.jsonl > ccnews_out/2022-01/all.jsonl
wc -l ccnews_out/2022-01/all.jsonl                       # total articles
# outlet mix:
python -c "import json,collections,sys; c=collections.Counter(json.loads(l)['domain'] for l in open(sys.argv[1])); [print(n,d) for d,n in c.most_common(30)]" ccnews_out/2022-01/all.jsonl
```

## The three questions this test answers

1. **Coverage** — do the outlets you care about show up, and how many articles each?
2. **Quality** — is `text` full-body or truncated/paywalled? (`n_chars` is a quick proxy.)
3. **Cost** — wall-time per WARC (in `logs/`) and articles-per-WARC → extrapolate
   storage/time for the full 2016→present pull.

## Notes

- Common Crawl's S3/HTTPS mirror is public and free — no AWS account, no egress cost.
- Run downloads as **batch tasks on compute nodes**, never on the login node.
- `--domains domains.txt` (one substring per line, e.g. `nytimes.com`) filters to
  specific outlets; omit it on the first run to see the full mix.
- `date` is trafilatura's best-guess publish date — spot-check it before trusting
  it as an information-time signal for the leakage rule.
