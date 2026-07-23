"""
Build a page-level corpus from the scraper's disk cache -- fully offline.

`newspaper_scraper.py` downloaded two things per newspaper page and cached both:
the page-resource JSON (metadata) and the page's full OCR text. It then threw
most of that away, keeping only the handful of sentences its regex matched. This
script recovers the whole thing: every cached page, with clean text and full
metadata, as data/pages.jsonl -- the input for whole-page LLM extraction.

Two things it fixes on the way through:

1. **The OCR text is parsed, not string-mangled.** The cached "full text" file is
   actually JSON: {"<segment_id>": {"full_text": "..."}}. `fetch_full_text()`
   decoded it as one big string and handed it straight to the regex extractor,
   so every page began with a literal '{"/service/ndnp/...":{"full_text":"'
   prefix and every real newline arrived as a two-character backslash-n. That is
   the source of the raw-NDNP-markup rows that
   handgrade_newspapers/eval_vs_consensus.py has to filter out as ungradeable,
   and it corrupted sentence splitting everywhere else. Here the JSON is parsed
   and the real text comes out.

2. **Pages the regex found nothing on are kept.** The cache holds ~2,190 pages
   but only ~1,250 produced a single claim, because a page only yielded rows if
   a search phrase happened to sit within two sentences of a future-tense verb.
   The other ~940 pages were downloaded, scanned, and silently dropped. They are
   in this corpus, and they are where a large part of the recall gap lives.

Usage:
    python build_page_corpus.py                      # cache/ -> data/pages.jsonl
    python build_page_corpus.py --limit 50           # quick check
    python build_page_corpus.py --cache-dir cache --out data/pages.jsonl

No network, no API key. Safe to rerun; overwrites the output.
"""

import argparse
import json
import re
from pathlib import Path

from newspaper_scraper import EPISODES

# Same transform newspaper_scraper._get() used to name cache files. Must stay
# byte-identical to it or nothing resolves.
CACHE_KEY_RE = re.compile(r"[^A-Za-z0-9]+")


def cache_key(url, cache_dir):
    """The cache filename newspaper_scraper._get() would have written for `url`."""
    return Path(cache_dir) / (CACHE_KEY_RE.sub("_", url)[-150:] + ".bin")


def parse_fulltext(raw):
    """Pull the real OCR text out of a cached word-coordinates-service payload.

    The payload is {"<segment>": {"full_text": "..."}}; a page can carry more
    than one segment, so join them in key order. Falls back to the decoded bytes
    if it isn't the JSON shape (older cache entries, or a service change), which
    is what the original code assumed unconditionally."""
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
    try:
        doc = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return text.strip()
    if not isinstance(doc, dict):
        return text.strip()
    parts = []
    for _, seg in sorted(doc.items()):
        if isinstance(seg, dict) and seg.get("full_text"):
            parts.append(seg["full_text"])
    return "\n".join(parts).strip() if parts else text.strip()


def _first(value):
    """LOC returns most metadata as single-element lists."""
    if isinstance(value, list):
        return value[0] if value else None
    return value


def episode_for(date, episodes=EPISODES):
    """(episode_name, kind) for a date, or (None, None) if it falls outside every
    scraped window. Kept separate from extraction on purpose: the episode label
    is corpus bookkeeping and must never reach the model, since names like
    "1929 Crash" state the outcome the model is supposed to be blind to."""
    if not date:
        return None, None
    for ep in episodes:
        if ep["start"] <= date <= ep["end"]:
            return ep["name"], ep.get("kind", "crisis")
    return None, None


def iter_page_records(cache_dir="cache", limit=None):
    """Yield one record per cached page that has resolvable full text."""
    cache_dir = Path(cache_dir)
    n = 0
    for f in sorted(cache_dir.glob("*.bin")):
        try:
            doc = json.loads(f.read_bytes())
        except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
            continue  # a full-text payload or a non-JSON blob, not a page record
        if not isinstance(doc, dict):
            continue
        resource = doc.get("resource") or {}
        fulltext_url = resource.get("fulltext_file")
        item = doc.get("item") or {}
        if not fulltext_url or not item:
            continue  # search-results JSON, or a resource with no OCR
        ft_file = cache_key(fulltext_url, cache_dir)
        if not ft_file.exists():
            continue
        text = parse_fulltext(ft_file.read_bytes())
        if not text:
            continue
        date = item.get("date")
        episode, kind = episode_for(date)
        yield {
            # segment_id is the one field guaranteed unique per physical page --
            # loc_url is per ISSUE, so a 12-page paper shares one of those.
            "page_id": doc.get("segment_id") or resource.get("url"),
            "loc_url": resource.get("url"),
            "date": date,
            "publisher": _first(item.get("partof_title")),
            "newspaper_title": _first(item.get("newspaper_title")),
            "state": "; ".join(item.get("location_state") or []),
            "city": "; ".join(item.get("location_city") or []),
            "lccn": _first(item.get("number_lccn")),
            "episode": episode,
            "kind": kind,
            "n_chars": len(text),
            "ocr_text": text,
        }
        n += 1
        if limit and n >= limit:
            return


def build(cache_dir="cache", out="data/pages.jsonl", limit=None):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n, chars, undated, no_episode = 0, 0, 0, 0
    by_episode = {}
    with open(out, "w", encoding="utf-8") as fh:
        for rec in iter_page_records(cache_dir, limit):
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            chars += rec["n_chars"]
            if not rec["date"]:
                undated += 1
            if not rec["episode"]:
                no_episode += 1
            by_episode[rec["episode"]] = by_episode.get(rec["episode"], 0) + 1
    print(f"{n} pages -> {out}  ({chars/1e6:.1f}M chars, "
          f"{chars/n if n else 0:,.0f} avg)")
    if undated:
        print(f"  {undated} pages with no date (cannot be placed in a window)")
    print(f"  {no_episode} pages outside every scraped episode window")
    for ep, count in sorted(by_episode.items(), key=lambda kv: (kv[0] is None, kv[0])):
        print(f"    {ep or '(outside all windows)':<28} {count:>5}")
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--out", default="data/pages.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    build(args.cache_dir, args.out, args.limit)
