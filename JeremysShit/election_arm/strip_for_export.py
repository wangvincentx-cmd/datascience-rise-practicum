"""
Strip article text from a predictions file so it is safe to Export from the
TDM Studio VM.

ProQuest forbids exporting full text or anything from which text could be
reconstructed. In a pred_*.jsonl the text-bearing field is the extracted
forecast sentence: `quote` in schema v2 (verbatim article text -- the whole
point of the hallucination guard is that it IS verbatim), and `claim_text` in
the older v1 files. Both are dropped. Everything the scorers need survives --
topic, direction, price/unemployment direction, scope, horizon_months,
horizon_hint, confidence, voice, date, window, source. `speaker_name` /
`attributed_to` are a person's name (a derived fact, not article text), so they
stay.

`horizon_hint` is why stripping the quote is safe for scoring: main's
score_predictions.py normally reads the quote's own time language to decide a
claim's horizon, which is impossible once the quote is gone. extract_gpt.py runs
that inference in-VM and leaves this one-token verdict behind.

THE TRIPWIRE: any schema change that adds a text-bearing field to the extractor
must add it to DROP_FIELDS here, in the same commit. Because that is easy to
forget and the failure is a licence breach rather than a crash, this script also
refuses to write when any surviving field looks like prose (see MAX_FIELD_CHARS).

Run this in the workbench, then Export the resulting .export.jsonl.

Usage:
  python strip_for_export.py data/predictions/pred_proquest_economy_gfc_2008.jsonl

Writes: data/predictions/pred_proquest_economy_gfc_2008.export.jsonl
"""

import json
import sys
from pathlib import Path

# v2 emits `quote`; v1 files still on disk use `claim_text`. Drop both.
DROP_FIELDS = {"quote", "claim_text"}

# No label is prose. Anything longer than this in a surviving field means a
# text-bearing field was added upstream and not listed above.
MAX_FIELD_CHARS = 200
# Identifiers, not article text: a NYT page_id is a long URL.
LENGTH_EXEMPT = {"page_id", "loc_url", "url"}


def check_no_prose(record, lineno):
    for k, v in record.items():
        if k in LENGTH_EXEMPT or not isinstance(v, str):
            continue
        if len(v) > MAX_FIELD_CHARS:
            raise SystemExit(
                f"\n*** REFUSING TO WRITE: line {lineno}, field {k!r} is "
                f"{len(v)} chars.\n"
                f"*** Fields that long are article text, which must not leave "
                f"the VM.\n"
                f"*** If the extractor gained a text-bearing field, add it to "
                f"DROP_FIELDS in this file.\n"
                f"*** Offending value starts: {v[:80]!r}\n")


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: python strip_for_export.py <pred_file.jsonl>")
    src = Path(sys.argv[1])
    if not src.exists():
        raise SystemExit(f"No such file: {src}")
    dst = src.with_suffix(".export.jsonl")

    records = []
    with open(src) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record = {k: v for k, v in record.items() if k not in DROP_FIELDS}
            check_no_prose(record, lineno)   # before anything is written
            records.append(record)

    with open(dst, "w") as out:
        for record in records:
            out.write(json.dumps(record) + "\n")

    size_kb = dst.stat().st_size / 1024
    dropped = ", ".join(sorted(DROP_FIELDS))
    print(f"wrote {len(records)} records ({dropped} removed) -> {dst}")
    print(f"size: {size_kb:.1f} KB  (Export cap is 15 MB, so this is fine)")


if __name__ == "__main__":
    main()
