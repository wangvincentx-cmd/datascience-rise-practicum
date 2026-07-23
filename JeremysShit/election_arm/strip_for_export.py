"""
Strip article text from a predictions file so it is safe to Export from the
TDM Studio VM.

ProQuest forbids exporting full text or anything from which text could be
reconstructed. The only field in a pred_*.jsonl that carries article-sentence
text is `claim_text` (the extracted/paraphrased forecast sentence). This drops
it and keeps everything the scorer needs -- date, horizon, predicted state,
hedged, voice, window, source, etc. `attributed_to` is a person's name (a
derived fact, not article text), so it stays.

Run this in the workbench, then Export the resulting .export.jsonl.

Usage:
  python strip_for_export.py data/predictions/pred_proquest_economy_gfc_2008.jsonl

Writes: data/predictions/pred_proquest_economy_gfc_2008.export.jsonl
"""

import json
import sys
from pathlib import Path

DROP_FIELDS = {"claim_text"}


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: python strip_for_export.py <pred_file.jsonl>")
    src = Path(sys.argv[1])
    if not src.exists():
        raise SystemExit(f"No such file: {src}")
    dst = src.with_suffix(".export.jsonl")

    kept = 0
    with open(src) as f, open(dst, "w") as out:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record = {k: v for k, v in record.items() if k not in DROP_FIELDS}
            out.write(json.dumps(record) + "\n")
            kept += 1

    size_kb = dst.stat().st_size / 1024
    print(f"wrote {kept} records (claim_text removed) -> {dst}")
    print(f"size: {size_kb:.1f} KB  (Export cap is 15 MB, so this is fine)")


if __name__ == "__main__":
    main()
