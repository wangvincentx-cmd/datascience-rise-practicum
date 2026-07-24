"""Map extract_llm.py's ProQuest export -> analyze_economy.py's schema.

Runs on the Mac (operates on the label-only .export.jsonl, no article text).
extract_llm.py was built for the LOC monthly corpus, so its output is NOT
drop-in for the economy scorer:

  scorer needs          extract_llm emits            this maps
  ------------------    -------------------------    ------------------------
  predicted_state_at_   direction (improve/worsen/   worsen->recession,
    horizon (rec/exp)     no_change/unclear)           improve/no_change->
                                                        expansion, unclear->drop
  hedged (bool)         confidence (assertive/hedged) hedged->True
  window/window_kind/   (nothing)                     injected from CLI +
    source                                              windows_economy.csv

It also applies the scope==national filter -- the free precision cleanup for the
weak 4o-mini extractor: industry/regional/foreign claims can't be compared to
national NBER statistics anyway, and most of 4o-mini's false positives are
exactly those (cattle/wool industry lines, foreign economies, regional notes).

JUDGMENT CALL: no_change -> expansion. A "no recession coming" call IS a real
expansion prediction for a recession-vs-not study, and it's the signal you most
want in the placebo (calm) windows. To drop no_change instead, delete it from
STATE below -- one line.

Usage (from election_arm/, one window per call):
    python adapt_llm_economy.py \
        data/exports_raw/pred_llm_proquest_economy_covid_2020.export.jsonl covid_2020

Writes data/predictions/pred_proquestllm_economy_<window>.jsonl, whose name
matches analyze_economy.py's glob (pred_*_economy_*.jsonl). Keep the raw
.export.jsonl OUT of data/predictions/ so it isn't double-counted by that glob.
"""

import csv
import json
import sys
from pathlib import Path

# direction -> predicted_state_at_horizon. See JUDGMENT CALL in the docstring.
STATE = {"worsen": "recession", "improve": "expansion", "no_change": "expansion"}

OUT_DIR = Path("data/predictions")


def window_kinds():
    with open("data/windows_economy.csv") as f:
        return {r["window_id"]: r["kind"] for r in csv.DictReader(f)}


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: python adapt_llm_economy.py <export.jsonl> <window_id>")
    src, window = Path(sys.argv[1]), sys.argv[2]
    if not src.exists():
        raise SystemExit(f"No such file: {src}")
    kinds = window_kinds()
    if window not in kinds:
        raise SystemExit(f"Unknown window '{window}'. Known: {sorted(kinds)}")
    kind = kinds[window]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dst = OUT_DIR / f"pred_proquestllm_economy_{window}.jsonl"
    kept = dropped_scope = dropped_unclear = empties = 0
    with open(src) as f, open(dst, "w") as out:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("no_predictions"):
                out.write(line + "\n")   # keep genuine empties (denominator honesty)
                empties += 1
                continue
            if r.get("scope") and r["scope"] != "national":
                dropped_scope += 1
                continue
            state = STATE.get(r.get("direction"))
            if state is None:            # unclear or missing direction
                dropped_unclear += 1
                continue
            out.write(json.dumps({**r,
                "predicted_state_at_horizon": state,
                "hedged": r.get("confidence") == "hedged",
                "window": window, "window_kind": kind, "source": "proquest",
            }) + "\n")
            kept += 1

    print(f"{window} ({kind}): kept {kept} claims"
          f" (+{empties} no-prediction pages), dropped {dropped_scope} non-national,"
          f" {dropped_unclear} unclear/missing-direction -> {dst}")


if __name__ == "__main__":
    main()
