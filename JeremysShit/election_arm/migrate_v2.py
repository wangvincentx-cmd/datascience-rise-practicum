"""
Bring the VM's data/predictions/ from schema v1 to v2, in one command.

Runs INSIDE the TDM Studio workbench, from election_arm/. After the v2 scripts
are pasted in, every window extracted under the old schema still sits in
data/predictions/ in the old vocabulary. extract_gpt.py refuses to append v2
records to those files (mixing two label sets in one file that
analyze_economy.py globs together is the quiet failure mode), so each one has to
be moved aside before its window can be re-extracted. This does that for all of
them at once, and tells you what it will cost.

IT NEVER DELETES ANYTHING. Files are RENAMED to *.v1.bak, which is reversible
with --restore. That matters more than it looks:

  * The label-only exports of all 9 v1 windows are already committed to git on
    `main` (data/proquest/*.export.jsonl), so the v1 LABELS are safe regardless.
  * But the VM copies are the only ones that still contain `claim_text` -- the
    article text that may never leave the VM. Human kappa validation
    (validate_kappa.py), the TF-IDF text model (model.py) and eyeballing
    (sample_claims.py) all need it. Delete it and the only way back is to spend
    the extraction quota again.

So: quarantine, don't delete. Disk is free; the daily LLM quota is not.

Usage (from election_arm/):
    python migrate_v2.py                  # DRY RUN -- shows the plan, changes nothing
    python migrate_v2.py --apply          # do it
    python migrate_v2.py --restore        # put the .v1.bak files back
"""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

PRED_DIR = Path("data/predictions")
BAK_SUFFIX = ".v1.bak"


def check_scripts_updated():
    """Fail fast if the v2 scripts were not pasted in yet.

    Running this against the OLD extract_gpt.py would quarantine every window
    and then re-extract them right back into v1 -- a full quota burn for no
    change at all."""
    problems = []
    try:
        import extract_gpt
        if getattr(extract_gpt, "SCHEMA_VERSION", 1) != 2:
            problems.append("extract_gpt.py is still the v1 copy "
                            "(no SCHEMA_VERSION == 2)")
    except ImportError as e:
        problems.append(f"cannot import extract_gpt.py ({e})")
    try:
        import strip_for_export
        if "quote" not in strip_for_export.DROP_FIELDS:
            problems.append("strip_for_export.py is still the v1 copy "
                            "(does not drop `quote`)")
    except ImportError as e:
        problems.append(f"cannot import strip_for_export.py ({e})")
    if problems:
        raise SystemExit(
            "\n*** The v2 scripts are not in place yet:\n"
            + "".join(f"      - {p}\n" for p in problems)
            + "*** Paste the transfer block from vm_update_commands_v2.txt first,\n"
              "*** then re-run this. Nothing has been changed.\n")


def classify(path):
    """(schema, n_claims, n_empty, n_fake_empty) for one predictions file.

    `fake_empty` counts the no_predictions rows written by the pre-fix
    extractor, which recorded FAILED calls as genuinely empty pages. They are
    indistinguishable from real empties in v1, which is one more reason those
    windows are worth redoing rather than keeping."""
    versions, claims, empty, fake = set(), 0, 0, 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            versions.add(r.get("schema_version", 1))
            if r.get("no_predictions"):
                empty += 1
                if not r.get("empty_source_text") and r.get("schema_version") != 2:
                    fake += 1
            else:
                claims += 1
    if not versions:
        return "empty-file", 0, 0, 0
    if versions == {2}:
        return "v2", claims, empty, fake
    if 2 in versions:
        return "MIXED", claims, empty, fake
    return "v1", claims, empty, fake


def survey():
    if not PRED_DIR.is_dir():
        raise SystemExit(f"No {PRED_DIR}/ here. Run this from election_arm/ "
                         f"inside the VM.")
    rows = []
    for p in sorted(PRED_DIR.glob("*.jsonl")):
        if p.name.endswith(".export.jsonl"):
            continue          # regenerable from the pred file; handled separately
        schema, claims, empty, fake = classify(p)
        rows.append({"path": p, "schema": schema, "claims": claims,
                     "empty": empty, "fake_empty": fake})
    return rows


def report(rows):
    print(f"\n{'file':<46}{'schema':<9}{'claims':>8}{'empty':>7}{'fake':>7}")
    print("-" * 77)
    for r in rows:
        print(f"{r['path'].name:<46}{r['schema']:<9}{r['claims']:>8}"
              f"{r['empty']:>7}{r['fake_empty']:>7}")
    by = Counter(r["schema"] for r in rows)
    print("-" * 77)
    print("  " + ", ".join(f"{v} {k}" for k, v in sorted(by.items())) or "  nothing")
    return by


def do_migrate(rows, apply):
    todo = [r for r in rows if r["schema"] in ("v1", "MIXED")]
    if not todo:
        print("\nNothing to migrate -- no v1 or MIXED files. You are already on v2.")
        return
    claims = sum(r["claims"] for r in todo)
    print(f"\n{len(todo)} file(s) hold {claims} v1 claims and must be moved aside "
          f"before their window can be re-extracted.")
    print("Their label-only exports are committed on main "
          "(data/proquest/*.export.jsonl), but the in-VM copies are the only "
          "ones with claim_text -- so they are RENAMED, never removed.\n")
    for r in todo:
        dst = r["path"].with_suffix(r["path"].suffix + BAK_SUFFIX)
        if dst.exists():
            print(f"  SKIP  {r['path'].name}  ({dst.name} already exists)")
            continue
        print(f"  {'mv  ' if apply else 'WOULD mv  '}{r['path'].name} -> {dst.name}")
        if apply:
            shutil.move(str(r["path"]), str(dst))
    # The stripped exports are derived, so they are stale the moment their
    # source moves. Move them too rather than leave a v1 export next to a v2
    # pred file, where run_all_economy.sh would bundle the wrong one.
    for r in todo:
        exp = r["path"].with_suffix(".export.jsonl")
        if exp.exists():
            dst = exp.with_suffix(exp.suffix + BAK_SUFFIX)
            if not dst.exists():
                print(f"  {'mv  ' if apply else 'WOULD mv  '}{exp.name} -> {dst.name}")
                if apply:
                    shutil.move(str(exp), str(dst))
    windows = sorted({r["path"].stem.replace("pred_proquest_economy_", "")
                      .replace("pred_nyt_economy_", "") for r in todo})
    if apply:
        print("\nDone. Re-extract these windows (quota resets daily):")
        print(f"  bash run_all_economy.sh {' '.join(windows)}")
        print("\nTo undo:  python migrate_v2.py --restore")
    else:
        print("\nDRY RUN -- nothing changed. Re-run with --apply to do it.")


def do_restore(apply):
    baks = sorted(PRED_DIR.glob(f"*{BAK_SUFFIX}"))
    if not baks:
        raise SystemExit(f"No *{BAK_SUFFIX} files in {PRED_DIR}/.")
    print(f"\nRestoring {len(baks)} quarantined file(s):")
    for b in baks:
        # NOT with_suffix(""): BAK_SUFFIX is two suffix components, so that
        # would leave "...jsonl.v1" behind. Strip the literal string.
        dst = b.with_name(b.name[:-len(BAK_SUFFIX)])
        if dst.exists():
            print(f"  SKIP  {b.name}  ({dst.name} exists -- would overwrite v2 output)")
            continue
        print(f"  {'mv  ' if apply else 'WOULD mv  '}{b.name} -> {dst.name}")
        if apply:
            shutil.move(str(b), str(dst))
    if not apply:
        print("\nDRY RUN -- nothing changed. Re-run with --apply.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true",
                    help="actually move the files (default is a dry run)")
    ap.add_argument("--restore", action="store_true",
                    help="move *.v1.bak files back and exit")
    ap.add_argument("--skip-script-check", action="store_true",
                    help="do not verify that the v2 scripts are installed")
    args = ap.parse_args()

    if args.restore:
        do_restore(args.apply)
        return
    if not args.skip_script_check:
        check_scripts_updated()
        print("v2 scripts confirmed in place (extract_gpt schema 2, "
              "strip_for_export drops `quote`).")
    rows = survey()
    if not rows:
        print(f"\n{PRED_DIR}/ has no prediction files yet -- nothing to migrate.")
        return
    report(rows)
    do_migrate(rows, args.apply)


if __name__ == "__main__":
    main()
