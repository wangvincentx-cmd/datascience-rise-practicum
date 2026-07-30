"""Run every model in poster_models and tee the output to outputs/.

    python poster_models/run_all.py            # default settings
    python poster_models/run_all.py --fast     # smaller bootstrap / subsample
    python poster_models/run_all.py --only m1 m4

Each model is a standalone script and can be run on its own; this exists so the
whole suite can be regenerated in one command and the console output kept as a
record next to the CSVs.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

MODELS = [
    ("m1", "m1_fixed_effects.py", "within-period comparison", [], []),
    ("m2", "m2_clustered.py", "honest error bars", ["--boot", "300"],
     ["--boot", "60"]),
    ("m3", "m3_dml.py", "macro increment with a CI", [], ["--folds", "3"]),
    ("m4", "m4_text.py", "the words themselves", [], []),
    ("m5a", "m5a_direction.py", "error direction", [], []),
    ("m5b", "m5b_survival.py", "time to realization", [],
     ["--sample", "3000"]),
    ("m5c", "m5c_press_probit.py", "press index vs recessions", [], []),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", help="subset of model ids to run")
    ap.add_argument("--fast", action="store_true",
                    help="reduced bootstrap/folds/sample for a quick pass")
    ap.add_argument("--rigid", action="store_true",
                    help="real-horizon claims only (passed through)")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    todo = [m for m in MODELS if not args.only or m[0] in args.only]
    if not todo:
        raise SystemExit(f"No models matched {args.only}. "
                         f"Known: {[m[0] for m in MODELS]}")

    status = C.fred_status()
    missing = [s for s, ok in status.items() if not ok]
    if missing:
        print(f"NOTE: FRED cache missing {', '.join(missing)}. Models degrade "
              f"to the\n      series that are available and say so. Run "
              f"`python poster_models/fetch_fred.py`\n      for the "
              f"full-coverage numbers.\n")

    results = []
    for mid, script, blurb, normal, fast in todo:
        extra = (fast if args.fast else normal)
        if args.rigid and mid != "m5c":  # m5c is monthly, not claim-level
            extra = extra + ["--rigid"]
        cmd = [sys.executable, str(here / script)] + extra
        print(f"\n{'#' * 78}\n# {mid}: {blurb}\n# {' '.join(cmd[1:])}\n{'#' * 78}")
        t0 = time.time()
        r = subprocess.run(cmd, cwd=C.ROOT, capture_output=True, text=True)
        dt = time.time() - t0
        log = C.OUT / f"{mid}_console.txt"
        log.write_text(r.stdout + ("\n[stderr]\n" + r.stderr if r.stderr else ""))
        print(r.stdout[-4000:] if len(r.stdout) > 4000 else r.stdout)
        if r.returncode != 0:
            print(f"  !! {mid} exited {r.returncode}")
            print(r.stderr[-2000:])
        results.append((mid, r.returncode, dt, log))

    print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
    for mid, rc, dt, log in results:
        mark = "ok  " if rc == 0 else f"FAIL"
        print(f"  {mark}  {mid:<5} {dt:>6.1f}s   {log.relative_to(C.ROOT)}")
    print(f"\n  CSVs in {C.OUT.relative_to(C.ROOT)}/")
    if any(rc != 0 for _, rc, _, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
