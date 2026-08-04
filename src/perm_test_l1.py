"""Block-permutation null for the L1 forecast model, at a usable replicate count.

make_l1_ladder_figure.py caches a 20-replicate null. That is not enough to
report a p-value: p = (1 + #{null >= observed}) / (1 + reps), so 20 replicates
cannot produce anything smaller than 1/21 = 0.048 no matter how strong the
signal is. The published "p = 0.048" is that floor, not a measurement.

This runs the same test at 200+ replicates and in parallel. The permutation
is identical to the cached one: `hit` shuffled WITHIN each 3-year block, so
the block structure and the class balance per block are preserved and only
the claim-to-outcome pairing is destroyed. The model refitted under each
shuffle is the full rung ("+ direction x economy"), L1 at C=0.5, scored with
the same leave-one-block-out CV.

Run from the repo root:
    python src/perm_test_l1.py --perm 200 [--jobs 8] [--out data/models/perm_l1_200.json]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hit_predictor as HP
from macro_context import FACTORS
from make_l1_ladder_figure import C_L1, build
from model_variants import clf_for


def _full_rung():
    """The last rung of HP.LADDER -- the model the poster reports."""
    name = cat = num = None
    for name_, cat_, num_ in HP.LADDER:
        if cat_ or num_:
            name, cat, num = name_, cat_, num_
    return name, cat, num


def _one_perm(i, X, y, groups, cat, num, seed):
    """One block-permuted refit. Its own RNG so results do not depend on order."""
    rng = np.random.default_rng(seed + i)
    yp = y.copy()
    for g in set(groups):
        idx = np.where(groups == g)[0]
        yp[idx] = rng.permutation(yp[idx])
    m = HP.metrics(yp, HP.oof_predict(X, yp, groups, cat, num,
                                      clf=clf_for("l1", C_L1)))
    return float(m["auc"]) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perm", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--out", default="data/models/perm_l1_200.json")
    a = ap.parse_args()

    X, y, groups = build()
    name, cat, num = _full_rung()
    n_blocks = len(set(groups))
    print(f"forecasts {len(y):,}   blocks {n_blocks}   features {X.shape[1]}")
    print(f"rung: {name}   penalty L1   C={C_L1}")

    t0 = time.time()
    observed = HP.metrics(y, HP.oof_predict(X, y, groups, cat, num,
                                            clf=clf_for("l1", C_L1)))["auc"]
    per_fit = time.time() - t0
    print(f"observed AUC {observed:.4f}   ({per_fit:.0f}s per refit)")
    print(f"running {a.perm} permutations on {a.jobs} workers "
          f"(~{per_fit * a.perm / max(a.jobs, 1) / 60:.0f} min)...", flush=True)

    null = Parallel(n_jobs=a.jobs, verbose=10)(
        delayed(_one_perm)(i, X, y, groups, cat, num, HP.SEED)
        for i in range(a.perm))
    null = [v for v in null if v is not None]

    beats = int(sum(v >= observed for v in null))
    p = (1 + beats) / (1 + len(null))
    res = {"penalty": "l1", "C": C_L1, "rung": name,
           "n": int(len(y)), "n_blocks": n_blocks,
           "observed_auc": float(observed),
           "perm_reps": len(null),
           "perm_null_mean": float(np.mean(null)),
           "perm_null_sd": float(np.std(null, ddof=1)),
           "perm_null_max": float(np.max(null)),
           "perm_null_q95": float(np.quantile(null, 0.95)),
           "n_null_ge_observed": beats,
           "p": float(p),
           "p_floor_at_this_rep_count": 1.0 / (1 + len(null)),
           "null": [float(v) for v in null]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))

    print(f"\nobserved AUC        {observed:.4f}")
    print(f"null mean           {res['perm_null_mean']:.4f} "
          f"(sd {res['perm_null_sd']:.4f})")
    print(f"null max / q95      {res['perm_null_max']:.4f} / {res['perm_null_q95']:.4f}")
    print(f"null >= observed    {beats} of {len(null)}")
    print(f"p                   {p:.4f}   (floor at this rep count: "
          f"{res['p_floor_at_this_rep_count']:.4f})")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
