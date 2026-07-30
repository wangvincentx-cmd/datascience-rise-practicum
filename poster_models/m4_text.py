"""MODEL 4 -- Read the sentence, not just its metadata.

The problem this solves
-----------------------
src/model_hit.py knows six things about each forecast: its topic, direction,
voice, scope, roughly how long it is, and whether it contains a digit. It never
looks at the words. Everything the extractor threw away -- the difference
between "trade will revive" and "a sharp reaction is inevitable" -- is invisible
to the poster's model.

This file puts the text back in and asks whether it predicts accuracy.

Why CHARACTER n-grams
---------------------
The corpus is 1900-1963 newspaper OCR and it is genuinely mangled: "busi ness",
"prosper f'ra", "31*00", "tho signs favorable". Word tokenisation turns each
distinct corruption into its own unique token, so the shared signal in
"business" is scattered across dozens of hapax forms and thrown away by any
min_df filter. Character n-grams of length 3-5 still match the intact
fragments, which is why they are the standard choice for noisy OCR. Word
n-grams are fitted too -- not because they score better, but because a list of
predictive PHRASES is readable on a poster and a list of predictive character
fragments is not.

Three guards against the obvious ways this could fool us
--------------------------------------------------------
1. Grouped CV. Every split holds out whole 3-year blocks, so the model is never
   scored on an era it was trained on. This is the difference between measuring
   "wording predicts accuracy" and measuring "the model memorised that
   1930-1932 went badly."
2. A digit-masked variant. Text contains years, and a year is a direct pointer
   to the macro outcome. Replacing every digit with '#' removes that channel;
   if the AUC survives, the signal was in the language.
3. A within-block permutation baseline. Shuffling the labels inside each block
   and refitting gives the AUC this pipeline produces from noise alone. High-
   dimensional sparse models on 22 effective clusters can score above 0.5 on
   pure noise, and the honest benchmark is that number, not 0.5.

LOC only. ProQuest rows arrive with `quote` stripped -- the verbatim text may
not leave the TDM VM -- so this model cannot be run on that corpus at all.

    python poster_models/m4_text.py
    python poster_models/m4_text.py --perm 20
"""

import argparse
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import _common as C

warnings.filterwarnings("ignore")

DIGIT_RE = re.compile(r"\d")


def clean(s, mask_digits=False):
    t = str(s).lower().strip()
    if mask_digits:
        t = DIGIT_RE.sub("#", t)
    return t


def char_vec(min_df):
    # char_wb keeps n-grams inside word boundaries, so a fragment never spans
    # two unrelated words -- which matters here because OCR inserts spaces in
    # the middle of words far more often than it deletes them.
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                           min_df=min_df, max_features=200_000,
                           sublinear_tf=True, lowercase=True)


def word_vec(min_df):
    return TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=min_df,
                           max_features=100_000, sublinear_tf=True,
                           lowercase=True, token_pattern=r"[a-z#][a-z#']+")


def oof_auc(build, y, groups, texts=None, dense=None, C_reg=1.0, penalty="l2"):
    """Pooled out-of-fold AUC under leave-one-block-out.

    Pooled rather than averaged over folds: blocks differ enormously in size,
    and a mean of per-fold AUCs would let a 200-claim block count as much as a
    1,000-claim one."""
    oof = np.full(len(y), np.nan)
    for tr, te in LeaveOneGroupOut().split(np.zeros(len(y)), y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        Xtr, Xte = build(tr, te, texts, dense)
        clf = LogisticRegression(penalty=penalty, C=C_reg, max_iter=3000,
                                 solver="liblinear" if penalty == "l1" else "lbfgs")
        clf.fit(Xtr, y[tr])
        oof[te] = clf.predict_proba(Xte)[:, 1]
    ok = np.isfinite(oof)
    if len(np.unique(y[ok])) < 2:
        return np.nan, oof
    return roc_auc_score(y[ok], oof[ok]), oof


def make_text_builder(vec_fn, min_df):
    def build(tr, te, texts, dense):
        v = vec_fn(min_df)
        Xtr = v.fit_transform(texts[tr])
        Xte = v.transform(texts[te])
        return Xtr, Xte
    return build


def make_dense_builder():
    def build(tr, te, texts, dense):
        sc = StandardScaler()
        return sc.fit_transform(dense[tr]), sc.transform(dense[te])
    return build


def make_combo_builder(vec_fn, min_df):
    from scipy import sparse

    def build(tr, te, texts, dense):
        v = vec_fn(min_df)
        Ttr, Tte = v.fit_transform(texts[tr]), v.transform(texts[te])
        sc = StandardScaler()
        Dtr, Dte = sc.fit_transform(dense[tr]), sc.transform(dense[te])
        return (sparse.hstack([Ttr, sparse.csr_matrix(Dtr)]).tocsr(),
                sparse.hstack([Tte, sparse.csr_matrix(Dte)]).tocsr())
    return build


def top_phrases(texts, y, min_df, k=25):
    """L1-sparse word model on the full corpus, for interpretation only.

    Fitted in-sample and NEVER used for a performance claim -- the AUCs above
    are the only performance numbers. This exists so the poster can show which
    phrases lean toward hits and which toward misses."""
    v = word_vec(min_df)
    X = v.fit_transform(texts)
    clf = LogisticRegression(penalty="l1", C=0.5, solver="liblinear",
                             max_iter=3000)
    clf.fit(X, y)
    coef = clf.coef_[0]
    names = np.array(v.get_feature_names_out())
    nz = np.where(coef != 0)[0]
    if not len(nz):
        return None
    order = nz[np.argsort(coef[nz])]
    bot = pd.DataFrame({"phrase": names[order[:k]], "coef": coef[order[:k]],
                        "leans": "MISS"})
    top = pd.DataFrame({"phrase": names[order[-k:]][::-1],
                        "coef": coef[order[-k:]][::-1], "leans": "HIT"})
    return pd.concat([top, bot], ignore_index=True), len(nz), X.shape[1]


def run(args):
    C.header("MODEL 4: text models on the forecast's own words",
             "Character n-grams for OCR robustness, word n-grams for a "
             "readable phrase list.\nEvery score is out-of-fold under "
             "leave-one-3-year-block-out.")

    df = C.load_scored(args.scored, args.rigid)
    if "quote" not in df.columns:
        raise SystemExit("This corpus has no `quote` column -- ProQuest rows "
                         "ship without verbatim text, so model 4 cannot run "
                         "on them. Use the LOC monthly corpus.")
    y = df["hit"].astype(int).values
    groups = C.time_blocks(df, args.block_years)

    texts = np.array([clean(q) for q in df["quote"]])
    texts_masked = np.array([clean(q, mask_digits=True) for q in df["quote"]])
    dense = C.standardize(C.drop_collinear(C.claim_design(df))).values

    n_empty = int((np.char.str_len(texts.astype(str)) < 10).sum())
    print(f"\n  claims {len(df):,}   hit rate {y.mean():.3f}   "
          f"blocks {len(set(groups))}")
    print(f"  median quote length {np.median([len(t.split()) for t in texts]):.0f} "
          f"words; {n_empty} quotes under 10 characters")

    rows = []

    def report(name, auc, note=""):
        rows.append({"model": name, "oof_auc": auc, "note": note})
        print(f"  {name:<40} AUC {auc:.3f}   {note}")

    print("\n=== out-of-fold AUC (leave-one-block-out) ===")
    auc_dense, _ = oof_auc(make_dense_builder(), y, groups, texts, dense)
    report("structured features only (the baseline)", auc_dense,
           "what model_hit already had")

    auc_word, _ = oof_auc(make_text_builder(word_vec, args.min_df), y, groups,
                          texts, dense)
    report("word 1-2 grams", auc_word, "readable, but OCR fragments the vocab")

    auc_char, _ = oof_auc(make_text_builder(char_vec, args.min_df), y, groups,
                          texts, dense)
    report("char 3-5 grams", auc_char, "OCR-robust")

    auc_mask, _ = oof_auc(make_text_builder(char_vec, args.min_df), y, groups,
                          texts_masked, dense)
    report("char 3-5 grams, digits masked", auc_mask,
           "guards against learning dates")

    auc_combo, oof_combo = oof_auc(make_combo_builder(char_vec, args.min_df),
                                   y, groups, texts, dense)
    report("char n-grams + structured", auc_combo, "the full model")

    print(f"\n  text adds over structured : {auc_combo - auc_dense:+.3f}")
    print(f"  cost of masking digits    : {auc_mask - auc_char:+.3f}  "
          f"({'signal is in the language' if auc_char - auc_mask < 0.01 else 'SOME signal was dates -- treat the unmasked number as contaminated'})")

    # --- what does this pipeline score on pure noise? -----------------------
    if args.perm:
        print(f"\n  permutation baseline: {args.perm} refits with labels "
              f"shuffled WITHIN block ...")
        rng = np.random.default_rng(0)
        null = []
        for _ in range(args.perm):
            yp = y.copy()
            for g in set(groups):
                i = np.where(groups == g)[0]
                yp[i] = rng.permutation(yp[i])
            a, _ = oof_auc(make_text_builder(char_vec, args.min_df), yp,
                           groups, texts, dense)
            if np.isfinite(a):
                null.append(a)
        if null:
            null = np.array(null)
            p = (1 + (null >= auc_char).sum()) / (1 + len(null))
            print(f"  noise AUC: mean {null.mean():.3f}, 95th pct "
                  f"{np.percentile(null, 95):.3f}   (n={len(null)})")
            print(f"  observed char-ngram AUC {auc_char:.3f}, p = {p:.3f}")
            print("  Compare the observed AUC against the 95th percentile of "
                  "noise, not\n  against 0.5 -- a sparse model on 22 clusters "
                  "beats 0.5 for free.")
            rows.append({"model": "PERMUTATION NULL (char n-grams)",
                         "oof_auc": float(null.mean()),
                         "note": f"95th pct {np.percentile(null, 95):.3f}, "
                                 f"p={p:.3f}"})

    C.save(pd.DataFrame(rows), "m4_text_auc.csv")

    # --- interpretation ------------------------------------------------------
    print("\n=== phrases that lean toward hits and misses (L1, in-sample) ===")
    print("  Interpretation only. These are NOT evidence of predictive power --")
    print("  the out-of-fold AUCs above are the only performance claim here.\n")
    res = top_phrases(texts, y, args.min_df, k=args.top)
    if res is None:
        print("  L1 kept no features at this penalty.")
    else:
        tab, n_kept, n_total = res
        print(f"  L1 kept {n_kept:,} of {n_total:,} word features\n")
        hit = tab[tab["leans"] == "HIT"].reset_index(drop=True)
        miss = tab[tab["leans"] == "MISS"].reset_index(drop=True)
        print(f"  {'leans HIT':<34}{'leans MISS':<34}")
        for i in range(max(len(hit), len(miss))):
            a = (f"{hit.loc[i, 'phrase'][:24]:<26}{hit.loc[i, 'coef']:>+7.2f}"
                 if i < len(hit) else "")
            b = (f"{miss.loc[i, 'phrase'][:24]:<26}{miss.loc[i, 'coef']:>+7.2f}"
                 if i < len(miss) else "")
            print(f"  {a:<34}{b:<34}")
        C.save(tab, "m4_top_phrases.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_common_args(ap)
    ap.add_argument("--min-df", type=int, default=20,
                    help="minimum document frequency for a feature")
    ap.add_argument("--top", type=int, default=20,
                    help="phrases to list per direction")
    ap.add_argument("--perm", type=int, default=0,
                    help="permutation refits for the noise baseline (slow)")
    run(ap.parse_args())
