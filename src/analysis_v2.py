"""
Deep analysis for the v2 poster. Three questions the v1 poster left open:

    1. WAS ANY NEWSPAPER ACTUALLY BETTER?  The v1 poster reported a 44%-56%
       spread across publishers and called it "all near chance" by eyeball.
       Here it is decomposed: how much of that spread is *composition* (which
       years a paper published in, and what direction it tended to predict --
       both of which fix the hit rate before any skill enters) versus residual
       skill, and whether the residual survives a binomial-noise null.

    2. WHICH FACTOR CONTRIBUTES MOST?  Every factor is scored twice: how far it
       separates hit rates *in sample*, and whether that separation *transfers*
       out of fold (leave-one-3-year-block-out). The gap between the two columns
       is the finding -- the largest in-sample effect in the dataset (topic)
       transfers at exactly chance.

    3. IS A NEURAL NET WORTH THE H200s?  Answered by measuring the ceiling
       instead of guessing: character/word n-grams of the raw forecast text fed
       to a linear model is a cheap upper bound on what a text encoder could
       find. If 40k TF-IDF features cannot beat 12 hand-coded ones out of fold,
       a fine-tuned transformer on 14k rows and ~21 independent blocks will not
       either.

No hindsight anywhere: nothing here is fitted with an episode name, an outcome,
or a recession flag as an input. `in_recession` is used only to *split* results
that are already scored, never as a feature.

Usage:
    python src/analysis_v2.py                       # all sections
    python src/analysis_v2.py --section publishers  # one section
    python src/analysis_v2.py --out data/scored/analysis_v2/
"""

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut

warnings.filterwarnings("ignore")

SCORED = "data/scored/monthly_scored.csv"
PRESS_INDEX = "data/scored/press_index.csv"
OUTDIR = "data/scored/analysis_v2"

# Minimum scorable claims for a publisher to be comparable at all. Below ~100
# the binomial standard error is >5 points and every "leader" is noise.
MIN_PUB_N = 100
BLOCK_YEARS = 3  # grouping unit for every out-of-fold number, matches model_hit.py


def in_recession(dates):
    """NBER contraction flag for each print month.

    Only ever used to SPLIT already-scored results, never as a model feature --
    NBER dates are announced 6-21 months late, so a contemporary forecaster
    could not have known them."""
    from truth_data import recession_months
    rec = set(recession_months())
    return pd.to_datetime(dates).dt.to_period("M").isin(rec).values


def load_scored(path=SCORED):
    d = pd.read_csv(path, low_memory=False)
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d["year"] = d["date"].dt.year
    d["decade"] = (d["year"] // 10) * 10
    d["block"] = (d["year"] // BLOCK_YEARS) * BLOCK_YEARS
    s = d[(d["scorable"] == True) & (d["hit"].isin([0, 1]))].copy()
    s["hit"] = s["hit"].astype(int)
    return d, s.reset_index(drop=True)


# ---------------------------------------------------------------- 1. publishers

def publishers(s, rng_seed=0, n_perm=400):
    """Split each paper's hit rate into what its calendar+mix guaranteed and
    what is left over.

    The expected rate is built from corpus-wide (year x predicted direction)
    cell means. That is the rate a paper would have scored with zero skill,
    given only *when* it published and *what direction* it called -- so the
    residual is the only place skill could live."""
    counts = s.groupby("publisher")["hit"].size()
    keep = counts[counts >= MIN_PUB_N].index
    sub = s[s["publisher"].isin(keep)].copy()

    cell = s.groupby(["year", "predicted_norm"])["hit"].mean().rename("exp")
    sub = sub.merge(cell, on=["year", "predicted_norm"], how="left")

    r = sub.groupby("publisher").agg(
        n=("hit", "size"), actual=("hit", "mean"), expected=("exp", "mean"),
        p_improve=("predicted_norm", lambda x: (x == "improve").mean()),
        y0=("year", "min"), y1=("year", "max"))
    r["skill"] = r["actual"] - r["expected"]
    r["se"] = np.sqrt(r["expected"] * (1 - r["expected"]) / r["n"])
    r["z"] = r["skill"] / r["se"]
    r = r.sort_values("actual")

    var_a = r["actual"].var(ddof=0)
    var_e = r["expected"].var(ddof=0)
    var_s = r["skill"].var(ddof=0)
    noise = (r["se"] ** 2).mean()
    chi2 = float((r["z"] ** 2).sum())
    p_chi2 = float(1 - stats.chi2.cdf(chi2, len(r)))

    # Permutation null: shuffle outcomes within (year x direction) so calendar
    # and mix are held exactly fixed and only the paper label moves.
    rng = np.random.default_rng(rng_seed)
    sub["_k"] = sub["year"].astype(str) + "|" + sub["predicted_norm"].astype(str)
    obs_sd = float(sub.groupby("publisher")["hit"].mean().std())
    null_sd = []
    for _ in range(n_perm):
        p = sub.copy()
        p["hit"] = p.groupby("_k")["hit"].transform(
            lambda x: rng.permutation(x.values))
        null_sd.append(float(p.groupby("publisher")["hit"].mean().std()))
    null_sd = np.array(null_sd)

    out = {
        "n_publishers": int(len(r)),
        "n_claims": int(len(sub)),
        "spread_lo": float(r["actual"].min()), "spread_hi": float(r["actual"].max()),
        "var_actual": float(var_a),
        "var_composition": float(var_e),
        "share_composition": float(var_e / var_a),
        "var_residual": float(var_s),
        "var_binomial_noise": float(noise),
        "share_residual_that_is_noise": float(noise / var_s),
        "chi2": chi2, "chi2_df": int(len(r)), "chi2_p": p_chi2,
        "perm_obs_sd": obs_sd,
        "perm_null_sd_mean": float(null_sd.mean()),
        "perm_p": float((null_sd >= obs_sd).mean()),
    }
    return r, out


# ------------------------------------------------------- 2. what drives a hit

def _oof_auc_single(s, col, groups, categorical=True):
    """Out-of-fold AUC using one factor alone, leave-one-block-out.

    Categorical factors are encoded by their TRAINING-fold hit rate (target
    encoding fitted inside the fold), which is exactly the "does this factor's
    ranking transfer to an unseen era" question."""
    y = s["hit"].values
    oof = np.full(len(y), np.nan)
    for tr, te in LeaveOneGroupOut().split(s, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        if categorical:
            m = s.iloc[tr].groupby(col)["hit"].mean()
            prior = y[tr].mean()
            oof[te] = s.iloc[te][col].map(m).fillna(prior).values
        else:
            x = pd.to_numeric(s[col], errors="coerce").fillna(0.0).values
            lr = LogisticRegression(max_iter=1000)
            lr.fit(x[tr].reshape(-1, 1), y[tr])
            oof[te] = lr.predict_proba(x[te].reshape(-1, 1))[:, 1]
    ok = ~np.isnan(oof)
    if len(np.unique(y[ok])) < 2 or np.nanstd(oof[ok]) == 0:
        return np.nan
    return float(roc_auc_score(y[ok], oof[ok]))


def drivers(s):
    """Rank every candidate factor by in-sample separation AND by transfer."""
    groups = s["block"].values
    factors = [
        ("what was predicted", "predicted_norm"),
        ("subject of the forecast", "topic"),
        ("era (decade printed)", "decade"),
        ("who is speaking", "voice"),
        ("hedged vs assertive", "confidence"),
        ("named forecaster", "_named"),
        ("horizon", "_horizon_b"),
        ("newspaper", "_pub"),
        ("quotes a forecaster", "is_quoted_forecaster"),
        ("conditional ('if...')", "_conditional"),
    ]
    s = s.copy()
    s["_named"] = np.where(
        s["speaker_name"].astype(str).str.lower().isin(["na", "nan", ""]),
        "anonymous", "named")
    s["_conditional"] = np.where(
        s["conditional_on"].astype(str).str.lower().isin(["na", "nan", ""]),
        "unconditional", "conditional")
    h = pd.to_numeric(s["horizon_used"], errors="coerce")
    s["_horizon_b"] = pd.cut(h, [-1, 3, 6, 12, 24, 999],
                             labels=["0-3mo", "4-6mo", "7-12mo", "13-24mo", ">24mo"]
                             ).astype(str)
    counts = s.groupby("publisher")["hit"].size()
    big = counts[counts >= MIN_PUB_N].index
    s["_pub"] = np.where(s["publisher"].isin(big), s["publisher"], "__other__")

    rows = []
    for label, col in factors:
        g = s.groupby(col)["hit"].agg(["size", "mean"])
        g = g[g["size"] >= 50]
        if len(g) < 2:
            continue
        w = g["size"] / g["size"].sum()
        sd = float(np.sqrt((w * (g["mean"] - (w * g["mean"]).sum()) ** 2).sum()))
        # In-sample AUC: encode each level by its hit rate over the WHOLE corpus
        # and score on that same corpus. This is the number a random-split
        # workflow would report, and the gap to oof_auc is the finding.
        m_all = s.groupby(col)["hit"].mean()
        ins = s[col].map(m_all).values
        insample = (float(roc_auc_score(s["hit"].values, ins))
                    if np.std(ins) > 0 else 0.5)
        rows.append({
            "factor": label, "column": col, "levels": int(len(g)),
            "spread_pts": float(100 * (g["mean"].max() - g["mean"].min())),
            "weighted_sd_pts": float(100 * sd),
            "insample_auc": insample,
            "oof_auc": _oof_auc_single(s, col, groups, categorical=True),
        })
    out = pd.DataFrame(rows).sort_values("spread_pts", ascending=False)
    return out, s


# ---------------------------------------------- 3. does the raw text help?

def text_ceiling(s, max_features=60000):
    """Upper bound on what any text model could extract, at ~0 compute cost.

    Word 1-2 grams over the forecast quote, L2 logistic regression, the same
    leave-one-block-out CV as everything else. This is generous: it sees every
    lexical cue a transformer would see, with no risk of underfitting from a
    short training run. Whatever this scores is roughly the ceiling for a
    fine-tuned encoder on the same 14k rows / ~21 independent blocks."""
    y = s["hit"].values
    groups = s["block"].values
    txt = s["quote"].astype(str).fillna("").values

    results = {}
    for name, kw in [
        ("text_word12", dict(ngram_range=(1, 2), min_df=3, sublinear_tf=True)),
        ("text_char35", dict(analyzer="char_wb", ngram_range=(3, 5), min_df=5,
                             sublinear_tf=True)),
    ]:
        oof = np.full(len(y), np.nan)
        for tr, te in LeaveOneGroupOut().split(txt, y, groups):
            if len(np.unique(y[tr])) < 2:
                continue
            vec = TfidfVectorizer(max_features=max_features, **kw)
            Xtr = vec.fit_transform(txt[tr])
            Xte = vec.transform(txt[te])
            lr = LogisticRegression(C=1.0, max_iter=3000)
            lr.fit(Xtr, y[tr])
            oof[te] = lr.predict_proba(Xte)[:, 1]
        ok = ~np.isnan(oof)
        results[name] = float(roc_auc_score(y[ok], oof[ok]))

        # In-sample (random-split) AUC, to show how much of the apparent signal
        # is era memorisation rather than anything generalisable.
        vec = TfidfVectorizer(max_features=max_features, **kw)
        X = vec.fit_transform(txt)
        lr = LogisticRegression(C=1.0, max_iter=3000).fit(X, y)
        results[name + "_insample"] = float(roc_auc_score(y, lr.predict_proba(X)[:, 1]))
        results[name + "_vocab"] = int(X.shape[1])
    return results


# --------------------------------------- 4. the disagreement definition fight

def disagreement(s, index_path=PRESS_INDEX):
    """Two disagreement measures live in this repo and they disagree. Settle it.

    A: model_hit.py  -- 1-|2p-1| where p = improve / ALL claims that month.
       Months with many non-directional claims look artificially "agreed".
    B: build_press_index.py -- p = improve / DIRECTIONAL claims only.

    B is the coherent one: disagreement about direction should be measured among
    claims that state a direction. A is contaminated by how many unscorable or
    flat claims happened to appear that month."""
    d = s.copy()
    d["_m"] = d["date"].dt.to_period("M")
    d["_imp"] = (d["predicted_norm"] == "improve").astype(int)
    d["_wor"] = (d["predicted_norm"] == "worsen").astype(int)
    g = d.groupby("_m").agg(n_all=("_imp", "size"), imp=("_imp", "sum"),
                            wor=("_wor", "sum"))
    g["n_dir"] = g["imp"] + g["wor"]
    g["disag_A"] = 1 - (2 * g["imp"] / g["n_all"] - 1).abs()
    g["disag_B"] = np.where(g["n_dir"] > 0,
                            1 - (2 * g["imp"] / g["n_dir"].replace(0, np.nan) - 1).abs(),
                            np.nan)
    d = d.merge(g[["disag_A", "disag_B", "n_dir"]], left_on="_m", right_index=True)

    d["_rec"] = in_recession(d["date"])
    out = {}
    for m in ["disag_A", "disag_B"]:
        for tag, v in [("", d[d["n_dir"] >= 5]),
                       ("_nofilter", d),
                       ("_expansions_only", d[(d["n_dir"] >= 5) & (~d["_rec"])]),
                       ("_improve_only", d[(d["n_dir"] >= 5) &
                                           (d["predicted_norm"] == "improve")])]:
            try:
                b = pd.qcut(v[m], 3, labels=["low", "mid", "high"],
                            duplicates="drop")
            except ValueError:
                continue
            tab = v.groupby(b)["hit"].agg(["size", "mean"])
            out[m + tag] = {str(k): {"n": int(r["size"]), "hit": float(r["mean"])}
                            for k, r in tab.iterrows()}
        v = d[d["n_dir"] >= 5]
        out[m + "_corr_with_hit"] = float(v[m].corr(v["hit"]))
        # Is this just a recession detector in disguise? If so the "uncertainty"
        # reading is spurious -- it would only be re-measuring the business cycle.
        out[m + "_corr_with_recession"] = float(
            v[m].corr(v["_rec"].astype(float)))
    return out, g


# --------------------------------------------------------------- 5. mechanism

def mechanism(s, d_all):
    """The core Panel-2 numbers, recomputed, plus the pieces the v2 poster needs."""
    out = {}
    s = s.copy()
    s["_rec"] = in_recession(s["date"])

    out["share_worsen_expansion"] = float(
        (s.loc[~s["_rec"], "predicted_norm"] == "worsen").mean())
    out["share_worsen_recession"] = float(
        (s.loc[s["_rec"], "predicted_norm"] == "worsen").mean())
    out["hit_expansion"] = float(s.loc[~s["_rec"], "hit"].mean())
    out["hit_recession"] = float(s.loc[s["_rec"], "hit"].mean())
    out["n_expansion"] = int((~s["_rec"]).sum())
    out["n_recession"] = int(s["_rec"].sum())
    out["overall_hit"] = float(s["hit"].mean())
    out["n_scorable"] = int(len(s))
    out["n_extracted"] = int(len(d_all))

    # Who was right when it counted. The average accuracy gap is a consequence
    # of the mix; this split shows the mix was the whole mechanism.
    for dname in ["improve", "worsen"]:
        m = s["predicted_norm"] == dname
        out[f"{dname}_hit_expansion"] = float(s.loc[m & ~s["_rec"], "hit"].mean())
        out[f"{dname}_hit_recession"] = float(s.loc[m & s["_rec"], "hit"].mean())
        out[f"{dname}_n"] = int(m.sum())

    # Block bootstrap on the accuracy gap: resample 3-year blocks, not claims,
    # because claims inside a block share wire copy and one macro reality.
    rng = np.random.default_rng(0)
    blocks = s["block"].unique()
    gaps = []
    for _ in range(2000):
        pick = rng.choice(blocks, size=len(blocks), replace=True)
        b = pd.concat([s[s["block"] == p] for p in pick])
        e, r = b.loc[~b["_rec"], "hit"], b.loc[b["_rec"], "hit"]
        if len(e) and len(r):
            gaps.append(e.mean() - r.mean())
    out["gap_pts"] = float(100 * (out["hit_expansion"] - out["hit_recession"]))
    out["gap_ci95"] = [float(100 * np.percentile(gaps, 2.5)),
                       float(100 * np.percentile(gaps, 97.5))]
    return out, s


# ------------------------------------------------- 6. did the press write more?

def attention(index_path=PRESS_INDEX):
    """Volume, not just direction: did economic coverage RISE in downturns?

    The naive metric -- claims per month -- says yes, emphatically. It is an
    artefact: LOC digitisation density varies by era, so months with more
    recession also happen to be months with more PAGES sampled. Normalising by
    pages sampled is the only honest version, and it kills the effect."""
    p = pd.read_csv(index_path)
    p["m"] = pd.PeriodIndex(p["month"], freq="M")
    p["rec"] = p["m"].isin(set(_recession_set()))
    p["year"] = p["m"].dt.year
    p["block"] = (p["year"] // BLOCK_YEARS) * BLOCK_YEARS

    out = {"n_months": int(len(p)), "n_recession_months": int(p["rec"].sum())}
    for col in ["attention", "n_all_claims", "n_pages"]:
        e, r = p.loc[~p["rec"], col].dropna(), p.loc[p["rec"], col].dropna()
        out[col] = {
            "expansion": float(e.mean()), "recession": float(r.mean()),
            "p": float(stats.ttest_ind(e, r, equal_var=False).pvalue)}

    # Months inside an era are not independent -- resample blocks, not months.
    rng = np.random.default_rng(0)
    blocks = p["block"].unique()
    diffs = []
    for _ in range(4000):
        b = pd.concat([p[p["block"] == q]
                       for q in rng.choice(blocks, len(blocks), replace=True)])
        e = b.loc[~b["rec"], "attention"].dropna()
        r = b.loc[b["rec"], "attention"].dropna()
        if len(e) and len(r):
            diffs.append(r.mean() - e.mean())
    diffs = np.array(diffs)
    out["attention_block_ci95"] = [float(np.percentile(diffs, 2.5)),
                                   float(np.percentile(diffs, 97.5))]
    p["att_dm"] = p["attention"] - p.groupby("block")["attention"].transform("mean")
    e, r = p.loc[~p["rec"], "att_dm"].dropna(), p.loc[p["rec"], "att_dm"].dropna()
    out["attention_within_era_p"] = float(
        stats.ttest_ind(e, r, equal_var=False).pvalue)
    return out, p


def _recession_set():
    from truth_data import recession_months
    return recession_months()


# --------------------------------------- 7. price forecasting: regime, not skill

def price_regime(s):
    """Was being right about prices skill, or just matching the era's regime?

    For each decade and each direction, compare the hit rate of forecasts
    predicting that direction against how often that direction ACTUALLY
    occurred. If a forecast carries information, its hit rate should beat the
    outcome's base rate. If the two are equal, the "accuracy" is the regime.

    Corrects two v1 claims: the up/down hit rates do NOT sum to 1 (prices are
    scored three ways, and 'flat' absorbs the rest -- 50% of 1950s outcomes),
    and 'prices will fall' after 1948 was right 9 of 139 times, not 0 of 93."""
    p = s[s["topic"] == "prices"].copy()
    rows = []
    for dec, g in p.groupby("decade"):
        base = g["realized"].value_counts(normalize=True)
        for d in ["up", "down"]:
            sub = g[g["predicted_norm"] == d]
            if len(sub) < 15:
                continue
            rows.append({"decade": int(dec), "direction": d, "n": len(sub),
                         "hit_rate": float(sub["hit"].mean()),
                         "base_rate": float(base.get(d, 0.0))})
    t = pd.DataFrame(rows)
    late = p[(p["predicted_norm"] == "down") & (p["year"] >= 1948)]
    out = {
        "corr_hit_vs_base": float(t["hit_rate"].corr(t["base_rate"])),
        "mean_abs_gap": float((t["hit_rate"] - t["base_rate"]).abs().mean()),
        "down_after_1948_n": int(len(late)),
        "down_after_1948_hits": int(late["hit"].sum()),
        "down_after_1948_rate": float(late["hit"].mean()),
    }
    return t, out


# ------------------------------------------------------------------- runner

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", default=SCORED)
    ap.add_argument("--out", default=OUTDIR)
    ap.add_argument("--section", default="all")
    ap.add_argument("--perm", type=int, default=400)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    d_all, s = load_scored(a.scored)
    print(f"extracted {len(d_all):,}  scorable {len(s):,}  hit {s['hit'].mean():.4f}")
    summary = {}

    if a.section in ("all", "publishers"):
        r, o = publishers(s, n_perm=a.perm)
        r.to_csv(f"{a.out}/publisher_skill.csv")
        summary["publishers"] = o
        print("\n=== 1. WAS ANY NEWSPAPER BETTER? ===")
        print(r[["n", "actual", "expected", "skill", "z", "y0", "y1"]].round(3).to_string())
        print(f"\nspread {o['spread_lo']:.3f}-{o['spread_hi']:.3f} across "
              f"{o['n_publishers']} papers")
        print(f"composition explains {o['share_composition']:.0%} of the variance; "
              f"{o['share_residual_that_is_noise']:.0%} of the remainder is "
              f"binomial noise")
        print(f"chi2 for ANY publisher skill = {o['chi2']:.1f} on {o['chi2_df']} df, "
              f"p = {o['chi2_p']:.3f}   permutation p = {o['perm_p']:.3f}")

    if a.section in ("all", "drivers"):
        t, s2 = drivers(s)
        t.to_csv(f"{a.out}/driver_ranking.csv", index=False)
        summary["drivers"] = t.to_dict("records")
        print("\n=== 2. WHAT DRIVES A HIT? in-sample separation vs transfer ===")
        print(t.round(3).to_string(index=False))

    if a.section in ("all", "text"):
        print("\n=== 3. DOES THE RAW TEXT CARRY SIGNAL? (neural-net ceiling) ===")
        tc = text_ceiling(s)
        summary["text_ceiling"] = tc
        for k, v in tc.items():
            print(f"  {k}: {v}")

    if a.section in ("all", "disagreement"):
        o, g = disagreement(s)
        g.to_csv(f"{a.out}/monthly_disagreement.csv")
        summary["disagreement"] = o
        print("\n=== 4. DISAGREEMENT: WHICH DEFINITION? ===")
        print(json.dumps(o, indent=2))

    if a.section in ("all", "attention"):
        o, pm = attention()
        pm.to_csv(f"{a.out}/attention_months.csv", index=False)
        summary["attention"] = o
        print("\n=== 6. DID THE PRESS WRITE MORE IN DOWNTURNS? ===")
        print(json.dumps(o, indent=2))

    if a.section in ("all", "prices"):
        t, o = price_regime(s)
        t.to_csv(f"{a.out}/price_regime.csv", index=False)
        summary["price_regime"] = o
        print("\n=== 7. PRICES: REGIME OR SKILL? ===")
        print(t.round(3).to_string(index=False))
        print(json.dumps(o, indent=2))

    if a.section in ("all", "mechanism"):
        o, _ = mechanism(s, d_all)
        summary["mechanism"] = o
        print("\n=== 5. MECHANISM (recomputed) ===")
        print(json.dumps(o, indent=2))

    # Merge rather than overwrite, so running one --section at a time does not
    # discard the other sections' results.
    path = f"{a.out}/summary.json"
    if os.path.exists(path):
        with open(path) as f:
            prev = json.load(f)
        prev.update(summary)
        summary = prev
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nwrote {a.out}/summary.json")


if __name__ == "__main__":
    main()
