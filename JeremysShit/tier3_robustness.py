"""
Tier 3 — statistical rigor (items 8 and 9 of the project plan).

8.  Diebold-Mariano test for the newspapers-vs-Livingston head-to-head,
    1946-1963. Both sides make directional calls; loss = 0/1 miss, pooled by
    half-year (Livingston surveys are semiannual). DM statistic uses a
    Newey-West HAC variance (lag h-1, h = 2 half-years for a 12-month
    horizon) with the Harvey-Leybourne-Newbold small-sample correction.

9.  Robustness suite:
    9a. Re-score everything with the no-change bands halved and doubled
        (newspaper side re-scored against cached FRED series; Livingston side
        via score_claims.livingston_directional(band_scale=...)).
    9b. Grade a 50-claim sample with a second LLM (Anthropic, a different
        vendor from the DeepSeek primary) and report Cohen's kappa agreement.
        Needs credentials; run AFTER grade_claims.py has produced real grades
        (agreement against the --heuristic keyword grades is meaningless).
    9c. Shift the Livingston era boundaries +/-3 years and check that the
        era ranking of forecast error survives.

Usage:
    python tier3_robustness.py                 # 8, 9a, 9c (all offline)
    python tier3_robustness.py --second-llm    # 9b (needs Anthropic access)
    python tier3_robustness.py --second-llm --limit 10   # cheap test first

Outputs: printed report; 9b also writes second_llm_grades.csv.
"""

import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

import score_claims
from score_claims import fred, livingston_directional, predicted_label, realized_direction

H2H_START, H2H_END = "1946-01-01", "1964-01-01"

# Era definitions copied from the Livingston notebook (section 3), so the
# script and notebook shift the same boundaries.
ERAS = {"Postwar boom": (1946, 1965), "Vietnam / stagflation": (1965, 1982),
        "Great Moderation": (1982, 2000), "Terror / fin. crisis": (2000, 2012),
        "Polarization / COVID": (2012, 2027)}


def half_year(dates):
    d = pd.to_datetime(dates)
    return d.dt.year.astype(str) + np.where(d.dt.month <= 6, "H1", "H2")


# ---------------------------------------------------------------- item 8: DM

def diebold_mariano(d, horizon=2):
    """DM test on a loss-differential series d (one value per period).
    Newey-West variance with lag horizon-1; HLN small-sample correction;
    two-sided p from t(n-1)."""
    d = np.asarray(d, dtype=float)
    n, h = len(d), horizon
    dbar = d.mean()
    e = d - dbar
    gamma = [np.mean(e * e)]
    for k in range(1, h):
        gamma.append(np.mean(e[k:] * e[:-k]))
    var = (gamma[0] + 2 * sum((1 - k / h) * gamma[k] for k in range(1, h))) / n
    if var <= 0:
        return float("nan"), float("nan"), n
    dm = dbar / np.sqrt(var)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)  # Harvey et al. 1997
    dm *= hln
    p = 2 * stats.t.sf(abs(dm), df=n - 1)
    return dm, p, n


def head_to_head_dm(scored):
    print("=" * 70)
    print("ITEM 8 — Diebold-Mariano: newspapers vs. Livingston, 1946-63")
    print("=" * 70)
    news = scored[(scored["date"] >= H2H_START) & (scored["date"] < H2H_END)].copy()
    if news.empty:
        print("No scored 1946-63 newspaper claims — scale up episodes 6-7 first.")
        return
    news["bucket"] = half_year(news["date"])
    news_loss = news.groupby("bucket").agg(n=("hit", "size"),
                                           loss=("hit", lambda x: 1 - x.mean()))

    liv = livingston_directional()
    liv = liv.assign(miss=(liv["pred"] != liv["act"]).astype(int),
                     bucket=half_year(liv["date"]))
    liv_loss = liv.groupby("bucket").agg(n=("miss", "size"), loss=("miss", "mean"))

    both = news_loss.join(liv_loss, lsuffix="_news", rsuffix="_liv", how="inner")
    print(f"matched half-year buckets: {len(both)} "
          f"(newspaper claims {int(both.n_news.sum())}, "
          f"Livingston forecasts {int(both.n_liv.sum())})")
    print(f"mean loss  newspapers {both.loss_news.mean():.3f}   "
          f"Livingston {both.loss_liv.mean():.3f}")
    if len(both) < 6:
        print("Fewer than 6 matched buckets — DM is not meaningful yet; "
              "report the raw losses only.")
        return
    dm, p, n = diebold_mariano(both.loss_news - both.loss_liv)
    print(f"DM statistic {dm:+.3f}   two-sided p = {p:.3f}   (n = {n} buckets, "
          f"NW lag 1, HLN-corrected)")
    verdict = ("newspapers significantly WORSE than Livingston" if dm > 0
               else "newspapers significantly BETTER than Livingston") \
        if p < 0.05 else "no significant difference — the honest headline"
    print(f"-> {verdict}")


# ------------------------------------------------------- item 9a: band scales

def rescore_with_bands(scored, scale, cpi, indpro, unrate):
    """Recompute realized labels and hits with the no-change bands scaled."""
    original = dict(score_claims.BANDS)
    score_claims.BANDS = {k: v * scale for k, v in original.items()}
    try:
        hits = []
        for _, r in scored.iterrows():
            realized, scorable, _ = realized_direction(
                r["topic"], r.get("price_direction", ""),
                r.get("unemployment_direction", ""), r["date"], int(r["months"]),
                cpi, indpro, unrate)
            pred = predicted_label(r)
            hits.append(int(pred == realized) if (scorable and pred) else np.nan)
    finally:
        score_claims.BANDS = original
    return pd.Series(hits, index=scored.index)


def band_sensitivity(scored):
    print("\n" + "=" * 70)
    print("ITEM 9a — no-change bands halved / headline / doubled")
    print("=" * 70)
    cpi, indpro, unrate = fred("CPIAUCNS"), fred("INDPRO"), fred("UNRATE")
    rows = []
    for scale in (0.5, 1.0, 2.0):
        hit = rescore_with_bands(scored, scale, cpi, indpro, unrate)
        ok = hit.dropna()
        by_kind = hit.groupby(scored["kind"]).mean() if "kind" in scored else {}
        try:
            liv = livingston_directional(band_scale=scale)
            liv_rate = (liv["pred"] == liv["act"]).mean()
        except Exception:
            liv_rate = float("nan")
        rows.append({"band_scale": scale, "n_scorable": len(ok),
                     "newspaper_hit_rate": ok.mean(),
                     "crisis_hit_rate": by_kind.get("crisis", float("nan")),
                     "control_hit_rate": by_kind.get("control", float("nan")),
                     "livingston_hit_rate_1946_63": liv_rate})
    tab = pd.DataFrame(rows).set_index("band_scale").round(3)
    print(tab.to_string())
    spread = tab["newspaper_hit_rate"].max() - tab["newspaper_hit_rate"].min()
    print(f"\nnewspaper hit-rate spread across band choices: {spread:.3f} "
          f"({'robust — coding choice does not drive the result' if spread < 0.05 else 'SENSITIVE — report all three on the poster'})")
    print("(NBER-based claims are unaffected by bands by construction; only "
          "CPI/INDPRO/UNRATE-scored claims move.)")


# ------------------------------------------------------ item 9c: era shifts

def livingston_errors():
    """Continuous 12-month |forecast - actual| per survey, all variables,
    same construction (and rebase-artifact filter) as the notebook."""
    xl = pd.ExcelFile("medians.xlsx")
    bounds = {"CPI": (-8, 99), "IP": (-18, 40)}
    frames = []
    for v in ("CPI", "IP", "UNPR"):
        d = xl.parse(v).sort_values("Date").reset_index(drop=True)
        bp, f12 = d[f"{v}_BP"], d[f"{v}_12M"]
        nxt = bp.shift(-2)
        if v == "UNPR":
            pred_chg, act_chg = f12 - bp, nxt - bp
        else:
            pred_chg = (f12 - bp) / bp * 100
            act_chg = (nxt / bp - 1) * 100
            g6 = (bp.shift(-1) / bp - 1) * 100
            lo, hi = bounds[v]
            bad = ((g6 < lo) | (g6 > hi)).fillna(False)
            act_chg = act_chg.mask(bad | bad.shift(-1).fillna(False))
        frames.append(pd.DataFrame({
            "year": pd.to_datetime(d["Date"]).dt.year, "variable": v,
            "abs_error": (pred_chg - act_chg).abs()}).dropna())
    return pd.concat(frames, ignore_index=True)


def era_shift_robustness():
    print("\n" + "=" * 70)
    print("ITEM 9c — Livingston era boundaries shifted +/-3 years")
    print("=" * 70)
    errors = livingston_errors()
    names = list(ERAS)
    results = {}
    for shift in range(-3, 4):
        # first boundary (data start) stays put; interior boundaries move
        edges = [ERAS[n][0] for n in names] + [ERAS[names[-1]][1]]
        edges = [edges[0]] + [e + shift for e in edges[1:-1]] + [edges[-1]]
        era = pd.cut(errors["year"], bins=edges, labels=names, right=False)
        mae = errors.groupby(era, observed=True)["abs_error"].mean()
        results[shift] = mae
    tab = pd.DataFrame(results).round(2)
    tab.columns = [f"{s:+d}y" for s in results]
    print("mean |12-month error| by era, at each boundary shift:")
    print(tab.to_string())
    rankings = {s: tuple(m.sort_values().index) for s, m in results.items()}
    stable = len(set(rankings.values())) == 1
    print(f"\nera ranking (best->worst) identical across all 7 shifts: "
          f"{'YES — era conclusions are not an artifact of boundary choice' if stable else 'NO — flag the unstable eras on the poster'}")
    if not stable:
        for s, r in rankings.items():
            print(f"  {s:+d}y: {' < '.join(r)}")


# --------------------------------------------------- item 9b: second LLM

def second_llm_agreement(limit):
    """Grade a sample with Claude (Anthropic) and report kappa vs. the
    primary (DeepSeek) grades — shows results don't depend on one vendor."""
    from grade_claims import GRADE_FIELDS, RUBRIC_PROMPT, cohens_kappa

    print("=" * 70)
    print(f"ITEM 9b — second-LLM agreement on a {limit}-claim sample")
    print("=" * 70)
    df = pd.read_csv("claims_scored.csv")
    if df["confidence"].nunique() <= 1 and df["voice"].nunique() <= 1:
        print("WARNING: current grades look like --heuristic keyword output "
              "(one confidence/voice value everywhere). Agreement against a "
              "heuristic is meaningless — run grade_claims.py (DeepSeek) "
              "first, re-score, then rerun this.")
    sample = df.sample(min(limit, len(df)), random_state=42)

    import anthropic
    client = anthropic.Anthropic()  # ANTHROPIC_API_KEY or `ant auth login` profile
    schema = {
        "type": "object", "additionalProperties": False,
        "required": GRADE_FIELDS,
        "properties": {
            "is_prediction": {"type": "string", "enum": ["yes", "no"]},
            "topic": {"type": "string", "enum": ["general_business", "prices",
                                                 "employment", "markets", "other"]},
            "direction": {"type": "string", "enum": ["improve", "worsen",
                                                     "no_change", "unclear"]},
            "price_direction": {"type": "string"},
            "unemployment_direction": {"type": "string"},
            "horizon_months": {"type": "string"},
            "confidence": {"type": "string", "enum": ["assertive", "hedged"]},
            "voice": {"type": "string", "enum": ["journalist", "expert",
                                                 "official", "layperson", "unclear"]},
            "speaker_name": {"type": "string"},
        },
    }

    graded = []
    for i, (_, r) in enumerate(sample.iterrows(), 1):
        prompt = RUBRIC_PROMPT.format(date=r["date"], episode=r["episode"],
                                      quote=r["quote"])
        try:
            resp = client.messages.create(
                model="claude-opus-4-8", max_tokens=1024,
                output_config={"format": {"type": "json_schema", "schema": schema}},
                messages=[{"role": "user", "content": prompt}])
            if resp.stop_reason == "refusal":
                raise RuntimeError("model refused")
            g = json.loads(next(b.text for b in resp.content if b.type == "text"))
        except (anthropic.AuthenticationError, TypeError):
            # missing credentials raise TypeError at request time
            raise SystemExit(
                "No Anthropic credentials. Set ANTHROPIC_API_KEY (or run "
                "`ant auth login`) and rerun with --second-llm.")
        except Exception as e:
            print(f"  claim {r['claim_id']}: FAILED ({e}) — skipped")
            g = {}
        graded.append({"claim_id": r["claim_id"],
                       **{f"claude_{k}": str(g.get(k, "")) for k in GRADE_FIELDS}})
        if i % 10 == 0:
            print(f"  {i}/{len(sample)} graded")

    out = sample.merge(pd.DataFrame(graded), on="claim_id")
    out.to_csv("second_llm_grades.csv", index=False)
    print(f"\nWrote second_llm_grades.csv ({len(out)} rows)")

    print("\nDeepSeek-vs-Claude agreement (Cohen's kappa):")
    for k in ("is_prediction", "topic", "direction", "confidence"):
        pairs = [(str(a).strip().lower(), str(b).strip().lower())
                 for a, b in zip(out[k], out[f"claude_{k}"]) if str(b).strip()]
        if pairs:
            agree = sum(1 for a, b in pairs if a == b) / len(pairs)
            print(f"  {k:15s} kappa = {cohens_kappa(pairs):+.2f}   "
                  f"raw agreement {agree:.0%}   (n={len(pairs)})")
    print("Target: kappa >= 0.7 on direction. At or above it, the results "
          "don't depend on DeepSeek specifically.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--second-llm", action="store_true",
                    help="run item 9b (needs Anthropic credentials)")
    ap.add_argument("--limit", type=int, default=50,
                    help="sample size for --second-llm (default 50)")
    args = ap.parse_args()

    if args.second_llm:
        second_llm_agreement(args.limit)
        return

    scored = pd.read_csv("claims_scored.csv", parse_dates=["date"]).dropna(subset=["hit"])
    scored["hit"] = scored["hit"].astype(int)
    head_to_head_dm(scored)
    band_sensitivity(scored)
    era_shift_robustness()


if __name__ == "__main__":
    main()
