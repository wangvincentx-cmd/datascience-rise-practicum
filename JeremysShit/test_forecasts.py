"""
Offline tests for the forecast-credibility pipeline (forecast_data.py,
forecast_credibility.py, greenbook_benchmark.py). No network, no large data files:
pure logic on tiny synthetic frames, with FRED monkeypatched. Mirrors the
test_offline.py convention -- run after any change to these modules.

    python test_forecasts.py
"""

import numpy as np
import pandas as pd

import forecast_data as fd
import forecast_credibility as fc
from spf_benchmark import direction_label
from greenbook_benchmark import _target_qindex, _edition_date

PASS = FAIL = 0


def check(cond, msg):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL: {msg}")


# 1. Direction banding (shared across all sources)
check(direction_label(2.0, 1.0) == "improve", "growth 2 > band -> improve")
check(direction_label(-2.0, 1.0) == "worsen", "growth -2 < -band -> worsen")
check(direction_label(0.5, 1.0) == "no_change", "|growth| < band -> no_change")
check(direction_label(float("nan"), 1.0) == "", "NaN growth -> empty")

# 2. Greenbook column-format parsing
check(_target_qindex("1966.1") == 1966 * 4 + 0, "1966.1 -> Q1 index")
check(_target_qindex("1966.4") == 1966 * 4 + 3, "1966.4 -> Q4 index")
check(_target_qindex("bad") is None, "non-quarter label -> None")
ed = _edition_date("gRGDP_19670329")
check(ed.year == 1967 and ed.month == 3 and ed.day == 29, "edition date parsed")

# 3. SPF ~1yr growth formula: 100*(level_+4q / level_nowcast - 1)
check(abs(100 * (110.0 / 100.0 - 1) - 10.0) < 1e-9, "SPF growth 100->110 = +10%")

# 4. Forecast revision = change from a forecaster's own previous call
df = pd.DataFrame({
    "forecaster_id": ["a", "a", "b"],
    "forecast_date": pd.to_datetime(["2000-01-01", "2000-04-01", "2000-01-01"]),
    "pred_growth": [2.0, 3.0, 5.0]})
rev = fc.add_pred_features(df).sort_values(["forecaster_id", "forecast_date"])["pred_revision"].tolist()
check(rev == [0.0, 1.0, 0.0], f"revision per forecaster (got {rev})")

# 5. Temporal split has NO leakage: every train date <= every test date
sub = pd.DataFrame({"forecast_date": pd.to_datetime([f"200{y}-01-01" for y in range(6)]),
                    "hit": [1, 0, 1, 0, 1, 0]})
s2, tr, te = fc._split(sub)
check(s2["forecast_date"].iloc[tr].max() <= s2["forecast_date"].iloc[te].min(),
      "temporal split: max(train date) <= min(test date)")
check(len(tr) + len(te) == len(sub), "split covers all rows")

# 6. Metrics: a perfect ranker scores ROC-AUC 1.0; a constant is NaN-safe
y = np.array([0, 0, 1, 1]); p = np.array([0.1, 0.2, 0.8, 0.9])
check(abs(fc._metrics(y, p)["roc_auc"] - 1.0) < 1e-9, "perfect ranker -> ROC-AUC 1.0")
check(np.isnan(fc._metrics(np.array([1, 1, 1]), np.array([.4, .5, .6]))["roc_auc"]),
      "single-class -> ROC-AUC NaN (no crash)")

# 7. State features computed as-of the forecast date (FRED monkeypatched, offline)
_idx = pd.period_range("1990-01", "2005-12", freq="M")
def _fake_fred(name):
    if name == "INDPRO":
        return pd.Series(np.linspace(80, 120, len(_idx)), index=_idx)   # rising
    if name == "UNRATE":
        return pd.Series(np.linspace(6.0, 5.0, len(_idx)), index=_idx)  # falling
    if name == "USREC":
        rec = pd.Series(0.0, index=_idx)          # a 1990 recession, expansion after
        rec.loc[pd.period_range("1990-01", "1990-12", freq="M")] = 1.0
        return rec
    if name == "GS10":
        return pd.Series(6.0, index=_idx)         # long rate 6%
    if name == "TB3MS":
        return pd.Series(4.0, index=_idx)         # short rate 4% -> curve +2
    raise ValueError(name)
fc.fred = _fake_fred
st = fc.add_state_features(pd.DataFrame({"forecast_date": pd.to_datetime(["2000-06-01"])}))
check(st["usrec_now"].iloc[0] == 0.0, "usrec_now 0 outside recession")
check(st["indpro_mom12"].iloc[0] > 0, "indpro momentum positive on rising IP")
check(st["unrate_chg12"].iloc[0] < 0, "unrate change negative on falling UNRATE")
check(abs(st["yield_curve"].iloc[0] - 2.0) < 1e-9, "yield_curve = GS10 - TB3MS = 2.0")
check(9 < st["expansion_age_yrs"].iloc[0] < 11, "expansion age ~10yrs since 1990 recession")
# and inside a recession, expansion age collapses to 0
st_rec = fc.add_state_features(pd.DataFrame({"forecast_date": pd.to_datetime(["1990-06-01"])}))
check(st_rec["usrec_now"].iloc[0] == 1.0 and st_rec["expansion_age_yrs"].iloc[0] == 0.0,
      "in recession: usrec_now 1, expansion age 0")

# 8. Harmonized schema contract: forecast_data.CORE columns are the promised set
for col in ["source", "forecaster_id", "variable_canonical", "forecast_date",
            "pred_direction", "realized_direction", "hit", "source_file", "variable_native"]:
    check(col in fd.CORE, f"CORE schema includes {col}")

# 9. Deployable (real-time) feature set must EXCLUDE the NBER final-vintage features
check("usrec_now" not in fc.RT_FEATURES and "expansion_age_yrs" not in fc.RT_FEATURES,
      "RT_FEATURES drops leaky NBER-final features")
check("usrec_now" in fc.SHARED_FEATURES, "retrospective SHARED_FEATURES keeps usrec_now")
check("yield_curve" in fc.RT_FEATURES, "RT_FEATURES includes real-time yield_curve")

# 10. Multi-origin CV returns a mean AUC in [0,1] with >=2 folds on a toy signal
_n = 240
_toy = pd.DataFrame({
    "forecast_date": pd.date_range("1980-01-01", periods=_n, freq="QS"),
    "hit": ([1, 1, 0, 1] * (_n // 4)),
    "pred_growth": np.linspace(-2, 4, _n), "pred_revision": 0.0,
    "indpro_mom12": np.linspace(-3, 5, _n), "unrate_level": 5.0,
    "unrate_chg12": 0.0, "yield_curve": 1.0, "usrec_now": 0.0, "expansion_age_yrs": 2.0})
_m = fc.cv_auc(_toy, fc.RT_FEATURES)
check(_m[3] >= 2 and 0.0 <= _m[0] <= 1.0, f"cv_auc returns mean in [0,1] over folds (got {_m})")

print(f"\n{PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
