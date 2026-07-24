"""
Ground truth for scoring economic predictions -- the objective half of the
project, rebuilt from scratch and deliberately kept separate from anything the
LLM touches.

THE ONE RULE: the language model extracts WHAT was predicted; this file decides
WHAT ACTUALLY HAPPENED. The two never mix. Correctness is a lookup against real
data (NBER business-cycle dates and Federal Reserve series), computed the same
way every time, with no hindsight-laden model judgement anywhere in it. That
separation is what makes "did they see it coming" a measurement rather than a
model grading itself.

What this provides
------------------
    realized_direction(topic, start, horizon_months) -> (label, scorable, basis)

Given a claim's topic, its print date, and how far ahead it looked, return what
the relevant real-world series actually did over that window, as one of a small
set of labels -- or (None, False, reason) when the claim cannot be scored.

Series and coverage (final revised values, from FRED's public CSV endpoint):
    INDPRO    industrial production, 1919-        general business / markets
    CPIAUCNS  consumer prices,        1913-        prices
    UNRATE    unemployment rate,      1948-        employment
    NBER      recession chronology,   full period  business, when INDPRO is absent

Two decisions worth stating plainly, because a reviewer will ask:

1. FINAL values, not vintage. We are scoring what the economy ACTUALLY did, so
   the latest revised series is correct. (Publication lag matters only for the
   separate question of what a forecaster could have KNOWN at print time -- that
   belongs to the prediction model's features, not here.)

2. NO-CHANGE BANDS. A forecast of "improve" should not be marked correct because
   production ticked up 0.1%. A move counts as real only past a threshold; below
   it the realized outcome is "flat". These bands are a documented modelling
   choice and are exposed as parameters so the headline result can be shown to
   survive plausible alternatives -- not hand-tuned to a desired answer.

Nothing here imports from the rest of the repo, and none of it is LLM-derived.
"""

import io
import ssl
from pathlib import Path

import pandas as pd

CACHE = Path("cache")
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
# FRED silently hangs on a non-browser user agent.
FRED_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# No-change thresholds: how big a move over the claim's window counts as the
# economy actually going somewhere, versus noise. Defaults chosen to be
# defensible and round, NOT calibrated to any outcome; pass overrides to
# realized_direction() to sensitivity-test them.
#   INDPRO 2.0%   industrial production is volatile; sub-2% over 6-12 months is noise
#   CPI    1.5%   distinguishes a real price move from measurement wobble
#   UNRATE 0.5pt  the Sahm-rule order of magnitude for a meaningful labour shift
DEFAULT_BANDS = {"INDPRO": 2.0, "CPI": 1.5, "UNRATE": 0.5}

# NBER US business-cycle contractions (peak month -> trough month). Used only
# before 1919, where INDPRO does not exist. Public chronology, nber.org.
NBER_RECESSIONS = [
    ("1902-09", "1904-08"), ("1907-05", "1908-06"), ("1910-01", "1912-01"),
    ("1913-01", "1914-12"), ("1918-08", "1919-03"), ("1920-01", "1921-07"),
    ("1923-05", "1924-07"), ("1926-10", "1927-11"), ("1929-08", "1933-03"),
    ("1937-05", "1938-06"), ("1945-02", "1945-10"), ("1948-11", "1949-10"),
    ("1953-07", "1954-05"), ("1957-08", "1958-04"), ("1960-04", "1961-02"),
]


def _ssl_context():
    try:
        import truststore
        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except ImportError:
        try:
            import certifi
            return ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            return ssl.create_default_context()


def load_fred(sid):
    """Monthly FRED series as a Period-indexed float Series, cached on disk."""
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"fred_{sid}.csv"
    if not f.exists():
        import urllib.request
        req = urllib.request.Request(FRED_CSV.format(sid=sid), headers=FRED_HEADERS)
        with urllib.request.urlopen(req, timeout=60, context=_ssl_context()) as r:
            f.write_bytes(r.read())
    df = pd.read_csv(io.StringIO(f.read_text(encoding="utf-8")))
    df.columns = ["date", "value"]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.dropna().set_index(pd.PeriodIndex(pd.to_datetime(df.dropna()["date"]),
                                             freq="M"))["value"]
    return s[~s.index.duplicated(keep="first")].sort_index()


def recession_months():
    months = set()
    for peak, trough in NBER_RECESSIONS:
        months.update(pd.period_range(peak, trough, freq="M"))
    return months


class TruthData:
    """Bundles the series once so a whole corpus can be scored without refetching.

    Kept as an object rather than module globals so tests can inject tiny
    synthetic series and check the scoring maths against known answers, which is
    the only way to be actually SURE the scorer is right rather than merely
    plausible."""

    def __init__(self, indpro=None, cpi=None, unrate=None, recessions=None):
        self.indpro = load_fred("INDPRO") if indpro is None else indpro
        self.cpi = load_fred("CPIAUCNS") if cpi is None else cpi
        self.unrate = load_fred("UNRATE") if unrate is None else unrate
        self.recession = recession_months() if recessions is None else recessions

    def _window_change_pct(self, series, start, months):
        """Percent change of a series from `start` to `start + months`.

        Returns None if either endpoint is outside coverage -- an unscorable
        claim, never a guessed one."""
        p0 = start.to_period("M")
        p1 = (start + pd.DateOffset(months=months)).to_period("M")
        if p0 < series.index.min() or p1 > series.index.max():
            return None
        try:
            v0 = series.asof(p0)
            v1 = series.asof(p1)
        except KeyError:
            return None
        if pd.isna(v0) or pd.isna(v1) or v0 == 0:
            return None
        return 100.0 * (v1 - v0) / v0

    def _band3(self, value, band, up, down):
        """Three-way classify a change: `up` above +band, `down` below -band,
        else 'flat'."""
        if value is None:
            return None
        if value > band:
            return up
        if value < -band:
            return down
        return "flat"

    def realized_direction(self, topic, start, horizon_months, bands=None):
        """What the economy actually did for this claim's topic over its window.

        Returns (label, scorable, basis):
          label    normalized outcome in the claim's own vocabulary --
                   improve/worsen/flat for business & markets,
                   up/down/flat for prices, up/down/flat for unemployment
          scorable False when no series covers this topic+date; label is None
          basis    which series produced it ('INDPRO' / 'CPI' / 'UNRATE' /
                   'NBER'), for transparency and stratified reporting
        """
        bands = bands or DEFAULT_BANDS
        start = pd.Timestamp(start)

        if topic in ("general_business", "markets", "other"):
            change = self._window_change_pct(self.indpro, start, horizon_months)
            if change is not None:
                return (self._band3(change, bands["INDPRO"], "improve", "worsen"),
                        True, "INDPRO")
            # Pre-1919: fall back to the NBER chronology. A window that is mostly
            # in contraction realized "worsen"; mostly in expansion "improve".
            p0 = start.to_period("M")
            p1 = (start + pd.DateOffset(months=horizon_months)).to_period("M")
            window = list(pd.period_range(p0, p1, freq="M"))
            if not window or p1 > pd.Period("1961-12", "M"):
                return (None, False, "no INDPRO / outside NBER chronology")
            share_recession = sum(m in self.recession for m in window) / len(window)
            return ("worsen" if share_recession >= 0.5 else "improve",
                    True, "NBER")

        if topic == "prices":
            change = self._window_change_pct(self.cpi, start, horizon_months)
            if change is None:
                return (None, False, "no CPI before 1913 / outside coverage")
            return (self._band3(change, bands["CPI"], "up", "down"), True, "CPI")

        if topic == "employment":
            p0 = start.to_period("M")
            p1 = (start + pd.DateOffset(months=horizon_months)).to_period("M")
            if p0 < self.unrate.index.min() or p1 > self.unrate.index.max():
                return (None, False, "no UNRATE before 1948 / outside coverage")
            v0, v1 = self.unrate.asof(p0), self.unrate.asof(p1)
            if pd.isna(v0) or pd.isna(v1):
                return (None, False, "UNRATE gap")
            # UNRATE rising = unemployment up. Report in unemployment's own terms.
            return (self._band3(v1 - v0, bands["UNRATE"], "up", "down"),
                    True, "UNRATE")

        return (None, False, f"topic '{topic}' has no ground-truth series")
