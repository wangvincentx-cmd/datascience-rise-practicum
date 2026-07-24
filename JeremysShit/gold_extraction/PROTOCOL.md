# Gold annotation protocol — claim extraction

Written **before** any page was read, so the labels are dictated by rules rather
than fitted to what the extractors happened to produce. The inclusion rules are
deliberately identical to `RUBRIC_PROMPT` in [../grade_claims.py](../grade_claims.py),
so this gold is comparable with the earlier 80-claim labeling gold
(`../handgrade_newspapers/`) rather than measuring a different construct.

## Unit of analysis

**The page, not the claim.** Every prediction on the page gets a record. A page
with zero predictions is still a gold record (`claims: []`) — that is what makes
false positives countable. This is the only difference from the earlier gold,
and it is the entire point: the previous gold could only see sentences an
extractor had already chosen, so it could measure precision and never recall.

## What counts as a prediction

A sentence (or minimal clause span) that makes a **falsifiable claim about
future economic conditions** — business conditions, prices, employment, markets,
prosperity, recession/panic.

**Include:**
- Quoted forecasts. A banker, official, or economist quoted predicting counts;
  the paper is the vehicle, the forecaster is the source.
- Real forecasts printed beside advertising or under a headline.
- Forecasts recoverable through OCR damage: if a named forecaster gives a dated
  directional call and the words can be reconstructed with confidence, it counts.
- Headlines, when the headline itself states a forecast.

**Exclude:**
- Advertisements and promotional copy, including price lists and sales pitches.
- Text too OCR-mangled to reconstruct what is being claimed.
- Explicit refusals to forecast ("it is too early to say when the recession will end").
- Retrospectives and descriptions of the present or past ("business is good today",
  "the panic ruined us"). **This is the highest-stakes boundary.** A paper printed
  the day after the 1929 crash saying "we are ruined" is not a forecast; scoring it
  as one manufactures a brilliant call out of hindsight.
- Conditional or hypothetical arithmetic with no committed direction.
- Metaphor, poetry, aphorism.
- Announcements that a speech/meeting/report about the outlook will occur — the
  outlook is the event's topic, not a forecast the sentence makes.
- Non-economic futures (weather, sport, a person's health), and predictions about
  a single company's stock (a stock tip is not a claim about the economy).

## Fields per claim

| Field | Values | Notes |
|---|---|---|
| `quote` | verbatim span, copied exactly from the page including OCR errors | Never corrected or paraphrased — it is the anchor every extractor is matched against |
| `topic` | `general_business`, `prices`, `employment`, `markets`, `other` | If it spans two, pick what it is *most* about |
| `direction` | `improve`, `worsen`, `no_change`, `unclear` | See rules below |
| `horizon_months` | `6`, `12`, or `vague` | Best estimate from the sentence's own time language |
| `confidence` | `assertive`, `hedged` | Judge the words, not the speaker's authority. Recorded but **known unreliable** (κ = 0.17–0.19 in prior work); the objective Hyland lexicon in `../hedging_lexicon.py` supersedes it |
| `voice` | `journalist`, `expert`, `official`, `layperson`, `unclear` | Judge the speaker, not the topic |
| `speaker_name` | personal name if stated or clearly implied, else `na` | People only, not organizations |

### `direction` rules

- Reassurance that conditions are sound or fears are unfounded ("nothing in the
  outlook to cause uneasiness") is **`improve`**, not `no_change`.
- `no_change` **only** when the sentence explicitly says conditions hold flat.
- `unclear` only when it is genuinely a forecast whose direction cannot be read.
  Never a default for "I'm not sure."
- For `prices` claims, ask what the sentence implies for **conditions overall**,
  using the sentence's own framing. Rising prices in a 1920 deflation read as
  recovery; the same words in 1945 read as ruinous inflation.

## Span rules

- The `quote` is the **minimal span that carries the prediction**, usually one
  sentence. Include the attribution clause when it is in the same sentence.
- Copy verbatim — including `tho` for `the`, broken words across column breaks,
  and stray characters. Matching is fuzzy (token Jaccard ≥ 0.6), so light OCR
  noise will not break a match, but rewriting the span would silently change
  what is being measured.
- Two predictions in one sentence → two records only if they have different
  directions or topics; otherwise one record.

## Honest provenance

These labels were produced by **Claude (Opus 4.8), in session, working from this
protocol**, reading each page's full OCR text before any extractor was run
against it. They are *not* human labels, and this file must be cited as such
wherever the numbers appear.

That ordering matters and is the one integrity property this gold does have: the
pages were annotated before any model output existed for them, so the gold could
not drift toward an extractor's answers — the failure mode the earlier gold's
reconciliation pass is explicitly caveated for
([../handgrade_newspapers/KAPPA_RESULTS.md](../handgrade_newspapers/KAPPA_RESULTS.md)).

**Before publication:** two humans should independently annotate ~40 of these
claims from this protocol, blind to both this gold and any model output, and the
human-vs-this-gold κ reported. Until that exists, every number computed against
this gold is a development metric, not a validation metric.

## Known limitation of the sample

The 16 pages are drawn from the cached corpus, which contains only crisis-window
pages — the 1905/1925/1955 calm-control windows were never cached. Extraction
quality is not outcome-dependent, so this does not bias precision/recall
estimates, but it does mean the gold has not been checked against the lower
forecast density of a calm period.
