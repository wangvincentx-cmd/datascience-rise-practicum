# Did Anyone See It Coming?
## Machine-Reading a Century of American Newspaper Economic Forecasts, 1900–1963

**Vincent Wang, Bode ___, Jonathan Liu — Boston University RISE**
*(author order and affiliations to confirm)*

> Short abstract (~300 words) for submission. Every number is reproducible from
> this repository; see `POSTER.md` for the panel-level detail and
> `RESULTS_MONTHLY.md` for the full monthly-corpus results. For the longer
> treatment including the forecast-credibility model and narrative coding, see
> `EXTENDED_ABSTRACT.md`.

---

### Abstract

Whether the press can anticipate economic turning points has never been measured
at scale: reading a century of newspapers by hand is infeasible, and the standard
shortcut — keyword search — loses most of the evidence. Against a hand-built gold
standard we show that keyword extraction recovers only **27%** of the forecasts
present on a page, while whole-page LLM reading recovers **73%**, and that keyword
retrieval fails *non-randomly* — surfacing advertisements, and in one documented
case inverting the meaning of a column reporting that forecasters had been wrong.

Using a direction-neutral query set held constant across all **768 months** from
1900 to 1963, we read **15,721** Library of Congress newspaper pages, extract
**30,765** structured forecasts, and score **14,251** US-national claims against
the NBER chronology and Federal Reserve series with a deterministic scorer. The
language model determines *what was predicted*; economic data determines *whether
it came true*.

The central result is a forecast mix that never responded to the economy:
downturn forecasts constitute **24.1%** of the corpus during expansions and
**24.3%** during recessions — statistically indistinguishable — with roughly 72%
of all forecasts upbeat throughout. Accuracy consequently falls from **58.8%** in
expansions to **39.7%** in recessions (gap +19.1 points, 95% CI [12.9, 24.4],
block-bootstrapped by three-year period). Pessimists were vindicated when it
mattered — "worsen" calls rise from 21% to 61% accuracy in downturns — but were
outnumbered roughly three to one regardless of conditions. Accuracy is flat across
six decades (53.7% → 49.9%), and no publisher outperforms.

This is not a failing peculiar to journalism. Scored on identical ground truth,
the Survey of Professional Forecasters (54.1%), the Livingston economists (54.4%),
Michigan households (≈55%) and the **Federal Reserve Board's own internal
Greenbook staff forecasts (54.0%, n=480)** all converge just above a coin flip.
The Fed's real-time record supplies the mechanism: across 490 editions, forecast
dispersion collapses from a 4.01-point standard deviation at the nowcast to 0.90
at eight quarters, while the mean holds near +2.8%. **Beyond about a year the
Greenbook is a trend forecast — at six or more quarters out the staff never once
projected negative growth across 54 years — and a constant cannot call a turning
point by construction.** Entering the 2008 financial crisis, its most pessimistic
one-year-ahead forecast on record was +1.45%.

The contribution is as much methodological as substantive: a validated,
reproducible pipeline that makes a century of qualitative forecasts quantitatively
scorable for roughly $25 in compute — and disciplined enough that building the
continuous corpus *overturned* a finding a smaller, outcome-selected sample had
suggested.

---

### Keywords

economic forecasting · business cycles · text as data · large language models ·
media economics · forecast evaluation · Chronicling America

---

### One-sentence version

Across sixty years and 14,251 scored forecasts, the American press predicted
improvement in booms and busts in the same proportion, with no improvement over
six decades and no paper better than any other — and the Federal Reserve's own
staff record shows the same structural blindness.

---

### 100-word version

We machine-read 15,721 Library of Congress newspaper pages spanning every month
from 1900 to 1963, extract 30,765 economic forecasts, and score 14,251 against
NBER and Federal Reserve data. Whole-page LLM reading recovers 73% of forecasts
against keyword search's 27%. The press's forecast mix never responded to the
economy — 24.1% predicted downturn in expansions versus 24.3% in recessions — so
accuracy fell from 58.8% to 39.7% when downturns arrived, flat across six decades
and uniform across publishers. Professional forecasters and the Federal Reserve's
own Greenbook staff converge at the same ~54%, for the same structural reason.
