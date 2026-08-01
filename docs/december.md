# December chart (Phase 5)

Evidence: `notebooks/07_december.py`. Implementation in `src/december.py`.

## The problem

`december_chart_inputs.csv` has seven columns. Missing: coordinates,
`market_index`, `quote_signal`. Only the date varies across its 31 rows.

Two different gaps, treated differently:

**Coordinates are a fact.** Every city maps to exactly one lat/lon, identical in
train and validation (verified in EDA). Looking up Lexington and Fort Wayne
recovers a static property, not a forecast.

**Market features are a forecast.** Their December values are not knowable at
quote time. `validation.csv` does contain them for those dates, but joining them
would use information a deployed quoting model would not have. They are
forecast from training data only, then used afterwards to **score** the forecast.

## Forecaster

Trailing level plus weekly profile. No extrapolated slope — the same finding as
`docs/damping.md`. Window length is chosen by backtesting the last 61 days of
training, never against December.

| Series | Level | Window | Weekly profile | Day-of-week signal/noise |
|---|---|---|---|---|
| `market_index` | 0.9390 | last 7d | yes | 9.0 |
| `quote_signal` | 2.0587 | last 56d | no | 0.3 |

The two series are structurally different. `market_index` has a dominant weekly
cycle (lag-7 autocorrelation 0.97, weekday swing −0.085 to +0.100 against noise
std 0.02). `quote_signal` has none (weekday deviations ~0.01 against noise std
0.07) but shifts regime every few months, so it takes a longer window and a flat
profile. The profile is included only where signal/noise clears 1.0.

## Audit against the withheld values

Actuals from `validation.csv`, not used to build the forecast:

| | forecast mean | actual mean | error | MAE | max abs error | vs naive |
|---|---|---|---|---|---|---|
| `market_index` | 0.9376 | 0.9344 | +0.35% | 0.0097 | 0.0193 | **93.5% better** |
| `quote_signal` | 2.0587 | 2.0475 | +0.54% | 0.0149 | 0.0494 | 14.6% better |

Ranges line up too: actual `market_index` spans [0.8306, 1.0445], forecast spans
[0.8426, 1.0387]. The amplitude of the weekly cycle is right, not just the level.

Baseline is the whole-training mean (1.0834), which would have been wrong by
0.1491 — the training average sits in a much tighter market than December.

## Predictions

Final model: hybrid, shrinkage 1.0, `best_config()` (trend damping 0.0), trained
on all 47,331 clean rows from Jan–Oct.

| | value |
|---|---|
| Mean | $834.13 |
| Range | $826.88 – $840.22 |
| Spread | $13.34 (1.6%) |
| Implied rate/mile | 2.317 |

Sanity check against history: the lane appears 32 times in training at a mean of
$856.59 and rpm 2.355. Predictions land 2.6% below that, consistent with December
sitting in a softer market (`market_index` 0.934) than the training average
(1.083). All 31 days fall inside the $800–1000 anchor band.

## Shape of the chart

Correlation of predicted rate with `market_index` is **+0.993**; with
`quote_signal`, 0.000 (it is forecast flat).

Weekday pattern, which is the whole shape of the line:

| Mon | Tue | Wed | Thu | Fri | Sat | Sun |
|---|---|---|---|---|---|---|
| 828.25 | 833.41 | 840.22 | **840.18** | 837.95 | 832.13 | **826.88** |

Midweek peak, weekend trough, ~$13 amplitude. The line repeats exactly every
seven days, so **the 31 predictions take only 7 distinct values** — the level is
flat by design and the weekly profile is the only varying input.

## What the flat level misses

The repetition is justified: the forecast explains **97.7% of the variance** in
December's actual daily `market_index`. The real series also oscillates on a
7-day period between 0.83 and 1.04, so a repeating weekly shape reflects the
series rather than oversimplifying it.

But the residuals are **not noise**. Lag-1 autocorrelation is +0.963, and they
drift monotonically from −0.019 on 1 December to +0.016 on 31 December:

| | Actual level |
|---|---|
| Dec 1–15 | 0.9207 |
| Dec 17–31 | 0.9487 |
| **Drift** | **+3.05%** |

The market rose steadily through December and the flat level caught none of it.

In dollars, on this lane:

| | |
|---|---|
| Rate sensitivity | ~$72 per 1.0 of `market_index` |
| Typical unmodelled movement | **$0.77** |
| Worst day | **$1.39** (0.17% of the $835.72 mean) |

**Not corrected.** The 7-day trailing level was chosen by backtesting the last 61
days of training, and December drifting upward was not knowable from October
data. Fitting that drift now, knowing the outcome, would be tuning to the test
set — the exact error the Fourier terms, recency weighting and trend damping
results each warned about. The ~$1 miss is the price of that discipline.

So the flat level was correct to within **$1.39 on the worst day**, not correct
in the abstract.

## Known limitation

No holiday effects. Training covers January–October and contains no December, so
Christmas week carries no special signal; 25 December is priced as an ordinary
Thursday. Real freight markets move sharply around the holidays. Adding a
hand-specified holiday effect was rejected: there is no in-sample evidence to
estimate its size, and every attempt in this project to impose structure the
data could not support (Fourier terms, recency weighting, trend extrapolation)
made results worse.
