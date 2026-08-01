# Features and drift (Phase 3)

Evidence: `notebooks/04_features.py`, implementation in `src/features.py`.
Same holdout as Phase 2b, so numbers are directly comparable.

## What each group buys

Cumulative, ridge on log-rate, uniform weights:

| Features | n | MAE | MAPE | MedAPE | Bias (clean) | Δ MAE |
|---|---|---|---|---|---|---|
| base | 13 | 161.43 | 7.00 | 4.34 | −100.38 | — |
| + time | 14 | 118.31 | 5.37 | 2.49 | −26.64 | **+43.13** |
| + geo | 25 | 108.76 | 4.82 | 1.95 | −26.88 | +9.55 |
| + lane | 31 | **106.99** | **4.67** | **1.81** | −31.63 | +1.78 |
| + interactions | 36 | 108.63 | 4.71 | 1.85 | −34.43 | **−1.65** |

Leave-one-out from the full set:

| Removed | MAE | Cost |
|---|---|---|
| time | 134.34 | +25.71 |
| geo | 113.93 | +5.30 |
| lane | 110.34 | +1.70 |
| interactions | 106.99 | **−1.65** |

**Interactions are dropped.** They cost $1.65 of MAE. The market_index × distance
effect is real (measured in EDA) but ridge already captures it through
log_distance, so the explicit terms only add variance.

Best configuration: **base + time + geo + lane, uniform weights, MAE $106.99**
against the $118.34 baseline — a 9.6% improvement.

## Annual Fourier terms are harmful here

Seasonality is normally modelled with Fourier terms. Measured:

| Time features | MAE | Bias (clean) |
|---|---|---|
| none | 161.43 | −100.38 |
| **trend only** | **118.31** | **−26.64** |
| trend + day-of-week | 120.30 | −33.30 |
| trend + fourier | 174.83 | −114.75 |
| trend + fourier + dow | 173.96 | −113.74 |

Cause: over an 8-month window the harmonics are near-collinear with the trend —
corr(trend_days, annual_cos) = **−0.92**, corr(trend_days, annual_sin) = −0.65.
Ridge splits the coefficient between them, which is harmless in-sample and
diverges as soon as prediction moves past the window edge.

Both are kept behind flags (`FeatureConfig.fourier`, `.day_of_week`), default off,
so the ablation demonstrates this rather than asserting it.

Month dummies were never an option: November and December do not appear in
training at all.

**This is a direct warning for Phase 5**, which planned to extrapolate
`market_index` into December with the same Fourier machinery.

## The drift fix is the trend feature, not recency weighting

Bias went from −$100.38 (no time features) to −$26.64 (trend), a 73% reduction.

Exponentially decayed sample weights were tested and made things worse at every
half-life:

| Weighting | MAE | Bias (clean) |
|---|---|---|
| **uniform** | **108.63** | **−34.43** |
| half-life 365d | 109.47 | −36.54 |
| half-life 180d | 110.27 | −38.47 |
| half-life 120d | 111.01 | −40.18 |
| half-life 90d | 111.71 | −41.74 |
| half-life 60d | 113.07 | −44.59 |
| half-life 30d | 118.27 | −53.66 |

Once the trend is an explicit feature, down-weighting old rows removes
information without adding any. The mechanism is kept in `src/features.py`
(`recency_weights`) with the measurement recorded here.

## Trees cannot extrapolate the trend

| Model | MAE | Bias (clean) |
|---|---|---|
| Ridge, uniform | **108.63** | **−34.43** |
| LightGBM, uniform | 112.45 | −48.90 |
| LightGBM, half-life 90d | 112.35 | −48.75 |

LightGBM loses to ridge, and its bias is 42% worse. Tree splits cannot produce a
value outside the training range, so a tree cannot follow `trend_days` past
August. A task defined by forward extrapolation gives the linear model a
structural advantage.

This sets up Phase 4: fit the linear structure (including trend) parametrically,
and give the trees only the residual, with no extrapolation demanded of them.

## Remaining

Bias is **−$31.63** in the best configuration — roughly −1.3% of mean rate. Not
eliminated, and validation sits two months further out than this holdout.
