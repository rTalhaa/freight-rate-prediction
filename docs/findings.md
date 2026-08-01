# EDA Findings

Evidence: `notebooks/01_eda.py`. Train = 48,000 rows (2025-01-01 → 2025-10-31).
Validation = 12,000 rows (2025-11-01 → 2025-12-31).

## Defects

| # | Defect | Train | Validation | Action |
|---|---|---|---|---|
| 1 | Corrupt labels (`rpm` > 4 or < 1) | 669 (1.39%) | n/a | Drop at fit time |
| 2 | Negative `weight` (sign flip) | 292 | 145 | `abs()` |
| 3 | Missing `weight` | 300 | 165 | Impute |
| 4 | Missing `market_index` | 374 | 249 | Impute from daily series |
| 5 | `weight` capped at 47,500 | 1,191 (2.5%) | 297 (2.5%) | Flag column |
| 6 | `distance` floored at 70 mi | 48 | 21 | Flag column |
| 7 | Distorted city coordinates | 22 rows circuity > 2 | 16 | Cap circuity feature |

Defects 2–6 appear in validation at the same rates, so **cleaning must live in the
prediction path, not just training**.

### 1. Corrupt labels — the important one
Rate-per-mile does not decay like a real tail; it plateaus:

| rpm > | 3 | 4 | 5 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|
| rows | 707 | 340 | 317 | 270 | 145 | 79 |

Only 23 rows between 4 and 5, then density holds out to 14. Mirrored at the bottom
(329 rows below rpm 1.0). ~0.7% each side, flat across all ten months — injected
noise, not a market event. Example: Chattanooga → Atlanta, 98 mi, $1,386.

Dropping (not winsorising) because these are corrupt, not extreme-but-real.

### 2. Negative weights
`abs()` of the negatives matches the healthy distribution (median 31,822 vs 31,494;
both bounded 5,000–47,500). Sign errors, fully recoverable. Do not drop.

### 5–6. Censoring
`weight` piles up at exactly 47,500 and `distance` at exactly 70.0. Both are clipped
bounds, not measurements. Binary flags let the model learn the discontinuity.

### 7. Distorted coordinates
Coordinates are *internally* consistent — corr(haversine, distance) = 0.9995, median
circuity 1.18 (the real road-vs-great-circle factor), one coordinate per city, and
zero train/validation mismatches. But they do not match true US geography: New York →
Allentown is 0.89 mi apart here, giving circuity 78. Usable as features; cap the
circuity ratio.

## Clean

No duplicate `load_id`, no duplicate rows, no null/non-positive `posted_rate`, no
whitespace or casing collisions, no `pickup == delivery`, no missing calendar days,
`load_id` format valid throughout, equipment mix stable (Dry Van 57% / Reefer 25% /
Flatbed 18%).

## Signal

- **Distance dominates.** corr = 0.91 with rate. Target skew 1.90 → −0.49 under log.
  **Model log-rate.**
- **Economies of scale.** rpm 2.82 (<250 mi) → 1.90 (3000+ mi).
- **Equipment.** Reefer 2.38 > Flatbed 2.29 > Dry Van 2.12 rpm.
- **Weight.** Monotonic but small: 2.13 → 2.29 across octiles.
- **Seasonality.** rpm +11% Jan→Jun, decaying to +6.6% by Oct. Day-of-week ~2.5%,
  negligible.

### `market_index` is real, not a decoy
Marginal correlation with rate is 0.034. That is confounding by distance. Within
bands, rpm rises monotonically across market_index quintiles, spread ~0.14 in every
band:

| distance | q1 | q2 | q3 | q4 | q5 |
|---|---|---|---|---|---|
| 0–500 | 2.504 | 2.529 | 2.555 | 2.591 | 2.648 |
| 500–1000 | 2.189 | 2.192 | 2.202 | 2.271 | 2.328 |
| 1000–2000 | 2.040 | 2.061 | 2.085 | 2.114 | 2.165 |
| 2000+ | 1.883 | 1.903 | 1.931 | 1.971 | 2.023 |

`market_index` is a **daily series** (within-day std 0.025 vs across-day 0.167) with a
weekly cycle — forecastable. `quote_signal` is the reverse (within-day std 2× across-day),
essentially per-load noise, weakly negative against rate.

## Generalisation risks

**Market regime shift.** Validation `market_index` averages 0.927 vs train 1.083
(−14.4%), and its whole range [0.724, 1.099] sits in the lower part of train's
[0.676, 1.468]. Validation is a softer market than the training average. Sep–Oct train
months share that regime — which is the argument for a late-window temporal holdout
rather than a random split. All other features are stable (< 1% drift).

**Unseen geography.** 8 validation cities absent from train (Allentown, Charlotte,
Chicago, Jackson, Knoxville, Laredo, Norfolk, San Diego); 736 of 4,214 validation
lanes (17.5%) unseen. Train lanes are thin — median 10 rows, p10 = 4. Lane target
encoding misses ~1 row in 6, so coordinates and distance must carry the fallback.

## December chart anchor

Lexington → Fort Wayne appears 32 times in train: mean $857, mean rpm 2.36, distance
~364 mi. The 31 December predictions should land near **$800–1000**. Anything far
outside that is a bug, not a forecast.

The chart file has only 7 columns — no lat/lon, no `market_index`, no `quote_signal`.
Those are imputed from the fitted daily series rather than joined from
`validation.csv`, because a deployed quoting model would not know a future date's
market index. The withheld values are used afterwards to score the imputation.
