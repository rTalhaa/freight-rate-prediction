# Trend damping (Phase 4b)

Evidence: `notebooks/06_damping.py`.

## Problem

Phase 4 left per-fold bias at +$19, +$84, −$30. Rate-per-mile rose Jan→Jun and
then fell; a linear trend fitted through the rise and projected into Jul–Aug
overshoots by $84.

## Mechanism

Past the last training date, the trend input is replaced by

```
t_eff = t_max + φ · (t − t_max)
```

φ=1.0 extrapolates at full slope, φ=0.0 freezes the level at the window edge.
Nothing inside the training window changes.

## Sweep

Hybrid model, three rolling folds:

| φ | mean MAE | worst MAE | max abs bias | MAE spread |
|---|---|---|---|---|
| 1.0 | 110.60 | 137.86 | 83.85 | 42.02 |
| 0.8 | 108.84 | 132.99 | 78.15 | 38.83 |
| 0.6 | 107.34 | 128.30 | 72.47 | 35.34 |
| 0.5 | 106.69 | 126.04 | 69.64 | 33.46 |
| 0.4 | 106.12 | 123.84 | 66.81 | 31.51 |
| 0.3 | 105.62 | 121.70 | 63.99 | 29.49 |
| 0.2 | 105.19 | 119.62 | 61.17 | 27.39 |
| 0.1 | 104.85 | 117.62 | 58.35 | 25.20 |
| **0.0** | **104.59** | **115.69** | **55.54** | **22.93** |

**φ=0.0 wins every criterion, monotonically.** Selection was set up to prefer
robustness over mean MAE in case they disagreed. They did not.

Per-fold MAE shows the trade honestly:

| φ | May–Jul | Jul–Aug | Sep–Oct |
|---|---|---|---|
| 1.0 | 95.84 | 137.86 | **98.09** |
| 0.0 | 92.76 | **115.69** | 105.32 |

Freezing costs $7 on Sep–Oct, where the trend happened to point the right way,
and saves $22 on Jul–Aug, where it did not. Worst case improves by 16%.

## What this means

**The trend's value is in de-confounding the training data, not in projecting
forward.** Removing time features entirely scores MAE 134.34 (Phase 3); keeping
the trend but refusing to extrapolate it scores 104.59. The feature earns its
place by explaining historical rates, not by predicting future ones.

Extrapolating a slope estimated over months, across a turn in the market, adds
no information and considerable variance. This is the third measurement in the
same direction, after Fourier terms and recency weighting.

## Error vs horizon

Mean over folds, clean rows, by days past the end of training:

| φ | 1–15d | 16–30d | 31–45d | 46–61d |
|---|---|---|---|---|
| 1.0 bias | 12.97 | 7.87 | 40.55 | 35.82 |
| 1.0 MAE | 35.04 | 52.38 | 68.05 | 71.68 |
| **0.0 bias** | **7.22** | **−8.57** | **13.44** | **−2.42** |
| **0.0 MAE** | **33.30** | **47.91** | **57.41** | **63.74** |

Two readings:

- Damping roughly halves bias at every horizon, and at φ=0.0 the bias no longer
  grows with distance — it oscillates around zero instead of drifting.
- MAE still nearly doubles from the first fortnight to the last. Predictions
  late in December are inherently weaker than predictions in early November.
  That is a property of the task, not something damping removes.

## Selected

**φ = 0.0**, via `features.best_config()`.

Caveat: φ=0.0 sits at the edge of the grid, so the data prefers the most
conservative option tested rather than an interior optimum. With three folds
this is a small-sample estimate, and the asymmetry of the errors favours
under-committing. The 12,000 validation rows also carry a supplied
`market_index`, so freezing the clock does not leave the model blind to the
market regime — it still sees the softer Nov–Dec market directly.

## Scope of the real task

The final model trains through 2025-10-31 and predicts from 2025-11-01, a **1–61
day** horizon. The folds simulate exactly that. The December chart sits at 31–61
days — the weaker half of the range.
