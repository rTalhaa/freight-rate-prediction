# Model selection (Phase 4)

Evidence: `notebooks/05_models.py`. Implementation in `src/model.py`,
`src/validation.py`.

## Validation design

Rolling-origin CV, not a single split. Three expanding-window folds, each with a
**61-day forward horizon** — the same length as the real Nov–Dec task:

| Fold | Train through | Test |
|---|---|---|
| 1 | 2025-05-01 | 2025-05-02 → 2025-07-01 |
| 2 | 2025-07-01 | 2025-07-02 → 2025-08-31 |
| 3 | 2025-08-31 | 2025-09-01 → 2025-10-31 |

Training folds have corrupt labels removed; test folds keep theirs. Feature
statistics are refitted per fold on that fold's training rows only.

## Results (fold-averaged)

| Model | RMSE | MAE | MAPE | MedAPE | Bias (clean) |
|---|---|---|---|---|---|
| **Hybrid (shrink 0.5)** | 629.5 | **110.32** | 4.66 | 2.01 | +21.12 |
| Hybrid (shrink 1.0) | 629.7 | 110.60 | 4.63 | 1.88 | +24.30 |
| Hybrid (shrink 0.3) | 629.8 | 111.87 | 4.75 | 2.12 | +19.98 |
| LightGBM | 633.0 | 114.40 | 4.78 | 1.92 | −19.39 |
| Ridge | 630.5 | 115.65 | 4.95 | 2.32 | +18.38 |

On the Sep–Oct holdout alone (comparable to Phase 3):

| Model | MAE | MAPE | MedAPE |
|---|---|---|---|
| **Hybrid (shrink 1.0)** | **98.09** | 4.23 | 1.47 |
| Hybrid (shrink 0.5) | 100.82 | 4.38 | 1.57 |
| Ridge | 106.99 | 4.67 | 1.81 |
| LightGBM | 110.36 | 4.69 | 1.95 |

## The hybrid works, and the reason is the one predicted

Ridge alone and LightGBM alone both lose. Splitting the job wins:

- **Stage 1, ridge on the full feature set.** Carries the trend — the only part
  that has to extrapolate — and does so linearly.
- **Stage 2, LightGBM on the stage-1 residual, with no time features.** Picks up
  the non-linear structure ridge cannot express, and is never asked for a value
  outside its training range.

The gain over ridge is 4.6% on CV and 8.3% on the Sep–Oct fold.

## Shrinkage is not distinguishable

Shrinkage 0.5 leads on the fold average by $0.27 — inside noise:

| Fold | 0.5 vs 1.0 |
|---|---|
| May–Jul | +2.31 |
| Jul–Aug | +1.23 |
| Sep–Oct | −2.72 |

Wins 2 of 3, mean +$0.27, std $2.65. **Shrinkage 1.0 is selected**: it is the
simpler model (one fewer tuned constant), and it wins clearly on the Sep–Oct
fold, which has the most training data and is the closest analogue to the real
task.

## Ridge alpha barely matters

MAE moves from 115.63 (α=0.1) to 116.92 (α=30) — 1.1% across a 300× range.
Kept at α=1.0.

## Open risk: trend extrapolation is unstable

Per-fold bias for the selected model:

| Fold | Bias (clean) |
|---|---|
| May–Jul | +18.78 |
| Jul–Aug | **+83.85** |
| Sep–Oct | −29.75 |

Rate-per-mile rose Jan→Jun and then fell. A linear trend fitted through the rise
and projected into Jul–Aug overshoots by $84; the same mechanism under-predicts
in Sep–Oct by $30. The trend feature fixes the *average* level error but its
extrapolation swings with whatever the recent slope happened to be.

The final model trains through 2025-10-31 and predicts 2025-11-01 onward, so the
real task is a **1–61 day** forward horizon — exactly what these folds simulate.
(An earlier draft described it as starting two months past training; that was
wrong. The folds are a faithful analogue, not an optimistic one.)

This is still the largest remaining source of error, and it lands hardest on the
December chart, which sits 31–61 days out with date as the only varying input.
Addressed in `docs/damping.md`.
