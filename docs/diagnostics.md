# Diagnostics (Phase 5b)

Evidence: `notebooks/08_diagnostics.py`. Checks that should have run earlier:
where the error actually is, whether the decisions survive more folds, and
whether the cleaning thresholds were arbitrary.

Reference: hybrid, `best_config()`, Sep–Oct holdout, clean rows —
**MAE $49.84 | MAPE 2.05% | MedAPE 1.79%**.

## Fixed: the lane feature leaked within-day

`_lane_block` built its expanding mean with `cumcount()`, which orders by row
position. Loads sharing a row's own date contributed to its lane feature — the
same leak the temporal split exists to prevent, at one-day resolution. Now
aggregated over strictly earlier **dates**.

Cost: ~$0.20 of MAE (104.59 → 104.80 on the damping sweep). All conclusions
unchanged; numbers in `docs/models.md` and `docs/damping.md` refreshed.

## Where the error is

**By distance band** — MAPE is flat, so there is no short-haul weakness:

| Band | n | MAE | MAPE | MedAPE | Bias |
|---|---|---|---|---|---|
| 0–250 mi | 545 | 10.13 | 2.31 | 1.82 | −4.97 |
| 250–500 | 1,511 | 17.51 | 1.83 | 1.61 | −13.06 |
| 500–1000 | 2,927 | 33.68 | 2.00 | 1.76 | −27.80 |
| 1000–2000 | 2,884 | 63.52 | 2.11 | 1.87 | −53.34 |
| 2000+ | 1,513 | 101.63 | 2.13 | 1.85 | −86.96 |

MAE scales with rate; percentage error does not. This matters because MAPE is a
plausible hidden metric and short hauls dominate it.

**By equipment** — Flatbed is the weak class:

| Equipment | n | MAE | MAPE |
|---|---|---|---|
| Dry Van | 5,276 | 42.31 | 1.88 |
| Reefer | 2,358 | 55.05 | 2.03 |
| **Flatbed** | 1,746 | **65.56** | **2.56** |

Flatbed is the smallest class (18% of rows) and the most variable.

**By defect flag** — imputation costs something, on few rows:

| Segment | n | MAE | MAPE |
|---|---|---|---|
| weight imputed | 65 | 76.66 | 2.79 |
| market imputed | 71 | 53.97 | 2.19 |
| weight at cap | 254 | 47.09 | 2.09 |
| no flags | 8,991 | 49.69 | 2.04 |

Rows with an imputed weight are 54% worse. Only 65 rows, so this is directional,
but it confirms the flags are worth carrying.

**Bias is negative in every segment** — roughly −1.8% of rate throughout. The
model under-prices uniformly rather than in any particular corner.

## Unseen geography — the case the holdout cannot test

`validation.csv` contains 8 cities and 17.5% of lanes never seen in training.
The Sep–Oct holdout contains **22 such rows (0.2%)**, because both windows cover
the same 64 cities. **The temporal split reproduces the market regime faithfully
but not this**, which is a real limitation of the validation design.

Simulated by holding 8 random cities out of training and predicting loads that
use them (3 trials):

| Trial | Unseen-city MAPE | Familiar MAPE |
|---|---|---|
| 1 | 2.43 | 2.05 |
| 2 | 2.76 | 2.07 |
| 3 | 2.15 | 2.06 |
| **Mean** | **2.45** | **2.06** |

**Penalty: +18.6% MAPE.** Degradation, not failure — an unseen city still has
coordinates, a distance and an equipment type. This is the payoff for the EDA
decision to make coordinates carry the fallback rather than lane statistics.

## Do the decisions survive more folds?

6 folds at a 30-day horizon, against the 3 folds at 61 days used for selection:

| Model | mean MAE | worst | std |
|---|---|---|---|
| **Hybrid (shrink 1.0)** | **98.59** | 112.51 | 12.48 |
| Hybrid (shrink 0.5) | 99.37 | 114.38 | 11.37 |
| Ridge | 105.82 | 123.53 | 12.17 |

Model choice holds. Damping is more horizon-dependent:

| φ | mean MAE | worst | max abs bias |
|---|---|---|---|
| 1.0 | 99.90 | **110.15** | 54.22 |
| 0.3 | 98.78 | 111.56 | 44.39 |
| **0.0** | **98.59** | 112.51 | **40.19** |

At 30 days φ=0.0 still wins mean MAE and bias, but worst-fold marginally favours
φ=1.0. Damping matters more the further ahead you predict — as expected. The
real task is 61 days, where φ=0.0 won every criterion, so it stands.

## Are the cleaning thresholds arbitrary?

Cross-validated at the 61-day horizon:

| Thresholds | rows dropped | mean MAE | MedAPE |
|---|---|---|---|
| rpm ∈ [1.0, 3.0] | 2.29% | 104.85 | 1.69 |
| rpm ∈ [1.0, 3.5] | 1.40% | 104.87 | 1.68 |
| **rpm ∈ [1.0, 4.0]** | **1.39%** | **104.80** | **1.68** |
| rpm ∈ [1.0, 4.5] | 1.38% | 105.81 | 1.69 |
| rpm ∈ [1.0, 5.0] | 1.35% | 107.59 | 1.72 |
| rpm ∈ [0.8, 4.0] | 1.26% | 107.53 | 1.84 |
| rpm ∈ [1.2, 4.0] | 1.41% | 105.01 | 1.68 |
| **no cleaning** | 0% | **143.20** | 3.15 |

The eyeballed thresholds turn out optimal, and the result is flat between 3.0
and 4.5 — the choice is not delicate. The lower bound matters more than the
upper: loosening to 0.8 costs $2.73.

Cleaning at all is worth **$38.40 of MAE** (143.20 → 104.80), the single largest
intervention in the project.
