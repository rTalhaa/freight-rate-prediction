# Baselines (Phase 2b)

Evidence: `notebooks/03_baselines.py`. Fit on Jan–Aug (37,951 rows, corrupt
labels removed), scored on the Sep–Oct holdout (9,523 rows, left contaminated).

## Scores

| Model | RMSE | MAE | MAPE | MedAPE | R² |
|---|---|---|---|---|---|
| Ridge on log(rate) + smearing | 635.3 | **118.34** | 5.37 | 2.49 | 0.827 |
| $/mile by band × equipment | 638.7 | 136.42 | 6.20 | 3.19 | 0.825 |
| $/mile by distance band | 654.3 | 190.48 | 8.33 | 5.43 | 0.816 |
| Flat $/mile ($2.182) | 697.7 | 274.96 | 11.85 | 8.27 | 0.791 |
| Global mean | 1526.8 | 1174.39 | 82.34 | 44.17 | −0.001 |

## Why no single metric

Reported as a family because Spotter does not publish the metric. RMSE is driven
by long expensive loads, MAPE by short cheap ones, and they rank models
differently. MedAPE is included because it is nearly immune to the corrupt labels.

## The corrupt rows cost about half the total error

143 corrupt rows (1.5% of the holdout) move best-model MAE from **$62.55 to
$118.34** (+89%).

| | clean subset | full holdout |
|---|---|---|
| RMSE | 86.6 | 635.3 |
| MAE | 62.55 | 118.34 |
| MedAPE | 2.45 | 2.49 |

RMSE absorbs a 7× hit; MedAPE moves 0.04 points. This is the evidence for
dropping them from training and for reporting more than one metric.

## Linear is already close

Ridge on log-rate reaches **R² 0.996, MAPE 2.88%** on the clean subset. The
generating process is close to log-linear, so gradient boosting is being asked
for a small remainder rather than a rescue.

## Duan smearing

Exponentiating a log-scale fit predicts the median, not the mean. The correction
factor here is **1.0006** — negligible, because log residuals are small. Applied
for correctness; it is not a fix for the drift bias below.

## Open issue carried into Phase 3

Bias is **−$26.60** on the clean holdout. A model fitted on Jan–Aug systematically
under-prices Sep–Oct, because rates drift upward through the year. Validation is
two months further out again.
