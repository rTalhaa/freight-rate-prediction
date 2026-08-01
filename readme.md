# Freight Rate Prediction

Solution for the Spotter ML Engineer assessment. Predicts `posted_rate` for
12,000 loads in `data/validation.csv` and for the 31 fixed-lane December rows in
`data/december_chart_inputs.csv`.

## Quick start

```bash
python -m pip install -r requirements.txt
python -m src.predict
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

`python -m src.predict` trains the final model and writes both submission files,
validating them against the scorer's requirements first. `score.py` (provided,
unmodified) then validates them again and writes
`scorer_results/candidate_december.png`.

Python 3.11. Runtime is about two minutes end to end.

## Approach

The data is a single year split forward in time: training covers 2025-01-01 to
2025-10-31, prediction covers 2025-11-01 to 2025-12-31. Everything below follows
from that.

**Validation** is a temporal split, never random. Holdout is Sep–Oct, with
rolling-origin folds at a 61-day horizon for model selection — the same horizon
as the real task. The holdout's mean `market_index` is 0.926 against
validation's 0.927, so it reproduces the market regime; a random split would
average it away.

**Cleaning** repairs 292 sign-flipped weights, imputes missing weights and
market values, and flags censoring at both the weight cap and the distance
floor. About 1.39% of training labels are corrupt — rate-per-mile does not decay
like a real tail, it plateaus — and are dropped from training folds only. The
holdout keeps its corrupt rows, because the real scoring will not be cleaned.

**Model** is a hybrid: ridge on log-rate carries the global structure including
the time trend, and LightGBM fits the residual using features with no time
component. Tree splits cannot emit values outside their training range, so trees
alone cannot follow a trend past the end of training; this split lets each part
do what it can.

**The trend is not extrapolated.** Its damping factor is 0, meaning the level
freezes at the last training date. This was measured, not assumed: projecting
the fitted slope forward loses on every criterion. The trend earns its place by
de-confounding historical rates, not by predicting future ones.

**December** needs coordinates and both market features, none of which the chart
file carries. Coordinates are looked up (a static city property). The market
series are forecast from training data alone rather than joined from
`validation.csv`, because a deployed quoting model would not know a future
date's market index. The withheld values are then used to score that forecast:
it landed within 0.35%.

## Results

Sep–Oct holdout, clean rows: **MAE $49.84 | MAPE 2.05% | MedAPE 1.79%**.

Metrics are reported as a family because the scoring metric is not published.
RMSE is driven by long expensive loads and MAPE by short cheap ones, and they
rank models differently.

| Stage | MAE (61-day CV) |
|---|---|
| No cleaning | 143.20 |
| Flat $/mile baseline | 274.96 |
| Ridge, base features | 118.34 |
| + time, geo, lane features | 106.99 |
| Hybrid + damped trend | **104.80** |

## Layout

```
src/
  data.py         loading, cleaning, temporal split
  features.py     feature blocks, damping, recency weights
  model.py        ridge, LightGBM, hybrid
  validation.py   rolling-origin cross-validation
  december.py     market forecast, December enrichment
  metrics.py      RMSE / MAE / MAPE / MedAPE / bias
  predict.py      entry point, writes both submission files
notebooks/        numbered analysis scripts, run in order
docs/             findings and decisions, one file per phase
```

The notebooks are plain scripts and run standalone (`python notebooks/01_eda.py`).
They are the evidence behind `docs/`; nothing in `src/` imports them.

## Documentation

| File | Contents |
|---|---|
| `docs/findings.md` | EDA: 7 defects, signal structure, generalisation risks |
| `docs/baselines.md` | Baselines and why the metric family |
| `docs/features.md` | Feature ablation; why Fourier terms and recency weighting fail |
| `docs/models.md` | Model selection under rolling CV |
| `docs/damping.md` | Trend damping, and error against horizon |
| `docs/december.md` | Market forecast and its audit |
| `docs/diagnostics.md` | Segment errors, unseen geography, threshold sensitivity |

## Outputs

- `validation_predictions.csv` — 12,000 rows of `load_id,predicted_rate`
- `data/december_chart_inputs.csv` — provided file with `predicted_rate` filled
- `scorer_results/candidate_december.png` — chart produced by `score.py`

## Assessment instructions

See `Freight_Rate_ML_Assessment.pdf`. `score.py`, the data files and the PDF are
as provided; only the filenames were normalised to the underscore names the
instructions reference.
