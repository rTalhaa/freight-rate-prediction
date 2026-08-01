"""
Phase 3 - feature ablation and the drift fix.

Run:  python notebooks/04_features.py

Two questions:
  1. What does each feature group actually buy?
  2. Can the -$26.60 under-prediction from Phase 2b be removed?

Same holdout as the baselines (Sep-Oct, contaminated) so numbers are comparable.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import corrupt_label_mask, prepare  # noqa: E402
from src.features import (  # noqa: E402
    FeatureConfig,
    build_features,
    fit_feature_stats,
    recency_weights,
)
from src.metrics import evaluate, smearing_factor  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None


def section(title: str) -> None:
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


bundle = prepare()
fit, holdout = bundle["fit"], bundle["holdout"]
stats = fit_feature_stats(fit)

y_fit_log = np.log(fit["posted_rate"].to_numpy())
y_holdout = holdout["posted_rate"].to_numpy()
clean_mask = ~corrupt_label_mask(holdout).to_numpy()

BASELINE_MAE = 118.34  # ridge + smearing, from notebooks/03_baselines.py


def fit_ridge(X, y, weights=None):
    mu, sigma = X.mean(axis=0), X.std(axis=0)
    sigma[sigma == 0] = 1.0
    model = Ridge(alpha=1.0).fit((X - mu) / sigma, y, sample_weight=weights)
    return model, mu, sigma


def predict_ridge(model, mu, sigma, X):
    return model.predict((X - mu) / sigma)


def fit_lgbm(X, y, weights=None):
    return lgb.LGBMRegressor(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=40,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        verbose=-1,
        random_state=0,
    ).fit(X, y, sample_weight=weights)


def run(config: FeatureConfig, half_life=None, model="ridge"):
    """Fit on the training window, score on the holdout. Returns metrics."""
    X_fit = build_features(fit, stats, config, expanding_lane=True)
    X_holdout = build_features(holdout, stats, config, expanding_lane=False)
    X_holdout = X_holdout[X_fit.columns]

    weights = recency_weights(fit["date"], half_life)
    Xf, Xh = X_fit.to_numpy(dtype=float), X_holdout.to_numpy(dtype=float)
    Xf = np.nan_to_num(Xf)
    Xh = np.nan_to_num(Xh)

    if model == "ridge":
        est, mu, sigma = fit_ridge(Xf, y_fit_log, weights)
        in_sample = predict_ridge(est, mu, sigma, Xf)
        pred_log = predict_ridge(est, mu, sigma, Xh)
    else:
        est = fit_lgbm(Xf, y_fit_log, weights)
        in_sample = est.predict(Xf)
        pred_log = est.predict(Xh)

    smear = smearing_factor(y_fit_log - in_sample)
    pred = np.clip(np.exp(pred_log) * smear, 1.0, None)

    metrics = evaluate(y_holdout, pred)
    metrics["MAE_clean"] = evaluate(y_holdout[clean_mask], pred[clean_mask])["MAE"]
    metrics["Bias_clean"] = evaluate(y_holdout[clean_mask], pred[clean_mask])["Bias"]
    metrics["n_features"] = X_fit.shape[1]
    return metrics


# --------------------------------------------------------------------------
# 1. Cumulative ablation
# --------------------------------------------------------------------------
section("1. FEATURE ABLATION (cumulative, ridge on log-rate)")
LADDER = [
    ("base", FeatureConfig(time=False, geo=False, lane=False, interactions=False)),
    ("+time", FeatureConfig(time=True, geo=False, lane=False, interactions=False)),
    ("+geo", FeatureConfig(time=True, geo=True, lane=False, interactions=False)),
    ("+lane", FeatureConfig(time=True, geo=True, lane=True, interactions=False)),
    ("+interactions", FeatureConfig(time=True, geo=True, lane=True, interactions=True)),
]

rows = {}
for name, config in LADDER:
    rows[name] = run(config, model="ridge")
ablation = pd.DataFrame(rows).T
ablation["gain_MAE"] = -ablation["MAE"].diff()
print(ablation[["n_features", "RMSE", "MAE", "MAPE", "MedAPE", "Bias_clean", "gain_MAE"]]
      .round(3).to_string())
print(f"\nPhase 2b ridge baseline MAE: ${BASELINE_MAE:.2f}")

# --------------------------------------------------------------------------
# 2. Time features: which ones survive extrapolation?
# --------------------------------------------------------------------------
section("2. TIME FEATURES UNDER EXTRAPOLATION")
print("Annual Fourier terms are the textbook way to model seasonality. Measured")
print("here, they are actively harmful, so the default is a bare linear trend.\n")
TIME_VARIANTS = [
    ("no time features", FeatureConfig(time=False, geo=False, lane=False, interactions=False)),
    ("trend only", FeatureConfig(time=True, geo=False, lane=False, interactions=False)),
    ("trend + day-of-week", FeatureConfig(time=True, geo=False, lane=False,
                                          interactions=False, day_of_week=True)),
    ("trend + fourier", FeatureConfig(time=True, geo=False, lane=False,
                                      interactions=False, fourier=True)),
    ("trend + fourier + dow", FeatureConfig(time=True, geo=False, lane=False,
                                            interactions=False, fourier=True,
                                            day_of_week=True)),
]
time_rows = {name: run(cfg, model="ridge") for name, cfg in TIME_VARIANTS}
print(pd.DataFrame(time_rows).T[["n_features", "MAE", "MAPE", "MAE_clean", "Bias_clean"]]
      .round(3).to_string())

days = (fit["date"] - pd.Timestamp("2025-01-01")).dt.days.to_numpy(float)
angle = 2 * np.pi * days / 365.25
print("\nwhy: inside an 8-month window the harmonics are near-collinear with the trend")
for label, values in [("annual_sin", np.sin(angle)), ("annual_cos", np.cos(angle))]:
    print(f"  corr(trend_days, {label}) = {np.corrcoef(days, values)[0, 1]:+.3f}")
print("  ridge splits the coefficient between them; past the window edge they")
print("  diverge and the extrapolation breaks. Same trap awaits Phase 5.")

# --------------------------------------------------------------------------
# 3. Leave-one-group-out - does each group carry its own weight?
# --------------------------------------------------------------------------
section("3. LEAVE-ONE-GROUP-OUT (from the full set)")
full = FeatureConfig()
full_metrics = run(full, model="ridge")
print(f"full set                    MAE ${full_metrics['MAE']:8.2f}")
for group in ("time", "geo", "lane", "interactions"):
    config = FeatureConfig(**{**full.__dict__, group: False})
    m = run(config, model="ridge")
    print(f"  without {group:14s}    MAE ${m['MAE']:8.2f}   "
          f"cost of removing: ${m['MAE'] - full_metrics['MAE']:+7.2f}")

# --------------------------------------------------------------------------
# 3. The drift fix: recency weighting
# --------------------------------------------------------------------------
section("4. RECENCY WEIGHTING vs BIAS (full feature set, ridge)")
print("Phase 2b left a -$26.60 bias on the clean holdout: a model fitted on")
print("Jan-Aug under-prices Sep-Oct because rates drift upward.\n")
weight_rows = {}
for half_life in [None, 365, 180, 120, 90, 60, 45, 30, 21]:
    label = "uniform" if half_life is None else f"half-life {half_life}d"
    weight_rows[label] = run(full, half_life=half_life, model="ridge")
weighting = pd.DataFrame(weight_rows).T
print(weighting[["RMSE", "MAE", "MAPE", "MedAPE", "MAE_clean", "Bias_clean"]]
      .round(3).to_string())

best_bias = weighting["Bias_clean"].abs().idxmin()
best_mae = weighting["MAE"].idxmin()
print(f"\nsmallest |bias| : {best_bias}  (${weighting.loc[best_bias, 'Bias_clean']:+.2f})")
print(f"smallest MAE    : {best_mae}  (${weighting.loc[best_mae, 'MAE']:.2f})")

# --------------------------------------------------------------------------
# 4. Does the time trend alone fix the bias, without down-weighting data?
# --------------------------------------------------------------------------
section("5. TREND FEATURES vs RECENCY WEIGHTING (isolating the mechanism)")
no_time = FeatureConfig(time=False)
print(f"{'time features off, uniform weights':42s} "
      f"bias ${run(no_time)['Bias_clean']:+8.2f}")
print(f"{'time features on,  uniform weights':42s} "
      f"bias ${run(full)['Bias_clean']:+8.2f}")
print(f"{'time features off, half-life 60d':42s} "
      f"bias ${run(no_time, half_life=60)['Bias_clean']:+8.2f}")
print(f"{'time features on,  half-life 60d':42s} "
      f"bias ${run(full, half_life=60)['Bias_clean']:+8.2f}")

# --------------------------------------------------------------------------
# 5. LightGBM on the same feature sets
# --------------------------------------------------------------------------
if lgb is not None:
    section("6. LIGHTGBM (same features, for comparison)")
    lgbm_rows = {}
    for label, half_life in [("uniform", None), ("half-life 90d", 90), ("half-life 60d", 60)]:
        lgbm_rows[f"lgbm {label}"] = run(full, half_life=half_life, model="lgbm")
    lgbm_rows["ridge best"] = weighting.loc[best_mae].to_dict()
    print(pd.DataFrame(lgbm_rows).T[["RMSE", "MAE", "MAPE", "MedAPE", "MAE_clean", "Bias_clean"]]
          .round(3).to_string())

section("SUMMARY")
print(f"Phase 2b baseline (ridge, base features)  MAE ${BASELINE_MAE:.2f}")
print(f"Phase 3  (full features, uniform weights) MAE ${full_metrics['MAE']:.2f}")
print(f"Phase 3  (full features, best weighting)  MAE ${weighting.loc[best_mae, 'MAE']:.2f}")

