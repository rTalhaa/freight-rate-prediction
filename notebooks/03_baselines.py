"""
Phase 2b - baselines.

Run:  python notebooks/03_baselines.py

Establishes what a naive model achieves before any gradient boosting, so the
final model has something to be measured against. Everything is scored on the
Sep-Oct holdout, which is left contaminated on purpose.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import corrupt_label_mask, prepare  # noqa: E402
from src.metrics import evaluate, scoreboard, smearing_factor  # noqa: E402


def section(title: str) -> None:
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


bundle = prepare()
fit, holdout = bundle["fit"], bundle["holdout"]

section("SETUP")
print(f"fit     {len(fit):6,} rows  {fit.date.min().date()} -> {fit.date.max().date()}"
      "   (corrupt labels removed)")
print(f"holdout {len(holdout):6,} rows  {holdout.date.min().date()} -> "
      f"{holdout.date.max().date()}   (left contaminated)")

y_holdout = holdout["posted_rate"].to_numpy()
clean_mask = ~corrupt_label_mask(holdout).to_numpy()
print(f"\nholdout contains {(~clean_mask).sum()} corrupt rows ({(~clean_mask).mean():.2%}). "
      "Scores are reported on\nthe full holdout; the clean subset is shown separately to "
      "size their drag.")

results, results_clean = {}, {}


def record(name: str, prediction) -> None:
    prediction = np.clip(np.asarray(prediction, dtype=float), 1.0, None)
    results[name] = evaluate(y_holdout, prediction)
    results_clean[name] = evaluate(y_holdout[clean_mask], prediction[clean_mask])


# --------------------------------------------------------------------------
# 0. Global mean - the floor any model must beat
# --------------------------------------------------------------------------
record("0. global mean", np.full(len(holdout), fit.posted_rate.mean()))

# --------------------------------------------------------------------------
# 1. Flat rate per mile
# --------------------------------------------------------------------------
flat_rpm = (fit.posted_rate / fit.distance).mean()
print(f"\nflat rate-per-mile fitted on training window: ${flat_rpm:.3f}/mi")
record("1. flat $/mile", flat_rpm * holdout.distance)

# --------------------------------------------------------------------------
# 2. Rate per mile by distance band - captures economies of scale
# --------------------------------------------------------------------------
BANDS = [0, 250, 500, 750, 1000, 1500, 2000, 3000, 10_000]
fit_band = pd.cut(fit.distance, BANDS)
band_rpm = (fit.posted_rate / fit.distance).groupby(fit_band, observed=True).mean()
print("\nrate-per-mile by distance band:")
print(band_rpm.round(3).to_string())
holdout_band = pd.cut(holdout.distance, BANDS)
record("2. $/mile by distance band", holdout_band.map(band_rpm).astype(float) * holdout.distance)

# --------------------------------------------------------------------------
# 3. Rate per mile by distance band x equipment
# --------------------------------------------------------------------------
band_eq_rpm = ((fit.posted_rate / fit.distance)
               .groupby([fit_band, fit.equipment], observed=True).mean())
holdout_key = pd.MultiIndex.from_arrays([holdout_band, holdout.equipment])
# Unseen band/equipment combinations fall back to the flat rate.
band_eq_lookup = band_eq_rpm.reindex(holdout_key).to_numpy(dtype=float)
record("3. $/mile by band x equipment",
       np.where(np.isnan(band_eq_lookup), flat_rpm, band_eq_lookup) * holdout.distance)

# --------------------------------------------------------------------------
# 4. Ridge on log(rate)
# --------------------------------------------------------------------------
NUMERIC = ["distance", "weight", "market_index", "quote_signal", "haversine", "circuity",
           "weight_missing", "weight_at_cap", "distance_at_floor", "market_missing"]


def design(frame: pd.DataFrame, encoder: OneHotEncoder, fitting: bool) -> np.ndarray:
    numeric = frame[NUMERIC].to_numpy(dtype=float)
    # log-distance matters: rate is closer to linear in log space, and it is the
    # dominant feature by a wide margin.
    extra = np.column_stack([
        np.log(frame["distance"].to_numpy(dtype=float)),
        frame["date"].dt.dayofyear.to_numpy(dtype=float),
    ])
    cats = frame[["equipment"]]
    encoded = encoder.fit_transform(cats) if fitting else encoder.transform(cats)
    return np.column_stack([numeric, extra, encoded])


encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_fit = design(fit, encoder, fitting=True)
X_holdout = design(holdout, encoder, fitting=False)

# Standardise using training statistics only.
mu, sigma = X_fit.mean(axis=0), X_fit.std(axis=0)
sigma[sigma == 0] = 1.0
X_fit_s, X_holdout_s = (X_fit - mu) / sigma, (X_holdout - mu) / sigma

y_log = np.log(fit["posted_rate"].to_numpy())
ridge = Ridge(alpha=1.0).fit(X_fit_s, y_log)

pred_log = ridge.predict(X_holdout_s)
record("4. ridge on log(rate)", np.exp(pred_log))

# Duan smearing: exponentiating a log-scale fit predicts the median, not the
# mean, so predictions come in systematically low. Correct it explicitly.
smear = smearing_factor(y_log - ridge.predict(X_fit_s))
print(f"\nDuan smearing factor: {smear:.4f}")
record("5. ridge + smearing", np.exp(pred_log) * smear)

# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
section("HOLDOUT SCORES (Sep-Oct, contaminated - the honest number)")
print(scoreboard(results).round(3).to_string())

section("HOLDOUT SCORES (clean subset only - for reference)")
print(scoreboard(results_clean).round(3).to_string())

section("READING THE GAP")
best = scoreboard(results).index[0]
full_mae = results[best]["MAE"]
clean_mae = results_clean[best]["MAE"]
print(f"best baseline: {best}")
print(f"  MAE on full holdout  ${full_mae:,.2f}")
print(f"  MAE on clean subset  ${clean_mae:,.2f}")
print(f"  drag from {(~clean_mask).sum()} corrupt rows: ${full_mae - clean_mae:,.2f} "
      f"({full_mae / clean_mae - 1:+.1%})")
print("\n1.5% of rows carrying that much error is why the metric family matters:")
print("RMSE absorbs most of the damage, MedAPE almost none.")
