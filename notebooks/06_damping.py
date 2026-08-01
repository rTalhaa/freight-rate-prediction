"""
Phase 4b - damping the trend extrapolation.

Run:  python notebooks/06_damping.py

Phase 4 left per-fold bias at +$19, +$84, -$30. Rate-per-mile rose through June
and then fell, so a linear trend projected forward overshoots when the market
turns. Validation starts two months past the end of training, further out than
any fold tested, and the December chart varies nothing but the date.

This sweeps the damping factor and measures how error grows with horizon.

Selection is deliberately not on mean MAE. The failure is variance across
regimes, and the mean is dominated by folds where the trend happened to point
the right way. Worst-fold MAE and max |bias| are the criteria.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import (  # noqa: E402
    clean,
    corrupt_label_mask,
    drop_corrupt_labels,
    fit_cleaning,
    load_raw,
    temporal_split,
)
from src.features import FeatureConfig, fit_feature_stats  # noqa: E402
from src.metrics import evaluate  # noqa: E402
from src.model import HybridModel  # noqa: E402
from src.validation import cross_validate, make_folds  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option("display.width", 220)


def section(title: str) -> None:
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


train_raw = load_raw("train_test.csv")
early, _ = temporal_split(train_raw)
development = clean(train_raw, fit_cleaning(early))
folds = make_folds(development, horizon_days=61, n_folds=4)

DAMPING_GRID = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]


def factory(damping: float):
    return lambda: HybridModel(config=FeatureConfig(trend_damping=damping), shrinkage=1.0)


# --------------------------------------------------------------------------
section("1. DAMPING SWEEP (hybrid, rolling folds)")
rows = {}
per_fold_mae, per_fold_bias = {}, {}
for damping in DAMPING_GRID:
    scores = cross_validate(factory(damping), development, folds)
    folds_only = scores.drop(index="MEAN")
    rows[f"phi {damping:.1f}"] = {
        "mean_MAE": scores.loc["MEAN", "MAE"],
        "worst_MAE": folds_only["MAE"].max(),
        "mean_MAPE": scores.loc["MEAN", "MAPE"],
        "mean_absBias": folds_only["Bias_clean"].abs().mean(),
        "max_absBias": folds_only["Bias_clean"].abs().max(),
        "MAE_spread": folds_only["MAE"].max() - folds_only["MAE"].min(),
    }
    per_fold_mae[f"phi {damping:.1f}"] = folds_only["MAE"]
    per_fold_bias[f"phi {damping:.1f}"] = folds_only["Bias_clean"]

sweep = pd.DataFrame(rows).T
print(sweep.round(3).to_string())

print("\nbest by each criterion:")
for column in ["mean_MAE", "worst_MAE", "max_absBias", "MAE_spread"]:
    print(f"  {column:14s} -> {sweep[column].idxmin():9s} ({sweep[column].min():.2f})")

# --------------------------------------------------------------------------
section("2. PER-FOLD DETAIL")
print("MAE by fold:")
print(pd.DataFrame(per_fold_mae).T.round(2).to_string())
print("\nbias (clean) by fold  <- the volatility that damping targets:")
print(pd.DataFrame(per_fold_bias).T.round(2).to_string())

# --------------------------------------------------------------------------
section("3. BIAS vs HORIZON")
print("How fast does error grow with distance past the end of training?")
print("Validation starts 1 day past training and runs 61 days; the December")
print("chart sits 31-61 days in.\n")

BUCKETS = [(1, 15), (16, 30), (31, 45), (46, 61)]


def horizon_profile(damping: float) -> pd.DataFrame:
    collected = {b: [] for b in BUCKETS}
    for fold in folds:
        train = drop_corrupt_labels(development.loc[development["date"] <= fold.train_end])
        test = development.loc[(development["date"] >= fold.test_start)
                               & (development["date"] <= fold.test_end)]
        model = HybridModel(config=FeatureConfig(trend_damping=damping),
                            shrinkage=1.0).fit(train, fit_feature_stats(train))
        prediction = model.predict(test)
        actual = test["posted_rate"].to_numpy()
        keep = ~corrupt_label_mask(test).to_numpy()
        horizon = (test["date"] - fold.train_end).dt.days.to_numpy()
        for low, high in BUCKETS:
            in_bucket = (horizon >= low) & (horizon <= high) & keep
            if in_bucket.sum():
                metrics = evaluate(actual[in_bucket], prediction[in_bucket])
                collected[(low, high)].append((metrics["Bias"], metrics["MAE"]))
    return pd.DataFrame({
        f"{low}-{high}d": {
            "bias": np.mean([v[0] for v in collected[(low, high)]]),
            "abs_bias": np.mean([abs(v[0]) for v in collected[(low, high)]]),
            "MAE": np.mean([v[1] for v in collected[(low, high)]]),
        }
        for low, high in BUCKETS
    })


for damping in [1.0, 0.5, 0.3, 0.0]:
    print(f"\ndamping phi = {damping:.1f}")
    print(horizon_profile(damping).round(2).to_string())

# --------------------------------------------------------------------------
section("4. SELECTION")
print("Chosen on robustness, not on the mean. Candidates within noise of each")
print("other resolve toward the more damped option, because over-committing to a")
print("trend that has turned is the expensive error.\n")
print(sweep[["mean_MAE", "worst_MAE", "max_absBias", "MAE_spread"]].round(2).to_string())
