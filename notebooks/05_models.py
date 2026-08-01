"""
Phase 4 - model selection under rolling-origin CV.

Run:  python notebooks/05_models.py

Phase 3 showed LightGBM losing to ridge because trees cannot extrapolate the
trend. This tests whether splitting the job - ridge for structure, trees for the
residual - recovers the non-linear gain without asking trees to extrapolate.

Selection uses rolling folds, not the single Sep-Oct holdout, so a $2 difference
can be told apart from noise.
"""
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import clean, fit_cleaning, load_raw, temporal_split  # noqa: E402
from src.features import FeatureConfig  # noqa: E402
from src.model import HybridModel, LGBMModel, RidgeModel  # noqa: E402
from src.validation import compare, cross_validate, make_folds  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option("display.width", 200)


def section(title: str) -> None:
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


# Cleaning is fitted on the earliest window only, so no fold's test months
# influence the imputation values used anywhere in this experiment.
train_raw = load_raw("train_test.csv")
early, _ = temporal_split(train_raw)
development = clean(train_raw, fit_cleaning(early))

folds = make_folds(development, horizon_days=61, n_folds=4)
section("FOLDS (expanding window, 61-day horizon)")
for fold in folds:
    print(f"  train <= {fold.train_end.date()}   test {fold.name}")

CONFIG = FeatureConfig()  # measured best: base + time + geo + lane

# --------------------------------------------------------------------------
section("1. MODEL COMPARISON (fold-averaged)")
models = {
    "ridge": lambda: RidgeModel(config=CONFIG),
    "lgbm": lambda: LGBMModel(config=CONFIG),
    "hybrid (shrink 1.0)": lambda: HybridModel(config=CONFIG, shrinkage=1.0),
    "hybrid (shrink 0.5)": lambda: HybridModel(config=CONFIG, shrinkage=0.5),
    "hybrid (shrink 0.3)": lambda: HybridModel(config=CONFIG, shrinkage=0.3),
}
summary = compare(models, development, folds)
print(summary[["RMSE", "MAE", "MAPE", "MedAPE", "MAE_clean", "Bias_clean"]].round(3).to_string())

# --------------------------------------------------------------------------
section("2. PER-FOLD DETAIL FOR THE TOP TWO")
for name in list(summary.index[:2]):
    print(f"\n{name}")
    scores = cross_validate(models[name], development, folds)
    print(scores[["MAE", "MAPE", "MedAPE", "MAE_clean", "Bias_clean", "n_train"]]
          .round(3).to_string())

# --------------------------------------------------------------------------
section("3. IS THE WINNER CONSISTENT, OR AVERAGING A LUCKY FOLD?")
best, runner_up = summary.index[0], summary.index[1]
a = cross_validate(models[best], development, folds).drop(index="MEAN")["MAE"]
b = cross_validate(models[runner_up], development, folds).drop(index="MEAN")["MAE"]
delta = b - a
print(f"{best} vs {runner_up}, MAE difference per fold (positive = winner is better):")
for fold_name, value in delta.items():
    print(f"  {fold_name}  {value:+8.2f}")
print(f"\nwins {int((delta > 0).sum())} of {len(delta)} folds | "
      f"mean {delta.mean():+.2f} | std {delta.std():.2f}")
if (delta > 0).all():
    print("consistent across every fold.")
else:
    print("not consistent - the averaged difference is inside fold-to-fold noise.")

# --------------------------------------------------------------------------
section("4. RIDGE ALPHA SWEEP (best model's linear stage)")
for alpha in [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]:
    factory = (lambda a=alpha: RidgeModel(alpha=a, config=CONFIG))
    scores = cross_validate(factory, development, folds).loc["MEAN"]
    print(f"  alpha {alpha:5.1f}   MAE {scores['MAE']:8.3f}   MAPE {scores['MAPE']:6.3f}   "
          f"bias {scores['Bias_clean']:+8.2f}")

# --------------------------------------------------------------------------
section("5. HOLDOUT CHECK (Sep-Oct, comparable to Phase 3)")
print("Phase 3 best (ridge, single holdout): MAE $106.99\n")
fit_part, holdout = temporal_split(development)
from src.data import drop_corrupt_labels  # noqa: E402
from src.features import fit_feature_stats  # noqa: E402
from src.metrics import evaluate  # noqa: E402

fit_part = drop_corrupt_labels(fit_part)
stats = fit_feature_stats(fit_part)
for name, factory in models.items():
    model = factory().fit(fit_part, stats)
    metrics = evaluate(holdout["posted_rate"].to_numpy(), model.predict(holdout))
    print(f"  {name:22s} MAE {metrics['MAE']:8.2f}  MAPE {metrics['MAPE']:6.3f}  "
          f"MedAPE {metrics['MedAPE']:6.3f}")
