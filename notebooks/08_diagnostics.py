"""
Phase 5b - diagnostics before generating the submission.

Run:  python notebooks/08_diagnostics.py

Three checks that should have happened earlier:

1. Where is the error? Every decision so far was made on aggregate metrics
   without asking which rows they are made of - in particular the 17.5% of
   validation lanes never seen in training, and the short hauls that dominate
   MAPE.

2. Do the decisions survive more folds? Model choice, damping and shrinkage all
   rest on three folds, and damping won at the edge of its grid.

3. Are the corrupt-label thresholds arbitrary? rpm < 1 and rpm > 4 were read off
   a plateau by eye and never tested.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data as data_module  # noqa: E402
from src.data import (  # noqa: E402
    clean,
    corrupt_label_mask,
    drop_corrupt_labels,
    fit_cleaning,
    load_raw,
    temporal_split,
)
from src.features import FeatureConfig, best_config, fit_feature_stats  # noqa: E402
from src.metrics import evaluate  # noqa: E402
from src.model import HybridModel, RidgeModel  # noqa: E402
from src.validation import cross_validate, make_folds  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option("display.width", 220)


def section(title: str) -> None:
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


train_raw = load_raw("train_test.csv")
early, _ = temporal_split(train_raw)
development = clean(train_raw, fit_cleaning(early))

fit_part, holdout = temporal_split(development)
fit_part = drop_corrupt_labels(fit_part)
feature_stats = fit_feature_stats(fit_part)

model = HybridModel(config=best_config(), shrinkage=1.0).fit(fit_part, feature_stats)
prediction = model.predict(holdout)
actual = holdout["posted_rate"].to_numpy()
keep = ~corrupt_label_mask(holdout).to_numpy()

overall = evaluate(actual[keep], prediction[keep])
print(f"Sep-Oct holdout, clean rows only: MAE ${overall['MAE']:.2f} | "
      f"MAPE {overall['MAPE']:.2f}% | MedAPE {overall['MedAPE']:.2f}%")


def by_segment(mask_series: pd.Series, label: str) -> pd.DataFrame:
    rows = {}
    for name, mask in mask_series.items():
        selected = mask & keep
        if selected.sum() < 30:
            continue
        metrics = evaluate(actual[selected], prediction[selected])
        rows[name] = {
            "n": int(selected.sum()),
            "MAE": metrics["MAE"],
            "MAPE": metrics["MAPE"],
            "MedAPE": metrics["MedAPE"],
            "Bias": metrics["Bias"],
            "mean_rate": float(actual[selected].mean()),
        }
    frame = pd.DataFrame(rows).T
    print(f"\n{label}")
    print(frame.round(2).to_string())
    return frame


# --------------------------------------------------------------------------
section("1. WHERE IS THE ERROR?")

lane_seen = holdout["lane"].isin(set(fit_part["lane"])).to_numpy()
by_segment(pd.Series({
    "lane seen in training": lane_seen,
    "lane NEVER seen": ~lane_seen,
}), "by lane familiarity")

bands = [(0, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 10_000)]
by_segment(pd.Series({
    f"{low}-{high} mi": ((holdout.distance > low) & (holdout.distance <= high)).to_numpy()
    for low, high in bands
}), "by distance band")

by_segment(pd.Series({
    equipment: (holdout.equipment == equipment).to_numpy()
    for equipment in sorted(holdout.equipment.unique())
}), "by equipment")

by_segment(pd.Series({
    "weight imputed": (holdout.weight_missing == 1).to_numpy(),
    "weight at cap": (holdout.weight_at_cap == 1).to_numpy(),
    "market imputed": (holdout.market_missing == 1).to_numpy(),
    "distance at floor": (holdout.distance_at_floor == 1).to_numpy(),
    "no flags": ((holdout.weight_missing == 0) & (holdout.weight_at_cap == 0)
                 & (holdout.market_missing == 0)).to_numpy(),
}), "by defect flag")

city_seen = (holdout["pickup"].isin(set(fit_part["pickup"]))
             & holdout["delivery"].isin(set(fit_part["delivery"]))).to_numpy()
by_segment(pd.Series({
    "both cities seen": city_seen,
    "a city unseen": ~city_seen,
}), "by city familiarity")

# --------------------------------------------------------------------------
section("2. DO THE DECISIONS SURVIVE MORE FOLDS?")
short_folds = make_folds(development, horizon_days=30, n_folds=8)
print(f"{len(short_folds)} folds at a 30-day horizon "
      f"(vs 3 folds at 61 days used for selection):")
for fold in short_folds:
    print(f"  train <= {fold.train_end.date()}   test {fold.name}")

print("\nmodel choice:")
candidates = {
    "ridge": lambda: RidgeModel(config=best_config()),
    "hybrid shrink 1.0": lambda: HybridModel(config=best_config(), shrinkage=1.0),
    "hybrid shrink 0.5": lambda: HybridModel(config=best_config(), shrinkage=0.5),
}
for name, factory in candidates.items():
    scores = cross_validate(factory, development, short_folds).drop(index="MEAN")
    print(f"  {name:20s} mean MAE {scores['MAE'].mean():7.2f} | "
          f"worst {scores['MAE'].max():7.2f} | std {scores['MAE'].std():6.2f}")

print("\ntrend damping (the decision that won at the edge of its grid):")
for damping in [1.0, 0.6, 0.3, 0.1, 0.0]:
    factory = (lambda d=damping: HybridModel(
        config=FeatureConfig(trend_damping=d), shrinkage=1.0))
    scores = cross_validate(factory, development, short_folds).drop(index="MEAN")
    print(f"  phi {damping:.1f}  mean MAE {scores['MAE'].mean():7.2f} | "
          f"worst {scores['MAE'].max():7.2f} | "
          f"max |bias| {scores['Bias_clean'].abs().max():7.2f}")

# --------------------------------------------------------------------------
section("3. ARE THE CORRUPT-LABEL THRESHOLDS ARBITRARY?")
print("Defaults are rpm < 1.0 and rpm > 4.0, read off the plateau in EDA.\n")
original_lower, original_upper = data_module.RPM_LOWER, data_module.RPM_UPPER
folds_61 = make_folds(development, horizon_days=61, n_folds=4)

for lower, upper in [(1.0, 3.0), (1.0, 3.5), (1.0, 4.0), (1.0, 4.5), (1.0, 5.0),
                     (0.8, 4.0), (1.2, 4.0), (0.0, np.inf)]:
    data_module.RPM_LOWER, data_module.RPM_UPPER = lower, upper
    dropped = corrupt_label_mask(train_raw).mean()
    scores = cross_validate(
        lambda: HybridModel(config=best_config(), shrinkage=1.0),
        development, folds_61,
    ).drop(index="MEAN")
    label = "no cleaning" if upper == np.inf else f"rpm in [{lower}, {upper}]"
    print(f"  {label:22s} drops {dropped:6.2%}  mean MAE {scores['MAE'].mean():7.2f}  "
          f"MedAPE {scores['MedAPE'].mean():5.2f}")

data_module.RPM_LOWER, data_module.RPM_UPPER = original_lower, original_upper
print(f"\nrestored defaults: [{data_module.RPM_LOWER}, {data_module.RPM_UPPER}]")

# --------------------------------------------------------------------------
section("4. UNSEEN GEOGRAPHY - THE CASE THE HOLDOUT CANNOT TEST")
print("validation.csv contains 8 cities and 17.5% of lanes never seen in")
print("training. The Sep-Oct holdout contains almost none, because both windows")
print("cover the same 64 cities. The temporal split reproduces the market regime")
print("faithfully but not this, so it is simulated directly: hold cities out of")
print("training, then predict loads that use them.\n")

cities = sorted(set(fit_part["pickup"]) | set(fit_part["delivery"]))
rng = np.random.default_rng(0)
penalties = []

for trial in range(3):
    held_out = set(rng.choice(cities, size=8, replace=False))
    touches = (fit_part["pickup"].isin(held_out) | fit_part["delivery"].isin(held_out))
    reduced = fit_part.loc[~touches]

    trial_model = HybridModel(config=best_config(), shrinkage=1.0).fit(
        reduced, fit_feature_stats(reduced))
    trial_prediction = trial_model.predict(holdout)

    affected = (holdout["pickup"].isin(held_out)
                | holdout["delivery"].isin(held_out)).to_numpy()
    unseen = evaluate(actual[affected & keep], trial_prediction[affected & keep])
    seen = evaluate(actual[~affected & keep], trial_prediction[~affected & keep])
    penalties.append((unseen["MAPE"], seen["MAPE"]))

    print(f"trial {trial + 1}: dropped {len(held_out)} cities "
          f"({(~touches).sum():,} of {len(fit_part):,} training rows kept)")
    print(f"  loads touching an unseen city  n={int((affected & keep).sum()):5d}  "
          f"MAE ${unseen['MAE']:7.2f}  MAPE {unseen['MAPE']:5.2f}%  "
          f"MedAPE {unseen['MedAPE']:5.2f}%")
    print(f"  loads on familiar cities       n={int((~affected & keep).sum()):5d}  "
          f"MAE ${seen['MAE']:7.2f}  MAPE {seen['MAPE']:5.2f}%  "
          f"MedAPE {seen['MedAPE']:5.2f}%")

unseen_mape = np.mean([p[0] for p in penalties])
seen_mape = np.mean([p[1] for p in penalties])
print(f"\nmean over trials: unseen {unseen_mape:.2f}% vs familiar {seen_mape:.2f}%"
      f"  -> penalty {unseen_mape / seen_mape - 1:+.1%}")
print("\nCoordinates carry the fallback: an unseen city still has a position, a")
print("distance and an equipment type, so the model degrades rather than fails.")
