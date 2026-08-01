"""
Phase 5 - December chart inputs, and auditing the forecast that produces them.

Run:  python notebooks/07_december.py

The chart file supplies only seven columns. Coordinates are looked up (a static
city property). market_index and quote_signal are forecast from training data
only, because a deployed quoting model pricing a future date would not know
them.

validation.csv does contain those values for December. They are used here to
score the forecast after the fact, never to produce it.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import clean, fit_cleaning, load_raw, prepare  # noqa: E402
from src.december import enrich_december, fit_market_forecast  # noqa: E402
from src.features import best_config, fit_feature_stats  # noqa: E402
from src.model import HybridModel  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option("display.width", 200)


def section(title: str) -> None:
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


bundle = prepare()
train_full = bundle["full"]
train_raw = load_raw("train_test.csv")
# Raw, not bundle["december"]: enrichment has to happen before cleaning so the
# forecast market values pass through the same transform as every other row.
december_raw = load_raw("december_chart_inputs.csv")

# --------------------------------------------------------------------------
section("1. MARKET FORECAST (fitted on training data only)")
forecast = fit_market_forecast(train_full)
print(forecast.describe())
print("\nWindow lengths come from backtesting the last 61 days of training.")
print("A trailing mean, not an extrapolated slope: three separate measurements")
print("in this project have shown projecting a fitted trend forward adds")
print("variance without adding information.")

# --------------------------------------------------------------------------
section("2. AUDIT: forecast vs the values withheld in validation.csv")
print("These actuals were not used to build the forecast.\n")
validation_raw = load_raw("validation.csv")
december_actual = validation_raw[validation_raw.date.dt.month == 12]

predicted = forecast.predict(pd.date_range("2025-12-01", "2025-12-31"))
for column in ["market_index", "quote_signal"]:
    actual = december_actual.groupby("date")[column].mean()
    pred = predicted[column]
    error = pred.to_numpy() - actual.to_numpy()
    print(f"{column}")
    print(f"  forecast mean {pred.mean():.4f}  vs  actual mean {actual.mean():.4f}   "
          f"({(pred.mean() / actual.mean() - 1):+.2%})")
    print(f"  MAE {np.mean(np.abs(error)):.4f} | RMSE {np.sqrt(np.mean(error ** 2)):.4f} | "
          f"max |error| {np.max(np.abs(error)):.4f}")
    print(f"  actual range [{actual.min():.4f}, {actual.max():.4f}] | "
          f"forecast range [{pred.min():.4f}, {pred.max():.4f}]")
    naive = float(train_full[column].mean())
    naive_mae = np.mean(np.abs(naive - actual.to_numpy()))
    print(f"  vs naive whole-training-mean ({naive:.4f}): MAE {naive_mae:.4f}   "
          f"-> {1 - np.mean(np.abs(error)) / naive_mae:+.1%} better\n")

# --------------------------------------------------------------------------
section("3. FINAL MODEL (trained on all of Jan-Oct)")
config = best_config()
print(f"config: {config}")
feature_stats = fit_feature_stats(train_full)
model = HybridModel(config=config, shrinkage=1.0).fit(train_full, feature_stats)
print(f"trained on {len(train_full):,} rows, "
      f"{train_full.date.min().date()} -> {train_full.date.max().date()}")

# --------------------------------------------------------------------------
section("4. DECEMBER PREDICTIONS")
december = enrich_december(december_raw, train_raw, forecast)
december = clean(december, fit_cleaning(train_raw))
december["predicted_rate"] = model.predict(december)

view = december[["date", "market_index", "quote_signal", "predicted_rate"]].copy()
view["dow"] = view["date"].dt.day_name().str[:3]
print(view.assign(date=view.date.dt.date).round(4).to_string(index=False))

# --------------------------------------------------------------------------
section("5. SANITY CHECK AGAINST THE HISTORICAL ANCHOR")
lane = train_raw[(train_raw.pickup == "Lexington") & (train_raw.delivery == "Fort Wayne")]
rate = december["predicted_rate"]
print(f"historical Lexington -> Fort Wayne: {len(lane)} loads, "
      f"mean ${lane.posted_rate.mean():,.2f}, rpm {(lane.posted_rate / lane.distance).mean():.3f}")
print(f"predicted December:  mean ${rate.mean():,.2f}, "
      f"range ${rate.min():,.2f} - ${rate.max():,.2f}, "
      f"rpm {(rate / 360).mean():.3f}")
print(f"\nspread across the month: ${rate.max() - rate.min():,.2f} "
      f"({(rate.max() / rate.min() - 1):.1%})")

anchor_low, anchor_high = 800, 1000
inside = ((rate >= anchor_low) & (rate <= anchor_high)).mean()
print(f"share of days inside the ${anchor_low}-${anchor_high} anchor band: {inside:.0%}")
if rate.min() <= 0:
    print("FAIL: non-positive predictions")
elif inside < 0.5:
    print("WARNING: most predictions sit outside the historical band - investigate")
else:
    print("OK: predictions sit in the historically plausible range")

# --------------------------------------------------------------------------
section("6. WHAT DRIVES THE SHAPE OF THE CHART?")
print("Only the date varies, so all movement comes from the market forecast.\n")
print("correlation of predicted_rate with:")
for column in ["market_index", "quote_signal"]:
    print(f"  {column:13s} {december['predicted_rate'].corr(december[column]):+.3f}")
print("\nmean predicted rate by weekday:")
by_dow = december.assign(dow=december.date.dt.day_name()).groupby("dow")["predicted_rate"]
order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
print(by_dow.mean().reindex(order).round(2).to_string())
print("\nThe weekly cycle in market_index (lag-7 autocorrelation 0.97 in training)")
print("is what gives the chart its shape. Without it the line would be flat,")
print("since the trend is deliberately not extrapolated.")

december.to_csv(Path(__file__).resolve().parents[1] / "data" / "december_predicted.csv",
                index=False)
print("\nwrote data/december_predicted.csv (working file; the submission copy is")
print("written by notebooks/08_predict.py)")
