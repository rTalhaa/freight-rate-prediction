"""
Generate the submission files.

Run:  python -m src.predict

Produces:
  validation_predictions.csv        12,000 rows of load_id,predicted_rate
  data/december_chart_inputs.csv    the provided file with predicted_rate filled

Trains on all of Jan-Oct and predicts Nov-Dec, a 1-61 day forward horizon. The
model, the feature set and the damping factor are the ones selected in docs/.
Every output is checked against the scorer's requirements before being written,
so a malformed file fails here rather than in score.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):  # allow `python src/predict.py` as well
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.data import clean, fit_cleaning, load_raw, prepare
    from src.december import enrich_december, fit_market_forecast
    from src.features import best_config, fit_feature_stats
    from src.model import HybridModel
else:
    from .data import clean, fit_cleaning, load_raw, prepare
    from .december import enrich_december, fit_market_forecast
    from .features import best_config, fit_feature_stats
    from .model import HybridModel

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

EXPECTED_ROWS = 12_000
DECEMBER_COLUMNS = ["pickup", "delivery", "distance", "equipment", "weight",
                    "date", "predicted_rate"]


def step(message: str) -> None:
    print(f"\n>>> {message}")


def check(label: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise SystemExit(f"FAILED: {label} {detail}")
    print(f"  ok   {label}{'  ' + detail if detail else ''}")


def main() -> None:
    bundle = prepare()
    train_full, validation = bundle["full"], bundle["validation"]
    train_raw = load_raw("train_test.csv")

    step("Training final model")
    model = HybridModel(config=best_config(), shrinkage=1.0).fit(
        train_full, fit_feature_stats(train_full))
    print(f"  hybrid (ridge + LightGBM residual), damping 0.0, shrinkage 1.0")
    print(f"  {len(train_full):,} clean rows, {train_full.date.min().date()} -> "
          f"{train_full.date.max().date()}")

    # ---------------------------------------------------------------- validation
    step("Predicting validation.csv")
    predicted = model.predict(validation)
    predictions = pd.DataFrame({
        "load_id": validation["load_id"].to_numpy(),
        "predicted_rate": np.round(predicted, 2),
    })

    template = pd.read_csv(DATA / "validation_predictions_template.csv")
    # Emit in the template's own order so the file is directly comparable to it.
    predictions = (template[["load_id"]]
                   .merge(predictions, on="load_id", how="left"))

    check("row count", len(predictions) == EXPECTED_ROWS, f"{len(predictions):,}")
    check("columns and order",
          list(predictions.columns) == ["load_id", "predicted_rate"])
    check("no missing predictions", predictions["predicted_rate"].notna().all())
    check("all finite", np.isfinite(predictions["predicted_rate"]).all())
    check("all positive", (predictions["predicted_rate"] > 0).all(),
          f"min ${predictions['predicted_rate'].min():,.2f}")
    check("load_ids match template",
          predictions["load_id"].tolist() == template["load_id"].tolist())
    check("no duplicate load_ids", not predictions["load_id"].duplicated().any())

    output = ROOT / "validation_predictions.csv"
    predictions.to_csv(output, index=False)
    print(f"  wrote {output.name}")

    # ---------------------------------------------------------------- december
    step("Predicting December chart inputs")
    forecast = fit_market_forecast(train_full)
    december = enrich_december(load_raw("december_chart_inputs.csv"),
                               train_raw, forecast)
    december = clean(december, fit_cleaning(train_raw))
    december["predicted_rate"] = np.round(model.predict(december), 2)

    december_out = december[DECEMBER_COLUMNS].copy()
    december_out["date"] = december_out["date"].dt.strftime("%Y-%m-%d")
    # Restore the integer formatting the provided file shipped with.
    december_out["distance"] = december_out["distance"].astype(int)
    december_out["weight"] = december_out["weight"].astype(int)

    original = load_raw("december_chart_inputs.csv")
    check("row count", len(december_out) == 31, str(len(december_out)))
    check("columns and order", list(december_out.columns) == DECEMBER_COLUMNS)
    check("fixed inputs unchanged",
          (december_out["pickup"].eq("Lexington").all()
           and december_out["delivery"].eq("Fort Wayne").all()
           and december_out["distance"].eq(360).all()
           and december_out["equipment"].eq("Dry Van").all()
           and december_out["weight"].eq(32_000).all()))
    check("dates unchanged",
          december_out["date"].tolist()
          == original["date"].dt.strftime("%Y-%m-%d").tolist())
    check("all positive", (december_out["predicted_rate"] > 0).all())

    december_out.to_csv(DATA / "december_chart_inputs.csv", index=False)
    print(f"  wrote data/december_chart_inputs.csv")

    # ---------------------------------------------------------------- summary
    step("Summary")
    rate = predictions["predicted_rate"]
    actual_train = train_full["posted_rate"]
    print(f"  validation predictions   mean ${rate.mean():,.2f}  "
          f"median ${rate.median():,.2f}  range ${rate.min():,.2f}-${rate.max():,.2f}")
    print(f"  training actuals         mean ${actual_train.mean():,.2f}  "
          f"median ${actual_train.median():,.2f}")
    implied = (predictions["predicted_rate"].to_numpy()
               / validation["distance"].to_numpy())
    print(f"  implied rate per mile    mean {implied.mean():.3f}  "
          f"(training actual {(actual_train / train_full.distance).mean():.3f})")

    december_rate = december_out["predicted_rate"]
    print(f"\n  december chart           mean ${december_rate.mean():,.2f}  "
          f"range ${december_rate.min():,.2f}-${december_rate.max():,.2f}")
    lane = train_raw[(train_raw.pickup == "Lexington")
                     & (train_raw.delivery == "Fort Wayne")]
    print(f"  historical anchor        mean ${lane.posted_rate.mean():,.2f} "
          f"over {len(lane)} loads")

    print("\nNext: python score.py --predictions validation_predictions.csv "
          "--december-predictions data/december_chart_inputs.csv")


if __name__ == "__main__":
    main()
