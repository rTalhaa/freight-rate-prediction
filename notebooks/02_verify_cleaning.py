"""
Phase 2 - verify the cleaning contract holds.

Run:  python notebooks/02_verify_cleaning.py

Checks the cleaned outputs against the defect counts recorded in
docs/findings.md, and asserts the leakage rules: cleaning parameters come from
the training window only, and corrupt labels are removed from training folds but
left in the holdout.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import (  # noqa: E402
    SPLIT_DATE,
    clean,
    corrupt_label_mask,
    fit_cleaning,
    load_all,
    prepare,
    temporal_split,
)

PASS, FAIL = "  ok   ", "  FAIL "
failures = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"{PASS if condition else FAIL} {label}{'  ' + detail if detail else ''}")
    if not condition:
        failures.append(label)


def section(title: str) -> None:
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


train_raw, valid_raw, december_raw = load_all()
bundle = prepare()
stats = bundle["stats"]

section("0. FITTED PARAMETERS")
print(stats.describe())

section("1. DEFECTS PRESENT IN RAW DATA (baseline from docs/findings.md)")
print(f"  train  negative weight {(train_raw.weight < 0).sum():4d} | "
      f"null weight {train_raw.weight.isna().sum():4d} | "
      f"null market_index {train_raw.market_index.isna().sum():4d}")
print(f"  valid  negative weight {(valid_raw.weight < 0).sum():4d} | "
      f"null weight {valid_raw.weight.isna().sum():4d} | "
      f"null market_index {valid_raw.market_index.isna().sum():4d}")

section("2. CLEANED OUTPUTS ARE DEFECT-FREE")
for name in ["fit", "holdout", "full", "validation"]:
    df = bundle[name]
    check(f"{name:11s} no negative weight", (df.weight < 0).sum() == 0)
    check(f"{name:11s} no null weight    ", df.weight.isna().sum() == 0)
    check(f"{name:11s} no null market_idx", df.market_index.isna().sum() == 0)
    check(f"{name:11s} circuity capped   ",
          df.circuity.max() <= stats.circuity_cap + 1e-9,
          f"max {df.circuity.max():.3f} <= {stats.circuity_cap:.3f}")

section("3. FLAGS PRESERVE THE DEFECT COUNTS")
valid_clean = bundle["validation"]
check("validation weight_missing matches raw nulls",
      valid_clean.weight_missing.sum() == valid_raw.weight.isna().sum(),
      f"{valid_clean.weight_missing.sum()} == {valid_raw.weight.isna().sum()}")
check("validation market_missing matches raw nulls",
      valid_clean.market_missing.sum() == valid_raw.market_index.isna().sum(),
      f"{valid_clean.market_missing.sum()} == {valid_raw.market_index.isna().sum()}")
check("validation weight_at_cap ~2.5%",
      0.02 < valid_clean.weight_at_cap.mean() < 0.03,
      f"{valid_clean.weight_at_cap.mean():.2%}")
check("validation distance_at_floor == 21",
      valid_clean.distance_at_floor.sum() == 21,
      f"{valid_clean.distance_at_floor.sum()}")

section("4. SIGN FLIPS REPAIRED, NOT DROPPED")
neg_idx = valid_raw.index[valid_raw.weight < 0]
repaired = valid_clean.loc[neg_idx, "weight"]
check("all sign-flipped rows retained", len(repaired) == len(neg_idx))
check("repaired weights equal absolute values",
      np.allclose(repaired.values, valid_raw.loc[neg_idx, "weight"].abs().values))
check("no rows lost in validation cleaning", len(valid_clean) == len(valid_raw),
      f"{len(valid_clean)} == {len(valid_raw)}")

section("5. LABEL HYGIENE: TRAIN CLEANED, HOLDOUT LEFT ALONE")
fit_raw, holdout_raw = temporal_split(train_raw)
n_corrupt_fit = corrupt_label_mask(fit_raw).sum()
n_corrupt_holdout = corrupt_label_mask(holdout_raw).sum()
print(f"  corrupt labels in raw fit window     : {n_corrupt_fit}")
print(f"  corrupt labels in raw holdout window : {n_corrupt_holdout}")
check("fit window corrupt labels removed",
      corrupt_label_mask(bundle["fit"]).sum() == 0)
check("holdout corrupt labels RETAINED",
      corrupt_label_mask(bundle["holdout"]).sum() == n_corrupt_holdout,
      f"{corrupt_label_mask(bundle['holdout']).sum()} kept")
check("fit row count = raw minus corrupt",
      len(bundle["fit"]) == len(fit_raw) - n_corrupt_fit,
      f"{len(fit_raw)} - {n_corrupt_fit} = {len(bundle['fit'])}")
check("total dropped across full train ~1.39%",
      0.013 < corrupt_label_mask(train_raw).mean() < 0.015,
      f"{corrupt_label_mask(train_raw).mean():.2%}")

section("6. NO LEAKAGE ACROSS THE SPLIT BOUNDARY")
check("fit window ends at split date",
      bundle["fit"].date.max() <= SPLIT_DATE, str(bundle["fit"].date.max().date()))
check("holdout starts after split date",
      bundle["holdout"].date.min() > SPLIT_DATE, str(bundle["holdout"].date.min().date()))
check("no date overlap",
      set(bundle["fit"].date) & set(bundle["holdout"].date) == set())
stats_holdout_free = fit_cleaning(fit_raw)
check("cleaning params identical when holdout is withheld",
      stats_holdout_free.describe() == stats.describe())

section("7. HOLDOUT SIZE AND MARKET REGIME")
h = bundle["holdout"]
print(f"  fit     {len(bundle['fit']):6,} rows  "
      f"{bundle['fit'].date.min().date()} -> {bundle['fit'].date.max().date()}")
print(f"  holdout {len(h):6,} rows  {h.date.min().date()} -> {h.date.max().date()}")
print(f"\n  market_index  fit {bundle['fit'].market_index.mean():.3f} | "
      f"holdout {h.market_index.mean():.3f} | "
      f"validation {bundle['validation'].market_index.mean():.3f}")
check("holdout market regime resembles validation",
      abs(h.market_index.mean() - bundle["validation"].market_index.mean()) < 0.05,
      "the reason for a late-window split")

section("8. DECEMBER FILE SURVIVES THE SHARED TRANSFORM")
dec = bundle["december"]
check("31 rows preserved", len(dec) == 31)
check("weight flags created", "weight_at_cap" in dec.columns)
check("geography skipped (no lat/lon supplied)", "circuity" not in dec.columns)
check("market_index absent, to be imputed in Phase 5", "market_index" not in dec.columns)
print("  columns:", list(dec.columns))

section("RESULT")
if failures:
    print(f"{len(failures)} check(s) failed:")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("All checks passed.")
