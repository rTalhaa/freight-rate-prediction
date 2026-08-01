"""
Loading, cleaning and splitting.

One transform serves all three datasets (train, validation, the 31 December chart
rows). Cleaning parameters are *fitted* on training data only and then *applied*
elsewhere, so the train-only discipline is structural rather than a convention
someone has to remember.

Defects handled here are catalogued in docs/findings.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

EARTH_RADIUS_MILES = 3958.8

# Corrupt-label thresholds. Rate-per-mile outside this range is injected noise,
# not a real load - see docs/findings.md section 1 for the evidence.
RPM_LOWER = 1.0
RPM_UPPER = 4.0

# Holdout boundary: train on everything up to this date, validate after it.
# Mirrors the real task (fit on history, predict the next two months) and lands
# the holdout in the softer market regime that validation.csv occupies.
SPLIT_DATE = pd.Timestamp("2025-08-31")


# --------------------------------------------------------------------------
# Fitted cleaning parameters
# --------------------------------------------------------------------------
@dataclass
class CleaningStats:
    """Everything learned from training data and reused unchanged downstream."""

    weight_median_by_equipment: dict[str, float] = field(default_factory=dict)
    weight_median_global: float = np.nan
    market_index_median_global: float = np.nan
    weight_cap: float = np.nan
    distance_floor: float = np.nan
    circuity_cap: float = np.nan

    def describe(self) -> str:
        lines = [
            "CleaningStats (fitted on training rows only)",
            f"  weight median by equipment : "
            + ", ".join(f"{k}={v:,.0f}" for k, v in sorted(self.weight_median_by_equipment.items())),
            f"  weight median (fallback)   : {self.weight_median_global:,.0f}",
            f"  market_index median        : {self.market_index_median_global:.4f}",
            f"  weight cap                 : {self.weight_cap:,.0f}",
            f"  distance floor             : {self.distance_floor:,.1f}",
            f"  circuity cap (p99.9)       : {self.circuity_cap:.3f}",
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Raw IO
# --------------------------------------------------------------------------
def load_raw(name: str) -> pd.DataFrame:
    """Read one of the provided CSVs with dates parsed. No cleaning."""
    return pd.read_csv(DATA / name, parse_dates=["date"])


def load_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        load_raw("train_test.csv"),
        load_raw("validation.csv"),
        load_raw("december_chart_inputs.csv"),
    )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def haversine_miles(lat1, lon1, lat2, lon2):
    """Great-circle distance in miles."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_MILES * np.arcsin(np.sqrt(a))


def _repaired_weight(frame: pd.DataFrame) -> pd.Series:
    """Sign-flipped weights are recoverable; their magnitudes are valid."""
    return frame["weight"].abs()


# --------------------------------------------------------------------------
# Fit
# --------------------------------------------------------------------------
def fit_cleaning(train: pd.DataFrame) -> CleaningStats:
    """Learn imputation values and clip bounds from training rows only."""
    weight = _repaired_weight(train)
    stats = CleaningStats(
        weight_median_by_equipment=(
            weight.groupby(train["equipment"]).median().to_dict()
        ),
        weight_median_global=float(weight.median()),
        market_index_median_global=float(train["market_index"].median()),
        weight_cap=float(weight.max()),
        distance_floor=float(train["distance"].min()),
    )

    # Circuity above this is a coordinate artefact (some city pairs sit almost on
    # top of each other), not a real detour. Clip the derived feature; the
    # underlying distance stays untouched.
    circuity = train["distance"] / haversine_miles(
        train.pickup_lat, train.pickup_lon, train.delivery_lat, train.delivery_lon
    )
    stats.circuity_cap = float(circuity.quantile(0.999))
    return stats


# --------------------------------------------------------------------------
# Transform
# --------------------------------------------------------------------------
def clean(frame: pd.DataFrame, stats: CleaningStats) -> pd.DataFrame:
    """
    Apply the fitted cleaning to any of the three datasets.

    Degrades gracefully: the December chart file carries only seven columns, so
    geography and market features are simply skipped rather than assumed. This
    keeps one code path for all callers instead of a second, drifting one.
    """
    out = frame.copy()

    # -- weight: repair sign flips, impute gaps, flag censoring -------------
    if "weight" in out.columns:
        raw = out["weight"]
        out["weight_missing"] = raw.isna().astype("int8")
        repaired = raw.abs()
        fallback = out["equipment"].map(stats.weight_median_by_equipment) if "equipment" in out else None
        if fallback is not None:
            repaired = repaired.fillna(fallback)
        out["weight"] = repaired.fillna(stats.weight_median_global)
        out["weight_at_cap"] = (out["weight"] >= stats.weight_cap).astype("int8")

    # -- distance: floored at 70 miles, value itself is usable --------------
    if "distance" in out.columns:
        out["distance"] = out["distance"].astype(float)
        out["distance_at_floor"] = (out["distance"] <= stats.distance_floor).astype("int8")

    # -- market_index: daily series, impute from same-day rows --------------
    # For validation this uses other rows sharing that date. That is information a
    # live system would genuinely hold at quote time, but it is transductive, so
    # it is stated explicitly in the report rather than done silently.
    if "market_index" in out.columns:
        out["market_missing"] = out["market_index"].isna().astype("int8")
        daily_median = out.groupby("date")["market_index"].transform("median")
        out["market_index"] = (
            out["market_index"].fillna(daily_median).fillna(stats.market_index_median_global)
        )

    # -- geography ----------------------------------------------------------
    geo_cols = {"pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"}
    if geo_cols.issubset(out.columns):
        out["haversine"] = haversine_miles(
            out.pickup_lat, out.pickup_lon, out.delivery_lat, out.delivery_lon
        )
        out["circuity"] = (out["distance"] / out["haversine"]).clip(upper=stats.circuity_cap)

    if {"pickup", "delivery"}.issubset(out.columns):
        out["lane"] = out["pickup"] + " -> " + out["delivery"]

    return out


# --------------------------------------------------------------------------
# Label hygiene - training folds only
# --------------------------------------------------------------------------
def corrupt_label_mask(frame: pd.DataFrame) -> pd.Series:
    """True where posted_rate is injected noise rather than a plausible quote."""
    rpm = frame["posted_rate"] / frame["distance"]
    return (rpm < RPM_LOWER) | (rpm > RPM_UPPER)


def drop_corrupt_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Remove corrupt labels. Call this on TRAINING folds only.

    The holdout is deliberately left contaminated: Spotter scores against
    uncleaned Nov-Dec data, so cleaning the holdout would make our reported
    metrics optimistic in exactly the way the real scoring will not be.
    """
    return frame.loc[~corrupt_label_mask(frame)].copy()


# --------------------------------------------------------------------------
# Temporal split
# --------------------------------------------------------------------------
def temporal_split(
    frame: pd.DataFrame, split_date: pd.Timestamp = SPLIT_DATE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by date, never at random.

    A random split would let September rates inform January predictions and post a
    flattering, meaningless score. The forward split reproduces the real problem.
    """
    fit_part = frame.loc[frame["date"] <= split_date].copy()
    holdout = frame.loc[frame["date"] > split_date].copy()
    return fit_part, holdout


def prepare() -> dict[str, pd.DataFrame | CleaningStats]:
    """
    Standard pipeline entry point.

    Cleaning is fitted on the pre-split training window only, so no information
    from the holdout months reaches the imputation values.
    """
    train_raw, valid_raw, december_raw = load_all()
    fit_raw, holdout_raw = temporal_split(train_raw)

    # Two sets of cleaning parameters, each fitted on everything that is history
    # for its consumer:
    #   stats      - Jan-Aug only, for the holdout experiment. Sep-Oct must stay
    #                unseen, including by the imputation medians.
    #   stats_full - all of Jan-Oct, for the model that predicts Nov-Dec. The
    #                whole training file is genuinely past at that point, so
    #                withholding Sep-Oct here would be stale, not safe.
    stats = fit_cleaning(fit_raw)
    stats_full = fit_cleaning(train_raw)

    return {
        "stats": stats,
        "stats_full": stats_full,
        # Training folds get cleaned labels; the holdout keeps its contamination.
        "fit": drop_corrupt_labels(clean(fit_raw, stats)),
        "holdout": clean(holdout_raw, stats),
        # Final-model inputs.
        "full": drop_corrupt_labels(clean(train_raw, stats_full)),
        "validation": clean(valid_raw, stats_full),
        "december": clean(december_raw, stats_full),
    }
