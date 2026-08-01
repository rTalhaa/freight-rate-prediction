"""
Feature engineering.

Grouped so each block can be switched off and its contribution measured (see
notebooks/04_features.py). Two constraints shape everything here:

1. Validation is Nov-Dec, months that never appear in training. Anything
   memorised per-month is dead weight - the time features have to *extrapolate*,
   which is why they are a smooth trend plus Fourier terms rather than dummies.

2. Lane statistics must be built from strictly prior rows. Computing a lane mean
   over the whole file leaks future rates backwards and is the subtlest version
   of the leak the temporal split exists to prevent.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Day zero for the trend feature. Fixed so train, validation and December all
# sit on one continuous axis.
ORIGIN = pd.Timestamp("2025-01-01")
DAYS_PER_YEAR = 365.25


@dataclass
class FeatureConfig:
    """Toggles for the ablation study."""

    time: bool = True
    geo: bool = True
    lane: bool = True
    # Off by default: the market_index x distance effect is real, but ridge
    # already captures it via log_distance, so the explicit terms cost $1.65 of
    # MAE. Defaults are the measured-best configuration.
    interactions: bool = False
    # Annual Fourier terms, off by default. Measured to be actively harmful here
    # - see _time_block and notebooks/04_features.py section 2.
    fourier: bool = False
    day_of_week: bool = False

    def label(self) -> str:
        on = [n for n in ("time", "geo", "lane", "interactions") if getattr(self, n)]
        return "base" + ("+" + "+".join(on) if on else "")


@dataclass
class FeatureStats:
    """Lane and city rate-per-mile levels, learned from the training window."""

    lane_rpm: dict[str, float] = field(default_factory=dict)
    pickup_rpm: dict[str, float] = field(default_factory=dict)
    delivery_rpm: dict[str, float] = field(default_factory=dict)
    lane_count: dict[str, int] = field(default_factory=dict)
    global_rpm: float = np.nan


def fit_feature_stats(train: pd.DataFrame) -> FeatureStats:
    """Aggregate rate-per-mile by lane and by city over the training window only."""
    rpm = train["posted_rate"] / train["distance"]
    return FeatureStats(
        lane_rpm=rpm.groupby(train["lane"]).mean().to_dict(),
        pickup_rpm=rpm.groupby(train["pickup"]).mean().to_dict(),
        delivery_rpm=rpm.groupby(train["delivery"]).mean().to_dict(),
        lane_count=train["lane"].value_counts().to_dict(),
        global_rpm=float(rpm.mean()),
    )


# --------------------------------------------------------------------------
# Feature blocks
# --------------------------------------------------------------------------
def _base_block(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    distance = frame["distance"].to_numpy(dtype=float)
    weight = frame["weight"].to_numpy(dtype=float)
    out = {
        "distance": distance,
        "log_distance": np.log(distance),
        "weight": weight,
        "weight_per_mile": weight / distance,
        "weight_missing": frame["weight_missing"].to_numpy(dtype=float),
        "weight_at_cap": frame["weight_at_cap"].to_numpy(dtype=float),
        "distance_at_floor": frame["distance_at_floor"].to_numpy(dtype=float),
    }
    # market_index / quote_signal are absent from the December chart file; they
    # are supplied by the imputation step before this function is called.
    for col in ("market_index", "quote_signal", "market_missing"):
        if col in frame.columns:
            out[col] = frame[col].to_numpy(dtype=float)
    for equipment in ("Dry Van", "Reefer", "Flatbed"):
        out[f"eq_{equipment.replace(' ', '_')}"] = (
            (frame["equipment"] == equipment).to_numpy(dtype=float)
        )
    return out


def _time_block(
    frame: pd.DataFrame, fourier: bool = False, day_of_week: bool = False
) -> dict[str, np.ndarray]:
    """
    Time features that survive extrapolation.

    A bare linear trend, by measurement rather than preference. Month dummies are
    not an option at all: November and December never appear in training.

    Annual Fourier terms are the textbook alternative and they made things much
    worse here (MAE $118 -> $175, holdout bias -$27 -> -$115). The training
    window spans eight months, so inside it corr(trend, annual_cos) = -0.92 and
    the terms are nearly collinear with the trend. Ridge splits the coefficient
    between them, which is harmless in-sample and diverges the moment prediction
    moves past the edge of the window. Both are kept behind flags so the
    ablation can show the damage instead of asserting it.

    Day-of-week is off for the same reason in miniature: EDA put the effect at
    ~2.5% and including it costs $2 of MAE.
    """
    days = (frame["date"] - ORIGIN).dt.days.to_numpy(dtype=float)
    out = {"trend_days": days}
    if fourier:
        angle = 2 * np.pi * days / DAYS_PER_YEAR
        out.update({
            "annual_sin": np.sin(angle),
            "annual_cos": np.cos(angle),
            "annual_sin2": np.sin(2 * angle),
            "annual_cos2": np.cos(2 * angle),
        })
    if day_of_week:
        out.update({
            "day_of_week": frame["date"].dt.dayofweek.to_numpy(dtype=float),
            "is_weekend": (frame["date"].dt.dayofweek >= 5).to_numpy(dtype=float),
        })
    return out


def _geo_block(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Coordinates are internally consistent, so they generalise to unseen cities."""
    if "haversine" not in frame.columns:
        return {}
    dlat = (frame["delivery_lat"] - frame["pickup_lat"]).to_numpy(dtype=float)
    dlon = (frame["delivery_lon"] - frame["pickup_lon"]).to_numpy(dtype=float)
    return {
        "pickup_lat": frame["pickup_lat"].to_numpy(dtype=float),
        "pickup_lon": frame["pickup_lon"].to_numpy(dtype=float),
        "delivery_lat": frame["delivery_lat"].to_numpy(dtype=float),
        "delivery_lon": frame["delivery_lon"].to_numpy(dtype=float),
        "haversine": frame["haversine"].to_numpy(dtype=float),
        "circuity": frame["circuity"].to_numpy(dtype=float),
        "delta_lat": dlat,
        "delta_lon": dlon,
        "bearing": np.arctan2(dlat, dlon),
        "is_northbound": (dlat > 0).astype(float),
        "is_eastbound": (dlon > 0).astype(float),
    }


def _lane_block(
    frame: pd.DataFrame, stats: FeatureStats, expanding: bool
) -> dict[str, np.ndarray]:
    """
    Lane and city rate levels.

    On training rows these are expanding means over *prior* rows only, so a row
    never contributes to its own feature. On inference rows the fitted training
    aggregates are used, which are wholly in the past by construction.

    17.5% of validation lanes are unseen, so every lookup degrades through city
    level to the global mean rather than emitting a null.
    """
    lanes = frame["lane"]
    if expanding:
        rpm = frame["posted_rate"] / frame["distance"]
        grouped = rpm.groupby(lanes)
        # Prior mean = (running total - this row) / (rows seen before this one).
        prior_sum = grouped.cumsum() - rpm
        prior_count = grouped.cumcount()
        lane_rpm = (prior_sum / prior_count.where(prior_count > 0)).to_numpy(dtype=float)
        lane_count = prior_count.to_numpy(dtype=float)
    else:
        lane_rpm = lanes.map(stats.lane_rpm).to_numpy(dtype=float)
        lane_count = lanes.map(stats.lane_count).fillna(0).to_numpy(dtype=float)

    pickup_rpm = frame["pickup"].map(stats.pickup_rpm).to_numpy(dtype=float)
    delivery_rpm = frame["delivery"].map(stats.delivery_rpm).to_numpy(dtype=float)

    # Cascade: lane -> mean of both city levels -> global.
    city_mean = np.nanmean(np.column_stack([pickup_rpm, delivery_rpm]), axis=1)
    filled = np.where(np.isnan(lane_rpm), city_mean, lane_rpm)
    filled = np.where(np.isnan(filled), stats.global_rpm, filled)

    return {
        "lane_rpm": filled,
        "lane_seen": (~np.isnan(lane_rpm)).astype(float),
        "lane_count": lane_count,
        "pickup_rpm": np.nan_to_num(pickup_rpm, nan=stats.global_rpm),
        "delivery_rpm": np.nan_to_num(delivery_rpm, nan=stats.global_rpm),
        # The strongest single predictor available: a lane's historical price
        # level multiplied by this load's distance.
        "lane_rate_estimate": filled * frame["distance"].to_numpy(dtype=float),
    }


def _interaction_block(blocks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Interactions the EDA measured rather than guessed.

    market_index shifts rate-per-mile by a near-constant amount inside every
    distance band, so its effect on total rate scales with distance.
    """
    out: dict[str, np.ndarray] = {}
    if "market_index" in blocks:
        out["market_x_distance"] = blocks["market_index"] * blocks["distance"]
        out["market_x_logdist"] = blocks["market_index"] * blocks["log_distance"]
    if "quote_signal" in blocks:
        out["quote_x_logdist"] = blocks["quote_signal"] * blocks["log_distance"]
    if "lane_rpm" in blocks and "market_index" in blocks:
        out["lane_x_market"] = blocks["lane_rpm"] * blocks["market_index"]
    if "trend_days" in blocks:
        out["trend_x_logdist"] = blocks["trend_days"] * blocks["log_distance"]
    return out


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------
def build_features(
    frame: pd.DataFrame,
    stats: FeatureStats,
    config: FeatureConfig | None = None,
    expanding_lane: bool = False,
) -> pd.DataFrame:
    """
    Assemble the design matrix.

    Set expanding_lane=True for training rows so lane statistics are built from
    prior rows only; leave it False for holdout, validation and December, where
    the fitted training aggregates already sit entirely in the past.
    """
    config = config or FeatureConfig()
    blocks = _base_block(frame)
    if config.time:
        blocks.update(_time_block(frame, config.fourier, config.day_of_week))
    if config.geo:
        blocks.update(_geo_block(frame))
    if config.lane:
        blocks.update(_lane_block(frame, stats, expanding=expanding_lane))
    if config.interactions:
        blocks.update(_interaction_block(blocks))
    return pd.DataFrame(blocks, index=frame.index)


# --------------------------------------------------------------------------
# Recency weighting - the answer to the drift problem
# --------------------------------------------------------------------------
def recency_weights(dates: pd.Series, half_life_days: float | None) -> np.ndarray:
    """
    Exponentially decayed sample weights, newest row weighted 1.0.

    Rates drift upward through the year, so a model fitted with equal weight on
    January under-prices October. Down-weighting old rows trades a little data
    volume for a level that tracks the recent market. half_life_days=None
    disables it (uniform weights).
    """
    if half_life_days is None:
        return np.ones(len(dates), dtype=float)
    age_days = (dates.max() - dates).dt.days.to_numpy(dtype=float)
    return np.power(0.5, age_days / half_life_days)
