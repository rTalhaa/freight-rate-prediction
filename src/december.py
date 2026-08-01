"""
December chart inputs.

`data/december_chart_inputs.csv` carries seven columns: pickup, delivery,
distance, equipment, weight, date, predicted_rate. Missing are the coordinates
and both market features, all of which the model needs.

Two different problems, handled differently:

Coordinates are a static property of a city. Every city maps to exactly one
lat/lon, identically in train and validation, so looking Lexington and Fort
Wayne up in the training data recovers a fact, not a forecast.

market_index and quote_signal are time series whose December values are not
knowable at quote time. They are forecast from training data only. The values
supplied in validation.csv for those same dates are deliberately not joined -
a deployed quoting model would not have them - and are used afterwards to score
the forecast instead. See notebooks/07_december.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Candidate averaging windows, in days, for the level estimate.
LEVEL_WINDOWS = (7, 14, 28, 56, 90, 120)

# A day-of-week profile is only used where the effect clearly beats the noise.
DOW_SIGNAL_THRESHOLD = 1.0


@dataclass
class MarketForecast:
    """Level-plus-weekly-profile forecaster for the market feature series."""

    level: dict[str, float] = field(default_factory=dict)
    window: dict[str, int] = field(default_factory=dict)
    dow_profile: dict[str, np.ndarray | None] = field(default_factory=dict)
    backtest_mae: dict[str, float] = field(default_factory=dict)
    dow_signal: dict[str, float] = field(default_factory=dict)

    def predict(self, dates: pd.Series | pd.DatetimeIndex) -> pd.DataFrame:
        dates = pd.DatetimeIndex(dates)
        out = {}
        for column, level in self.level.items():
            values = np.full(len(dates), level, dtype=float)
            profile = self.dow_profile.get(column)
            if profile is not None:
                values = values + profile[dates.dayofweek.to_numpy()]
            out[column] = values
        return pd.DataFrame(out, index=dates)

    def describe(self) -> str:
        lines = ["MarketForecast (fitted on training data only)"]
        for column in self.level:
            profile = self.dow_profile.get(column)
            shape = ("weekly profile" if profile is not None else "flat, no weekly profile")
            lines.append(
                f"  {column:13s} level {self.level[column]:.4f} "
                f"(mean of last {self.window[column]}d) | {shape} | "
                f"dow signal/noise {self.dow_signal[column]:.1f} | "
                f"backtest MAE {self.backtest_mae[column]:.4f}"
            )
        return "\n".join(lines)


def _daily(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.groupby("date")[column].mean().sort_index()


def _dow_profile(series: pd.Series) -> tuple[np.ndarray, float]:
    """
    Average deviation from a 28-day centred rolling mean, by weekday.

    Returns the seven offsets and a signal-to-noise ratio: the spread of the
    weekday effect against the typical scatter within a weekday.
    """
    detrended = series - series.rolling(28, center=True, min_periods=7).mean()
    grouped = detrended.groupby(detrended.index.dayofweek)
    profile = grouped.mean().reindex(range(7)).fillna(0.0).to_numpy(dtype=float)
    noise = float(grouped.std().mean())
    spread = float(profile.max() - profile.min())
    return profile, (spread / noise if noise > 0 else 0.0)


def _backtest_window(series: pd.Series, window: int, profile: np.ndarray | None,
                     horizon: int = 61) -> float:
    """
    Score one window length by forecasting the tail of the training series.

    Fit on everything before the last `horizon` days, predict those days, and
    return mean absolute error. This is how the window is chosen - never by
    looking at December.
    """
    if len(series) <= horizon + window:
        return np.inf
    history, actual = series.iloc[:-horizon], series.iloc[-horizon:]
    level = float(history.tail(window).mean())
    predicted = np.full(len(actual), level, dtype=float)
    if profile is not None:
        predicted = predicted + profile[actual.index.dayofweek.to_numpy()]
    return float(np.mean(np.abs(predicted - actual.to_numpy())))


def fit_market_forecast(
    train: pd.DataFrame, columns=("market_index", "quote_signal")
) -> MarketForecast:
    """
    Fit the forecaster on training data only.

    The level is a trailing mean rather than an extrapolated trend, and the
    window length is chosen by backtesting the last 61 days of training. This
    follows the same finding as the model itself: across this data, projecting a
    fitted slope forward has consistently added variance without adding
    information.
    """
    forecast = MarketForecast()
    for column in columns:
        series = _daily(train, column)
        profile, signal = _dow_profile(series)
        use_profile = profile if signal >= DOW_SIGNAL_THRESHOLD else None

        scored = {w: _backtest_window(series, w, use_profile) for w in LEVEL_WINDOWS}
        best_window = min(scored, key=scored.get)

        forecast.level[column] = float(series.tail(best_window).mean())
        forecast.window[column] = best_window
        forecast.dow_profile[column] = use_profile
        forecast.backtest_mae[column] = scored[best_window]
        forecast.dow_signal[column] = signal
    return forecast


def city_coordinates(train: pd.DataFrame) -> pd.DataFrame:
    """
    One row per city. Verified in EDA: every city has exactly one coordinate,
    consistent between train and validation.
    """
    pickups = train[["pickup", "pickup_lat", "pickup_lon"]].rename(
        columns={"pickup": "city", "pickup_lat": "lat", "pickup_lon": "lon"})
    deliveries = train[["delivery", "delivery_lat", "delivery_lon"]].rename(
        columns={"delivery": "city", "delivery_lat": "lat", "delivery_lon": "lon"})
    return pd.concat([pickups, deliveries]).drop_duplicates("city").set_index("city")


def enrich_december(
    december: pd.DataFrame, train: pd.DataFrame, forecast: MarketForecast
) -> pd.DataFrame:
    """Attach the looked-up coordinates and the forecast market features."""
    coords = city_coordinates(train)
    out = december.copy()

    missing = set(out["pickup"]) | set(out["delivery"]) - set(coords.index)
    missing = {c for c in (set(out["pickup"]) | set(out["delivery"])) if c not in coords.index}
    if missing:
        raise ValueError(f"cities absent from training coordinates: {sorted(missing)}")

    out["pickup_lat"] = out["pickup"].map(coords["lat"]).to_numpy()
    out["pickup_lon"] = out["pickup"].map(coords["lon"]).to_numpy()
    out["delivery_lat"] = out["delivery"].map(coords["lat"]).to_numpy()
    out["delivery_lon"] = out["delivery"].map(coords["lon"]).to_numpy()

    predicted = forecast.predict(out["date"])
    for column in predicted.columns:
        out[column] = predicted[column].to_numpy()
    out["market_missing"] = 0
    return out
