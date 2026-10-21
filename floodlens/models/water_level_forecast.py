"""
Water level forecasting — the "time-to-failure" model, adapted from the
same modeling pattern used in Cluster Heartbeat (GPU cluster anomaly /
time-to-failure estimation): given a recent history + a leading indicator
(there: workload/thermal trends; here: rainfall), project forward and
estimate time-to-threshold.

Model choice: a damped-trend exponential smoothing model (Holt's linear
trend, damped) fit on the recent gauge history, with an additive rainfall
coupling term. This is intentionally NOT a deep model — for a 72-hour
hydrological forecast on a single gauge, a transparent statistical model
is both more defensible in a judge Q&A ("how did it decide that?") and
more appropriate given the short lead-time data available. The interface
is written so a learned model (e.g. an LSTM trained on real multi-year
BWDB data) can be substituted later without changing callers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class ForecastResult:
    forecast: pd.DataFrame            # timestamp, predicted_level_m
    hours_to_danger: float | None     # hours until DANGER_LEVEL_M is crossed, or None
    hours_to_severe: float | None     # hours until SEVERE_LEVEL_M is crossed, or None
    confidence_band_m: float          # +/- meters, a simple uncertainty estimate


def _holt_damped_fit(levels: np.ndarray, alpha: float = 0.35, beta: float = 0.15, phi: float = 0.98):
    """Minimal Holt's damped-trend exponential smoothing, fit via simple
    recursive updates (no external stats library needed)."""
    level = levels[0]
    trend = levels[1] - levels[0]
    for y in levels[1:]:
        prev_level = level
        level = alpha * y + (1 - alpha) * (prev_level + phi * trend)
        trend = beta * (level - prev_level) + (1 - beta) * phi * trend
    return level, trend, phi


def forecast_water_level(
    gauge_history: pd.DataFrame,
    rainfall_forecast: pd.DataFrame,
    danger_level_m: float,
    severe_level_m: float,
    rainfall_coupling: float = 0.018,
    max_hourly_rise_m: float = 0.28,
    plateau_level_m: float = 10.5,
) -> ForecastResult:
    """
    Project river level forward across the rainfall_forecast horizon.

    rainfall_coupling: meters of additional rise per (mm of rainfall in the
    upstream catchment), a simplified proxy for the real world's rainfall
    -> runoff -> gauge response relationship. Calibrated to produce
    plausible monsoon rise rates (roughly matching the ~0.1-0.3 m/hr
    rises seen in real BWDB flood-event records), not fit to real data.

    max_hourly_rise_m / plateau_level_m: physical guardrails. Real rivers
    don't rise indefinitely at an accelerating rate — once water tops the
    banks it spreads across the floodplain, which slows the *gauge* rise
    rate even as the flood extent keeps growing. We model that as a hard
    per-hour cap plus a soft saturation toward plateau_level_m (roughly
    the historical extreme-flood stage for this class of gauge), rather
    than letting the trend term compound unchecked.
    """
    levels = gauge_history["level_m"].to_numpy()
    level, trend, phi = _holt_damped_fit(levels)
    trend = float(np.clip(trend, -0.05, max_hourly_rise_m))

    horizon = len(rainfall_forecast)
    predicted = np.zeros(horizon)
    cur_level, cur_trend = level, trend
    cumulative_rain_effect = 0.0

    for i in range(horizon):
        cur_trend *= phi
        # Rainfall contributes a smoothed, lagged effect (runoff takes time).
        recent_rain = rainfall_forecast["rainfall_mm"].iloc[max(0, i - 5):i + 1].mean()
        rain_effect = rainfall_coupling * recent_rain
        cumulative_rain_effect = 0.7 * cumulative_rain_effect + rain_effect

        raw_increment = cur_trend + cumulative_rain_effect
        raw_increment = float(np.clip(raw_increment, -0.05, max_hourly_rise_m))

        # Soft saturation: as the river approaches the plateau (extreme
        # flood / floodplain-absorption regime), the effective rise rate
        # tapers off instead of compounding.
        headroom = max(plateau_level_m - cur_level, 0.0)
        saturation_factor = min(1.0, headroom / (plateau_level_m - danger_level_m + 1e-6))
        saturation_factor = max(saturation_factor, 0.05)  # never fully flatlines

        cur_level = cur_level + raw_increment * saturation_factor
        predicted[i] = cur_level

    forecast_df = pd.DataFrame({
        "timestamp": rainfall_forecast["timestamp"].to_numpy(),
        "predicted_level_m": predicted,
    })

    def _hours_to_threshold(threshold: float) -> float | None:
        above = np.where(predicted >= threshold)[0]
        if len(above) == 0:
            return None
        return float(above[0] + 1)  # +1 because index 0 = hour 1 of forecast

    # Simple uncertainty estimate: grows with horizon, anchored to recent
    # gauge volatility (residual std of last 48h vs. a 6h rolling mean).
    recent = gauge_history["level_m"].iloc[-48:]
    residual_std = float((recent - recent.rolling(6, min_periods=1).mean()).std())
    confidence_band_m = round(max(0.05, residual_std) * 2, 2)

    return ForecastResult(
        forecast=forecast_df,
        hours_to_danger=_hours_to_threshold(danger_level_m),
        hours_to_severe=_hours_to_threshold(severe_level_m),
        confidence_band_m=confidence_band_m,
    )
