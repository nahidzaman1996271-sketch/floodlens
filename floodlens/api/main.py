"""
FloodLens API — FastAPI app exposing the end-to-end pipeline.

Run with: uvicorn floodlens.api.main:app --reload

Endpoints:
  GET  /health
  GET  /scenario/summary      -> locality info, current gauge level, thresholds
  GET  /predict                -> full forecast + impact assessment + KPI summary
  GET  /alerts?limit=10        -> plain-language alerts for the most urgent buildings
  GET  /flood-map/{hour}       -> boolean flood grid (JSON) for a given forecast hour
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from floodlens.data.synthetic import (
    generate_scenario, DANGER_LEVEL_M, SEVERE_LEVEL_M, RIVER_BASELINE_M,
)
from floodlens.models.water_level_forecast import forecast_water_level
from floodlens.models.flood_propagation import flood_timeline
from floodlens.models.impact_assessment import assess_buildings, assess_roads, summarize_impact
from floodlens.alerts.claude_alert import generate_alerts_for_critical

app = FastAPI(
    title="FloodLens API",
    description="From flood warning to flood action — street-level flood impact prediction.",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@lru_cache(maxsize=1)
def _pipeline():
    """Runs the full pipeline once and caches it. In production this would
    be a scheduled job re-run whenever new gauge/rainfall data arrives, not
    a per-request computation."""
    scenario = generate_scenario(seed=42)
    forecast = forecast_water_level(
        scenario.gauge_history, scenario.rainfall_forecast, DANGER_LEVEL_M, SEVERE_LEVEL_M,
    )
    timeline = flood_timeline(scenario.dem, scenario.river_mask, forecast.forecast, RIVER_BASELINE_M)
    building_impacts = assess_buildings(scenario.buildings, scenario.shelters, timeline)
    road_impacts = assess_roads(scenario.roads, timeline)
    summary = summarize_impact(building_impacts, road_impacts)
    return {
        "scenario": scenario, "forecast": forecast, "timeline": timeline,
        "building_impacts": building_impacts, "road_impacts": road_impacts, "summary": summary,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/scenario/summary")
def scenario_summary():
    p = _pipeline()
    s = p["scenario"]
    return {
        "locality_name": s.locality_name,
        "current_gauge_level_m": round(float(s.gauge_history["level_m"].iloc[-1]), 2),
        "danger_level_m": DANGER_LEVEL_M,
        "severe_level_m": SEVERE_LEVEL_M,
        "total_buildings": len(s.buildings),
        "total_shelters": len(s.shelters),
        "total_roads": len(s.roads),
        "data_note": "Synthetic scenario calibrated to plausible haor/floodplain hydrology. See README.",
    }


@app.get("/predict")
def predict():
    p = _pipeline()
    f = p["forecast"]
    return {
        "hours_to_danger_level": f.hours_to_danger,
        "hours_to_severe_level": f.hours_to_severe,
        "confidence_band_m": f.confidence_band_m,
        "impact_summary": p["summary"],
        "roads": [r.__dict__ for r in p["road_impacts"]],
        "top_critical_buildings": [
            b.__dict__ for b in sorted(
                [b for b in p["building_impacts"] if b.hours_to_flood is not None],
                key=lambda b: b.hours_to_flood,
            )[:15]
        ],
    }


@app.get("/alerts")
def alerts(limit: int = 10):
    p = _pipeline()
    generated = generate_alerts_for_critical(p["building_impacts"], limit=limit)
    return {"count": len(generated), "alerts": [a.__dict__ for a in generated]}


@app.get("/flood-map/{hour}")
def flood_map(hour: int):
    p = _pipeline()
    timeline = p["timeline"]
    available_hours = sorted(timeline.keys())
    nearest = min(available_hours, key=lambda h: abs(h - hour))
    if nearest != hour and hour not in available_hours:
        pass  # snap to nearest sampled hour; documented via response field below
    grid = timeline[nearest]
    return {
        "requested_hour": hour,
        "returned_hour": nearest,
        "flooded_cell_count": int(grid.sum()),
        "total_cells": int(grid.size),
        "grid": grid.astype(int).tolist(),
    }
