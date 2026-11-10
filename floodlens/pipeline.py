"""
Shared, cached end-to-end pipeline.

Both the FastAPI service (floodlens/api/main.py) and the MCP server
(floodlens/mcp_server.py) call run_pipeline(), so every interface serves
exactly the same forecast, flood timeline and impact assessment.
"""

from __future__ import annotations

from functools import lru_cache

from floodlens.data.synthetic import (
    generate_scenario, DANGER_LEVEL_M, SEVERE_LEVEL_M, RIVER_BASELINE_M,
)
from floodlens.models.water_level_forecast import forecast_water_level
from floodlens.models.flood_propagation import flood_timeline
from floodlens.models.impact_assessment import assess_buildings, assess_roads, summarize_impact


@lru_cache(maxsize=1)
def run_pipeline() -> dict:
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
