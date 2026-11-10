"""
FloodLens MCP server: exposes household flood-risk predictions as MCP tools,
so any MCP host (Claude Code, Claude Desktop, an agent) can ask questions like
"which homes flood first?" or "is road R1 still passable?" and get answers
computed by the same deterministic pipeline that powers the API.

Run over stdio:
    python -m floodlens.mcp_server

All tools are read-only. They never call an LLM themselves, except
get_flood_alerts, which uses the same grounded Claude alert layer as the API
(and falls back to deterministic templates when ANTHROPIC_API_KEY is unset).
"""

from __future__ import annotations

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from floodlens.alerts.claude_alert import generate_alerts_for_critical
from floodlens.data.synthetic import DANGER_LEVEL_M, SEVERE_LEVEL_M
from floodlens.pipeline import run_pipeline

READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True)

mcp = MCPServer(
    name="floodlens",
    title="FloodLens Flood Impact Server",
    description="Street-level flood forecasts, household risk and evacuation guidance for a Bangladesh floodplain.",
    instructions=(
        "FloodLens predicts when each building floods, which roads close, and the nearest "
        "high-ground shelter. All numbers come from a deterministic forecast and flood "
        "propagation model over a synthetic haor-floodplain scenario. Quote them exactly; "
        "never estimate your own timings or distances."
    ),
    version="0.1.0",
)


def _building_dict(b) -> dict:
    return {
        "building_id": b.id,
        "type": b.type,
        "elevation_m": b.elevation_m,
        "hours_to_flood": b.hours_to_flood,
        "risk_level": b.risk_level,
        "nearest_shelter_id": b.nearest_shelter_id,
        "nearest_shelter_name": b.nearest_shelter_name,
        "distance_to_shelter_km": b.distance_to_shelter_km,
        "priority": b.priority,
    }


@mcp.tool(annotations=READ_ONLY)
def get_scenario_summary() -> dict:
    """Locality name, current river gauge level, danger/severe thresholds and counts of
    buildings, shelters and roads in the modelled area."""
    s = run_pipeline()["scenario"]
    return {
        "locality_name": s.locality_name,
        "current_gauge_level_m": round(float(s.gauge_history["level_m"].iloc[-1]), 2),
        "danger_level_m": DANGER_LEVEL_M,
        "severe_level_m": SEVERE_LEVEL_M,
        "total_buildings": len(s.buildings),
        "total_shelters": len(s.shelters),
        "total_roads": len(s.roads),
        "data_note": "Synthetic scenario calibrated to plausible haor/floodplain hydrology.",
    }


@mcp.tool(annotations=READ_ONLY)
def get_flood_forecast() -> dict:
    """72-hour river forecast: hours until the danger and severe stages, the forecast
    confidence band, and the overall impact summary (buildings and roads at risk)."""
    p = run_pipeline()
    f = p["forecast"]
    return {
        "hours_to_danger_level": f.hours_to_danger,
        "hours_to_severe_level": f.hours_to_severe,
        "confidence_band_m": f.confidence_band_m,
        "impact_summary": p["summary"],
    }


@mcp.tool(annotations=READ_ONLY)
def list_at_risk_buildings(limit: int = 15, risk_level: str | None = None) -> dict:
    """Buildings that will flood within 72 hours, most urgent first.

    Args:
        limit: maximum number of buildings to return (1-200).
        risk_level: optional filter: CRITICAL, HIGH or WATCH.
    """
    limit = max(1, min(int(limit), 200))
    impacts = [b for b in run_pipeline()["building_impacts"] if b.hours_to_flood is not None]
    if risk_level:
        wanted = risk_level.strip().upper()
        if wanted not in ("CRITICAL", "HIGH", "WATCH"):
            raise ToolError("risk_level must be CRITICAL, HIGH or WATCH")
        impacts = [b for b in impacts if b.risk_level == wanted]
    impacts.sort(key=lambda b: b.hours_to_flood)
    return {"count": len(impacts[:limit]), "buildings": [_building_dict(b) for b in impacts[:limit]]}


@mcp.tool(annotations=READ_ONLY)
def get_building_risk(building_id: str) -> dict:
    """Flood risk for one building: hours until water reaches it, risk tier, and the
    nearest high-ground shelter with distance. Use list_at_risk_buildings to find IDs."""
    for b in run_pipeline()["building_impacts"]:
        if b.id == building_id:
            return _building_dict(b)
    raise ToolError(f"Unknown building_id: {building_id!r}")


@mcp.tool(annotations=READ_ONLY)
def get_road_status() -> dict:
    """Every evacuation road with the forecast hour it becomes impassable (None = stays
    passable for the whole 72-hour window)."""
    return {"roads": [r.__dict__ for r in run_pipeline()["road_impacts"]]}


@mcp.tool(annotations=READ_ONLY)
def get_flood_extent(hour: int) -> dict:
    """How much of the area is underwater at a forecast hour (snaps to the nearest
    computed hour). Returns counts and percentage, not the full grid."""
    timeline = run_pipeline()["timeline"]
    nearest = min(timeline.keys(), key=lambda h: abs(h - int(hour)))
    grid = timeline[nearest]
    flooded = int(grid.sum())
    return {
        "requested_hour": int(hour),
        "returned_hour": nearest,
        "flooded_cell_count": flooded,
        "total_cells": int(grid.size),
        "flooded_pct": round(100 * flooded / grid.size, 1),
    }


@mcp.tool(annotations=READ_ONLY)
def get_flood_alerts(limit: int = 5) -> dict:
    """Plain-language English + Bangla alerts for the most urgent CRITICAL/HIGH
    households. Uses Claude when ANTHROPIC_API_KEY is set, otherwise deterministic
    templates; each alert's `source` field says which."""
    limit = max(1, min(int(limit), 20))
    alerts = generate_alerts_for_critical(run_pipeline()["building_impacts"], limit=limit)
    return {"count": len(alerts), "alerts": [a.__dict__ for a in alerts]}


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
