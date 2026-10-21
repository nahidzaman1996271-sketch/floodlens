"""
Impact assessment — the layer that turns "this grid cell floods at hour X"
into "your house will probably be flooded in 7 hours, nearest shelter is
2.4km away". This is the actual product: prediction -> action, not just
prediction.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict

CELL_METERS_DEFAULT = 50


def _first_flood_hour(row: int, col: int, timeline: dict[int, np.ndarray]) -> int | None:
    for hour in sorted(timeline.keys()):
        if timeline[hour][row, col]:
            return hour
    return None


def _euclidean_distance_m(r1, c1, r2, c2, cell_m: int) -> float:
    return math.hypot(r1 - r2, c1 - c2) * cell_m


@dataclass
class BuildingImpact:
    id: str
    type: str
    row: int
    col: int
    elevation_m: float
    hours_to_flood: int | None
    risk_level: str                 # "SAFE", "WATCH", "HIGH", "CRITICAL"
    nearest_shelter_id: str | None
    nearest_shelter_name: str | None
    distance_to_shelter_km: float | None
    priority: str                   # "standard", "medical_priority" etc.


@dataclass
class RoadImpact:
    id: str
    name: str
    hours_to_inaccessible: int | None
    status: str                     # "PASSABLE", "AT_RISK", "INACCESSIBLE"


def _risk_level(hours_to_flood: int | None) -> str:
    if hours_to_flood is None:
        return "SAFE"
    if hours_to_flood <= 10:
        return "CRITICAL"
    if hours_to_flood <= 24:
        return "HIGH"
    return "WATCH"


def assess_buildings(
    buildings: pd.DataFrame,
    shelters: pd.DataFrame,
    timeline: dict[int, np.ndarray],
    cell_meters: int = CELL_METERS_DEFAULT,
) -> list[BuildingImpact]:
    results = []
    for _, b in buildings.iterrows():
        hours = _first_flood_hour(int(b["row"]), int(b["col"]), timeline)
        risk = _risk_level(hours)

        # Nearest shelter (straight-line distance; a production version
        # would use road-network routing distance, not straight-line).
        best = None
        for _, s in shelters.iterrows():
            d = _euclidean_distance_m(b["row"], b["col"], s["row"], s["col"], cell_meters)
            if best is None or d < best[0]:
                best = (d, s["id"], s["name"])

        priority = "standard"
        if b["type"] == "clinic":
            priority = "medical_priority"
        elif b["type"] == "school":
            priority = "vulnerable_group_priority"

        results.append(BuildingImpact(
            id=b["id"], type=b["type"], row=int(b["row"]), col=int(b["col"]),
            elevation_m=round(float(b["elevation_m"]), 2),
            hours_to_flood=hours,
            risk_level=risk,
            nearest_shelter_id=best[1] if best else None,
            nearest_shelter_name=best[2] if best else None,
            distance_to_shelter_km=round(best[0] / 1000, 2) if best else None,
            priority=priority,
        ))
    return results


def assess_roads(
    roads: list,
    timeline: dict[int, np.ndarray],
) -> list[RoadImpact]:
    results = []
    for road in roads:
        hours_list = []
        for (r, c) in road["points"]:
            h = _first_flood_hour(r, c, timeline)
            if h is not None:
                hours_list.append(h)
        hours_to_inaccessible = min(hours_list) if hours_list else None
        if hours_to_inaccessible is None:
            status = "PASSABLE"
        elif hours_to_inaccessible <= 12:
            status = "INACCESSIBLE"
        else:
            status = "AT_RISK"
        results.append(RoadImpact(
            id=road["id"], name=road["name"],
            hours_to_inaccessible=hours_to_inaccessible,
            status=status,
        ))
    return results


def summarize_impact(building_impacts: list[BuildingImpact], road_impacts: list[RoadImpact]) -> dict:
    """Aggregate stats used both in the API response and the Story/Proof
    sections — this is the KPI block for the submission."""
    total = len(building_impacts)
    at_risk = [b for b in building_impacts if b.risk_level != "SAFE"]
    critical = [b for b in building_impacts if b.risk_level == "CRITICAL"]
    medical_priority_at_risk = [b for b in at_risk if b.priority == "medical_priority"]

    inaccessible_roads = [r for r in road_impacts if r.status == "INACCESSIBLE"]
    at_risk_roads = [r for r in road_impacts if r.status == "AT_RISK"]

    return {
        "total_buildings": total,
        "buildings_at_risk": len(at_risk),
        "buildings_at_risk_pct": round(100 * len(at_risk) / total, 1) if total else 0.0,
        "buildings_critical_lt_10h": len(critical),
        "medical_facilities_at_risk": len(medical_priority_at_risk),
        "roads_inaccessible": len(inaccessible_roads),
        "roads_at_risk": len(at_risk_roads),
        "total_roads": len(road_impacts),
    }
