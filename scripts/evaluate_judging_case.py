"""Reproduce the offline checks cited in the CloudCamp judging brief.

Run from the repository root with python -m scripts.evaluate_judging_case.
"""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from floodlens.alerts.claude_alert import generate_alert
from floodlens.data.synthetic import (
    DANGER_LEVEL_M,
    RIVER_BASELINE_M,
    SEVERE_LEVEL_M,
    generate_scenario,
)
from floodlens.models.flood_propagation import flood_timeline
from floodlens.models.impact_assessment import assess_buildings, assess_roads, summarize_impact
from floodlens.models.water_level_forecast import forecast_water_level
from floodlens.pipeline import run_pipeline

SEEDS = range(42, 54)


def check(condition, seed, subject):
    if not condition:
        raise ValueError(f"Seed {seed}: {subject}")


def evaluate_seed(seed):
    scenario = generate_scenario(seed)
    forecast = forecast_water_level(
        scenario.gauge_history, scenario.rainfall_forecast, DANGER_LEVEL_M, SEVERE_LEVEL_M
    )
    timeline = flood_timeline(
        scenario.dem, scenario.river_mask, forecast.forecast, RIVER_BASELINE_M
    )
    buildings = assess_buildings(scenario.buildings, scenario.shelters, timeline)
    roads = assess_roads(scenario.roads, timeline)
    times = np.array(sorted(timeline))
    maps = np.stack([timeline[hour] for hour in times])
    counts = {"buildings": len(buildings), "roads": len(roads), "bilingual_alerts": 0}

    for building in buildings:
        flooded_hours = times[maps[:, building.row, building.col]]
        first_hour = int(flooded_hours[0]) if flooded_hours.size else None
        check(building.hours_to_flood == first_hour, seed, building.id)
        if building.risk_level not in {"CRITICAL", "HIGH"}:
            continue

        alert = generate_alert(building, use_api=False, use_ollama=False)
        check(alert.source == "template_fallback", seed, f"{building.id} fallback source")
        check(alert.building_id == building.id, seed, f"{building.id} alert identity")
        check(alert.risk_level == building.risk_level, seed, f"{building.id} alert risk")
        check(
            all(
                value in message
                for value, message in (
                    (f"{first_hour} hour", alert.message),
                    (f"{first_hour} ঘণ্টা", alert.message_bn),
                    (building.nearest_shelter_name, alert.message),
                    (building.nearest_shelter_name, alert.message_bn),
                    (f"{building.distance_to_shelter_km} km", alert.message),
                    (f"{building.distance_to_shelter_km} কিমি", alert.message_bn),
                )
            ),
            seed,
            f"{building.id} alert fields",
        )
        counts["bilingual_alerts"] += 1

    road_points = {road["id"]: road["points"] for road in scenario.roads}
    for road in roads:
        rows, cols = zip(*road_points[road.id])
        flooded_hours = times[maps[:, rows, cols].any(axis=1)]
        first_hour = int(flooded_hours[0]) if flooded_hours.size else None
        check(road.hours_to_inaccessible == first_hour, seed, road.id)

    return counts


def main():
    runs = {seed: evaluate_seed(seed) for seed in SEEDS}
    case = run_pipeline()
    home = next(building for building in case["building_impacts"] if building.id == "B0002")
    result = {
        "scope": "Synthetic consistency and deterministic fallback checks",
        "seeds": list(runs),
        "totals": {field: sum(run[field] for run in runs.values()) for field in runs[42]},
        "runs": runs,
        "demo_case": {
            "seed": 42,
            "forecast_hours": len(case["forecast"].forecast),
            "impact": summarize_impact(case["building_impacts"], case["road_impacts"]),
            "home": asdict(home),
            "roads": [asdict(road) for road in case["road_impacts"]],
            "template_alert": asdict(generate_alert(home, use_api=False, use_ollama=False)),
        },
    }
    output = Path(__file__).resolve().parents[1] / "results" / "judging_evaluation.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["totals"]))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
