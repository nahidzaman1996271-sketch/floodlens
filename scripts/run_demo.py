"""
Runs the full FloodLens pipeline end to end and writes demo artifacts to
results/: a console summary in the "6 hours -> road inaccessible, 10 hours
-> 70% flood probability" pitch style, flood-extent map PNGs at three
horizons, an impact table PNG, and a JSON export of the full prediction
(useful for the Proof / AI-Native Build submission sections).

Run: python3 scripts/run_demo.py
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from floodlens.data.synthetic import generate_scenario, DANGER_LEVEL_M, SEVERE_LEVEL_M, RIVER_BASELINE_M
from floodlens.models.water_level_forecast import forecast_water_level
from floodlens.models.flood_propagation import flood_timeline, flood_extent_for_level
from floodlens.models.impact_assessment import assess_buildings, assess_roads, summarize_impact
from floodlens.alerts.claude_alert import generate_alerts_for_critical
from floodlens.visualization.map_renderer import render_flood_map, render_impact_table_png


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("FloodLens — From Flood Warning to Flood Action")
    print("=" * 70)

    print("\n[1/6] Generating synthetic locality scenario (seeded, reproducible)...")
    scenario = generate_scenario(seed=42)
    print(f"      Locality: {scenario.locality_name}")
    print(f"      Grid: {scenario.dem.shape}, Buildings: {len(scenario.buildings)}, "
          f"Roads: {len(scenario.roads)}, Shelters: {len(scenario.shelters)}")

    print("\n[2/6] Forecasting river water level (72h horizon)...")
    forecast = forecast_water_level(
        scenario.gauge_history, scenario.rainfall_forecast, DANGER_LEVEL_M, SEVERE_LEVEL_M,
    )
    current_level = scenario.gauge_history["level_m"].iloc[-1]
    print(f"      Current gauge level: {current_level:.2f} m "
          f"(danger: {DANGER_LEVEL_M} m, severe: {SEVERE_LEVEL_M} m)")
    print(f"      Hours to danger level: {forecast.hours_to_danger}")
    print(f"      Hours to severe level: {forecast.hours_to_severe}")
    print(f"      Forecast uncertainty: +/- {forecast.confidence_band_m} m")

    print("\n[3/6] Propagating flood extent across terrain (hydrologically-connected fill)...")
    timeline = flood_timeline(scenario.dem, scenario.river_mask, forecast.forecast, RIVER_BASELINE_M)
    print(f"      Computed flood extent at {len(timeline)} sampled hours")

    print("\n[4/6] Assessing impact on buildings and roads...")
    building_impacts = assess_buildings(scenario.buildings, scenario.shelters, timeline)
    road_impacts = assess_roads(scenario.roads, timeline)
    summary = summarize_impact(building_impacts, road_impacts)

    print(f"\n      --- IMPACT SUMMARY ---")
    print(f"      {summary['buildings_at_risk']}/{summary['total_buildings']} buildings at risk "
          f"({summary['buildings_at_risk_pct']}%)")
    print(f"      {summary['buildings_critical_lt_10h']} buildings CRITICAL (flood predicted within 10h)")
    print(f"      {summary['medical_facilities_at_risk']} medical facilities at risk")
    print(f"      {summary['roads_inaccessible']}/{summary['total_roads']} roads predicted inaccessible")

    print(f"\n      --- ROAD-LEVEL DETAIL ---")
    for r in road_impacts:
        h = f"{r.hours_to_inaccessible}h" if r.hours_to_inaccessible is not None else "passable (72h+)"
        print(f"      {r.name}: {r.status} (impassable in {h})")

    print("\n[5/6] Generating plain-language alerts for the most urgent buildings (Bilingual EN & BN)...")
    alerts = generate_alerts_for_critical(building_impacts, limit=5)
    for a in alerts:
        source_tag = "[Claude API]" if a.source == "claude_api" else "[offline template]"
        print(f"\n      {source_tag} {a.building_id} ({a.risk_level}):")
        print(f"      EN: \"{a.message}\"")
        if getattr(a, "message_bn", ""):
            print(f"      BN: \"{a.message_bn}\"")

    print("\n[6/6] Rendering flood maps and impact table...")
    for hour in [1, 24, max(timeline.keys())]:
        nearest = min(timeline.keys(), key=lambda h: abs(h - hour))
        buildings_with_risk = scenario.buildings.copy()
        risk_by_id = {b.id: b.risk_level for b in building_impacts}
        buildings_with_risk["risk_level"] = buildings_with_risk["id"].map(risk_by_id)
        out_path = os.path.join(RESULTS_DIR, f"flood_map_hour_{nearest}.png")
        render_flood_map(
            scenario.dem, timeline[nearest], buildings_with_risk, scenario.roads, scenario.shelters,
            hour_label=f"+{nearest}h", out_path=out_path,
        )
        print(f"      Wrote {out_path}")

    table_path = os.path.join(RESULTS_DIR, "impact_table_top15.png")
    render_impact_table_png(building_impacts, table_path)
    print(f"      Wrote {table_path}")

    export = {
        "locality_name": scenario.locality_name,
        "current_gauge_level_m": round(float(current_level), 2),
        "danger_level_m": DANGER_LEVEL_M,
        "severe_level_m": SEVERE_LEVEL_M,
        "hours_to_danger_level": forecast.hours_to_danger,
        "hours_to_severe_level": forecast.hours_to_severe,
        "confidence_band_m": forecast.confidence_band_m,
        "impact_summary": summary,
        "roads": [asdict(r) for r in road_impacts],
        "top_critical_buildings": [
            asdict(b) for b in sorted(
                [b for b in building_impacts if b.hours_to_flood is not None],
                key=lambda b: b.hours_to_flood,
            )[:15]
        ],
        "sample_alerts": [asdict(a) for a in alerts],
        "data_provenance": (
            "Synthetic scenario: DEM, river gauge history, rainfall forecast, buildings, "
            "roads and shelters are generated to be hydrologically and demographically "
            "plausible for a Bangladeshi riverine/haor floodplain locality, calibrated to "
            "public BWDB-style gauge behavior and open-data-style terrain patterns -- they "
            "are NOT real observations. Forecasting model: damped-trend exponential "
            "smoothing (Holt's method) with a rainfall-coupling term, adapted from the "
            "time-to-failure modeling approach used in the Cluster Heartbeat project. "
            "Flood propagation: hydrologically-connected elevation flood-fill "
            "(BFS from river channel), not a full hydrodynamic model."
        ),
    }
    json_path = os.path.join(RESULTS_DIR, "prediction_export.json")
    with open(json_path, "w") as f:
        json.dump(export, f, indent=2, default=str)
    print(f"      Wrote {json_path}")

    print("\n" + "=" * 70)
    print("Demo complete. See results/ for maps, table, and JSON export.")
    print("=" * 70)


if __name__ == "__main__":
    main()
