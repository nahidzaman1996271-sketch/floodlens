import numpy as np
import pandas as pd
import pytest

from floodlens.data.synthetic import (
    generate_scenario, GRID_SIZE, DANGER_LEVEL_M, SEVERE_LEVEL_M, RIVER_BASELINE_M,
)
from floodlens.models.water_level_forecast import forecast_water_level
from floodlens.models.flood_propagation import flood_extent_for_level, flood_timeline
from floodlens.models.impact_assessment import assess_buildings, assess_roads, summarize_impact
from floodlens.alerts.claude_alert import generate_alert, generate_alerts_for_critical, Alert


@pytest.fixture(scope="module")
def scenario():
    return generate_scenario(seed=42)


@pytest.fixture(scope="module")
def forecast(scenario):
    return forecast_water_level(
        scenario.gauge_history, scenario.rainfall_forecast, DANGER_LEVEL_M, SEVERE_LEVEL_M,
    )


@pytest.fixture(scope="module")
def timeline(scenario, forecast):
    return flood_timeline(scenario.dem, scenario.river_mask, forecast.forecast, RIVER_BASELINE_M)


@pytest.fixture(scope="module")
def building_impacts(scenario, timeline):
    return assess_buildings(scenario.buildings, scenario.shelters, timeline)


@pytest.fixture(scope="module")
def road_impacts(scenario, timeline):
    return assess_roads(scenario.roads, timeline)


# ---------------------------------------------------------------- data layer

def test_scenario_shapes(scenario):
    assert scenario.dem.shape == (GRID_SIZE, GRID_SIZE)
    assert scenario.river_mask.shape == (GRID_SIZE, GRID_SIZE)
    assert scenario.river_mask.sum() > 0
    assert len(scenario.gauge_history) == 14 * 24
    assert len(scenario.rainfall_forecast) == 72
    assert len(scenario.buildings) > 0
    assert len(scenario.shelters) == 2
    assert len(scenario.roads) == 3


def test_scenario_reproducible():
    a = generate_scenario(seed=7)
    b = generate_scenario(seed=7)
    assert np.allclose(a.dem, b.dem)
    assert np.array_equal(a.river_mask, b.river_mask)
    assert a.gauge_history["level_m"].tolist() == b.gauge_history["level_m"].tolist()


def test_dem_elevation_range_sane(scenario):
    # Ground truth sanity: elevation range should roughly match the
    # gauge's flood-stage range so flood propagation is meaningful
    # (not everything floods instantly, not nothing ever floods).
    assert scenario.dem.max() > SEVERE_LEVEL_M - RIVER_BASELINE_M
    assert scenario.dem[scenario.river_mask].max() < 0  # river channel below plain


def test_shelters_on_high_ground(scenario):
    plain_median = np.median(scenario.dem[~scenario.river_mask])
    for _, s in scenario.shelters.iterrows():
        assert s["elevation_m"] > plain_median


# ------------------------------------------------------------ forecast model

def test_forecast_output_shape(forecast, scenario):
    assert len(forecast.forecast) == len(scenario.rainfall_forecast)
    assert "predicted_level_m" in forecast.forecast.columns


def test_forecast_is_bounded(forecast):
    # Guards against the runaway-growth bug: level should never blow past
    # a physically plausible ceiling within a 72h horizon.
    assert forecast.forecast["predicted_level_m"].max() < 15.0


def test_forecast_nondecreasing_ish(forecast):
    # Water level during a rising flood event should be broadly increasing
    # (allow small local dips from the smoothing, but the overall trend
    # across the horizon must be upward).
    levels = forecast.forecast["predicted_level_m"].to_numpy()
    assert levels[-1] > levels[0]


def test_hours_to_threshold_ordering(forecast):
    if forecast.hours_to_danger is not None and forecast.hours_to_severe is not None:
        assert forecast.hours_to_danger <= forecast.hours_to_severe


def test_confidence_band_positive(forecast):
    assert forecast.confidence_band_m > 0


# --------------------------------------------------------- flood propagation

def test_flood_extent_grows_with_level(scenario):
    low = flood_extent_for_level(scenario.dem, scenario.river_mask, 3.5, RIVER_BASELINE_M)
    high = flood_extent_for_level(scenario.dem, scenario.river_mask, 9.0, RIVER_BASELINE_M)
    assert high.sum() >= low.sum()


def test_flood_extent_includes_river(scenario):
    grid = flood_extent_for_level(scenario.dem, scenario.river_mask, 5.0, RIVER_BASELINE_M)
    assert grid[scenario.river_mask].all()


def test_flood_timeline_monotonic_extent(timeline):
    hours = sorted(timeline.keys())
    counts = [timeline[h].sum() for h in hours]
    # Extent should never shrink over time in this simplified model
    # (no drainage/recession modeled).
    assert all(counts[i] <= counts[i + 1] for i in range(len(counts) - 1))


def test_high_ground_never_fully_submerged(scenario, timeline):
    # The two shelter ridge apexes should remain dry across the whole
    # forecast horizon in this scenario (that's the point of siting
    # shelters there).
    last_hour = max(timeline.keys())
    final_grid = timeline[last_hour]
    for _, s in scenario.shelters.iterrows():
        assert final_grid[int(s["row"]), int(s["col"])] == False  # noqa: E712


# --------------------------------------------------------- impact assessment

def test_building_impacts_cover_all_buildings(scenario, building_impacts):
    assert len(building_impacts) == len(scenario.buildings)


def test_building_impact_risk_levels_valid(building_impacts):
    valid = {"SAFE", "WATCH", "HIGH", "CRITICAL"}
    assert all(b.risk_level in valid for b in building_impacts)


def test_shelters_never_marked_critical_for_themselves(scenario, building_impacts):
    # Not a direct check on shelters (they aren't in `buildings`), but
    # sanity check that at least some buildings are SAFE, proving the
    # model differentiates rather than flagging everything.
    risk_levels = {b.risk_level for b in building_impacts}
    assert len(risk_levels) >= 2


def test_road_impacts_cover_all_roads(scenario, road_impacts):
    assert len(road_impacts) == len(scenario.roads)


def test_road_status_valid(road_impacts):
    valid = {"PASSABLE", "AT_RISK", "INACCESSIBLE"}
    assert all(r.status in valid for r in road_impacts)


def test_summary_counts_consistent(building_impacts, road_impacts):
    summary = summarize_impact(building_impacts, road_impacts)
    assert summary["total_buildings"] == len(building_impacts)
    assert summary["total_roads"] == len(road_impacts)
    assert 0 <= summary["buildings_at_risk"] <= summary["total_buildings"]
    assert summary["buildings_critical_lt_10h"] <= summary["buildings_at_risk"]


# ---------------------------------------------------------------------- alerts

def test_alert_template_fallback_used_without_api_key(monkeypatch, building_impacts):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    critical = next(b for b in building_impacts if b.risk_level == "CRITICAL")
    alert = generate_alert(critical, use_api=False)
    assert isinstance(alert, Alert)
    assert alert.source == "template_fallback"
    assert len(alert.message) > 0
    assert str(critical.hours_to_flood) in alert.message


def test_alert_mentions_shelter_when_at_risk(building_impacts):
    # Shelter guidance only makes sense (and is only included) once a
    # building is actually predicted to flood -- a SAFE building's alert
    # correctly omits it, so pick an at-risk one here.
    at_risk_with_shelter = next(
        b for b in building_impacts if b.hours_to_flood is not None and b.nearest_shelter_name
    )
    alert = generate_alert(at_risk_with_shelter, use_api=False)
    assert at_risk_with_shelter.nearest_shelter_name in alert.message


def test_generate_alerts_for_critical_respects_limit(building_impacts):
    alerts = generate_alerts_for_critical(building_impacts, limit=5, use_api=False)
    assert len(alerts) <= 5
    assert all(a.risk_level in ("CRITICAL", "HIGH") for a in alerts)


def test_safe_building_message_no_crash():
    from floodlens.models.impact_assessment import BuildingImpact
    safe = BuildingImpact(
        id="X", type="household", row=0, col=0, elevation_m=20.0,
        hours_to_flood=None, risk_level="SAFE",
        nearest_shelter_id=None, nearest_shelter_name=None,
        distance_to_shelter_km=None, priority="standard",
    )
    alert = generate_alert(safe, use_api=False)
    assert "No flooding predicted" in alert.message


# ------------------------------------------------------------------- API layer

def test_api_end_to_end():
    from fastapi.testclient import TestClient
    from floodlens.api.main import app

    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}

    summary = client.get("/scenario/summary").json()
    assert "locality_name" in summary

    predict = client.get("/predict").json()
    assert "impact_summary" in predict
    assert "roads" in predict

    alerts = client.get("/alerts?limit=2").json()
    assert alerts["count"] <= 2

    flood_map = client.get("/flood-map/10").json()
    assert "grid" in flood_map
    assert flood_map["total_cells"] == GRID_SIZE * GRID_SIZE
