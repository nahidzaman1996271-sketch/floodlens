"""
Synthetic data generation for the FloodLens prototype.

WHY SYNTHETIC:
Real-time satellite imagery and hyper-local elevation/building data were not
obtainable within the hackathon build window. Instead, this module generates
a physically-plausible synthetic scenario modeled after a real flood-prone
riverine/haor locality in Bangladesh (e.g. a union in the Sunamganj /
Jamalpur floodplain):

  - A digital elevation model (DEM) with a river channel, a floodplain that
    slopes away from the river, and a few higher "safe" ridges — this
    mirrors real haor topography, not a random height field.
  - A river gauge time series with a realistic monsoon rise pattern
    (slow build-up, then a fast rise during a storm event), similar in
    shape to publicly available BWDB (Bangladesh Water Development Board)
    gauge charts.
  - Buildings, roads and shelters placed with a density gradient (denser
    near the river/market, sparser on high ground), which is how
    real riverine settlements are distributed.

For a production version, the DEM would come from open SRTM/Copernicus
data, the gauge series from BWDB's real-time feed, and building/road
footprints from OpenStreetMap — the interfaces below are written so those
can be swapped in without changing any downstream module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


GRID_SIZE = 60  # 60x60 cells, each cell ~ 50m -> ~3km x 3km locality
CELL_METERS = 50
RIVER_BASELINE_M = 3.0        # normal dry-season river stage (meters)
DANGER_LEVEL_M = 6.5          # official "danger level" stage for this gauge
SEVERE_LEVEL_M = 8.0          # severe flooding stage


@dataclass
class SyntheticScenario:
    dem: np.ndarray                      # (GRID_SIZE, GRID_SIZE) elevation in meters
    river_mask: np.ndarray                # boolean grid, True where the river channel is
    gauge_history: pd.DataFrame           # hourly river level, past 14 days
    rainfall_forecast: pd.DataFrame       # hourly rainfall forecast, next 72 hours
    buildings: pd.DataFrame               # id, row, col, type, name
    roads: list                           # list of dicts: id, points [(row,col), ...]
    shelters: pd.DataFrame                # id, row, col, name, capacity
    locality_name: str = "Char Bandar Union (synthetic, haor-floodplain profile)"


def _generate_dem(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Generate a floodplain DEM: a river channel on one side, elevation
    rising smoothly with distance from the river, plus a couple of raised
    ridges (natural high ground / embankments) that matter for shelter
    siting."""
    rows, cols = GRID_SIZE, GRID_SIZE
    dem = np.zeros((rows, cols))

    # River channel: a gently meandering line down the left-ish side.
    river_mask = np.zeros((rows, cols), dtype=bool)
    center_col = 10 + 4 * np.sin(np.linspace(0, 3 * np.pi, rows))
    for r in range(rows):
        c = int(round(center_col[r]))
        c = max(1, min(cols - 2, c))
        river_mask[r, max(0, c - 1):c + 2] = True

    # Distance-from-river field via simple iterative propagation.
    dist = np.full((rows, cols), fill_value=1e6)
    dist[river_mask] = 0.0
    for _ in range(rows + cols):
        shifted = [
            np.roll(dist, 1, axis=0), np.roll(dist, -1, axis=0),
            np.roll(dist, 1, axis=1), np.roll(dist, -1, axis=1),
        ]
        new_dist = np.minimum(dist, np.min(np.stack(shifted), axis=0) + 1)
        if np.allclose(new_dist, dist):
            break
        dist = new_dist

    # Base elevation rises with distance from river (floodplain slope),
    # capped, plus mild smooth noise for realism. Scaled so that the
    # plain's elevation range (0-9m above the river channel bed) is
    # comparable to the gauge's realistic flood-stage range (baseline
    # ~3m, severe ~8m, extreme plateau ~10.5m) -- otherwise the whole
    # plain floods on the very first forecast hour, which is not how
    # real floodplains behave.
    dem = np.minimum(dist * 0.25, 9.0)

    # Add two raised ridges (e.g. old embankments / high ground) — good
    # shelter locations, similar to how villages cluster on natural levees.
    # Heights are set so ridge apex elevation comfortably exceeds the
    # extreme-scenario flood surface (~7.5m rise above baseline), i.e.
    # shelters sited here stay dry even in the severe/extreme case.
    for (rr, rc, radius, height) in [(15, 45, 10, 5.0), (45, 40, 8, 4.5)]:
        yy, xx = np.mgrid[0:rows, 0:cols]
        d = np.sqrt((yy - rr) ** 2 + (xx - rc) ** 2)
        bump = height * np.exp(-(d ** 2) / (2 * radius ** 2))
        dem += bump

    noise = rng.normal(0, 0.08, size=dem.shape)
    dem = np.clip(dem + noise, 0, None)
    dem[river_mask] = -0.5  # river channel itself sits below plain level

    return dem, river_mask


def _generate_gauge_history(rng: np.random.Generator) -> pd.DataFrame:
    """14 days of hourly river gauge readings ending 'now', shaped like a
    real monsoon build-up: slow rise, then an accelerating rise in the
    final ~36 hours consistent with an upstream storm event."""
    hours = pd.date_range(end=pd.Timestamp.now().floor("h"), periods=14 * 24, freq="h")
    n = len(hours)
    t = np.arange(n)

    # Tuned so that "now" (end of history) sits clearly below the danger
    # threshold, with a rainfall-driven storm surge just beginning --
    # this gives the forecast a real story arc (currently safe -> crosses
    # danger in a few hours -> approaches severe over 1-2 days) rather
    # than starting the demo already at the threshold.
    slow_rise = RIVER_BASELINE_M + 0.8 * (t / n)  # gentle seasonal rise
    accel_start = n - 18
    accel = np.zeros(n)
    accel[accel_start:] = 1.4 * ((t[accel_start:] - accel_start) / 18) ** 1.5
    diurnal = 0.05 * np.sin(2 * np.pi * t / 24)
    noise = rng.normal(0, 0.03, size=n)

    level = slow_rise + accel + diurnal + noise
    level = np.maximum(level, 0.2)

    return pd.DataFrame({"timestamp": hours, "level_m": level})


def _generate_rainfall_forecast(rng: np.random.Generator, gauge_df: pd.DataFrame) -> pd.DataFrame:
    """72-hour rainfall forecast (mm/hour). Correlated with the accelerating
    gauge trend so the forecast is internally consistent (heavy rain
    upstream is *why* the river keeps rising)."""
    hours = pd.date_range(
        start=gauge_df["timestamp"].iloc[-1] + pd.Timedelta(hours=1),
        periods=72, freq="h",
    )
    n = len(hours)
    t = np.arange(n)
    heavy_band = np.exp(-((t - 10) ** 2) / (2 * 8 ** 2)) * 14.0
    background = 1.5 + 1.0 * np.sin(2 * np.pi * t / 24)
    noise = rng.normal(0, 0.6, size=n)
    rainfall = np.clip(heavy_band + background + noise, 0, None)
    return pd.DataFrame({"timestamp": hours, "rainfall_mm": rainfall})


def _place_buildings(rng: np.random.Generator, dem: np.ndarray, river_mask: np.ndarray) -> pd.DataFrame:
    """Place households and a few critical facilities with density highest
    near the river/market (realistic riverine settlement pattern) and
    thinning out on high ground."""
    rows, cols = dem.shape
    records = []
    building_id = 0
    for r in range(rows):
        for c in range(cols):
            if river_mask[r, c]:
                continue
            dist_to_river = 1  # cheap proxy; density governed by column band below
            near_river_band = 0 < c < 22
            density_prob = 0.075 if near_river_band else 0.02
            if dem[r, c] > 4.5:
                density_prob *= 0.3  # sparser on high ridges
            if rng.random() < density_prob:
                building_id += 1
                btype = "household"
                if rng.random() < 0.03:
                    btype = rng.choice(["clinic", "school", "market_stall"])
                records.append({
                    "id": f"B{building_id:04d}",
                    "row": r, "col": c,
                    "elevation_m": float(dem[r, c]),
                    "type": btype,
                })
    return pd.DataFrame(records)


def _place_roads(rng: np.random.Generator, dem: np.ndarray) -> list:
    """A handful of roads with deliberately different flood exposure, so
    the demo shows differentiated risk rather than every route failing
    at once (which is unrealistic and also a weaker demo):

      - Road 1 & 2: village access roads that cross the low-lying
        floodplain near the river — the realistic first-to-fail routes.
      - Road 3: the upazila connector that runs along higher ground
        (skirting the settlement rather than crossing near the river),
        which should stay passable much longer — this is the road an
        evacuation plan would route people toward.
    """
    rows, cols = dem.shape
    roads = []
    for i, r0 in enumerate([12, 28]):
        points = [(r0 + int(2 * np.sin(c / 8)), c) for c in range(0, cols, 3)]
        points = [(max(0, min(rows - 1, r)), c) for r, c in points]
        roads.append({"id": f"R{i+1}", "name": f"Village Access Road {i+1}", "points": points})

    # High-ground connector: only spans the higher-elevation half of the
    # grid (cols 28-59), avoiding the low near-river band entirely.
    bypass_points = [(46 + int(2 * np.sin(c / 10)), c) for c in range(28, cols, 3)]
    bypass_points = [(max(0, min(rows - 1, r)), c) for r, c in bypass_points]
    roads.append({"id": "R3", "name": "Upazila Connector Road (high ground)", "points": bypass_points})
    return roads


def _place_shelters(dem: np.ndarray, buildings: pd.DataFrame) -> pd.DataFrame:
    """Shelters sited on the highest local ground — mirrors real practice
    of using schools/cyclone shelters built on raised plinths or natural
    high ground."""
    rows, cols = dem.shape
    flat = dem.copy()
    candidates = []
    # Two known ridge centers from _generate_dem, plus the single highest cell.
    for (rr, rc, name, cap) in [
        (15, 45, "Char Bandar Primary School Shelter", 400),
        (45, 40, "Union Cyclone Shelter #12", 600),
    ]:
        candidates.append({
            "id": f"S{len(candidates)+1}",
            "row": rr, "col": rc,
            "name": name,
            "capacity": cap,
            "elevation_m": float(dem[rr, rc]),
        })
    return pd.DataFrame(candidates)


def generate_scenario(seed: int = 42) -> SyntheticScenario:
    """Deterministic (seeded) synthetic scenario generation, so the demo
    and tests are fully reproducible."""
    rng = np.random.default_rng(seed)
    dem, river_mask = _generate_dem(rng)
    gauge_history = _generate_gauge_history(rng)
    rainfall_forecast = _generate_rainfall_forecast(rng, gauge_history)
    buildings = _place_buildings(rng, dem, river_mask)
    roads = _place_roads(rng, dem)
    shelters = _place_shelters(dem, buildings)

    return SyntheticScenario(
        dem=dem,
        river_mask=river_mask,
        gauge_history=gauge_history,
        rainfall_forecast=rainfall_forecast,
        buildings=buildings,
        roads=roads,
        shelters=shelters,
    )
