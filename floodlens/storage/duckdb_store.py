"""
DuckDB analytical store for FloodLens.

Persists every pipeline run (gauge history, rainfall forecast, water-level
forecast, building and road impacts) into an embedded DuckDB file so runs
can be queried with SQL and compared over time, instead of living only in
an in-memory cache and a JSON export.

Default location: results/floodlens.duckdb

Tables (all keyed by run_id):
    runs               one row per pipeline run + headline metrics
    gauge_history      hourly observed river level
    rainfall_forecast  hourly forecast rainfall (mm)
    level_forecast     hourly predicted river level (m)
    building_impacts   per-building risk, hours-to-flood, nearest shelter
    road_impacts       per-road status, hours-to-inaccessible
"""

from __future__ import annotations

import os
import uuid
from dataclasses import asdict
from datetime import datetime, timezone

import pandas as pd

DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "results", "floodlens.duckdb")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id VARCHAR PRIMARY KEY,
    created_at TIMESTAMP,
    locality_name VARCHAR,
    current_gauge_level_m DOUBLE,
    hours_to_danger DOUBLE,
    hours_to_severe DOUBLE,
    confidence_band_m DOUBLE,
    buildings_at_risk INTEGER,
    total_buildings INTEGER,
    roads_inaccessible INTEGER
);
CREATE TABLE IF NOT EXISTS gauge_history (run_id VARCHAR, timestamp TIMESTAMP, level_m DOUBLE);
CREATE TABLE IF NOT EXISTS rainfall_forecast (run_id VARCHAR, timestamp TIMESTAMP, rainfall_mm DOUBLE);
CREATE TABLE IF NOT EXISTS level_forecast (run_id VARCHAR, timestamp TIMESTAMP, predicted_level_m DOUBLE);
CREATE TABLE IF NOT EXISTS building_impacts (
    run_id VARCHAR, id VARCHAR, type VARCHAR, "row" INTEGER, col INTEGER, elevation_m DOUBLE,
    hours_to_flood INTEGER, risk_level VARCHAR, nearest_shelter_id VARCHAR,
    nearest_shelter_name VARCHAR, distance_to_shelter_km DOUBLE, priority VARCHAR
);
CREATE TABLE IF NOT EXISTS road_impacts (
    run_id VARCHAR, id VARCHAR, name VARCHAR, hours_to_inaccessible INTEGER, status VARCHAR
);
"""


def connect(db_path: str = DEFAULT_DB_PATH):
    """Open (or create) the DuckDB file and ensure the schema exists."""
    import duckdb  # imported lazily so the rest of FloodLens runs without it

    if db_path != ":memory:":
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    con = duckdb.connect(db_path)
    con.execute(_SCHEMA)
    return con


def _insert_df(con, table: str, df: pd.DataFrame, run_id: str, columns: list[str]) -> None:
    frame = df[columns].copy()
    frame.insert(0, "run_id", run_id)
    con.register("_tmp_frame", frame)
    quoted = ", ".join(['run_id'] + [f'"{c}"' for c in columns])
    con.execute(f"INSERT INTO {table} ({quoted}) SELECT {quoted} FROM _tmp_frame")
    con.unregister("_tmp_frame")


def save_run(result: dict, db_path: str = DEFAULT_DB_PATH, con=None) -> str:
    """Persist one pipeline result (the dict returned by run_pipeline()).
    Returns the new run_id."""
    own_con = con is None
    con = con or connect(db_path)
    run_id = uuid.uuid4().hex[:12]

    scenario = result["scenario"]
    forecast = result["forecast"]
    summary = result["summary"]

    con.execute(
        "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            run_id,
            datetime.now(timezone.utc).replace(tzinfo=None),
            scenario.locality_name,
            float(scenario.gauge_history["level_m"].iloc[-1]),
            forecast.hours_to_danger,
            forecast.hours_to_severe,
            forecast.confidence_band_m,
            summary["buildings_at_risk"],
            summary["total_buildings"],
            summary["roads_inaccessible"],
        ],
    )
    _insert_df(con, "gauge_history", scenario.gauge_history, run_id, ["timestamp", "level_m"])
    _insert_df(con, "rainfall_forecast", scenario.rainfall_forecast, run_id, ["timestamp", "rainfall_mm"])
    _insert_df(con, "level_forecast", forecast.forecast, run_id, ["timestamp", "predicted_level_m"])

    buildings = pd.DataFrame([asdict(b) for b in result["building_impacts"]])
    if not buildings.empty:
        _insert_df(con, "building_impacts", buildings, run_id, list(buildings.columns))
    roads = pd.DataFrame([asdict(r) for r in result["road_impacts"]])
    if not roads.empty:
        _insert_df(con, "road_impacts", roads, run_id, list(roads.columns))

    if own_con:
        con.close()
    return run_id


def query(sql: str, params: list | None = None, db_path: str = DEFAULT_DB_PATH, con=None) -> pd.DataFrame:
    """Run a SQL query against the store and return a DataFrame."""
    own_con = con is None
    con = con or connect(db_path)
    try:
        return con.execute(sql, params or []).df()
    finally:
        if own_con:
            con.close()


# Example analytical query used by the demo: risk breakdown for the latest run.
RISK_BREAKDOWN_SQL = """
SELECT risk_level, COUNT(*) AS buildings, MIN(hours_to_flood) AS earliest_flood_h
FROM building_impacts
WHERE run_id = (SELECT run_id FROM runs ORDER BY created_at DESC LIMIT 1)
GROUP BY risk_level
ORDER BY earliest_flood_h NULLS LAST
"""
