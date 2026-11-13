"""Tests for the DuckDB analytical store. Skipped if duckdb is not installed."""

import pytest

duckdb = pytest.importorskip("duckdb")

from floodlens.pipeline import run_pipeline
from floodlens.storage import duckdb_store


@pytest.fixture
def con():
    c = duckdb_store.connect(":memory:")
    yield c
    c.close()


def test_save_run_persists_all_tables(con):
    result = run_pipeline()
    run_id = duckdb_store.save_run(result, con=con)

    runs = duckdb_store.query("SELECT * FROM runs", con=con)
    assert list(runs["run_id"]) == [run_id]
    assert runs["total_buildings"].iloc[0] == result["summary"]["total_buildings"]

    counts = {
        t: duckdb_store.query(f"SELECT COUNT(*) AS n FROM {t} WHERE run_id = ?", [run_id], con=con)["n"].iloc[0]
        for t in ["gauge_history", "rainfall_forecast", "level_forecast", "building_impacts", "road_impacts"]
    }
    assert counts["gauge_history"] == len(result["scenario"].gauge_history)
    assert counts["rainfall_forecast"] == len(result["scenario"].rainfall_forecast)
    assert counts["level_forecast"] == len(result["forecast"].forecast)
    assert counts["building_impacts"] == len(result["building_impacts"])
    assert counts["road_impacts"] == len(result["road_impacts"])


def test_risk_breakdown_matches_summary(con):
    result = run_pipeline()
    duckdb_store.save_run(result, con=con)
    breakdown = duckdb_store.query(duckdb_store.RISK_BREAKDOWN_SQL, con=con)
    assert breakdown["buildings"].sum() == result["summary"]["total_buildings"]


def test_multiple_runs_are_kept_separately(con):
    result = run_pipeline()
    a = duckdb_store.save_run(result, con=con)
    b = duckdb_store.save_run(result, con=con)
    assert a != b
    assert duckdb_store.query("SELECT COUNT(*) AS n FROM runs", con=con)["n"].iloc[0] == 2
