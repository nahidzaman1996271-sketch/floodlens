"""Tests for the FloodLens MCP server (floodlens/mcp_server.py)."""

import asyncio
import json

import pytest

pytest.importorskip("mcp")

from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402

from floodlens.mcp_server import mcp  # noqa: E402

EXPECTED_TOOLS = {
    "get_scenario_summary", "get_flood_forecast", "list_at_risk_buildings",
    "get_building_risk", "get_road_status", "get_flood_extent", "get_flood_alerts",
}


def _call(name, args=None):
    result = asyncio.run(mcp.call_tool(name, args or {}))
    return result


def _json(name, args=None):
    result = _call(name, args)
    assert not result.is_error, result.content
    return json.loads(result.content[0].text)


def test_all_tools_registered_and_read_only():
    tools = asyncio.run(mcp.list_tools())
    tools = getattr(tools, "tools", tools)
    assert {t.name for t in tools} == EXPECTED_TOOLS
    for t in tools:
        assert t.annotations is not None and t.annotations.read_only_hint is True


def test_forecast_matches_pipeline_export():
    data = _json("get_flood_forecast")
    assert data["hours_to_danger_level"] == 6.0
    assert data["hours_to_severe_level"] == 13.0
    assert data["impact_summary"]["total_buildings"] == 102


def test_at_risk_buildings_sorted_and_filtered():
    data = _json("list_at_risk_buildings", {"limit": 10, "risk_level": "critical"})
    hours = [b["hours_to_flood"] for b in data["buildings"]]
    assert hours == sorted(hours)
    assert all(b["risk_level"] == "CRITICAL" for b in data["buildings"])


def test_building_risk_roundtrip_and_unknown_id():
    first = _json("list_at_risk_buildings", {"limit": 1})["buildings"][0]
    one = _json("get_building_risk", {"building_id": first["building_id"]})
    assert one == first
    # In-process the error surfaces as ToolError; over the protocol it becomes is_error=True.
    with pytest.raises(ToolError, match="Unknown building_id"):
        _call("get_building_risk", {"building_id": "NOPE"})


def test_flood_extent_snaps_and_grows():
    early = _json("get_flood_extent", {"hour": 1})
    late = _json("get_flood_extent", {"hour": 72})
    assert 0 <= early["flooded_pct"] <= late["flooded_pct"] <= 100


def test_alerts_use_template_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    data = _json("get_flood_alerts", {"limit": 2})
    assert data["count"] == 2
    assert all(a["source"] == "template_fallback" and a["message_bn"] for a in data["alerts"])
