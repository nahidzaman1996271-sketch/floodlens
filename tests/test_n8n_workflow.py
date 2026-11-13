"""Structural checks for the n8n workflow export in workflows/."""

import json
import os

WF_PATH = os.path.join(os.path.dirname(__file__), "..", "workflows", "floodlens_n8n_workflow.json")


def _wf():
    with open(WF_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_workflow_is_valid_n8n_export():
    wf = _wf()
    assert wf["name"]
    assert isinstance(wf["nodes"], list) and wf["nodes"]
    for n in wf["nodes"]:
        assert {"id", "name", "type", "typeVersion", "position", "parameters"} <= n.keys()
        assert n["type"].startswith("n8n-nodes-base.")


def test_all_connections_point_to_existing_nodes():
    wf = _wf()
    names = {n["name"] for n in wf["nodes"]}
    for src, outputs in wf["connections"].items():
        assert src in names
        for branch in outputs["main"]:
            for target in branch:
                assert target["node"] in names


def test_workflow_calls_floodlens_api_on_a_schedule():
    wf = _wf()
    types = [n["type"] for n in wf["nodes"]]
    assert "n8n-nodes-base.scheduleTrigger" in types
    urls = [n["parameters"].get("url", "") for n in wf["nodes"] if n["type"] == "n8n-nodes-base.httpRequest"]
    assert any(u.endswith("/predict") for u in urls)
    assert any(u.endswith("/alerts") for u in urls)
