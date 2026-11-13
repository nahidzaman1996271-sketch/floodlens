"""
FloodLens API — FastAPI app exposing the end-to-end pipeline.

Run with: uvicorn floodlens.api.main:app --reload

Endpoints:
  GET  /health
  GET  /scenario/summary      -> locality info, current gauge level, thresholds
  GET  /predict                -> full forecast + impact assessment + KPI summary
  GET  /alerts?limit=10        -> plain-language alerts for the most urgent buildings
  GET  /flood-map/{hour}       -> boolean flood grid (JSON) for a given forecast hour
  GET  /ask?q=...              -> RAG answer sources from the flood-guidance knowledge base
  GET  /buildings/{id}/context -> Graph RAG facts + retrieved guidance for one building
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from floodlens.data.synthetic import DANGER_LEVEL_M, SEVERE_LEVEL_M
from floodlens.pipeline import run_pipeline as _pipeline
from floodlens.alerts.claude_alert import generate_alerts_for_critical
from floodlens.rag.retriever import get_default_rag


def _rag_enabled() -> bool:
    return os.environ.get("FLOODLENS_RAG", "1") != "0"

app = FastAPI(
    title="FloodLens API",
    description="From flood warning to flood action — street-level flood impact prediction.",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/scenario/summary")
def scenario_summary():
    p = _pipeline()
    s = p["scenario"]
    return {
        "locality_name": s.locality_name,
        "current_gauge_level_m": round(float(s.gauge_history["level_m"].iloc[-1]), 2),
        "danger_level_m": DANGER_LEVEL_M,
        "severe_level_m": SEVERE_LEVEL_M,
        "total_buildings": len(s.buildings),
        "total_shelters": len(s.shelters),
        "total_roads": len(s.roads),
        "data_note": "Synthetic scenario calibrated to plausible haor/floodplain hydrology. See README.",
    }


@app.get("/predict")
def predict():
    p = _pipeline()
    f = p["forecast"]
    return {
        "hours_to_danger_level": f.hours_to_danger,
        "hours_to_severe_level": f.hours_to_severe,
        "confidence_band_m": f.confidence_band_m,
        "impact_summary": p["summary"],
        "roads": [r.__dict__ for r in p["road_impacts"]],
        "top_critical_buildings": [
            b.__dict__ for b in sorted(
                [b for b in p["building_impacts"] if b.hours_to_flood is not None],
                key=lambda b: b.hours_to_flood,
            )[:15]
        ],
    }


@app.get("/alerts")
def alerts(limit: int = 10):
    p = _pipeline()
    rag = get_default_rag(p) if _rag_enabled() else None
    generated = generate_alerts_for_critical(p["building_impacts"], limit=limit, rag=rag)
    return {"count": len(generated), "alerts": [a.__dict__ for a in generated]}


@app.get("/flood-map/{hour}")
def flood_map(hour: int):
    p = _pipeline()
    timeline = p["timeline"]
    available_hours = sorted(timeline.keys())
    nearest = min(available_hours, key=lambda h: abs(h - hour))
    if nearest != hour and hour not in available_hours:
        pass  # snap to nearest sampled hour; documented via response field below
    grid = timeline[nearest]
    return {
        "requested_hour": hour,
        "returned_hour": nearest,
        "flooded_cell_count": int(grid.sum()),
        "total_cells": int(grid.size),
        "grid": grid.astype(int).tolist(),
    }


@app.get("/ask")
def ask(q: str, k: int = 3):
    rag = get_default_rag(_pipeline())
    res = rag.retrieve(q, k=k)
    return {
        "query": q,
        "results": [
            {"chunk_id": r.chunk.chunk_id, "document": r.chunk.doc_title, "section": r.chunk.section,
             "text": r.chunk.text, "context": r.chunk.context, "rerank_score": r.rerank_score}
            for r in res.results
        ],
        "trace": res.trace,
        "index": rag.build_info,
    }


@app.get("/buildings/{building_id}/context")
def building_context(building_id: str):
    p = _pipeline()
    impact = next((b for b in p["building_impacts"] if b.id == building_id), None)
    if impact is None:
        raise HTTPException(status_code=404, detail=f"Unknown building {building_id}")
    return get_default_rag(p).context_for_building(impact)
