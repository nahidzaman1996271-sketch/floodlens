"""
Builds the FloodLens RAG index and evaluates retrieval quality.

    python scripts/build_rag_index.py

With Ollama running (OLLAMA_HOST=http://localhost:11434, `ollama pull
nomic-embed-text` and `ollama pull qwen2.5:3b`) the index uses local
embeddings and model-written contextual chunk headers; with ANTHROPIC_API_KEY
the contexts are written by Claude with prompt caching. Without either it
still runs with the offline fallbacks.

Outputs:
    results/floodlens.duckdb      table rag_chunks (chunks, contexts, embeddings)
    results/rag_index_sample.json index stats + every chunk with its context
    results/rag_eval.json         hit@1 / hit@3 / MRR for naive vs contextual
                                  hybrid vs full pipeline, + a Graph RAG sample
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from floodlens.pipeline import run_pipeline
from floodlens.rag.retriever import FloodRAG

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# (question, the knowledge-base document that should answer it)
EVAL_SET = [
    ("What should my family carry when we leave the house?", "01_household_evacuation"),
    ("Who should be evacuated first from a home?", "01_household_evacuation"),
    ("Is it safe to walk through moving flood water?", "01_household_evacuation"),
    ("The shelter is almost full, where do new families go?", "02_shelter_operations"),
    ("How do we keep drinking water safe in the shelter?", "02_shelter_operations"),
    ("Where should women and girls stay inside the shelter?", "02_shelter_operations"),
    ("How should a clinic move vaccines and insulin before flooding?", "03_medical_facility_evacuation"),
    ("Where can people get treatment for snake bites during a flood?", "03_medical_facility_evacuation"),
    ("What do we do with our cattle and goats?", "04_livestock_and_assets"),
    ("How should we store rice seed so it is not lost?", "04_livestock_and_assets"),
    ("The village road is going under water, how do we evacuate?", "05_roads_and_boats"),
    ("What is the emergency phone number in Bangladesh?", "05_roads_and_boats"),
    ("How many people can safely ride in an evacuation boat?", "05_roads_and_boats"),
    ("What does the danger level at a river gauge mean?", "06_warning_levels"),
    ("Why can the forecast get worse very quickly?", "06_warning_levels"),
    ("What does CRITICAL risk mean in FloodLens?", "06_warning_levels"),
]

CONFIGS = {
    "naive_rag": dict(hyde=False, rerank=False, corrective=False, hybrid=False),
    "contextual_hybrid": dict(hyde=False, rerank=False, corrective=False, hybrid=True),
    "full_pipeline": dict(hyde=True, rerank=True, corrective=True, hybrid=True),
}


def evaluate(rag: FloodRAG, cfg: dict) -> dict:
    hits1 = hits3 = 0
    rr = 0.0
    misses = []
    for q, expected in EVAL_SET:
        docs = [r.chunk.doc_id for r in rag.retrieve(q, k=3, **cfg).results]
        if docs and docs[0] == expected:
            hits1 += 1
        if expected in docs:
            hits3 += 1
            rr += 1.0 / (docs.index(expected) + 1)
        else:
            misses.append({"question": q, "expected": expected, "got": docs})
    n = len(EVAL_SET)
    return {"hit@1": round(hits1 / n, 3), "hit@3": round(hits3 / n, 3), "mrr@3": round(rr / n, 3),
            "misses": misses}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    db_path = os.path.join(RESULTS_DIR, "floodlens.duckdb")

    t0 = time.perf_counter()
    contextual_rag = FloodRAG(db_path=db_path).build(contextual=True)
    build_s = time.perf_counter() - t0
    plain_rag = FloodRAG(embedder=contextual_rag.embedder, use_llm=contextual_rag.use_llm).build(contextual=False)
    print(f"Index built in {build_s:.1f}s: {contextual_rag.build_info}")

    report = {"index": contextual_rag.build_info, "build_seconds": round(build_s, 1), "configs": {}}
    report["configs"]["naive_rag"] = evaluate(plain_rag, CONFIGS["naive_rag"])
    for name in ("contextual_hybrid", "full_pipeline"):
        t = time.perf_counter()
        report["configs"][name] = evaluate(contextual_rag, CONFIGS[name])
        report["configs"][name]["seconds"] = round(time.perf_counter() - t, 1)
    for name, r in report["configs"].items():
        print(f"{name:18s} hit@1={r['hit@1']:.2f} hit@3={r['hit@3']:.2f} mrr@3={r['mrr@3']:.2f}")

    result = run_pipeline()
    contextual_rag.attach_graph(result)
    clinic = next(b for b in result["building_impacts"] if b.priority == "medical_priority")
    report["graph"] = contextual_rag.graph.stats()
    report["graph_rag_sample"] = {"building_id": clinic.id, **contextual_rag.context_for_building(clinic)}
    sample_q = "The shelter is almost full, where do new families go?"
    report["full_pipeline_trace_sample"] = {"question": sample_q,
                                            **contextual_rag.retrieve(sample_q).trace}

    with open(os.path.join(RESULTS_DIR, "rag_eval.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(os.path.join(RESULTS_DIR, "rag_index_sample.json"), "w", encoding="utf-8") as f:
        json.dump({"index": contextual_rag.build_info, "chunks": [
            {"chunk_id": c.chunk_id, "section": c.section, "chars": len(c.text),
             "context": c.context, "context_backend": c.context_backend, "text": c.text}
            for c in contextual_rag.chunks]}, f, ensure_ascii=False, indent=2)
    print("Wrote results/rag_eval.json and results/rag_index_sample.json")


if __name__ == "__main__":
    main()
