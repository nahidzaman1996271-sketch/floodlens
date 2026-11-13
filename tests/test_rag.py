"""Tests for the RAG layer (offline: hashing embeddings, no LLM)."""

import pytest

from floodlens.pipeline import run_pipeline
from floodlens.rag import llm
from floodlens.rag.bm25 import BM25
from floodlens.rag.chunking import Document, load_markdown_docs, semantic_chunk
from floodlens.rag.graph import build_graph, graph_context
from floodlens.rag.retriever import KB_DIR, FloodRAG
from floodlens.alerts.claude_alert import generate_alerts_for_critical


@pytest.fixture(scope="module")
def rag():
    r = FloodRAG(embedder=llm.Embedder(prefer_ollama=False), use_llm=False).build()
    r.attach_graph(run_pipeline())
    return r


def test_knowledge_base_loads():
    docs = load_markdown_docs(KB_DIR)
    assert len(docs) >= 6
    assert all(d.sections for d in docs)


def test_semantic_chunking_produces_variable_sizes():
    docs = load_markdown_docs(KB_DIR)
    chunks = [c for d in docs for c in semantic_chunk(d, llm.hashing_embed)]
    sizes = {len(c.text) for c in chunks}
    assert len(chunks) > len(docs)
    assert max(sizes) - min(sizes) > 100       # variable, not fixed-size
    assert all(len(c.text) <= 900 for c in chunks)


def test_semantic_chunking_splits_on_topic_shift():
    doc = Document("t", "T", "", [("S", "Cattle need dry fodder. Goats need fodder too. "
                                       "Cattle and goats go to the killa. "
                                       "Insulin must stay cold. Vaccines need ice packs. "
                                       "Cold boxes keep insulin and vaccines cold.")])
    chunks = semantic_chunk(doc, llm.hashing_embed, percentile=50, min_chars=20)
    assert len(chunks) >= 2


def test_contextual_headers_added(rag):
    assert all(c.context for c in rag.chunks)
    assert all(c.doc_title in c.contextual_text for c in rag.chunks)


def test_bm25_prefers_matching_doc():
    bm = BM25(["boats carry people across water", "vaccines need cold boxes"])
    s = bm.scores("cold vaccines")
    assert s[1] > s[0]


@pytest.mark.parametrize("q,doc", [
    ("How should a clinic move vaccines and insulin?", "03_medical_facility_evacuation"),
    ("The shelter is almost full", "02_shelter_operations"),
    ("What does the danger level mean?", "06_warning_levels"),
    ("treatment for snake bites", "03_medical_facility_evacuation"),
])
def test_hybrid_retrieval_finds_right_document(rag, q, doc):
    res = rag.retrieve(q, k=3)
    assert doc in [r.chunk.doc_id for r in res.results]
    assert "hybrid_rrf" in res.trace["steps"] and "rerank" in res.trace["steps"]


def test_corrective_rag_rewrites_on_weak_results(rag):
    res = rag.retrieve("zzz qqq unrelated words", k=2)
    assert "crag_rewrite" in res.trace["steps"]


def test_knowledge_graph_and_graph_rag():
    result = run_pipeline()
    g = build_graph(result)
    assert g.stats()["nodes"] > len(result["building_impacts"])
    clinic = next(b for b in result["building_impacts"] if b.priority == "medical_priority")
    ctx = graph_context(g, clinic.id)
    assert ctx["topics"][0] == "03_medical_facility_evacuation"
    assert any("shelter" in f.lower() for f in ctx["facts"])
    assert any("road" in f.lower() for f in ctx["facts"])


def test_building_context_combines_graph_and_retrieval(rag):
    clinic = next(b for b in run_pipeline()["building_impacts"] if b.priority == "medical_priority")
    ctx = rag.context_for_building(clinic)
    assert ctx["facts"] and ctx["guidance"]
    assert any(s.startswith("03_medical") for s in ctx["sources"])


def test_alerts_with_rag_still_work_offline(rag):
    impacts = run_pipeline()["building_impacts"]
    alerts = generate_alerts_for_critical(impacts, limit=3, use_api=False, rag=rag)
    assert len(alerts) == 3
    assert all(a.source == "template_fallback" for a in alerts)


def test_vector_store_rejects_mixed_embedding_dimensions():
    """Regression: a 768-d query against a 512-d index must fail clearly,
    not deep inside DuckDB's cosine-similarity call."""
    import numpy as np
    from floodlens.rag.store import VectorStore
    from floodlens.rag.chunking import Chunk
    store = VectorStore(":memory:")
    chunks = [Chunk(f"d#{i}", "d", "D", "S", f"text {i}") for i in range(3)]
    store.add(chunks, llm.hashing_embed([c.text for c in chunks]))
    assert store.search(llm.hashing_embed(["text 1"])[0], k=1)[0][0] == 1
    with pytest.raises(ValueError, match="dims"):
        store.search(np.ones(768, dtype=np.float32), k=1)


def test_ollama_embedder_never_falls_back_to_hashing_mid_run(monkeypatch):
    """Regression: one transient Ollama failure used to return 512-d hashing
    vectors while the index held 768-d Ollama vectors."""
    import numpy as np
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def flaky(self, texts):
        calls["n"] += 1
        if calls["n"] == 2:  # first embed after the probe fails once, then recovers
            raise TimeoutError("busy")
        return np.ones((len(texts), 768), dtype=np.float32) / np.sqrt(768)

    monkeypatch.setattr(llm.Embedder, "_ollama_embed", flaky)
    e = llm.Embedder(prefer_ollama=True)
    assert e.embed(["a", "b"]).shape == (2, 768)

    monkeypatch.setattr(llm.Embedder, "_ollama_embed",
                        lambda self, t: (_ for _ in ()).throw(TimeoutError("down")))
    with pytest.raises(RuntimeError, match="dimension mismatch"):
        e.embed(["c"])
