"""
FloodRAG: the retrieval pipeline that grounds FloodLens alerts.

    query
      -> Query rewriting / HyDE      (LLM writes a hypothetical guidance passage;
                                      its embedding is averaged with the query's)
      -> Hybrid search               (DuckDB vector search  +  BM25 keywords,
                                      both over contextualised chunks,
                                      fused with Reciprocal Rank Fusion)
      -> Reranker                    (LLM scores each candidate 0-10 for
                                      relevance; lexical cross-scorer offline)
      -> Corrective RAG (CRAG)       (if the best candidate is graded weak,
                                      rewrite the query once and retry; if still
                                      weak, fall back to graph-selected topics)
      -> top-k chunks + trace of which steps ran and which model ran them

Graph RAG (graph.py) supplies structured facts and topic hints per building;
context_for_building() combines both into the text given to the alert model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from floodlens.rag import llm
from floodlens.rag.bm25 import BM25
from floodlens.rag.chunking import Chunk, load_markdown_docs, semantic_chunk
from floodlens.rag.contextual import contextualize
from floodlens.rag.graph import KnowledgeGraph, build_graph, graph_context
from floodlens.rag.store import VectorStore

KB_DIR = os.path.join(os.path.dirname(__file__), "knowledge")
RRF_K = 60
CRAG_MIN_SCORE = 5.0  # reranker score (0-10) below which retrieval is "incorrect"

HYDE_SYSTEM = ("You write short passages for a flood-preparedness guide used in rural Bangladesh. "
               "Write 2-3 factual sentences that would answer the question. No preamble.")
RERANK_SYSTEM = ("You grade whether a passage helps answer a question about flood preparedness. "
                 'Reply as JSON: {"score": <integer 0-10>}.')
REWRITE_SYSTEM = ("Rewrite the user's question into a clearer search query for a flood "
                  "preparedness knowledge base. Reply with the query only.")


@dataclass
class Retrieved:
    chunk: Chunk
    score: float
    rerank_score: float | None = None


@dataclass
class RetrievalResult:
    query: str
    results: list[Retrieved]
    trace: dict = field(default_factory=dict)

    def as_text(self) -> str:
        return "\n".join(f"- ({r.chunk.doc_title} / {r.chunk.section}) {r.chunk.text}" for r in self.results)


class FloodRAG:
    def __init__(self, embedder: llm.Embedder | None = None, db_path: str = ":memory:",
                 use_llm: bool | None = None):
        self.embedder = embedder or llm.Embedder()
        self.use_llm = (llm.claude_enabled() or llm.ollama_enabled() or llm.local_openai_enabled()) \
            if use_llm is None else use_llm
        self.store = VectorStore(db_path)
        self.chunks: list[Chunk] = []
        self.bm25: BM25 | None = None
        self.graph: KnowledgeGraph | None = None
        self.build_info: dict = {}

    # ------------------------------------------------------------- indexing
    def build(self, kb_dir: str = KB_DIR, contextual: bool = True) -> "FloodRAG":
        docs = load_markdown_docs(kb_dir)
        chunks: list[Chunk] = []
        for d in docs:
            dc = semantic_chunk(d, self.embedder.embed)
            if contextual:
                contextualize(d, dc, use_llm=self.use_llm)
            chunks.extend(dc)
        self.chunks = chunks
        texts = [c.contextual_text for c in chunks]
        self.store.add(chunks, self.embedder.embed(texts))
        self.bm25 = BM25(texts)
        backends = sorted({c.context_backend for c in chunks if c.context_backend})
        self.build_info = {
            "documents": len(docs), "chunks": len(chunks),
            "chunk_chars_min": min(len(c.text) for c in chunks),
            "chunk_chars_max": max(len(c.text) for c in chunks),
            "embedding_backend": self.embedder.backend, "vector_store": self.store.backend,
            "context_backends": backends, "contextual": contextual,
        }
        return self

    def attach_graph(self, pipeline_result: dict) -> KnowledgeGraph:
        self.graph = build_graph(pipeline_result)
        return self.graph

    # ------------------------------------------------------------ retrieval
    def _hyde_vector(self, query: str, trace: dict) -> np.ndarray:
        qv = self.embedder.embed([query])[0]
        if not self.use_llm:
            return qv
        passage, backend = llm.complete(HYDE_SYSTEM, query, max_tokens=150)
        if not passage:
            return qv
        trace["hyde"] = {"backend": backend, "passage": passage[:300]}
        hv = self.embedder.embed([passage])[0]
        v = qv + hv
        return v / (np.linalg.norm(v) or 1.0)

    def _hybrid(self, query: str, qvec: np.ndarray, n: int = 12) -> list[tuple[int, float]]:
        vec_hits = self.store.search(qvec, k=n)
        bm = self.bm25.scores(query)
        kw_order = [i for i in np.argsort(-np.asarray(bm)) if bm[i] > 0][:n]
        fused: dict[int, float] = {}
        for rank, (i, _) in enumerate(vec_hits):
            fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, i in enumerate(kw_order):
            fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (RRF_K + rank + 1)
        return sorted(fused.items(), key=lambda x: -x[1])

    def _lexical_score(self, query: str, chunk: Chunk) -> float:
        """Offline cross-scorer: IDF-weighted share of query terms found in the chunk."""
        from floodlens.rag.bm25 import _terms
        q = set(_terms(query))
        c = set(_terms(chunk.contextual_text))
        idf = self.bm25.idf if self.bm25 else {}
        total = sum(idf.get(t, 1.0) for t in q) or 1.0
        return round(10.0 * sum(idf.get(t, 1.0) for t in q & c) / total, 2)

    def _rerank(self, query: str, cands: list[Retrieved], trace: dict) -> list[Retrieved]:
        backend_used = "lexical"
        for r in cands:
            score = None
            if self.use_llm:
                text, backend = llm.complete(
                    RERANK_SYSTEM, f"Question: {query}\n\nPassage: {r.chunk.contextual_text}",
                    json_mode=True, max_tokens=20)
                data = llm.parse_json(text)
                if data and isinstance(data.get("score"), (int, float)):
                    score, backend_used = float(data["score"]), backend
            r.rerank_score = score if score is not None else self._lexical_score(query, r.chunk)
        trace["reranker"] = backend_used
        return sorted(cands, key=lambda r: -(r.rerank_score or 0))

    def _rewrite(self, query: str) -> str:
        if self.use_llm:
            text, _ = llm.complete(REWRITE_SYSTEM, query, max_tokens=60)
            if text:
                return text.strip().strip('"')
        return query + " flood evacuation shelter safety guidance"

    def retrieve(self, query: str, k: int = 3, hyde: bool = True, rerank: bool = True,
                 corrective: bool = True, topic_hint: list[str] | None = None,
                 candidates: int = 6, hybrid: bool = True) -> RetrievalResult:
        assert self.bm25 is not None, "call build() first"
        trace: dict = {"steps": []}

        def run(q: str) -> list[Retrieved]:
            qvec = self._hyde_vector(q, trace) if hyde else self.embedder.embed([q])[0]
            trace["steps"].append("hyde" if hyde else "embed")
            if hybrid:
                fused = self._hybrid(q, qvec)
                trace["steps"].append("hybrid_rrf")
            else:
                fused = self.store.search(qvec, k=12)
                trace["steps"].append("vector_only")
            if topic_hint:  # Graph RAG: boost chunks from topics the graph linked to this case
                def boost(doc_id: str) -> float:
                    if doc_id not in topic_hint:
                        return 0.0
                    return 0.03 if doc_id == topic_hint[0] else 0.015
                extra = [(i, 0.0) for i, c in enumerate(self.chunks)
                         if c.doc_id in topic_hint and i not in {j for j, _ in fused}]
                fused = sorted(fused + extra, key=lambda x: -(x[1] + boost(self.chunks[x[0]].doc_id)))
            cands = [Retrieved(self.chunks[i], s) for i, s in fused[:candidates]]
            if rerank:
                cands = self._rerank(q, cands, trace)
                trace["steps"].append("rerank")
            return cands

        results = run(query)
        if corrective and rerank and results and (results[0].rerank_score or 0) < CRAG_MIN_SCORE:
            trace["steps"].append("crag_rewrite")
            new_q = self._rewrite(query)
            trace["crag"] = {"reason": "low relevance", "rewritten_query": new_q}
            retry = run(new_q)
            if retry and (retry[0].rerank_score or 0) > (results[0].rerank_score or 0):
                results = retry
            if (results[0].rerank_score or 0) < CRAG_MIN_SCORE and topic_hint:
                trace["steps"].append("crag_graph_fallback")
                on_topic = [r for r in results if r.chunk.doc_id in topic_hint]
                if not on_topic:
                    on_topic = [Retrieved(c, 0.0, None) for c in self.chunks if c.doc_id == topic_hint[0]]
                results = on_topic + [r for r in results if r not in on_topic]
        return RetrievalResult(query=query, results=results[:k], trace=trace)

    # -------------------------------------------------------- per building
    def context_for_building(self, impact, k: int = 3) -> dict:
        g = graph_context(self.graph, impact.id) if self.graph else {"facts": [], "topics": []}
        if impact.hours_to_flood is None:
            q = "No flooding is forecast. How should households keep monitoring warnings?"
        else:
            q = (f"A {impact.type.replace('_', ' ')} at {impact.risk_level} flood risk will flood in "
                 f"{impact.hours_to_flood} hours. What should people do and what should they carry?")
            if impact.priority == "medical_priority":
                q += " It is a health facility with patients."
            elif impact.priority == "vulnerable_group_priority":
                q += " Children and vulnerable people are present."
        res = self.retrieve(q, k=k, topic_hint=g["topics"])
        return {
            "query": q, "facts": g["facts"], "topics": g["topics"],
            "guidance": [r.chunk.text for r in res.results],
            "sources": [r.chunk.chunk_id for r in res.results], "trace": res.trace,
        }


_DEFAULT: FloodRAG | None = None


def get_default_rag(pipeline_result: dict | None = None) -> FloodRAG:
    """Lazily built shared instance (used by the API and alert layer)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = FloodRAG().build()
    if pipeline_result is not None and _DEFAULT.graph is None:
        _DEFAULT.attach_graph(pipeline_result)
    return _DEFAULT
