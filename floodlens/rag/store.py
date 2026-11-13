"""
Vector store for chunk embeddings.

Uses DuckDB (the same embedded database as the run store) with a FLOAT[]
embedding column and list_cosine_similarity() for nearest-neighbour search,
so the knowledge base lives next to the forecasts in results/floodlens.duckdb.
Falls back to an in-memory numpy matrix if duckdb is not installed.
"""

from __future__ import annotations

import numpy as np

from floodlens.rag.chunking import Chunk

_SCHEMA = """
CREATE TABLE IF NOT EXISTS rag_chunks (
    chunk_id VARCHAR PRIMARY KEY, doc_id VARCHAR, doc_title VARCHAR, section VARCHAR,
    text VARCHAR, context VARCHAR, context_backend VARCHAR, embedding FLOAT[]
)
"""


class VectorStore:
    def __init__(self, db_path: str = ":memory:"):
        self.backend = "numpy"
        self.con = None
        self.chunks: list[Chunk] = []
        self.matrix = None
        try:
            import duckdb
            self.con = duckdb.connect(db_path)
            self.con.execute(_SCHEMA)
            self.backend = "duckdb"
        except ImportError:
            pass

    def add(self, chunks: list[Chunk], vectors: np.ndarray):
        self.chunks = list(chunks)
        self.matrix = np.asarray(vectors, dtype=np.float32)
        if self.con is not None:
            self.con.execute("DELETE FROM rag_chunks")
            self.con.executemany(
                "INSERT INTO rag_chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [[c.chunk_id, c.doc_id, c.doc_title, c.section, c.text, c.context, c.context_backend,
                  v.astype(float).tolist()] for c, v in zip(chunks, self.matrix)],
            )

    def search(self, query_vec: np.ndarray, k: int = 10) -> list[tuple[int, float]]:
        """Returns [(chunk_index, cosine_similarity)] best first."""
        if not self.chunks:
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        if self.con is not None:
            rows = self.con.execute(
                "SELECT chunk_id, list_cosine_similarity(embedding, ?::FLOAT[]) AS sim "
                "FROM rag_chunks ORDER BY sim DESC LIMIT ?",
                [q.astype(float).tolist(), k],
            ).fetchall()
            index = {c.chunk_id: i for i, c in enumerate(self.chunks)}
            return [(index[cid], float(sim)) for cid, sim in rows]
        sims = self.matrix @ q
        order = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in order]
