"""
Vector store for chunk embeddings.

Uses DuckDB (the same embedded database as the run store) with a fixed-size
FLOAT[dim] ARRAY embedding column and array_cosine_similarity() for
nearest-neighbour search, so the knowledge base lives next to the forecasts in
results/floodlens.duckdb. The fixed size makes DuckDB itself reject vectors of
the wrong dimension (an open FLOAT[] list accepted mixed 512-d hashing and
768-d Ollama vectors, which only failed later at query time).
Falls back to an in-memory numpy matrix if duckdb is not installed.
"""

from __future__ import annotations

import numpy as np

from floodlens.rag.chunking import Chunk


def _schema(dim: int) -> str:
    return f"""
CREATE TABLE rag_chunks (
    chunk_id VARCHAR PRIMARY KEY, doc_id VARCHAR, doc_title VARCHAR, section VARCHAR,
    text VARCHAR, context VARCHAR, context_backend VARCHAR, embedding FLOAT[{dim}]
)
"""


class VectorStore:
    def __init__(self, db_path: str = ":memory:"):
        self.backend = "numpy"
        self.con = None
        self.chunks: list[Chunk] = []
        self.matrix = None
        self.dim: int | None = None
        try:
            import duckdb
            self.con = duckdb.connect(db_path)
            self.backend = "duckdb"
        except ImportError:
            pass

    def add(self, chunks: list[Chunk], vectors: np.ndarray):
        self.chunks = list(chunks)
        self.matrix = np.asarray(vectors, dtype=np.float32)
        self.dim = int(self.matrix.shape[1]) if self.matrix.ndim == 2 and len(self.matrix) else None
        if self.con is not None and self.dim:
            # The index is rebuilt from scratch on every build, so recreate the
            # table with the current embedding size (older files may hold FLOAT[]).
            self.con.execute("DROP TABLE IF EXISTS rag_chunks")
            self.con.execute(_schema(self.dim))
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
        if self.dim is not None and q.shape[-1] != self.dim:
            raise ValueError(
                f"query embedding has {q.shape[-1]} dims but the index was built with {self.dim}; "
                "use the same Embedder for indexing and querying"
            )
        if self.con is not None:
            rows = self.con.execute(
                f"SELECT chunk_id, array_cosine_similarity(embedding, ?::FLOAT[{self.dim}]) AS sim "
                "FROM rag_chunks ORDER BY sim DESC LIMIT ?",
                [q.astype(float).tolist(), k],
            ).fetchall()
            index = {c.chunk_id: i for i, c in enumerate(self.chunks)}
            return [(index[cid], float(sim)) for cid, sim in rows]
        sims = self.matrix @ q
        order = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in order]
