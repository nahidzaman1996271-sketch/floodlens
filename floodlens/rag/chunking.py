"""
Variable-size semantic chunking.

Documents are first split on their markdown sections, then into sentences.
Consecutive sentences are embedded and a chunk boundary is placed wherever
the cosine distance between neighbouring sentences is in the top
(100 - percentile)% for the document -- i.e. where the topic shifts --
subject to min/max chunk sizes. Chunks therefore vary in length with the
content instead of being cut every N characters.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np

_SENT_RE = re.compile(r"(?<=[.!?।])\s+")


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    sections: list[tuple[str, str]] = field(default_factory=list)  # (heading, body)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    section: str
    text: str
    context: str = ""          # filled by contextual retrieval
    context_backend: str = ""  # which model wrote the context

    @property
    def contextual_text(self) -> str:
        return f"{self.context}\n\n{self.text}" if self.context else self.text


def load_markdown_docs(kb_dir: str) -> list[Document]:
    docs = []
    for name in sorted(os.listdir(kb_dir)):
        if not name.endswith(".md"):
            continue
        with open(os.path.join(kb_dir, name), encoding="utf-8") as f:
            text = f.read()
        title_m = re.search(r"^#\s+(.+)$", text, re.M)
        title = title_m.group(1).strip() if title_m else name
        sections = []
        for m in re.finditer(r"^##\s+(.+?)\n(.*?)(?=^##\s|\Z)", text, re.M | re.S):
            body = re.sub(r"\s+", " ", m.group(2)).strip()
            if body:
                sections.append((m.group(1).strip(), body))
        docs.append(Document(doc_id=name[:-3], title=title, text=text, sections=sections))
    return docs


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


def semantic_chunk(doc: Document, embed_fn, percentile: float = 70.0,
                   min_chars: int = 120, max_chars: int = 700) -> list[Chunk]:
    chunks: list[Chunk] = []
    for heading, body in doc.sections:
        sents = split_sentences(body)
        if not sents:
            continue
        if len(sents) == 1:
            groups = [sents]
        else:
            vecs = np.asarray(embed_fn(sents), dtype=np.float32)
            sims = np.sum(vecs[:-1] * vecs[1:], axis=1)
            dists = 1.0 - sims
            threshold = float(np.percentile(dists, percentile))
            groups, current = [], [sents[0]]
            for i, s in enumerate(sents[1:]):
                cur_len = sum(len(x) for x in current)
                topic_shift = dists[i] >= threshold and cur_len >= min_chars
                if topic_shift or cur_len + len(s) > max_chars:
                    groups.append(current)
                    current = [s]
                else:
                    current.append(s)
            groups.append(current)
            # merge a too-short tail into the previous chunk
            if len(groups) > 1 and sum(len(x) for x in groups[-1]) < min_chars:
                groups[-2].extend(groups.pop())
        for g in groups:
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}#{len(chunks)}", doc_id=doc.doc_id,
                doc_title=doc.title, section=heading, text=" ".join(g),
            ))
    return chunks
