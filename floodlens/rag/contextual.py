"""
Contextual Retrieval (Anthropic-style).

Before indexing, every chunk gets a short, model-written context that
situates it inside its source document ("This chunk is from the shelter
operations guide and explains when to redirect families to a second
shelter..."). The context is prepended to the chunk for BOTH the embedding
and the BM25 index, which fixes chunks that are ambiguous on their own.

With the Claude API, the full document is sent with prompt caching so it is
paid for once per document, not once per chunk. With Ollama or another local
server the same prompt runs on-device. Without any model, a deterministic
title/section header is used so indexing still works.
"""

from __future__ import annotations

from floodlens.rag import llm
from floodlens.rag.chunking import Chunk, Document

CONTEXT_SYSTEM = (
    "You write short retrieval contexts for a flood early-warning knowledge base in Bangladesh. "
    "Answer with the context only, one or two sentences, no preamble."
)

CONTEXT_USER = (
    "Here is the chunk we want to situate within the whole document above:\n"
    "<chunk>\n{chunk}\n</chunk>\n"
    "Please give a short succinct context to situate this chunk within the overall document "
    "for the purposes of improving search retrieval of the chunk. Answer only with the succinct "
    "context and nothing else."
)


def heuristic_context(chunk: Chunk) -> str:
    return f"From the FloodLens guide '{chunk.doc_title}', section '{chunk.section}'."


def contextualize(doc: Document, chunks: list[Chunk], use_llm: bool = True) -> list[Chunk]:
    document_block = f"<document>\n{doc.text}\n</document>"
    for c in chunks:
        text, backend = (None, None)
        if use_llm:
            text, backend = llm.complete(
                CONTEXT_SYSTEM, CONTEXT_USER.format(chunk=c.text),
                max_tokens=120, cached_document=document_block,
            )
        if text:
            # keep the heuristic header too: it anchors the doc title for BM25
            c.context = f"{heuristic_context(c)} {text.strip()}"
            c.context_backend = backend
        else:
            c.context = heuristic_context(c)
            c.context_backend = "heuristic"
    return chunks
