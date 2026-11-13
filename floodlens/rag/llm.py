"""
Model access for the RAG layer: chat completion and embeddings.

Chat backends, tried in order:
    1. Claude API              (ANTHROPIC_API_KEY set)
    2. Ollama                  (OLLAMA_HOST set or FLOODLENS_USE_OLLAMA=1)
    3. OpenAI-compatible local (LOCAL_LLM_BASE_URL set: llama.cpp `llama-server`,
                                LM Studio, vLLM ... e.g. http://localhost:8080/v1)
If none is configured or all fail, callers get (None, None) and use a
deterministic fallback, so retrieval keeps working offline and in tests.

Embeddings: Ollama /api/embed (OLLAMA_EMBED_MODEL, default nomic-embed-text)
when Ollama is enabled, otherwise a deterministic hashing embedder (numpy only).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.request

import numpy as np

from floodlens.alerts.claude_alert import ollama_chat, OLLAMA_DEFAULT_HOST

OLLAMA_EMBED_DEFAULT_MODEL = "nomic-embed-text"
HASH_DIM = 512


# --------------------------------------------------------------------- config

def ollama_enabled() -> bool:
    return bool(os.environ.get("OLLAMA_HOST") or os.environ.get("FLOODLENS_USE_OLLAMA") == "1")


def claude_enabled() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def local_openai_enabled() -> bool:
    return bool(os.environ.get("LOCAL_LLM_BASE_URL"))


def _ollama_host() -> str:
    host = (os.environ.get("OLLAMA_HOST") or OLLAMA_DEFAULT_HOST).rstrip("/")
    return host if host.startswith("http") else "http://" + host


# ----------------------------------------------------------------------- chat

def openai_compatible_chat(messages: list[dict], json_mode: bool = False, base_url: str | None = None,
                           model: str | None = None, timeout: float = 120.0) -> str:
    """Chat with any OpenAI-compatible local server (llama.cpp llama-server,
    LM Studio, vLLM, ...). Standard library only."""
    base_url = (base_url or os.environ["LOCAL_LLM_BASE_URL"]).rstrip("/")
    body = {
        "model": model or os.environ.get("LOCAL_LLM_MODEL", "local-model"),
        "messages": messages,
        "temperature": 0.2,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(
        f"{base_url}/chat/completions", data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {os.environ.get('LOCAL_LLM_API_KEY', 'not-needed')}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def _claude_chat(system: str, user: str, max_tokens: int, cached_document: str | None) -> str:
    import anthropic
    client = anthropic.Anthropic()
    content = []
    if cached_document:
        # Anthropic prompt caching: the whole document is cached once and reused
        # for every chunk of that document (contextual retrieval cost trick).
        content.append({"type": "text", "text": cached_document, "cache_control": {"type": "ephemeral"}})
    content.append({"type": "text", "text": user})
    resp = client.messages.create(
        model=os.environ.get("FLOODLENS_CLAUDE_MODEL", "claude-sonnet-4-6"),
        max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": content}],
    )
    return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


def complete(system: str, user: str, json_mode: bool = False, max_tokens: int = 300,
             cached_document: str | None = None) -> tuple[str | None, str | None]:
    """Returns (text, backend_name) or (None, None) if no backend answered."""
    attempts = []
    if claude_enabled():
        attempts.append(("claude_api", lambda: _claude_chat(system, user, max_tokens, cached_document)))
    full_user = f"{cached_document}\n\n{user}" if cached_document else user
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": full_user}]
    if ollama_enabled():
        if json_mode:
            attempts.append(("ollama_local", lambda: ollama_chat(msgs)))
        else:
            attempts.append(("ollama_local", lambda: _ollama_text(msgs)))
    if local_openai_enabled():
        attempts.append(("openai_compatible_local", lambda: openai_compatible_chat(msgs, json_mode=json_mode)))
    for name, fn in attempts:
        try:
            text = (fn() or "").strip()
            if text:
                return text, name
        except Exception:
            continue
    return None, None


def _ollama_text(messages: list[dict], timeout: float = 120.0) -> str:
    """Ollama chat without JSON mode (for free-text answers like HyDE / context)."""
    body = json.dumps({
        "model": os.environ.get("OLLAMA_MODEL", "qwen2.5:3b"),
        "messages": messages, "stream": False, "options": {"temperature": 0.2},
    }).encode("utf-8")
    req = urllib.request.Request(f"{_ollama_host()}/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8")).get("message", {}).get("content", "")


def parse_json(text: str | None) -> dict | None:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


# ------------------------------------------------------------------ embeddings

_TOKEN_RE = re.compile(r"[a-z0-9]+|[ঀ-৿]+")


def _stem(t: str) -> str:
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 4 and t.endswith("ing"):
        return t[:-3]
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN_RE.findall(text.lower())]


def hashing_embed(texts: list[str], dim: int = HASH_DIM) -> np.ndarray:
    """Deterministic bag-of-words + bigram hashing embedding, L2-normalised."""
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        toks = tokenize(t)
        feats = toks + [a + "_" + b for a, b in zip(toks, toks[1:])]
        for f in feats:
            h = int(hashlib.md5(f.encode()).hexdigest(), 16)
            out[i, h % dim] += 1.0 if (h >> 64) & 1 else -1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return out / norms


class Embedder:
    """Ollama embeddings when available, hashing fallback otherwise."""

    def __init__(self, prefer_ollama: bool | None = None):
        self.use_ollama = ollama_enabled() if prefer_ollama is None else prefer_ollama
        self.model = os.environ.get("OLLAMA_EMBED_MODEL", OLLAMA_EMBED_DEFAULT_MODEL)
        self.backend = "hashing"
        if self.use_ollama:
            try:
                self._dim = self._ollama_embed(["probe"]).shape[1]
                self.backend = f"ollama:{self.model}"
            except Exception:
                self.use_ollama = False

    def _ollama_embed(self, texts: list[str]) -> np.ndarray:
        body = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        req = urllib.request.Request(f"{_ollama_host()}/api/embed", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            vecs = np.asarray(json.loads(resp.read().decode("utf-8"))["embeddings"], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    @property
    def dim(self) -> int | None:
        return getattr(self, "_dim", None) if self.use_ollama else HASH_DIM

    def embed(self, texts: list[str]) -> np.ndarray:
        if not self.use_ollama:
            return hashing_embed(texts) if texts else np.zeros((0, HASH_DIM), dtype=np.float32)
        if not texts:
            return np.zeros((0, self.dim or HASH_DIM), dtype=np.float32)
        # Once this embedder has committed to Ollama vectors (e.g. 768-d for
        # nomic-embed-text) it must never hand back 512-d hashing vectors for a
        # later call: a single transient Ollama timeout used to silently mix
        # dimensions, and DuckDB's cosine search then failed with
        # "list dimensions must be equal". Retry, then fail loudly instead.
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                vecs = self._ollama_embed(texts)
                self._dim = vecs.shape[1]
                return vecs
            except Exception as exc:  # timeout / model reload / busy machine
                last_exc = exc
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(
            f"Ollama embedding with {self.model} failed after 3 attempts ({last_exc}); "
            "refusing to fall back to hashing vectors mid-run (dimension mismatch)."
        )
