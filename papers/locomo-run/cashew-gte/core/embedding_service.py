"""
Embedding service: the single entry point every caller should use.

Strategy per call:
    1. Look up each text in the content-hash cache.
    2. For cache misses, try the warm daemon over its unix socket.
    3. If the daemon is unreachable, load the model in-process and embed.
    4. Write misses back to the cache.

Callers never see the model directly. That means any future change to the
backend (larger model, remote inference, GPU offload) only touches this file
plus the cache, and every consumer benefits transparently.

The service is deliberately stateless at the module level — no hidden
singletons. Inject a `Backend` protocol for tests.
"""

from __future__ import annotations

from typing import List, Optional, Protocol, Sequence, Union

import numpy as np

from .embedding_cache import EmbeddingCache

import os as _os
DEFAULT_MODEL = _os.environ.get("CASHEW_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM = int(_os.environ.get("CASHEW_EMBEDDING_DIM", "384"))

TextLike = Union[str, Sequence[str]]


class Backend(Protocol):
    """Minimal contract for something that can encode texts into vectors."""

    def encode(self, texts: List[str]) -> np.ndarray: ...


class LocalBackend:
    """In-process sentence-transformer. Lazy-loads to keep import cheap."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._model = None

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model.encode(texts, convert_to_numpy=True).astype(np.float32)


class DaemonBackend:
    """Talks to the warm daemon over its unix socket. Returns empty array on
    any transport failure so the service can fall back cleanly."""

    def __init__(self, socket_path: Optional[str] = None) -> None:
        self.socket_path = socket_path

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        from .daemon import client_request
        resp = client_request(
            {"op": "embed_batch", "texts": list(texts)},
            socket_path=self.socket_path,
        )
        if resp is None or not resp.get("ok"):
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        vectors = resp.get("result") or []
        if len(vectors) != len(texts):
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        return np.asarray(vectors, dtype=np.float32)


class EmbeddingService:
    """Cache-first, daemon-second, local-last embedding service."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        cache: Optional[EmbeddingCache] = None,
        daemon: Optional[Backend] = None,
        local: Optional[Backend] = None,
    ) -> None:
        self.model = model
        self.cache = cache if cache is not None else EmbeddingCache()
        self.daemon = daemon if daemon is not None else DaemonBackend()
        self.local = local if local is not None else LocalBackend(model)

    def embed(self, text: TextLike) -> Union[List[float], List[List[float]]]:
        """Embed one string or a list of strings.

        Returns a single vector for a string input, a list of vectors for a
        sequence input. Empty/whitespace strings yield the zero vector.
        """
        single = isinstance(text, str)
        texts = [text] if single else list(text)
        vectors = self._embed_batch(texts)
        if single:
            return vectors[0].tolist()
        return [v.tolist() for v in vectors]

    def embed_np(self, texts: Sequence[str]) -> np.ndarray:
        """Same as embed() but returns an (N, D) ndarray for numeric callers."""
        vectors = self._embed_batch(list(texts))
        if not vectors:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        return np.stack(vectors)

    def _embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        if not texts:
            return []
        # Empty strings -> zero vectors, never hit model/cache.
        results: List[Optional[np.ndarray]] = [None] * len(texts)
        nonempty_idx: List[int] = []
        nonempty_texts: List[str] = []
        for i, t in enumerate(texts):
            if not t or not t.strip():
                results[i] = np.zeros(EMBEDDING_DIM, dtype=np.float32)
            else:
                nonempty_idx.append(i)
                nonempty_texts.append(t)

        if nonempty_texts:
            cached = self.cache.get_many(self.model, nonempty_texts)
            miss_idx: List[int] = []
            miss_texts: List[str] = []
            for pos, vec in enumerate(cached):
                if vec is not None:
                    results[nonempty_idx[pos]] = vec
                else:
                    miss_idx.append(pos)
                    miss_texts.append(nonempty_texts[pos])

            if miss_texts:
                computed = self._compute(miss_texts)
                to_store = []
                for pos, vec in zip(miss_idx, computed):
                    results[nonempty_idx[pos]] = vec
                    to_store.append((nonempty_texts[pos], vec))
                self.cache.put_many(self.model, to_store)

        # mypy/humans: no Nones remain
        return [r for r in results if r is not None]

    def _compute(self, texts: List[str]) -> List[np.ndarray]:
        """Daemon first, then local. Returns one vector per input text."""
        vecs = self.daemon.encode(texts)
        if len(vecs) == len(texts):
            return [vecs[i] for i in range(len(texts))]
        vecs = self.local.encode(texts)
        return [vecs[i] for i in range(len(texts))]


_default_service: Optional[EmbeddingService] = None


def get_default_service() -> EmbeddingService:
    """Module-level singleton used by the thin convenience wrappers. Tests
    should construct their own EmbeddingService with injected dependencies
    rather than touching this."""
    global _default_service
    if _default_service is None:
        _default_service = EmbeddingService()
    return _default_service


def reset_default_service() -> None:
    """Drop the cached default service — used by tests to force rebuild."""
    global _default_service
    _default_service = None


def embed(text: TextLike) -> Union[List[float], List[List[float]]]:
    return get_default_service().embed(text)


def embed_np(texts: Sequence[str]) -> np.ndarray:
    return get_default_service().embed_np(texts)
