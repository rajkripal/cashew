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

Model selection: the default service reads `CASHEW_EMBEDDING_MODEL` via
`core.config.get_embedding_model()`. The dimension is derived from the loaded
model on first encode, not hardcoded — swapping models from MiniLM (384) to
gte-large (1024) just works.
"""

from __future__ import annotations

from typing import List, Optional, Protocol, Sequence, Union

import numpy as np

from .embedding_cache import EmbeddingCache

# Legacy constant kept for back-compat with code that imports it directly
# (daemon, tests, permanence). It reflects MiniLM-L6-v2's dimension. New code
# should use the service's `dim` property instead.
EMBEDDING_DIM = 384


def _resolve_default_model() -> str:
    """Resolve the configured embedding model name. Falls back to MiniLM if
    config is unavailable (e.g. tests that import this module standalone)."""
    try:
        from .config import get_embedding_model
        return get_embedding_model()
    except Exception:
        return "all-MiniLM-L6-v2"


# Kept for back-compat with imports — but tests/daemon now read this lazily.
DEFAULT_MODEL = "all-MiniLM-L6-v2"

# Best-effort known dims so we can size zero vectors correctly even before
# the model loads. If a model isn't listed, we lazily probe it at first use.
_KNOWN_DIMS = {
    "all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "thenlper/gte-large": 1024,
    "thenlper/gte-base": 768,
    "thenlper/gte-small": 384,
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-small-en-v1.5": 384,
}


def resolve_embedding_dim(model_name: Optional[str] = None) -> int:
    """Resolve the embedding dimension for ``model_name``.

    Checks the known-dims table first (cheap, no model load). Falls back to
    constructing a LocalBackend and probing the loaded SentenceTransformer.
    Used by callers (e.g. sqlite-vec table creation in core/embeddings.py)
    that need the dim before any embed call has happened. Single source of
    truth: keep this in sync with _KNOWN_DIMS rather than duplicating the
    table elsewhere.
    """
    name = model_name or _resolve_default_model()
    if name in _KNOWN_DIMS:
        return _KNOWN_DIMS[name]
    # Unknown model — probe by loading it. LocalBackend.dim handles this.
    return LocalBackend(name).dim


TextLike = Union[str, Sequence[str]]


class Backend(Protocol):
    """Minimal contract for something that can encode texts into vectors."""

    def encode(self, texts: List[str]) -> np.ndarray: ...


class LocalBackend:
    """In-process sentence-transformer. Lazy-loads to keep import cheap."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name or _resolve_default_model()
        self._model = None
        self._dim: Optional[int] = _KNOWN_DIMS.get(self.model_name)

    @property
    def dim(self) -> int:
        """Embedding dimension for this backend's model. Loads the model if
        the dim isn't known a priori."""
        if self._dim is None:
            self._ensure_model()
            self._dim = self._model.get_sentence_embedding_dimension()
        return self._dim

    def _ensure_model(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        self._ensure_model()
        # Update dim from the actual loaded model (authoritative).
        self._dim = self._model.get_sentence_embedding_dimension()
        return self._model.encode(texts, convert_to_numpy=True).astype(np.float32)


class DaemonBackend:
    """Talks to the warm daemon over its unix socket. Returns empty array on
    any transport failure so the service can fall back cleanly.

    Note: the daemon serves whatever model it was started with. Callers that
    set CASHEW_EMBEDDING_MODEL to override should ensure the daemon is
    restarted, otherwise the daemon's model wins for daemon-served requests.
    """

    def __init__(
        self,
        socket_path: Optional[str] = None,
        model_name: Optional[str] = None,
        dim: Optional[int] = None,
    ) -> None:
        self.socket_path = socket_path
        self.model_name = model_name or _resolve_default_model()
        self._dim = dim if dim is not None else _KNOWN_DIMS.get(self.model_name, EMBEDDING_DIM)

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        from .daemon import client_request
        resp = client_request(
            {"op": "embed_batch", "texts": list(texts)},
            socket_path=self.socket_path,
        )
        if resp is None or not resp.get("ok"):
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = resp.get("result") or []
        if len(vectors) != len(texts):
            return np.zeros((0, self.dim), dtype=np.float32)
        arr = np.asarray(vectors, dtype=np.float32)
        # Authoritative: trust the daemon's actual output dim.
        if arr.ndim == 2 and arr.shape[1] > 0:
            self._dim = arr.shape[1]
        return arr


class EmbeddingService:
    """Cache-first, daemon-second, local-last embedding service."""

    def __init__(
        self,
        model: Optional[str] = None,
        cache: Optional[EmbeddingCache] = None,
        daemon: Optional[Backend] = None,
        local: Optional[Backend] = None,
    ) -> None:
        self.model = model or _resolve_default_model()
        self.cache = cache if cache is not None else EmbeddingCache()
        self.daemon = daemon if daemon is not None else DaemonBackend(model_name=self.model)
        self.local = local if local is not None else LocalBackend(self.model)

    @property
    def dim(self) -> int:
        """Embedding dimension. Prefers the local backend's known/probed dim,
        falls back to the daemon's, then to the legacy constant."""
        for b in (self.local, self.daemon):
            d = getattr(b, "dim", None)
            if d:
                return d
        return _KNOWN_DIMS.get(self.model, EMBEDDING_DIM)

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
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack(vectors)

    def _embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        if not texts:
            return []
        # Empty strings -> zero vectors (correct dim), never hit model/cache.
        results: List[Optional[np.ndarray]] = [None] * len(texts)
        nonempty_idx: List[int] = []
        nonempty_texts: List[str] = []
        for i, t in enumerate(texts):
            if not t or not t.strip():
                results[i] = np.zeros(self.dim, dtype=np.float32)
            else:
                nonempty_idx.append(i)
                nonempty_texts.append(t)

        if nonempty_texts:
            cached = self.cache.get_many(self.model, nonempty_texts)
            # Defensive: prior versions of the bug could have cached a
            # wrong-dim vector under this model's namespace (e.g. MiniLM 384
            # stored under "thenlper/gte-large"). Treat dim-mismatched hits
            # as misses so they get recomputed and overwritten on put_many.
            expected_dim = self.dim
            miss_idx: List[int] = []
            miss_texts: List[str] = []
            for pos, vec in enumerate(cached):
                if vec is not None and vec.shape[0] == expected_dim:
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

        # Fix up any zero-vector slots whose dim doesn't match the now-known
        # model dim (rare: dim was unknown when we placed zeros, then a real
        # encode revealed it).
        actual_dim = self.dim
        for i, r in enumerate(results):
            if r is not None and r.shape[0] != actual_dim:
                # Only zeros could have wrong dim here; resize them.
                if not np.any(r):
                    results[i] = np.zeros(actual_dim, dtype=np.float32)

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
