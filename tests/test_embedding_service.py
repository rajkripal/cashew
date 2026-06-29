"""Tests for the cache/daemon/local orchestration in EmbeddingService.

Backends are injected as fakes so we can assert the strategy (cache-first,
daemon-second, local-last) and count calls, without loading the real model.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pytest

from core.embedding_cache import EmbeddingCache
from core.embedding_service import EMBEDDING_DIM, EmbeddingService


class FakeBackend:
    """Deterministic backend: vector[0] = len(text), vector[1] = hash%100."""

    def __init__(self) -> None:
        self.calls: List[List[str]] = []

    def encode(self, texts: List[str]) -> np.ndarray:
        self.calls.append(list(texts))
        out = np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i, 0] = float(len(t))
            out[i, 1] = float(hash(t) % 100)
        return out


class FailingBackend:
    """Simulates daemon being unreachable: returns empty array."""

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts: List[str]) -> np.ndarray:
        self.calls += 1
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)


@pytest.fixture
def service_with_fakes(tmp_path):
    cache = EmbeddingCache(str(tmp_path / "cache.db"))
    daemon = FakeBackend()
    local = FakeBackend()
    svc = EmbeddingService(
        model="fake-model", cache=cache, daemon=daemon, local=local
    )
    return svc, cache, daemon, local


class TestEmbeddingService:
    def test_single_text_returns_list(self, service_with_fakes):
        svc, *_ = service_with_fakes
        out = svc.embed("hello")
        assert isinstance(out, list)
        assert len(out) == EMBEDDING_DIM
        assert out[0] == 5.0  # len("hello")

    def test_batch_returns_list_of_lists(self, service_with_fakes):
        svc, *_ = service_with_fakes
        out = svc.embed(["a", "bb"])
        assert isinstance(out, list)
        assert len(out) == 2
        assert out[0][0] == 1.0
        assert out[1][0] == 2.0

    def test_daemon_used_before_local(self, service_with_fakes):
        svc, cache, daemon, local = service_with_fakes
        svc.embed(["a", "b"])
        assert len(daemon.calls) == 1
        assert len(local.calls) == 0

    def test_cache_hit_skips_both_backends(self, service_with_fakes):
        svc, cache, daemon, local = service_with_fakes
        svc.embed(["a", "b"])  # populates cache
        daemon.calls.clear()
        local.calls.clear()
        svc.embed(["a", "b"])  # full hit
        assert daemon.calls == []
        assert local.calls == []

    def test_partial_cache_hit_only_computes_misses(self, service_with_fakes):
        svc, cache, daemon, local = service_with_fakes
        svc.embed(["a"])
        daemon.calls.clear()
        svc.embed(["a", "b"])
        assert daemon.calls == [["b"]]

    def test_falls_back_to_local_when_daemon_empty(self, tmp_path):
        cache = EmbeddingCache(str(tmp_path / "cache.db"))
        daemon = FailingBackend()
        local = FakeBackend()
        svc = EmbeddingService(
            model="fake-model", cache=cache, daemon=daemon, local=local
        )
        out = svc.embed(["x"])
        assert daemon.calls == 1
        assert len(local.calls) == 1
        assert out[0][0] == 1.0  # len("x")

    def test_empty_string_returns_zero_vector_no_backend_call(
        self, service_with_fakes
    ):
        svc, cache, daemon, local = service_with_fakes
        out = svc.embed("")
        assert out == [0.0] * EMBEDDING_DIM
        assert daemon.calls == []
        assert local.calls == []

    def test_mix_of_empty_and_real_texts(self, service_with_fakes):
        svc, cache, daemon, local = service_with_fakes
        out = svc.embed(["", "hi", "", "there"])
        assert out[0] == [0.0] * EMBEDDING_DIM
        assert out[1][0] == 2.0
        assert out[2] == [0.0] * EMBEDDING_DIM
        assert out[3][0] == 5.0
        # Only nonempty strings hit the backend.
        assert daemon.calls == [["hi", "there"]]

    def test_model_namespace_isolates_cache(self, tmp_path):
        cache = EmbeddingCache(str(tmp_path / "cache.db"))
        b1 = FakeBackend()
        b2 = FakeBackend()
        svc_a = EmbeddingService(model="m1", cache=cache, daemon=b1, local=b1)
        svc_b = EmbeddingService(model="m2", cache=cache, daemon=b2, local=b2)
        svc_a.embed(["hello"])
        svc_b.embed(["hello"])  # different model → miss → computes again
        assert len(b1.calls) == 1
        assert len(b2.calls) == 1

    def test_embed_np_returns_ndarray(self, service_with_fakes):
        svc, *_ = service_with_fakes
        arr = svc.embed_np(["a", "bb"])
        assert arr.shape == (2, EMBEDDING_DIM)
        assert arr.dtype == np.float32
        assert arr[0, 0] == 1.0
        assert arr[1, 0] == 2.0

    def test_embed_np_empty_input(self, service_with_fakes):
        svc, *_ = service_with_fakes
        arr = svc.embed_np([])
        assert arr.shape == (0, EMBEDDING_DIM)


# ── _model_dim shim (sentence-transformers <5.5 vs >=5.5) ────────────────


class _ModelNewApi:
    def get_embedding_dimension(self):
        return 1024


class _ModelOldApi:
    def get_sentence_embedding_dimension(self):
        return 384


class _ModelBothApis:
    def get_embedding_dimension(self):
        return 1024

    def get_sentence_embedding_dimension(self):
        return 999  # should NOT be picked when new API exists


class _ModelNoApi:
    pass


class TestModelDimShim:
    """The shim must support both the new sentence-transformers >=5.5 API
    (`get_embedding_dimension`) and the deprecated pre-5.5 API
    (`get_sentence_embedding_dimension`). v1.2.0 shipped a call to the
    new method without pinning the dep, breaking every install on <5.5."""

    def test_uses_new_api_when_available(self):
        from core.embedding_service import _model_dim
        assert _model_dim(_ModelNewApi()) == 1024

    def test_falls_back_to_deprecated_api(self):
        from core.embedding_service import _model_dim
        assert _model_dim(_ModelOldApi()) == 384

    def test_prefers_new_api_when_both_exist(self):
        from core.embedding_service import _model_dim
        assert _model_dim(_ModelBothApis()) == 1024

    def test_raises_when_neither_api_exists(self):
        from core.embedding_service import _model_dim
        with pytest.raises(AttributeError, match="get_embedding_dimension"):
            _model_dim(_ModelNoApi())
