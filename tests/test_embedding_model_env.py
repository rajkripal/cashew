"""Tests for CASHEW_EMBEDDING_MODEL env override propagation.

Bug history: `core/embedding_service.py` used to hardcode the model name and
dimension, ignoring the `CASHEW_EMBEDDING_MODEL` env that `core/config.py`
already honored. Result: gte-large benchmarks silently embedded with MiniLM.
These tests pin the contract that the service reads the configured model and
sizes vectors to match.
"""

from __future__ import annotations

import importlib
import os
from typing import List

import numpy as np
import pytest

from core.embedding_cache import EmbeddingCache
from core.embedding_service import (
    EmbeddingService,
    LocalBackend,
    _resolve_default_model,
)


class DimmedFakeBackend:
    """Fake backend that emits vectors of a configured dim, identity-ish so
    different models produce different vectors for the same text."""

    def __init__(self, model_name: str, dim: int) -> None:
        self.model_name = model_name
        self.dim = dim
        self.calls: List[List[str]] = []

    def encode(self, texts: List[str]) -> np.ndarray:
        self.calls.append(list(texts))
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        # Encode model identity into the vector so cross-model collisions are visible.
        sig = float(sum(ord(c) for c in self.model_name) % 997)
        for i, t in enumerate(texts):
            out[i, 0] = float(len(t))
            out[i, 1] = sig
        return out


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("CASHEW_EMBEDDING_MODEL", raising=False)
    yield


class TestEnvOverridePropagation:
    def test_resolve_default_model_picks_up_env(self, monkeypatch):
        monkeypatch.setenv("CASHEW_EMBEDDING_MODEL", "thenlper/gte-large")
        # core.config caches a singleton; force a reload so it re-reads env.
        import core.config as cfg
        importlib.reload(cfg)
        from core.embedding_service import _resolve_default_model as resolve
        assert resolve() == "thenlper/gte-large"

    def test_local_backend_default_follows_env(self, monkeypatch):
        monkeypatch.setenv("CASHEW_EMBEDDING_MODEL", "thenlper/gte-large")
        import core.config as cfg
        importlib.reload(cfg)
        b = LocalBackend()
        assert b.model_name == "thenlper/gte-large"
        # Known-dim table should give us 1024 without loading the model.
        assert b.dim == 1024

    def test_service_default_follows_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CASHEW_EMBEDDING_MODEL", "thenlper/gte-large")
        import core.config as cfg
        importlib.reload(cfg)
        cache = EmbeddingCache(str(tmp_path / "cache.db"))
        svc = EmbeddingService(
            cache=cache,
            daemon=DimmedFakeBackend("thenlper/gte-large", 1024),
            local=DimmedFakeBackend("thenlper/gte-large", 1024),
        )
        assert svc.model == "thenlper/gte-large"
        assert svc.dim == 1024
        vec = svc.embed("hello")
        assert len(vec) == 1024


class TestZeroFillDimMatchesModel:
    def test_empty_string_zero_vector_uses_model_dim(self, tmp_path):
        cache = EmbeddingCache(str(tmp_path / "cache.db"))
        backend = DimmedFakeBackend("thenlper/gte-large", 1024)
        svc = EmbeddingService(
            model="thenlper/gte-large",
            cache=cache,
            daemon=backend,
            local=backend,
        )
        # Empty string never hits the backend, but the zero vector must still
        # be 1024-dim, not 384.
        out = svc.embed("")
        assert len(out) == 1024
        assert all(v == 0.0 for v in out)

    def test_embed_np_empty_input_uses_model_dim(self, tmp_path):
        cache = EmbeddingCache(str(tmp_path / "cache.db"))
        backend = DimmedFakeBackend("thenlper/gte-large", 1024)
        svc = EmbeddingService(
            model="thenlper/gte-large",
            cache=cache,
            daemon=backend,
            local=backend,
        )
        arr = svc.embed_np([])
        assert arr.shape == (0, 1024)


class TestCacheModelKeying:
    def test_same_text_two_models_yields_two_cache_entries(self, tmp_path):
        cache = EmbeddingCache(str(tmp_path / "cache.db"))
        b_mini = DimmedFakeBackend("all-MiniLM-L6-v2", 384)
        b_gte = DimmedFakeBackend("thenlper/gte-large", 1024)
        svc_mini = EmbeddingService(
            model="all-MiniLM-L6-v2", cache=cache, daemon=b_mini, local=b_mini
        )
        svc_gte = EmbeddingService(
            model="thenlper/gte-large", cache=cache, daemon=b_gte, local=b_gte
        )

        v_mini = svc_mini.embed("hello world")
        v_gte = svc_gte.embed("hello world")

        # Different dims and different signatures → cache cannot have crossed
        # over.
        assert len(v_mini) == 384
        assert len(v_gte) == 1024
        assert v_mini[1] != v_gte[1]

        # Cache holds one entry per (model, text).
        assert cache.size("all-MiniLM-L6-v2") == 1
        assert cache.size("thenlper/gte-large") == 1
        assert cache.size() == 2

    def test_warm_cache_for_one_model_does_not_serve_other(self, tmp_path):
        cache = EmbeddingCache(str(tmp_path / "cache.db"))
        b_mini = DimmedFakeBackend("all-MiniLM-L6-v2", 384)
        b_gte = DimmedFakeBackend("thenlper/gte-large", 1024)
        svc_mini = EmbeddingService(
            model="all-MiniLM-L6-v2", cache=cache, daemon=b_mini, local=b_mini
        )
        svc_gte = EmbeddingService(
            model="thenlper/gte-large", cache=cache, daemon=b_gte, local=b_gte
        )

        svc_mini.embed("the quick brown fox")
        # gte must compute, not pull MiniLM's cached vector.
        b_gte.calls.clear()
        svc_gte.embed("the quick brown fox")
        assert b_gte.calls == [["the quick brown fox"]]


class TestPoisonedCacheRecovery:
    def test_wrong_dim_cache_entry_is_treated_as_miss(self, tmp_path):
        """Pre-fix bug: MiniLM (384) vectors got cached under the gte-large
        namespace because the service ignored the env. Post-fix, those
        poisoned entries should be silently recomputed instead of returned."""
        cache = EmbeddingCache(str(tmp_path / "cache.db"))
        # Plant a 384-dim vector under thenlper/gte-large for "hello".
        bad = np.ones(384, dtype=np.float32) * 0.5
        cache.put("thenlper/gte-large", "hello", bad)

        backend = DimmedFakeBackend("thenlper/gte-large", 1024)
        svc = EmbeddingService(
            model="thenlper/gte-large",
            cache=cache,
            daemon=backend,
            local=backend,
        )
        out = svc.embed("hello")
        assert len(out) == 1024  # not 384
        assert backend.calls == [["hello"]]  # forced a recompute
        # Cache now has the correct entry (overwrite).
        fresh = cache.get("thenlper/gte-large", "hello")
        assert fresh is not None and fresh.shape[0] == 1024


class TestEndToEndCrossModel:
    def test_two_models_yield_different_vectors_for_same_text(self, tmp_path):
        cache = EmbeddingCache(str(tmp_path / "cache.db"))
        b_a = DimmedFakeBackend("model-a", 384)
        b_b = DimmedFakeBackend("model-b", 384)
        svc_a = EmbeddingService(model="model-a", cache=cache, daemon=b_a, local=b_a)
        svc_b = EmbeddingService(model="model-b", cache=cache, daemon=b_b, local=b_b)

        v_a = svc_a.embed("identical input")
        v_b = svc_b.embed("identical input")

        assert v_a != v_b, "different models must produce different vectors"
