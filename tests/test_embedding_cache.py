"""Unit tests for the content-hash embedding cache."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from core.embedding_cache import EmbeddingCache


@pytest.fixture
def cache_path(tmp_path):
    return str(tmp_path / "cache.db")


@pytest.fixture
def cache(cache_path):
    return EmbeddingCache(cache_path)


def _vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(384, dtype=np.float32)


class TestEmbeddingCache:
    def test_miss_returns_none(self, cache):
        assert cache.get("model-a", "unseen") is None

    def test_put_then_get_roundtrip(self, cache):
        v = _vec(1)
        cache.put("model-a", "hello", v)
        got = cache.get("model-a", "hello")
        assert got is not None
        assert np.allclose(got, v)
        assert got.dtype == np.float32

    def test_model_namespace_isolates(self, cache):
        cache.put("model-a", "hello", _vec(1))
        assert cache.get("model-b", "hello") is None

    def test_put_is_idempotent_and_updatable(self, cache):
        cache.put("m", "k", _vec(1))
        cache.put("m", "k", _vec(2))
        got = cache.get("m", "k")
        assert np.allclose(got, _vec(2))
        assert cache.size() == 1

    def test_get_many_preserves_order_and_misses(self, cache):
        cache.put("m", "a", _vec(1))
        cache.put("m", "c", _vec(3))
        results = cache.get_many("m", ["a", "b", "c", "d"])
        assert len(results) == 4
        assert np.allclose(results[0], _vec(1))
        assert results[1] is None
        assert np.allclose(results[2], _vec(3))
        assert results[3] is None

    def test_put_many_bulk_insert(self, cache):
        pairs = [(f"t{i}", _vec(i)) for i in range(5)]
        n = cache.put_many("m", pairs)
        assert n == 5
        assert cache.size() == 5
        for t, v in pairs:
            got = cache.get("m", t)
            assert np.allclose(got, v)

    def test_invalidate_model_clears_namespace(self, cache):
        cache.put("m1", "a", _vec(1))
        cache.put("m1", "b", _vec(2))
        cache.put("m2", "a", _vec(3))
        removed = cache.invalidate_model("m1")
        assert removed == 2
        assert cache.get("m1", "a") is None
        assert cache.get("m2", "a") is not None

    def test_persists_across_instances(self, cache_path):
        a = EmbeddingCache(cache_path)
        a.put("m", "persistent", _vec(7))
        b = EmbeddingCache(cache_path)
        got = b.get("m", "persistent")
        assert np.allclose(got, _vec(7))

    def test_empty_get_many_returns_empty(self, cache):
        assert cache.get_many("m", []) == []

    def test_empty_put_many_is_noop(self, cache):
        assert cache.put_many("m", []) == 0
        assert cache.size() == 0

    def test_operations_do_not_leak_connections(self, cache, monkeypatch):
        """Every cache call must close its connection.

        Regression for the daemon fd leak: the old ``_connect`` returned a bare
        connection and relied on ``with conn:`` (commit-only, never closes), so
        a long-lived process leaked one fd per call until it hit the fd cap and
        every open failed with "unable to open database file".
        """
        import sqlite3

        open_count = 0
        close_count = 0
        real_connect = sqlite3.connect

        class _TrackedConn(sqlite3.Connection):
            def close(self):
                nonlocal close_count
                close_count += 1
                super().close()

        def _tracked(*args, **kwargs):
            nonlocal open_count
            open_count += 1
            kwargs["factory"] = _TrackedConn
            return real_connect(*args, **kwargs)

        monkeypatch.setattr(sqlite3, "connect", _tracked)

        for i in range(25):
            cache.put("m", f"text-{i}", _vec(i))
            cache.get("m", f"text-{i}")
            cache.get_many("m", [f"text-{i}", "miss"])
            cache.size()

        assert open_count > 0
        assert close_count == open_count, (
            f"leaked {open_count - close_count} connections across {open_count} opens"
        )
