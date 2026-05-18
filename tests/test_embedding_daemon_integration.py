"""End-to-end: run the daemon on a temp socket, verify embed/embed_batch ops.

Loads the real sentence-transformer model (one-time, cached on disk) and
asserts that a service pointed at the daemon produces the same vectors as a
service with an in-process LocalBackend. This is the test that proves the
socket path is wired correctly without recursing into itself.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from core import daemon as daemon_mod
from core.embedding_cache import EmbeddingCache
from core.embedding_service import (
    DaemonBackend,
    EmbeddingService,
    LocalBackend,
    resolve_embedding_dim,
)

# Derive expected dimension from the configured embedding model.
EXPECTED_DIM = resolve_embedding_dim()


@pytest.fixture
def socket_path():
    # AF_UNIX paths on macOS are capped at ~104 bytes, which pytest's tmp_path
    # blows past. /tmp is short enough and the socket is unlinked on teardown.
    import tempfile, uuid
    path = f"/tmp/cashew-test-{uuid.uuid4().hex[:8]}.sock"
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
def running_daemon(socket_path):
    daemon_mod.serve_in_thread(socket_path=socket_path, warm=False)
    # daemon is a daemon thread; no teardown needed — tmp_path cleanup handles socket
    # give the server a moment to actually bind
    for _ in range(50):
        if os.path.exists(socket_path):
            break
        time.sleep(0.02)
    assert os.path.exists(socket_path), "daemon socket never appeared"
    yield socket_path


class TestDaemonIntegration:
    def test_embed_op_roundtrips(self, running_daemon):
        resp = daemon_mod.client_request(
            {"op": "embed", "text": "hello world"}, socket_path=running_daemon
        )
        assert resp is not None and resp["ok"] is True
        vec = resp["result"]
        assert isinstance(vec, list)
        assert len(vec) == EXPECTED_DIM

    def test_embed_batch_op_returns_per_input_vector(self, running_daemon):
        resp = daemon_mod.client_request(
            {"op": "embed_batch", "texts": ["hello", "world", "cashew"]},
            socket_path=running_daemon,
        )
        assert resp is not None and resp["ok"] is True
        vectors = resp["result"]
        assert len(vectors) == 3
        for v in vectors:
            assert len(v) == EXPECTED_DIM

    def test_embed_batch_empty_list(self, running_daemon):
        resp = daemon_mod.client_request(
            {"op": "embed_batch", "texts": []}, socket_path=running_daemon
        )
        assert resp == {"ok": True, "result": []}

    def test_embed_batch_empty_strings_are_zero_vectors(self, running_daemon):
        resp = daemon_mod.client_request(
            {"op": "embed_batch", "texts": ["", "hello"]},
            socket_path=running_daemon,
        )
        assert resp["ok"] is True
        out = resp["result"]
        assert out[0] == [0.0] * EXPECTED_DIM
        assert out[1] != [0.0] * EXPECTED_DIM

    def test_service_via_daemon_matches_local(self, running_daemon, tmp_path):
        """The same model loaded in the daemon and in-process must produce
        identical vectors. Proves the daemon is not a semantic detour."""
        texts = ["redis bug hunt", "cashew is the brain"]

        cache_a = EmbeddingCache(str(tmp_path / "a.db"))
        via_daemon = EmbeddingService(
            cache=cache_a,
            daemon=DaemonBackend(socket_path=running_daemon),
            local=LocalBackend(),
        )
        a = np.asarray(via_daemon.embed(texts), dtype=np.float32)

        cache_b = EmbeddingCache(str(tmp_path / "b.db"))

        class BrokenDaemon:
            def encode(self, texts):
                return np.zeros((0, EXPECTED_DIM), dtype=np.float32)

        in_process = EmbeddingService(
            cache=cache_b, daemon=BrokenDaemon(), local=LocalBackend()
        )
        b = np.asarray(in_process.embed(texts), dtype=np.float32)

        # Same model, same input → identical vectors. Allow tiny float jitter.
        assert np.allclose(a, b, atol=1e-5)

    def test_unknown_op_returns_error(self, running_daemon):
        resp = daemon_mod.client_request(
            {"op": "nope"}, socket_path=running_daemon
        )
        assert resp["ok"] is False
        assert "unknown op" in resp["error"]
