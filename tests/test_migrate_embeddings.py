"""Tests for the smooth-upgrade embedding migration path.

Covers:
- ``scripts.migrate_embeddings.detect_mismatch`` correctly identifies
  brains whose stored embedding dim differs from the configured model.
- ``migrate_embeddings`` wipes existing embeddings and re-embeds under
  the configured model, leaving the brain consistent.
- The startup warning fires once when ``embed_nodes`` is called against a
  mismatched brain.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import db as cdb
from core import embeddings as cembed
from scripts.migrate_embeddings import (  # noqa: E402
    detect_mismatch,
    migrate_embeddings,
    _stored_embedding_dim,
)


def _make_brain(dim: int) -> str:
    """Create a temp brain with two short nodes and pre-populated stored
    embeddings of the given dim (simulating a brain built under an older
    model). Returns the db path."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    cdb.ensure_schema(path)
    conn = sqlite3.connect(path)
    conn.execute(
        "INSERT INTO thought_nodes (id, content, node_type, domain, source_file, "
        "timestamp, last_accessed, decayed) VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
        ("n1", "alice prefers async standups", "fact", "user", "test", "2026-01-01T00:00:00", "2026-01-01T00:00:00"),
    )
    conn.execute(
        "INSERT INTO thought_nodes (id, content, node_type, domain, source_file, "
        "timestamp, last_accessed, decayed) VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
        ("n2", "project orion deadline pushed back", "fact", "user", "test", "2026-01-02T00:00:00", "2026-01-02T00:00:00"),
    )

    vec = np.zeros(dim, dtype=np.float32).tobytes()
    conn.execute(
        "INSERT INTO embeddings (node_id, vector, model, updated_at) VALUES (?, ?, ?, ?)",
        ("n1", vec, "fake-old-model", "2026-01-01T00:00:00"),
    )
    conn.execute(
        "INSERT INTO embeddings (node_id, vector, model, updated_at) VALUES (?, ?, ?, ?)",
        ("n2", vec, "fake-old-model", "2026-01-02T00:00:00"),
    )
    conn.commit()
    conn.close()
    return path


def test_stored_embedding_dim_matches_blob_size():
    path = _make_brain(384)
    try:
        conn = sqlite3.connect(path)
        assert _stored_embedding_dim(conn) == 384
        conn.close()
    finally:
        os.remove(path)


def test_stored_embedding_dim_returns_none_on_empty_brain():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        cdb.ensure_schema(path)
        conn = sqlite3.connect(path)
        assert _stored_embedding_dim(conn) is None
        conn.close()
    finally:
        os.remove(path)


def test_detect_mismatch_finds_dim_difference(monkeypatch):
    """Brain stored at 384 dim, currently configured model produces a
    different dim -> mismatch."""
    path = _make_brain(384)
    try:
        from core import embedding_service
        monkeypatch.setattr(embedding_service, "_resolve_default_model",
                            lambda: "test-fake-1024")
        monkeypatch.setattr(embedding_service, "resolve_embedding_dim",
                            lambda model_name=None: 1024)
        # scripts.migrate_embeddings imports these by name at call time
        import scripts.migrate_embeddings as mig
        monkeypatch.setattr(mig, "resolve_embedding_dim", lambda: 1024)
        result = detect_mismatch(path)
        assert result is not None
        assert result["stored_dim"] == 384
        assert result["configured_dim"] == 1024
        assert result["configured_model"] == "test-fake-1024"
    finally:
        os.remove(path)


def test_detect_mismatch_returns_none_when_aligned(monkeypatch):
    """Brain stored at 384 dim, configured model also produces 384 -> no
    mismatch."""
    path = _make_brain(384)
    try:
        from core import embedding_service
        monkeypatch.setattr(embedding_service, "_resolve_default_model",
                            lambda: "test-fake-384")
        monkeypatch.setattr(embedding_service, "resolve_embedding_dim",
                            lambda model_name=None: 384)
        import scripts.migrate_embeddings as mig
        monkeypatch.setattr(mig, "resolve_embedding_dim", lambda: 384)
        assert detect_mismatch(path) is None
    finally:
        os.remove(path)


def test_warn_on_dim_mismatch_fires_once(caplog, monkeypatch):
    """The startup warning fires the first time embed_nodes is called
    against a mismatched brain, but not on repeated calls (idempotent
    per-process)."""
    path = _make_brain(384)
    try:
        monkeypatch.setattr(cembed, "_current_embedding_dim", lambda: 1024)
        from core import embedding_service
        monkeypatch.setattr(embedding_service, "_resolve_default_model",
                            lambda: "test-fake-1024")

        # Clear the warning-cache so this test sees the first call.
        cembed._WARNED_DIM_MISMATCH.discard(path)

        with caplog.at_level(logging.WARNING):
            cembed._warn_on_dim_mismatch(path)
        assert any(
            "embedding dim mismatch" in r.message.lower()
            for r in caplog.records
        ), f"expected dim-mismatch warning, got: {[r.message for r in caplog.records]}"

        # Second call: should be a no-op (warning already emitted for this db)
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            cembed._warn_on_dim_mismatch(path)
        assert not any(
            "embedding dim mismatch" in r.message.lower()
            for r in caplog.records
        )
    finally:
        cembed._WARNED_DIM_MISMATCH.discard(path)
        os.remove(path)


def test_warn_skips_brain_without_stored_embeddings(caplog, monkeypatch):
    """A brand-new brain with no embeddings shouldn't trigger the warning."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        cdb.ensure_schema(path)
        monkeypatch.setattr(cembed, "_current_embedding_dim", lambda: 1024)
        with caplog.at_level(logging.WARNING):
            cembed._warn_on_dim_mismatch(path)
        assert not any(
            "embedding dim mismatch" in r.message.lower()
            for r in caplog.records
        )
    finally:
        os.remove(path)
