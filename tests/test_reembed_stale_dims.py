#!/usr/bin/env python3
"""Tests for scripts/reembed_stale_dims.py — the targeted stale-dim repair
script that complements PR #130's retrieval guard by actually fixing the
underlying data (nodes stuck at a legacy embedding dim)."""

import os
import sqlite3
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.embeddings import embed_nodes, ensure_schema
from core.embedding_service import resolve_embedding_dim
from scripts.reembed_stale_dims import find_stale_nodes, reembed_stale

EXPECTED_DIM = resolve_embedding_dim()


@pytest.fixture
def temp_db():
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE thought_nodes (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            node_type TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            confidence REAL NOT NULL,
            mood_state TEXT,
            metadata TEXT,
            source_file TEXT,
            decayed INTEGER DEFAULT 0
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE derivation_edges (
            parent_id TEXT NOT NULL,
            child_id TEXT NOT NULL,
            relation TEXT NOT NULL,
            weight REAL NOT NULL,
            reasoning TEXT,
            PRIMARY KEY (parent_id, child_id, relation)
        )
        """
    )
    cursor.execute(
        "INSERT INTO thought_nodes (id, content, node_type, timestamp, confidence, source_file) "
        "VALUES ('good1', 'a healthy current-dim node', 'fact', '2026-01-01T00:00:00', 0.9, 'test')"
    )
    cursor.execute(
        "INSERT INTO thought_nodes (id, content, node_type, timestamp, confidence, source_file) "
        "VALUES ('stale1', 'a node stuck on the old embedding model', 'fact', '2026-01-01T00:00:00', 0.9, 'test')"
    )
    conn.commit()
    conn.close()

    # Embed 'good1' normally so it ends up at the current dim.
    ensure_schema(db_path)
    embed_nodes(db_path)

    # Simulate a legacy 384-dim row for 'stale1' as if left over from an old model.
    conn = sqlite3.connect(db_path)
    fake_vec = np.zeros(384, dtype=np.float32).tobytes()
    conn.execute(
        "INSERT OR REPLACE INTO embeddings (node_id, vector, model, updated_at) VALUES (?, ?, ?, ?)",
        ("stale1", fake_vec, "all-MiniLM-L6-v2", "2026-01-01T00:00:00"),
    )
    conn.commit()
    conn.close()

    yield db_path
    os.unlink(db_path)


def test_find_stale_nodes_flags_only_mismatched_dim(temp_db):
    stale = find_stale_nodes(temp_db, EXPECTED_DIM)
    stale_ids = {node_id for node_id, _content, _dim in stale}
    assert stale_ids == {"stale1"}


def test_reembed_stale_dry_run_does_not_write(temp_db):
    summary = reembed_stale(temp_db, dry_run=True)
    assert summary["stale_found"] == 1
    assert summary["reembedded"] == 0

    # Vector for stale1 is untouched.
    conn = sqlite3.connect(temp_db)
    row = conn.execute("SELECT LENGTH(vector) FROM embeddings WHERE node_id = 'stale1'").fetchone()
    conn.close()
    assert row[0] // 4 == 384


def test_reembed_stale_fixes_dim_in_place(temp_db):
    summary = reembed_stale(temp_db, dry_run=False)
    assert summary["stale_found"] == 1
    assert summary["reembedded"] == 1
    assert summary["failed"] == 0

    conn = sqlite3.connect(temp_db)
    row = conn.execute("SELECT LENGTH(vector), model FROM embeddings WHERE node_id = 'stale1'").fetchone()
    good_row = conn.execute("SELECT LENGTH(vector) FROM embeddings WHERE node_id = 'good1'").fetchone()
    conn.close()

    assert row[0] // 4 == EXPECTED_DIM
    assert row[1] != "all-MiniLM-L6-v2"
    # The already-correct node was left alone.
    assert good_row[0] // 4 == EXPECTED_DIM

    # Re-running finds nothing left to fix.
    assert find_stale_nodes(temp_db, EXPECTED_DIM) == []
