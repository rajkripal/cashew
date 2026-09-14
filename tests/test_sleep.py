#!/usr/bin/env python3
"""Tests for core.sleep.

The vectorized pipeline (cross-link, dedup, GC, permanence, dream) is covered
in test_sleep_refactor.py. This file keeps the SleepProtocol shim smoke check
and the permanence-integrity tests that drive core.permanence / core.decay
end-to-end — the invariant that a permanent node is never decayed.
"""

import os
import sqlite3
import tempfile

import pytest

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.sleep import SleepProtocol, DEDUP_THRESHOLD


def test_dedup_threshold_matches_active_profile():
    """The shim exposes the calibrated dedup threshold from the active model."""
    proto = SleepProtocol(":memory:", tempfile.mktemp(suffix=".json"))
    assert proto.dedup_threshold == DEDUP_THRESHOLD


class TestPermanenceIntegrity:
    """Permanent nodes must survive every decay path. Regression for the
    invariant validate_permanence_integrity enforces (permanent_but_decayed==0).
    """

    @pytest.fixture
    def permanence_db(self):
        fd, db_path = tempfile.mkstemp(suffix='.db')
        os.close(fd)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE thought_nodes (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                node_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                confidence REAL NOT NULL,
                mood_state TEXT,
                metadata TEXT,
                source_file TEXT,
                decayed INTEGER DEFAULT 0,
                permanent INTEGER DEFAULT 0,
                access_count INTEGER DEFAULT 0,
                last_updated TEXT,
                last_accessed TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE derivation_edges (
                parent_id TEXT NOT NULL,
                child_id TEXT NOT NULL,
                weight REAL NOT NULL,
                reasoning TEXT,
                FOREIGN KEY (parent_id) REFERENCES thought_nodes(id),
                FOREIGN KEY (child_id) REFERENCES thought_nodes(id),
                PRIMARY KEY (parent_id, child_id)
            )
        """)

        nodes = [
            ("high_access_1", "Frequently accessed thought", "derived", "2023-01-01T00:00:00", 0.8, "confident", "{}", "test", 0, 0, 25),
            ("high_access_2", "Another frequent thought", "derived", "2023-01-02T00:00:00", 0.7, "stable", "{}", "test", 0, 0, 15),
            ("medium_access", "Moderately accessed", "derived", "2023-01-03T00:00:00", 0.6, "neutral", "{}", "test", 0, 0, 8),
            ("low_access", "Rarely accessed", "derived", "2023-01-04T00:00:00", 0.5, "uncertain", "{}", "test", 0, 0, 2),
            ("zero_access", "Never accessed", "derived", "2023-01-05T00:00:00", 0.4, "doubtful", "{}", "test", 0, 0, 0),
            ("already_permanent", "Already permanent node", "derived", "2023-01-06T00:00:00", 0.9, "certain", "{}", "test", 0, 1, 20),
            ("core_memory_1", "Core memory node", "core_memory", "2023-01-07T00:00:00", 0.8, "stable", "{}", "test", 0, 0, 5),
            ("decayed_node", "Decayed node", "derived", "2023-01-08T00:00:00", 0.3, "forgotten", "{}", "test", 1, 0, 12),
        ]
        cursor.executemany("""
            INSERT INTO thought_nodes
            (id, content, node_type, timestamp, confidence, mood_state, metadata, source_file, decayed, permanent, access_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, nodes)
        conn.commit()
        conn.close()

        yield db_path
        try:
            os.unlink(db_path)
        except OSError:
            pass

    def test_promotion_by_access_count(self, permanence_db):
        from core.permanence import promote_permanent_nodes

        result = promote_permanent_nodes(permanence_db, access_threshold=10)
        assert result["nodes_promoted"] == 2
        assert result["nodes_evaluated"] == 2
        assert result["access_threshold"] == 10

        conn = sqlite3.connect(permanence_db)
        permanent = {r[0] for r in conn.execute(
            "SELECT id FROM thought_nodes WHERE permanent = 1"
        ).fetchall()}
        conn.close()
        assert permanent == {"high_access_1", "high_access_2", "already_permanent"}

    def test_permanent_protected_from_decay(self, permanence_db):
        from core.decay import auto_decay
        from core.permanence import promote_permanent_nodes

        promote_permanent_nodes(permanence_db, access_threshold=10)
        auto_decay(permanence_db, min_age_days=0, enable_cascading=False)

        conn = sqlite3.connect(permanence_db)
        alive_permanent = conn.execute(
            "SELECT id FROM thought_nodes "
            "WHERE permanent = 1 AND (decayed IS NULL OR decayed = 0)"
        ).fetchall()
        conn.close()
        assert len(alive_permanent) == 3

    def test_permanence_is_irreversible(self, permanence_db):
        from core.permanence import promote_permanent_nodes

        promote_permanent_nodes(permanence_db, access_threshold=10)
        conn = sqlite3.connect(permanence_db)
        conn.execute("UPDATE thought_nodes SET access_count = 0 WHERE id = 'high_access_1'")
        conn.commit()
        promote_permanent_nodes(permanence_db, access_threshold=10)
        still = conn.execute(
            "SELECT permanent FROM thought_nodes WHERE id = 'high_access_1'"
        ).fetchone()[0]
        conn.close()
        assert still == 1

    def test_stats_and_validation(self, permanence_db):
        from core.permanence import (
            get_permanence_stats,
            validate_permanence_integrity,
            promote_permanent_nodes,
        )

        initial = get_permanence_stats(permanence_db)
        assert initial["permanent_count"] == 1
        assert initial["non_permanent_count"] == 7

        promote_permanent_nodes(permanence_db, access_threshold=10)

        updated = get_permanence_stats(permanence_db)
        assert updated["permanent_count"] == 3
        assert updated["non_permanent_count"] == 5

        integrity = validate_permanence_integrity(permanence_db)
        assert integrity["permanent_but_decayed"] == 0
        assert integrity["integrity_ok"] is True


def test_embed_orphans_uses_injected_model_and_batch_protocol(tmp_path):
    """Orphan repair uses the caller-owned client and records its model."""
    import sqlite3
    import numpy as np
    from core import sleep as S
    captured = {}

    class FakeClient:
        def encode(self, texts):
            captured["texts"] = texts
            return np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

    db = str(tmp_path / "orphan.db")
    conn = sqlite3.connect(db)
    pytest.importorskip("sqlite_vec")
    from core.embeddings import _load_vec
    _load_vec(conn)
    conn.execute("CREATE TABLE thought_nodes (id TEXT PRIMARY KEY, content TEXT, decayed INTEGER DEFAULT 0)")
    conn.execute("CREATE TABLE embeddings (node_id TEXT PRIMARY KEY, vector BLOB, model TEXT, updated_at TEXT)")
    conn.execute("CREATE VIRTUAL TABLE vec_embeddings USING vec0(node_id text primary key, embedding float[3])")
    conn.execute("INSERT INTO thought_nodes (id, content) VALUES ('n1', 'an orphan node')")
    conn.commit()

    n = S._embed_orphans(conn, embedding_client=FakeClient(),
                          embedding_model="sentinel/custom-model",
                          expected_dimension=3)
    assert n == 1
    assert captured["texts"] == ["an orphan node"]
    stored = conn.execute("SELECT model FROM embeddings WHERE node_id='n1'").fetchone()[0]
    assert stored == "sentinel/custom-model"
    conn.close()


def test_embed_orphans_writes_vec_index_row(tmp_path):
    """A valid injected vector is written as bytes to both stores."""
    import sqlite3
    import numpy as np
    from core import sleep as S

    class FakeClient:
        def encode(self, texts):
            return np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)

    db = str(tmp_path / "orphan.db")
    conn = sqlite3.connect(db)
    pytest.importorskip("sqlite_vec")
    from core.embeddings import _load_vec
    _load_vec(conn)
    conn.execute("CREATE TABLE thought_nodes (id TEXT PRIMARY KEY, content TEXT, decayed INTEGER DEFAULT 0)")
    conn.execute("CREATE TABLE embeddings (node_id TEXT PRIMARY KEY, vector BLOB, model TEXT, updated_at TEXT)")
    conn.execute("CREATE VIRTUAL TABLE vec_embeddings USING vec0(node_id text primary key, embedding float[4])")
    conn.execute("INSERT INTO thought_nodes (id, content) VALUES ('n1', 'an orphan node')")
    conn.commit()

    n = S._embed_orphans(conn, embedding_client=FakeClient(),
                          embedding_model="test-model", expected_dimension=4)
    assert n == 1
    row = conn.execute("SELECT embedding FROM vec_embeddings WHERE node_id='n1'").fetchone()
    assert row is not None and isinstance(row[0], (bytes, bytearray))
    conn.close()

def test_promote_core_memories_never_promotes_decayed_node():
    """Regression: a decayed node must never be marked permanent, even if it
    ranks in the top-√N by fitness. metrics are computed pre-GC, so Phase 5 can
    decay a node that Phase 7 then sees as high-fitness — promoting it would
    violate the permanence invariant (permanent_but_decayed must be 0)."""
    import sqlite3
    from core.sleep import _promote_core_memories

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE thought_nodes (id TEXT PRIMARY KEY, node_type TEXT, "
                 "permanent INTEGER DEFAULT 0, decayed INTEGER DEFAULT 0)")
    for nid, dec in [("dead", 1), ("live", 0), ("c", 0), ("d", 0)]:
        conn.execute("INSERT INTO thought_nodes (id, node_type, decayed) VALUES (?, 'derived', ?)", (nid, dec))
    conn.commit()
    # target = int(sqrt(4)) = 2; the two highest-fitness are the decayed one and 'live'.
    metrics = {"dead": {"fitness": 0.99}, "live": {"fitness": 0.98},
               "c": {"fitness": 0.10}, "d": {"fitness": 0.10}}
    _promote_core_memories(conn, metrics)
    dead_perm = conn.execute("SELECT permanent FROM thought_nodes WHERE id='dead'").fetchone()[0]
    live_perm = conn.execute("SELECT permanent FROM thought_nodes WHERE id='live'").fetchone()[0]
    conn.close()
    assert dead_perm in (0, None), "decayed node must not be marked permanent"
    assert live_perm == 1, "live top-fitness node should still be promoted"


def test_load_embedding_matrix_is_float64(monkeypatch, tmp_path):
    """The sleep similarity matrix must be float64. Cosine matmul on float32
    raises spurious FPE warnings (divide-by-zero/overflow) under Apple's BLAS
    even though results are correct; float64 fires none. Asserts dtype rather
    than warning-absence, which is platform-dependent."""
    import sqlite3
    import numpy as np
    from core import sleep as S

    monkeypatch.setattr(S, "_resolve_expected_dim", lambda: 4)
    db = str(tmp_path / "m.db")
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE thought_nodes (id TEXT PRIMARY KEY, decayed INTEGER DEFAULT 0)")
    conn.execute("CREATE TABLE embeddings (node_id TEXT PRIMARY KEY, vector BLOB)")
    for nid, vec in [("a", [1, 0, 0, 0]), ("b", [0, 1, 0, 0])]:
        conn.execute("INSERT INTO thought_nodes (id) VALUES (?)", (nid,))
        conn.execute("INSERT INTO embeddings (node_id, vector) VALUES (?, ?)",
                     (nid, np.array(vec, dtype=np.float32).tobytes()))
    conn.commit()
    ids, matrix = S._load_embedding_matrix(conn, ["a", "b"])
    conn.close()
    assert matrix.dtype == np.float64
    assert len(ids) == 2
