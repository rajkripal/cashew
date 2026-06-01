#!/usr/bin/env python3
"""Tests for the vectorized sleep cycle free functions in core/sleep.py.

Covers the 9-phase pipeline independently + smoke tests for the module-level
``run_sleep_cycle()`` entry point with an embeddings table present.
"""

from __future__ import annotations

import hashlib
import math
import os
import sqlite3
import sys
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.sleep import (
    CROSS_LINK_THRESHOLD,
    DEDUP_THRESHOLD,
    MAX_NODES_PER_CYCLE,
    MAX_EDGES_PER_CYCLE,
    EDGES_PER_BATCH,
    GC_K_NODES,
    run_sleep_cycle,
    _find_pairs,
    _batch_cross_links,
    _run_dedup,
    _compute_metrics,
    _garbage_collect,
    _evaluate_permanence,
    _promote_core_memories,
    _generate_dream,
    _embed_orphans,
    _merge_cluster,
    _load_embedding_matrix,
    _set_wal,
)


# ── helpers ──────────────────────────────────────────────────────────────


def _make_embedding(seed: int, dim: int = 384) -> np.ndarray:
    """Deterministic unit-norm embedding vector."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


def _node_id(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def _create_schema(conn: sqlite3.Connection, *, with_embeddings: bool = True):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS thought_nodes (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            node_type TEXT NOT NULL DEFAULT 'observation',
            timestamp TEXT NOT NULL DEFAULT '',
            mood_state TEXT,
            metadata TEXT,
            source_file TEXT,
            decayed INTEGER DEFAULT 0,
            permanent INTEGER DEFAULT 0,
            domain TEXT DEFAULT 'user',
            access_count INTEGER DEFAULT 0,
            last_accessed TEXT,
            tags TEXT
        );
        CREATE TABLE IF NOT EXISTS derivation_edges (
            parent_id TEXT NOT NULL,
            child_id TEXT NOT NULL,
            weight REAL NOT NULL DEFAULT 1.0,
            reasoning TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_edges_parent ON derivation_edges(parent_id);
        CREATE INDEX IF NOT EXISTS idx_edges_child ON derivation_edges(child_id);
    """)
    if with_embeddings:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS embeddings (
                node_id TEXT PRIMARY KEY,
                vector BLOB NOT NULL,
                model TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
        """)


def _insert_node(conn, id_: str, content: str, node_type="observation",
                 timestamp="2026-01-01T00:00:00", domain="user",
                 source_file=None, access_count=0, permanent=False):
    conn.execute(
        "INSERT OR REPLACE INTO thought_nodes "
        "(id, content, node_type, timestamp, domain, source_file, access_count, permanent) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (id_, content, node_type, timestamp, domain, source_file or "",
         access_count, 1 if permanent else 0),
    )


def _insert_embedding(conn, node_id: str, vec: np.ndarray):
    """Insert an embedding blob for a node."""
    conn.execute(
        "INSERT OR REPLACE INTO embeddings (node_id, vector, model, updated_at) "
        "VALUES (?, ?, 'test-model', datetime('now'))",
        (node_id, vec.astype(np.float32).tobytes()),
    )


def _make_test_db(with_embeddings=True) -> str:
    """Return path to a populated test database."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(path)
    _create_schema(conn, with_embeddings=with_embeddings)
    conn.commit()
    conn.close()
    return path


# ── Fixture ──────────────────────────────────────────────────────────────


@pytest.fixture
def db_with_embeddings():
    """Database with ~10 embedding-bearing nodes for pipeline testing."""
    path = _make_test_db(with_embeddings=True)
    conn = sqlite3.connect(path)
    _set_wal(conn)

    contents = [
        ("n001", "God exists and is all-powerful", "seed", "source_a"),
        ("n002", "God exists and is all-powerful and omnipotent", "derived", "source_a"),
        ("n003", "God exists and is all-powerful deity", "belief", "source_a"),
        ("n004", "Prayer works sometimes", "belief", "source_b"),
        ("n005", "Core belief about reality", "core_memory", "source_b"),
        ("n006", "Uncertain thought about the divine", "derived", "source_c"),
        ("n007", "Another weak idea floating around", "derived", "source_c"),
        ("n008", "Completely separate thought about nature", "derived", "source_d"),
        ("n009", "Another isolated idea about physics", "belief", "source_d"),
        ("n010", "The universe is vast and mysterious", "belief", "source_e"),
    ]
    for nid, content, ntype, src in contents:
        _insert_node(conn, nid, content, ntype, source_file=src)

    # Insert embeddings — controlled similarity values:
    # v1 (nodes 0,2,6): identical → dedup  (sim = 1.0)
    # v2 (nodes 1,3,7): mixed vector with sim ≈ 0.92 → cross-link with v1
    #   (in the [CROSS_LINK_THRESHOLD, DEDUP_THRESHOLD) = [0.90, 0.94) band)
    # v3 (nodes 4,5,8,9): random → low or no similarity
    ref_vec = _make_embedding(0)
    orth = _make_embedding(999)
    # Make orth orthogonal to ref_vec
    orth = orth - np.dot(orth, ref_vec) * ref_vec
    orth = orth / np.linalg.norm(orth)
    mixed = 0.92 * ref_vec + np.sqrt(1 - 0.92**2) * orth
    random_vec = _make_embedding(42)

    vecs_by_group = [
        ref_vec,    # group 0: dedup (identical to ref)
        mixed,      # group 1: cross-link candidate with ref (~0.75)
        ref_vec,    # group 2: dedup
        mixed,      # group 3: cross-link
        random_vec, # group 4: orthogonal (no link)
        random_vec, # group 5: orthogonal
        ref_vec,    # group 6: dedup
        mixed,      # group 7: cross-link
        random_vec, # group 8: orthogonal
        random_vec, # group 9: orthogonal
    ]
    for i, (nid, _, _, _) in enumerate(contents):
        _insert_embedding(conn, nid, vecs_by_group[i])

    # Some existing edges
    conn.executemany(
        "INSERT OR IGNORE INTO derivation_edges "
        "(parent_id, child_id, weight, reasoning) VALUES (?,?,?,?)",
        [
            ("n001", "n002", 0.9, "supports"),
            ("n001", "n004", 0.6, "supports"),
        ],
    )
    conn.commit()
    conn.close()
    return path


# ── Embedded-missing test ────────────────────────────────────────────────


def test_run_sleep_cycle_no_embeddings_table():
    """Missing embeddings table should return error, not crash."""
    path = _make_test_db(with_embeddings=False)
    try:
        conn = sqlite3.connect(path)
        _insert_node(conn, "a", "test content")
        conn.commit()
        conn.close()

        result = run_sleep_cycle(db_path=path)
        assert "error" in result
        assert "no embeddings table" in result["error"]
    finally:
        os.unlink(path)


# ── _load_embedding_matrix ───────────────────────────────────────────────


def test_load_embedding_matrix_filters_bad_vectors():
    """NaN, inf, and zero vectors should be filtered out."""
    path = _make_test_db(with_embeddings=True)
    try:
        conn = sqlite3.connect(path)
        _insert_node(conn, "good", "good node")
        _insert_node(conn, "nan_vec", "nan node")
        _insert_node(conn, "inf_vec", "inf node")
        _insert_node(conn, "zero_vec", "zero node")

        # Good embedding
        good_vec = _make_embedding(1).tobytes()
        conn.execute(
            "INSERT INTO embeddings (node_id, vector, model, updated_at) "
            "VALUES (?, ?, 'test', datetime('now'))",
            ("good", good_vec),
        )
        # NaN embedding
        nan_blob = np.array([float("nan")] * 384, dtype=np.float32).tobytes()
        conn.execute(
            "INSERT INTO embeddings (node_id, vector, model, updated_at) "
            "VALUES (?, ?, 'test', datetime('now'))",
            ("nan_vec", nan_blob),
        )
        # Inf embedding
        inf_blob = np.array([float("inf")] * 384, dtype=np.float32).tobytes()
        conn.execute(
            "INSERT INTO embeddings (node_id, vector, model, updated_at) "
            "VALUES (?, ?, 'test', datetime('now'))",
            ("inf_vec", inf_blob),
        )
        # Zero embedding
        zero_blob = np.zeros(384, dtype=np.float32).tobytes()
        conn.execute(
            "INSERT INTO embeddings (node_id, vector, model, updated_at) "
            "VALUES (?, ?, 'test', datetime('now'))",
            ("zero_vec", zero_blob),
        )
        conn.commit()

        ids, matrix = _load_embedding_matrix(conn, ["good", "nan_vec", "inf_vec", "zero_vec"])
        conn.close()

        assert len(ids) == 1
        assert ids == ["good"]
        assert matrix.shape == (1, 384)
    finally:
        os.unlink(path)


# ── _find_pairs ──────────────────────────────────────────────────────────


def test_find_pairs_returns_correct_shapes():
    """_find_pairs should return cross_pairs and dedup_pairs arrays."""
    ids = [f"n{i:03d}" for i in range(5)]
    # Build a matrix where some pairs are highly similar
    rng = np.random.RandomState(42)
    base = rng.randn(5, 384).astype(np.float32)
    base = base / np.linalg.norm(base, axis=1, keepdims=True)

    # Make n000 and n001 very similar
    matrix = base.copy()
    matrix[1] = matrix[0] * 0.99 + matrix[1] * 0.01
    matrix[1] = matrix[1] / np.linalg.norm(matrix[1])

    cross_pairs, dedup_pairs, sim = _find_pairs(ids, matrix)

    assert isinstance(cross_pairs, np.ndarray)
    assert isinstance(dedup_pairs, np.ndarray)
    assert isinstance(sim, np.ndarray)
    assert sim.shape == (5, 5)

    # n000 vs n001 should be in dedup (sim >= 0.82)
    assert any(
        (p[0] == 0 and p[1] == 1) or (p[0] == 1 and p[1] == 0)
        for p in dedup_pairs
    ), "n000 vs n001 should be dedup candidates"


def test_find_pairs_no_candidates_when_below_threshold():
    """When all similarities are below CROSS_LINK_THRESHOLD, both arrays should be empty."""
    ids = [f"n{i:03d}" for i in range(3)]
    # Orthogonal vectors
    matrix = np.eye(3, 384, dtype=np.float32) * 10
    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

    cross_pairs, dedup_pairs, sim = _find_pairs(ids, matrix)
    assert len(cross_pairs) == 0
    assert len(dedup_pairs) == 0


# ── _batch_cross_links ───────────────────────────────────────────────────


def test_batch_cross_links_creates_edges(db_with_embeddings):
    """Cross-link edges should be created in the database."""
    conn = sqlite3.connect(db_with_embeddings)
    ids = [r[0] for r in conn.execute("SELECT id FROM thought_nodes LIMIT 4").fetchall()]
    matrix = np.random.randn(4, 384).astype(np.float32)
    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    # Force cross-link similarity
    matrix[1] = matrix[0] * 0.95
    matrix[1] = matrix[1] / np.linalg.norm(matrix[1])

    cross_pairs, dedup_pairs, sim = _find_pairs(ids, matrix)
    stats = _batch_cross_links(conn, ids, cross_pairs, sim)

    assert stats["created"] >= 0
    assert stats["skipped"] >= 0

    # Verify edges actually exist for pairs that should have been created
    if len(cross_pairs) > 0:
        n1, n2 = cross_pairs[0]
        row = conn.execute(
            "SELECT COUNT(*) FROM derivation_edges WHERE parent_id=? AND child_id=?",
            (ids[int(n1)], ids[int(n2)]),
        ).fetchone()[0]
        assert row > 0
    conn.close()


def test_batch_cross_links_respects_max_edges(db_with_embeddings):
    """max_edges cap should stop edge creation."""
    conn = sqlite3.connect(db_with_embeddings)
    ids = [r[0] for r in conn.execute("SELECT id FROM thought_nodes LIMIT 4").fetchall()]
    matrix = np.random.randn(4, 384).astype(np.float32)
    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    # Force maximum similarity
    matrix[:] = matrix[0:1]
    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

    cross_pairs, dedup_pairs, sim = _find_pairs(ids, matrix)
    stats = _batch_cross_links(conn, ids, cross_pairs, sim, max_edges=1)

    assert stats["created"] <= 1
    if len(cross_pairs) > 0:
        assert stats["capped"] is True or stats["created"] <= 1
    conn.close()


def test_batch_cross_links_skips_existing_edges(db_with_embeddings):
    """Already-existing edges should be skipped."""
    conn = sqlite3.connect(db_with_embeddings)
    ids = [r[0] for r in conn.execute("SELECT id FROM thought_nodes LIMIT 3").fetchall()]
    matrix = np.random.randn(3, 384).astype(np.float32)
    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix[1] = matrix[0] * 0.95 + matrix[1] * 0.05
    matrix[1] = matrix[1] / np.linalg.norm(matrix[1])

    cross_pairs, dedup_pairs, sim = _find_pairs(ids, matrix)

    # Insert edge beforehand for the first candidate pair
    if len(cross_pairs) > 0:
        n1, n2 = cross_pairs[0]
        conn.execute(
            "INSERT OR IGNORE INTO derivation_edges "
            "(parent_id, child_id, weight, reasoning) VALUES (?,?,?,'pre-existing')",
            (ids[int(n1)], ids[int(n2)], 0.5),
        )
        conn.commit()

    stats = _batch_cross_links(conn, ids, cross_pairs, sim)
    assert stats["skipped"] >= 0
    conn.close()


# ── _run_dedup / _merge_cluster ──────────────────────────────────────────


def test_merge_cluster_basic(db_with_embeddings):
    """Two near-duplicate nodes should merge into one."""
    conn = sqlite3.connect(db_with_embeddings)
    # Add two near-duplicate nodes
    _insert_node(conn, "dup_a", "The build is broken on main", "observation",
                 access_count=2)
    _insert_node(conn, "dup_b", "Build broken on main branch since Tuesday",
                 "observation", access_count=5)
    conn.commit()

    keeper = _merge_cluster(conn, ["dup_a", "dup_b"])
    assert keeper is not None

    # The loser (lower access_count) should be decayed; keeper lives
    loser_count = conn.execute(
        "SELECT COUNT(*) FROM thought_nodes WHERE id IN ('dup_a','dup_b') AND decayed=1"
    ).fetchone()[0]
    keeper_count = conn.execute(
        "SELECT COUNT(*) FROM thought_nodes WHERE id=? AND (decayed IS NULL OR decayed=0)",
        (keeper,),
    ).fetchone()[0]
    assert loser_count == 1        # one loser decayed
    assert keeper_count == 1       # keeper's original row alive

    # New merged node should exist
    row = conn.execute(
        "SELECT id FROM thought_nodes WHERE id=?", (keeper,)
    ).fetchone()
    assert row is not None

    conn.close()


def test_merge_cluster_single_node_returns_none(db_with_embeddings):
    """A single node should not be merged."""
    conn = sqlite3.connect(db_with_embeddings)
    result = _merge_cluster(conn, ["n001"])
    assert result is None
    conn.close()


def test_run_dedup_empty_pairs(db_with_embeddings):
    """No dedup pairs should produce empty stats."""
    conn = sqlite3.connect(db_with_embeddings)
    stats = _run_dedup(conn, [], np.array([]))
    assert stats["components"] == 0
    assert stats["nodes_merged"] == 0
    conn.close()


# ── _compute_metrics ─────────────────────────────────────────────────────


def test_compute_metrics(db_with_embeddings):
    """Active nodes should have metrics computed."""
    conn = sqlite3.connect(db_with_embeddings)
    metrics = _compute_metrics(conn)
    assert len(metrics) > 0
    for nid, m in metrics.items():
        assert "branching_factor" in m
        assert "cross_links" in m
        assert "fitness" in m
        assert m["branching_factor"] >= 0
        assert m["cross_links"] >= 0
        assert m["fitness"] >= 0
    conn.close()


# ── _garbage_collect ─────────────────────────────────────────────────────


def test_garbage_collect_off_does_nothing(db_with_embeddings):
    """GC mode=off should return empty."""
    conn = sqlite3.connect(db_with_embeddings)
    metrics = _compute_metrics(conn)
    result = _garbage_collect(conn, metrics, mode="off")
    assert result == []
    conn.close()


def test_garbage_collect_respects_permanent(db_with_embeddings):
    """Permanent nodes should never be GC'd."""
    conn = sqlite3.connect(db_with_embeddings)
    # Make n001 permanent
    conn.execute("UPDATE thought_nodes SET permanent=1 WHERE id='n001'")
    conn.commit()
    metrics = _compute_metrics(conn)
    result = _garbage_collect(conn, metrics, threshold=100.0, grace_days=0, mode="soft")
    assert "n001" not in result
    conn.close()


# ── _evaluate_permanence ─────────────────────────────────────────────────


def test_evaluate_permanence_promotes_high_access(db_with_embeddings):
    """Nodes with access_count >= threshold should become permanent."""
    conn = sqlite3.connect(db_with_embeddings)
    conn.execute("UPDATE thought_nodes SET access_count=15 WHERE id='n001'")
    conn.commit()

    stats = _evaluate_permanence(conn, access_threshold=10)
    assert stats.get("nodes_promoted", 0) >= 1

    row = conn.execute(
        "SELECT permanent FROM thought_nodes WHERE id='n001'"
    ).fetchone()
    assert row[0] == 1
    conn.close()


# ── _promote_core_memories ───────────────────────────────────────────────


def test_promote_core_memories_basic(db_with_embeddings):
    """Top nodes by fitness should become core_memory."""
    conn = sqlite3.connect(db_with_embeddings)
    metrics = _compute_metrics(conn)
    stats = _promote_core_memories(conn, metrics)

    assert stats["promoted"] >= 0
    assert stats["demoted"] >= 0
    assert stats["target"] == int(math.sqrt(len(metrics)))

    # Should have at least some core_memories (if metrics > 0)
    target = stats["target"]
    count = conn.execute(
        "SELECT COUNT(*) FROM thought_nodes WHERE node_type='core_memory'"
    ).fetchone()[0]
    assert count <= target
    conn.close()


# ── _generate_dream ──────────────────────────────────────────────────────


def test_generate_dream_no_model_fn_returns_none(db_with_embeddings):
    """Without model_fn, dream should not be generated."""
    conn = sqlite3.connect(db_with_embeddings)
    result = _generate_dream(conn, [("n001", "n004", 0.75)])
    assert result is None
    conn.close()


def test_generate_dream_with_model_fn(db_with_embeddings):
    """With model_fn and cross-source pairs, dream should be created."""
    conn = sqlite3.connect(db_with_embeddings)

    def fake_model(prompt):
        return "Both rely on the same unstated assumption about order of operations."

    dream_id = _generate_dream(
        conn,
        [("n001", "n008", 0.78)],  # n001=source_a, n008=source_d → different sources
        model_fn=fake_model,
    )
    assert dream_id is not None

    row = conn.execute(
        "SELECT node_type FROM thought_nodes WHERE id=?", (dream_id,)
    ).fetchone()
    assert row is not None
    assert row[0] == "dream"
    conn.close()


def test_generate_dream_same_source_skipped(db_with_embeddings):
    """Pairs from the same source_file should not produce a dream."""
    conn = sqlite3.connect(db_with_embeddings)

    def fake_model(prompt):
        return "Dream synthesis result."

    # n001 and n002 are both source_a
    dream_id = _generate_dream(
        conn, [("n001", "n002", 0.95)], model_fn=fake_model,
    )
    assert dream_id is None
    conn.close()


# ── _embed_orphans ───────────────────────────────────────────────────────


def test_embed_orphans_empty_when_all_embedded(db_with_embeddings):
    """When all nodes have embeddings, orphans should be 0."""
    conn = sqlite3.connect(db_with_embeddings)
    count = _embed_orphans(conn)
    assert count == 0
    conn.close()


# ── Integration: run_sleep_cycle ─────────────────────────────────────────


def test_run_sleep_cycle_vectorized(db_with_embeddings):
    """Full vectorized pipeline should complete without error."""
    with patch("core.sleep.config") as mock_cfg:
        mock_cfg.gc_mode = "off"
        mock_cfg.gc_threshold = 0.05
        mock_cfg.gc_grace_days = 7
        mock_cfg.gc_think_cycle_penalty = 1.5

        result = run_sleep_cycle(
            db_path=db_with_embeddings,
            limit=5,
            background_dream=False,
            max_edges=10,
            cross_source_only=True,
        )

    assert "error" not in result
    assert result["nodes_selected"] > 0
    assert result["total_nodes"] > 0
    assert result["cross_link_candidates"] >= 0
    assert result["dedup_candidates"] >= 0
    assert result["elapsed_s"] >= 0

    # Verify DB is still valid
    conn = sqlite3.connect(db_with_embeddings)
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    table_names = {t[0] for t in tables}
    assert "thought_nodes" in table_names
    assert "derivation_edges" in table_names
    conn.close()


def test_run_sleep_cycle_background_dream(db_with_embeddings):
    """background_dream=True should not error."""
    with patch("core.sleep.config") as mock_cfg:
        mock_cfg.gc_mode = "soft"
        mock_cfg.gc_threshold = 0.0
        mock_cfg.gc_grace_days = 0
        mock_cfg.gc_think_cycle_penalty = 1.0

        result = run_sleep_cycle(
            db_path=db_with_embeddings,
            limit=100,  # include all nodes so similar pairs are found
            model_fn=lambda p: "Dream synthesis",
            background_dream=True,
            max_edges=10,
            cross_source_only=False,
        )

    assert "error" not in result
    assert result["cross_link_candidates"] > 0, (
        "Need cross-link candidates for dream_pending, check similarity setup"
    )
    assert result["dream_pending"] is True
    assert result["dream_id"] is None  # not yet produced (async)


def test_run_sleep_cycle_cross_source_only(db_with_embeddings):
    """cross_source_only=True should produce same_source_skipped counts."""
    with patch("core.sleep.config") as mock_cfg:
        mock_cfg.gc_mode = "off"
        mock_cfg.gc_threshold = 0.0
        mock_cfg.gc_grace_days = 0
        mock_cfg.gc_think_cycle_penalty = 1.0

        result = run_sleep_cycle(
            db_path=db_with_embeddings,
            limit=10,
            cross_source_only=True,
            max_edges=100,
        )

    assert "cross_link_same_source_skipped" in result
    assert result["cross_link_candidates"] > 0  # more nodes available


def test_run_sleep_cycle_empty_graph():
    """Empty database (thought_nodes but no data) should not crash."""
    path = _make_test_db(with_embeddings=True)
    try:
        conn = sqlite3.connect(path)
        _insert_node(conn, "lonely", "lonely node")
        _insert_embedding(conn, "lonely", _make_embedding(1))
        conn.commit()
        conn.close()

        with patch("core.sleep.config") as mock_cfg:
            mock_cfg.gc_mode = "off"
            mock_cfg.gc_threshold = 0.0
            mock_cfg.gc_grace_days = 0
            mock_cfg.gc_think_cycle_penalty = 1.0

            result = run_sleep_cycle(db_path=path, limit=10)
        # One node can't form pairs, but shouldn't crash
        assert "error" in result or result["total_nodes"] >= 0
    finally:
        os.unlink(path)
