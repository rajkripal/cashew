#!/usr/bin/env python3
"""
Tests for decay functionality including cascading tree decay.

Post-confidence-removal, post-node_type-removal gate (both direct and cascade):
  - access_count == 0
  - age >= min_age_days (default 14 direct, 30 cascade)
  - edge_degree == 0 (no edges to live neighbors)

node_type plays no role in decay logic; it's purely a display tag.
"""

import pytest
import sqlite3
import tempfile
import os
from datetime import datetime, timezone, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.decay import auto_decay, get_decay_candidates, cascade_decay, simulate_cascade_decay


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ago(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


@pytest.fixture
def decay_test_db():
    """Create a test database structured for the degree+access gate.

    Layout:
        root (old, ac=0, edges to child1/child2)
            child1 (old, ac=0)        — orphan once root decays → cascades
                grandchild1 (old, ac=0, also has anchored_parent) — anchored, won't cascade
            child2 (old, ac=0)        — orphan once root decays → cascades
                grandchild2 (old, ac=0) — orphan once child2 decays → cascades
        anchored_parent (recent, ac=0, permanent=1) — protected
        isolated_old (old, ac=0, no edges) — direct decay
        recent_obs (recent, ac=5)     — recent + accessed, won't decay
        accessed_old (old, ac=3, no edges) — accessed, protected from direct decay
        connected_old (old, ac=0, edge to recent_obs) — has live neighbor, protected
    """
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)

    conn = sqlite3.connect(path)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE thought_nodes (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            node_type TEXT NOT NULL,
            domain TEXT,
            timestamp TEXT,
            access_count INTEGER DEFAULT 0,
            last_accessed TEXT,
            source_file TEXT,
            decayed INTEGER DEFAULT 0,
            metadata TEXT DEFAULT '{}',
            last_updated TEXT,
            mood_state TEXT,
            permanent INTEGER DEFAULT 0
        )
    ''')

    cursor.execute('''
        CREATE TABLE derivation_edges (
            parent_id TEXT,
            child_id TEXT,
            weight REAL,
            reasoning TEXT,
            timestamp TEXT,
            PRIMARY KEY (parent_id, child_id),
            FOREIGN KEY (parent_id) REFERENCES thought_nodes(id),
            FOREIGN KEY (child_id) REFERENCES thought_nodes(id)
        )
    ''')

    old = _ago(60)
    recent = _now()

    # node_type values are arbitrary — they don't gate anything anymore.
    nodes = [
        # id,               type,           timestamp,  ac, perm
        ("root",            "observation",  old,        0,  0),
        ("child1",          "insight",      old,        0,  0),
        ("child2",          "observation",  old,        0,  0),
        ("grandchild1",     "observation",  old,        0,  0),
        ("grandchild2",     "observation",  old,        0,  0),
        ("anchored_parent", "observation",  recent,     0,  1),
        ("isolated_old",    "observation",  old,        0,  0),
        ("recent_obs",      "observation",  recent,     5,  0),
        ("accessed_old",    "observation",  old,        3,  0),
        ("connected_old",   "observation",  old,        0,  0),
    ]

    for node_id, node_type, ts, ac, perm in nodes:
        cursor.execute("""
            INSERT INTO thought_nodes
            (id, content, node_type, timestamp, access_count, source_file, metadata, permanent)
            VALUES (?, ?, ?, ?, ?, 'test', '{}', ?)
        """, (node_id, f"content {node_id}", node_type, ts, ac, perm))

    edges = [
        ("root", "child1", "parent_of"),
        ("root", "child2", "parent_of"),
        ("child1", "grandchild1", "parent_of"),
        ("child2", "grandchild2", "parent_of"),
        ("anchored_parent", "grandchild1", "supports"),
        ("connected_old", "recent_obs", "references"),
    ]
    for parent_id, child_id, relation in edges:
        cursor.execute("""
            INSERT INTO derivation_edges (parent_id, child_id, weight, reasoning, timestamp)
            VALUES (?, ?, 1.0, ?, ?)
        """, (parent_id, child_id, relation, _now()))

    conn.commit()
    conn.close()

    yield path
    os.unlink(path)


def test_cascade_propagation_basic(decay_test_db):
    """After root decays, its now-orphaned children that match the gate cascade."""
    conn = sqlite3.connect(decay_test_db)
    cursor = conn.cursor()
    cursor.execute("UPDATE thought_nodes SET decayed = 1 WHERE id = 'root'")
    conn.commit()
    conn.close()

    result = cascade_decay(decay_test_db, "root")

    # child1 and child2 both had only root as a parent — both should cascade
    # (regardless of node_type — 'insight' vs 'observation' is irrelevant).
    assert result['cascaded'] >= 2

    conn = sqlite3.connect(decay_test_db)
    cursor = conn.cursor()
    cursor.execute("SELECT decayed FROM thought_nodes WHERE id = 'child1'")
    assert cursor.fetchone()[0] == 1
    cursor.execute("SELECT decayed FROM thought_nodes WHERE id = 'child2'")
    assert cursor.fetchone()[0] == 1
    conn.close()


def test_multi_parent_anchoring(decay_test_db):
    """Nodes with another live parent are NOT cascaded even if they match the gate."""
    conn = sqlite3.connect(decay_test_db)
    cursor = conn.cursor()
    cursor.execute("UPDATE thought_nodes SET decayed = 1 WHERE id = 'root'")
    conn.commit()
    conn.close()

    cascade_decay(decay_test_db, "root")

    # grandchild1 has anchored_parent (live, permanent) as a second parent — must stay alive.
    conn = sqlite3.connect(decay_test_db)
    cursor = conn.cursor()
    cursor.execute("SELECT decayed FROM thought_nodes WHERE id = 'grandchild1'")
    assert cursor.fetchone()[0] == 0
    conn.close()


def test_node_type_does_not_gate_decay(decay_test_db):
    """node_type is informational — 'insight' nodes decay just like 'observation' nodes
    when they fail the deterministic gate."""
    conn = sqlite3.connect(decay_test_db)
    cursor = conn.cursor()
    cursor.execute("UPDATE thought_nodes SET decayed = 1 WHERE id = 'root'")
    conn.commit()
    conn.close()

    cascade_decay(decay_test_db, "root")

    # child1 is an 'insight' but it's old, never accessed, and orphaned — decays.
    conn = sqlite3.connect(decay_test_db)
    cursor = conn.cursor()
    cursor.execute("SELECT decayed FROM thought_nodes WHERE id = 'child1'")
    assert cursor.fetchone()[0] == 1
    conn.close()


def test_access_count_gate_protects_node(decay_test_db):
    """access_count > 0 protects a node from direct decay."""
    auto_decay(decay_test_db, min_age_days=14)
    conn = sqlite3.connect(decay_test_db)
    cursor = conn.cursor()
    cursor.execute("SELECT decayed FROM thought_nodes WHERE id = 'accessed_old'")
    assert cursor.fetchone()[0] == 0
    conn.close()


def test_edge_degree_protects_node(decay_test_db):
    """A node with a live neighbor is protected from direct decay even if old + unaccessed."""
    auto_decay(decay_test_db, min_age_days=14, enable_cascading=False)
    conn = sqlite3.connect(decay_test_db)
    cursor = conn.cursor()
    # connected_old has an edge to recent_obs (live, accessed) — must NOT direct-decay.
    cursor.execute("SELECT decayed FROM thought_nodes WHERE id = 'connected_old'")
    assert cursor.fetchone()[0] == 0
    conn.close()


def test_auto_decay_with_cascading_integration(decay_test_db):
    """Direct decay grabs the orphan; cascade picks up children orphaned by the prune."""
    result = auto_decay(decay_test_db, min_age_days=14, enable_cascading=True)

    # Direct: only isolated_old qualifies for direct decay (root has live edges).
    assert result['pruned'] >= 1
    assert 'cascaded' in result
    assert 'total' in result
    assert result['total'] == result['pruned'] + result['cascaded']


def test_auto_decay_without_cascading(decay_test_db):
    """Disabling cascading skips the DFS walk."""
    result = auto_decay(decay_test_db, min_age_days=14, enable_cascading=False)

    assert result['pruned'] >= 1
    assert result.get('cascaded', 0) == 0
    assert result['total'] == result['pruned']


def test_dry_run_preview(decay_test_db):
    """get_decay_candidates with cascade preview reports counts without mutating."""
    conn = sqlite3.connect(decay_test_db)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM thought_nodes WHERE decayed = 1")
    before = cursor.fetchone()[0]
    conn.close()

    candidates = get_decay_candidates(decay_test_db, min_age_days=14, show_cascade_preview=True)

    assert 'cascade_preview' in candidates
    assert 'total_preview' in candidates
    assert candidates['total_preview'] >= candidates['candidates']

    conn = sqlite3.connect(decay_test_db)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM thought_nodes WHERE decayed = 1")
    after = cursor.fetchone()[0]
    conn.close()

    assert after == before


def test_simulate_cascade_decay_does_not_mutate(decay_test_db):
    """simulate_cascade_decay returns a count and leaves the DB untouched."""
    would = simulate_cascade_decay(decay_test_db, "root")
    assert isinstance(would, int)
    assert would >= 0

    conn = sqlite3.connect(decay_test_db)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM thought_nodes WHERE decayed = 1")
    assert cursor.fetchone()[0] == 0
    conn.close()


def test_empty_database_edge_cases(temp_db):
    """Decay functions handle empty databases gracefully."""
    result = auto_decay(temp_db)
    assert result['pruned'] == 0
    assert result.get('cascaded', 0) == 0

    candidates = get_decay_candidates(temp_db)
    assert candidates['candidates'] == 0

    cascade_result = cascade_decay(temp_db, "nonexistent")
    assert cascade_result['cascaded'] == 0


def test_circular_references_handling(decay_test_db):
    """Cascade visits each node at most once even with cycles."""
    conn = sqlite3.connect(decay_test_db)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO derivation_edges (parent_id, child_id, weight, reasoning, timestamp)
        VALUES ('child1', 'root', 1.0, 'circular reference', ?)
    """, (_now(),))
    conn.commit()
    conn.close()

    cascade_result = cascade_decay(decay_test_db, "root")
    assert isinstance(cascade_result['cascaded'], int)
    assert cascade_result['cascaded'] >= 0
