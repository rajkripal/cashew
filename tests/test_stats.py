#!/usr/bin/env python3
"""Tests for core.stats — consolidated graph query functions."""

import sqlite3
import pytest
from datetime import datetime, timezone

from core.stats import (
    get_connection,
    get_active_node_count,
    get_total_node_count,
    get_edge_count,
    get_embedding_coverage,
    get_orphan_count,
    get_node_edge_count,
    get_think_node_count,
    get_domain_counts,
    get_graph_summary,
)


# ── helpers ──────────────────────────────────────────────────────────

def _cursor(db_path):
    conn = sqlite3.connect(db_path)
    return conn, conn.cursor()


def _add_node(cursor, node_id, content="test", node_type="fact",
              domain="raj", decayed=0, source_file="test"):
    now = datetime.now(timezone.utc).isoformat()
    cursor.execute("""
        INSERT INTO thought_nodes
        (id, content, node_type, domain, timestamp, source_file,
         decayed, access_count, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, 0, '{}')
    """, (node_id, content, node_type, domain, now, source_file, decayed))


def _add_edge(cursor, parent_id, child_id, relation="supports", weight=0.7):
    now = datetime.now(timezone.utc).isoformat()
    cursor.execute("""
        INSERT INTO derivation_edges
        (parent_id, child_id, weight, reasoning, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """, (parent_id, child_id, weight, f"{relation} - test", now))


def _add_embedding(cursor, node_id):
    now = datetime.now(timezone.utc).isoformat()
    cursor.execute("""
        INSERT INTO embeddings (node_id, vector, model, updated_at)
        VALUES (?, X'00', 'test-model', ?)
    """, (node_id, now))


# ── empty database tests ─────────────────────────────────────────────

class TestEmptyDB:
    def test_active_node_count(self, temp_db):
        conn, cur = _cursor(temp_db)
        assert get_active_node_count(cur) == 0
        conn.close()

    def test_total_node_count(self, temp_db):
        conn, cur = _cursor(temp_db)
        assert get_total_node_count(cur) == 0
        assert get_total_node_count(cur, include_decayed=True) == 0
        conn.close()

    def test_edge_count(self, temp_db):
        conn, cur = _cursor(temp_db)
        assert get_edge_count(cur) == 0
        conn.close()

    def test_embedding_coverage(self, temp_db):
        conn, cur = _cursor(temp_db)
        embedded, total = get_embedding_coverage(cur)
        assert embedded == 0
        assert total == 0
        conn.close()

    def test_orphan_count(self, temp_db):
        conn, cur = _cursor(temp_db)
        assert get_orphan_count(cur) == 0
        conn.close()

    def test_think_node_count(self, temp_db):
        conn, cur = _cursor(temp_db)
        assert get_think_node_count(cur) == 0
        conn.close()

    def test_domain_counts(self, temp_db):
        conn, cur = _cursor(temp_db)
        assert get_domain_counts(cur) == {}
        conn.close()

    def test_graph_summary(self, temp_db):
        summary = get_graph_summary(temp_db)
        assert summary["active_nodes"] == 0
        assert summary["edges"] == 0
        assert summary["orphan_count"] == 0


# ── populated database tests (using fixture) ─────────────────────────

class TestPopulatedDB:
    """Uses temp_db_with_data: 5 nodes, 2 edges, 0 embeddings."""

    def test_active_node_count(self, temp_db_with_data):
        conn, cur = _cursor(temp_db_with_data)
        assert get_active_node_count(cur) == 5
        conn.close()

    def test_total_node_count_active_only(self, temp_db_with_data):
        conn, cur = _cursor(temp_db_with_data)
        assert get_total_node_count(cur) == 5
        conn.close()

    def test_total_node_count_with_decayed(self, temp_db_with_data):
        conn, cur = _cursor(temp_db_with_data)
        assert get_total_node_count(cur, include_decayed=True) == 5
        conn.close()

    def test_edge_count(self, temp_db_with_data):
        conn, cur = _cursor(temp_db_with_data)
        assert get_edge_count(cur) == 2
        conn.close()

    def test_embedding_coverage_none(self, temp_db_with_data):
        conn, cur = _cursor(temp_db_with_data)
        embedded, total = get_embedding_coverage(cur)
        assert embedded == 0
        assert total == 5
        conn.close()

    def test_orphan_count(self, temp_db_with_data):
        # node4 has no edges at all → orphan
        # node1→node2 (edge), node3→node5 (edge) → 4 connected, 1 orphan
        conn, cur = _cursor(temp_db_with_data)
        assert get_orphan_count(cur) == 1
        conn.close()

    def test_node_edge_count(self, temp_db_with_data):
        conn, cur = _cursor(temp_db_with_data)
        assert get_node_edge_count(cur, "node1") == 1  # node1→node2
        assert get_node_edge_count(cur, "node3") == 1  # node3→node5
        assert get_node_edge_count(cur, "node4") == 0  # orphan
        conn.close()

    def test_think_node_count_zero(self, temp_db_with_data):
        conn, cur = _cursor(temp_db_with_data)
        assert get_think_node_count(cur) == 0  # source_file='test', not 'system_generated'
        conn.close()

    def test_domain_counts(self, temp_db_with_data):
        conn, cur = _cursor(temp_db_with_data)
        domains = get_domain_counts(cur)
        assert domains["tech"] == 2
        assert domains["health"] == 1
        assert domains["philosophy"] == 1
        assert domains["meta"] == 1
        conn.close()

    def test_graph_summary_structure(self, temp_db_with_data):
        summary = get_graph_summary(temp_db_with_data)
        expected_keys = {
            "active_nodes", "total_nodes", "decayed_nodes", "edges",
            "edge_node_ratio", "embedded_nodes", "embedding_coverage",
            "orphan_count", "orphan_pct", "domains",
            "think_node_count", "permanent_nodes", "permanent_ratio",
            "auto_permanent", "manually_pinned",
        }
        assert set(summary.keys()) == expected_keys
        assert summary["active_nodes"] == 5
        assert summary["edges"] == 2
        assert summary["decayed_nodes"] == 0
        assert isinstance(summary["domains"], dict)


# ── decayed node handling ────────────────────────────────────────────

class TestDecayedNodes:
    def test_decayed_excluded_from_active(self, temp_db):
        conn, cur = _cursor(temp_db)
        _add_node(cur, "alive", decayed=0)
        _add_node(cur, "dead", decayed=1)
        conn.commit()

        assert get_active_node_count(cur) == 1
        conn.close()

    def test_decayed_included_when_requested(self, temp_db):
        conn, cur = _cursor(temp_db)
        _add_node(cur, "alive", decayed=0)
        _add_node(cur, "dead", decayed=1)
        conn.commit()

        assert get_total_node_count(cur, include_decayed=True) == 2
        assert get_total_node_count(cur, include_decayed=False) == 1
        conn.close()

    def test_decayed_excluded_from_embedding_coverage(self, temp_db):
        conn, cur = _cursor(temp_db)
        _add_node(cur, "alive", decayed=0)
        _add_node(cur, "dead", decayed=1)
        _add_embedding(cur, "alive")
        _add_embedding(cur, "dead")
        conn.commit()

        embedded, total = get_embedding_coverage(cur)
        assert embedded == 1  # only the alive node's embedding counts
        assert total == 1
        conn.close()

    def test_decayed_excluded_from_orphan_count(self, temp_db):
        conn, cur = _cursor(temp_db)
        _add_node(cur, "alive_orphan", decayed=0)
        _add_node(cur, "dead_orphan", decayed=1)
        conn.commit()

        assert get_orphan_count(cur) == 1
        conn.close()

    def test_decayed_excluded_from_domain_counts(self, temp_db):
        conn, cur = _cursor(temp_db)
        _add_node(cur, "a", domain="raj", decayed=0)
        _add_node(cur, "b", domain="raj", decayed=1)
        _add_node(cur, "c", domain="bunny", decayed=0)
        conn.commit()

        domains = get_domain_counts(cur)
        assert domains.get("raj", 0) == 1
        assert domains.get("bunny", 0) == 1
        conn.close()


# ── hotspot counting ─────────────────────────────────────────────────

# ── edge counts ──────────────────────────────────────────────────────

class TestEdgeCounts:
    def test_total_edge_count(self, temp_db):
        conn, cur = _cursor(temp_db)
        _add_node(cur, "a")
        _add_node(cur, "b")
        _add_node(cur, "c")
        _add_edge(cur, "a", "b")
        _add_edge(cur, "a", "c")
        conn.commit()

        assert get_edge_count(cur) == 2
        conn.close()

    def test_per_node_edge_count(self, temp_db):
        conn, cur = _cursor(temp_db)
        _add_node(cur, "a")
        _add_node(cur, "b")
        _add_node(cur, "c")
        _add_edge(cur, "a", "b")
        _add_edge(cur, "a", "c")
        conn.commit()

        assert get_node_edge_count(cur, "a") == 2
        assert get_node_edge_count(cur, "b") == 0  # outgoing only
        conn.close()


# ── embedding coverage ───────────────────────────────────────────────

class TestEmbeddingCoverage:
    def test_partial_coverage(self, temp_db):
        conn, cur = _cursor(temp_db)
        _add_node(cur, "a")
        _add_node(cur, "b")
        _add_node(cur, "c")
        _add_embedding(cur, "a")
        conn.commit()

        embedded, total = get_embedding_coverage(cur)
        assert embedded == 1
        assert total == 3
        conn.close()

    def test_full_coverage(self, temp_db):
        conn, cur = _cursor(temp_db)
        _add_node(cur, "a")
        _add_node(cur, "b")
        _add_embedding(cur, "a")
        _add_embedding(cur, "b")
        conn.commit()

        embedded, total = get_embedding_coverage(cur)
        assert embedded == 2
        assert total == 2
        conn.close()


# ── orphan detection ─────────────────────────────────────────────────

class TestOrphanDetection:
    def test_all_orphans(self, temp_db):
        conn, cur = _cursor(temp_db)
        _add_node(cur, "a")
        _add_node(cur, "b")
        conn.commit()

        assert get_orphan_count(cur) == 2
        conn.close()

    def test_no_orphans(self, temp_db):
        conn, cur = _cursor(temp_db)
        _add_node(cur, "a")
        _add_node(cur, "b")
        _add_edge(cur, "a", "b")
        conn.commit()

        assert get_orphan_count(cur) == 0  # both participate in an edge
        conn.close()

    def test_child_is_not_orphan(self, temp_db):
        conn, cur = _cursor(temp_db)
        _add_node(cur, "parent")
        _add_node(cur, "child")
        _add_node(cur, "loner")
        _add_edge(cur, "parent", "child")
        conn.commit()

        assert get_orphan_count(cur) == 1  # only "loner"
        conn.close()


# ── think node count ─────────────────────────────────────────────────

class TestThinkNodeCount:
    def test_counts_system_generated(self, temp_db):
        conn, cur = _cursor(temp_db)
        _add_node(cur, "t1", source_file="system_generated")
        _add_node(cur, "t2", source_file="system_generated")
        _add_node(cur, "n1", source_file="conversation.txt")
        conn.commit()

        assert get_think_node_count(cur) == 2
        conn.close()


# ── get_connection ───────────────────────────────────────────────────

class TestGetConnection:
    def test_returns_connection(self, temp_db):
        conn = get_connection(temp_db)
        assert isinstance(conn, sqlite3.Connection)
        conn.close()


# ── get_graph_summary interfaces ─────────────────────────────────────

class TestGraphSummaryInterfaces:
    def test_accepts_db_path(self, temp_db_with_data):
        summary = get_graph_summary(temp_db_with_data)
        assert summary["active_nodes"] == 5

    def test_accepts_cursor(self, temp_db_with_data):
        conn, cur = _cursor(temp_db_with_data)
        summary = get_graph_summary(cur)
        assert summary["active_nodes"] == 5
        conn.close()

    def test_edge_node_ratio(self, temp_db_with_data):
        summary = get_graph_summary(temp_db_with_data)
        assert summary["edge_node_ratio"] == 0.4  # 2 edges / 5 nodes

    def test_orphan_pct(self, temp_db_with_data):
        summary = get_graph_summary(temp_db_with_data)
        assert summary["orphan_pct"] == 20.0  # 1 orphan / 5 nodes * 100
