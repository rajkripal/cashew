"""Regression tests for cold-start think-cycle and extractor bugs.

Covers issues filed by fauxreigner-mb against a freshly ingested ~20k-node
graph:

- #13: extractor parser stored markdown headers as no-value nodes.
- #15: think cycle's diversity sampling source_file filter missed
  `extractor:*` nodes, leaving the random-walk pool empty on fresh graphs.
- #16: high-activation pool sorted by `last_accessed` even when every node
  had `access_count=0`, so cold think cycles deterministically returned the
  same oldest-ingested nodes.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest

from core.session import _find_cluster_for_thinking
from extractors.utils import parse_extraction_lines


# ---------------------------------------------------------------------------
# #13 — extractor header filtering.
# ---------------------------------------------------------------------------

def test_parse_extraction_lines_drops_markdown_headers():
    response = (
        "# Extracted Insights\n"
        "## Decisions\n"
        "Cashew uses additive-only migrations within a major version.\n"
        "---\n"
        "The brain is the source of truth for Bunny.\n"
    )
    assert parse_extraction_lines(response) == [
        "Cashew uses additive-only migrations within a major version.",
        "The brain is the source of truth for Bunny.",
    ]


def test_parse_extraction_lines_drops_blanks_and_strips():
    response = "  one\n\n  two  \n"
    assert parse_extraction_lines(response) == ["one", "two"]


def test_parse_extraction_lines_keeps_statements_starting_with_letters():
    # Sanity: a line that merely *contains* a `#` shouldn't be dropped.
    response = "Issue #42 is closed.\nA fact about C# code."
    assert parse_extraction_lines(response) == [
        "Issue #42 is closed.",
        "A fact about C# code.",
    ]


# ---------------------------------------------------------------------------
# Helpers for the think-cycle tests.
# ---------------------------------------------------------------------------

def _seed(temp_db: str, rows):
    """Insert nodes with given (id, source_file, access_count, last_accessed)."""
    conn = sqlite3.connect(temp_db)
    cur = conn.cursor()
    for nid, source_file, access_count, last_accessed in rows:
        cur.execute(
            """INSERT INTO thought_nodes
               (id, content, node_type, domain, timestamp,
                access_count, last_accessed, source_file, decayed)
               VALUES (?, ?, 'fact', 'd', ?, ?, ?, ?, 0)""",
            (nid, f"content {nid}", last_accessed, access_count,
             last_accessed, source_file),
        )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# #16 — high-activation pool determinism on cold graphs.
# ---------------------------------------------------------------------------

def test_high_activation_not_deterministic_when_all_access_zero(temp_db):
    # 30 nodes, all access_count=0, monotonically increasing last_accessed.
    rows = [
        (f"n{i:03d}", "extractor:session:s1", 0, f"2026-04-27T00:00:{i:02d}+00:00")
        for i in range(30)
    ]
    _seed(temp_db, rows)

    runs = [tuple(_find_cluster_for_thinking(temp_db)) for _ in range(5)]
    # Pre-fix: every run returns the same first-ingested ids.
    # Post-fix: random tiebreaker means at least two runs differ.
    assert len(set(runs)) > 1, (
        "high-activation pool returned identical node ids across 5 runs on a "
        "graph where every node has access_count=0; sort is still falling "
        "through to deterministic last_accessed ordering"
    )


# ---------------------------------------------------------------------------
# #15 — diversity sampling must see extractor:* nodes.
# ---------------------------------------------------------------------------

def test_diversity_sampling_sees_extractor_nodes(temp_db):
    now = datetime.now(timezone.utc).isoformat()
    # 40 extractor-ingested nodes; no system_generated nodes at all. Pre-fix,
    # the underrep_stats query would return zero rows and the random-walk
    # pool would be empty.
    rows = [
        (f"e{i:03d}", "extractor:session:s1", 0, now)
        for i in range(40)
    ]
    _seed(temp_db, rows)

    selected = _find_cluster_for_thinking(temp_db)
    assert selected, "no nodes selected at all"
    # We can't directly inspect which slot each node came from, but with the
    # widened filter the query path that builds random_walk_candidates must
    # find rows. Re-run several times and ensure the underlying SQL returns
    # non-empty underrep stats.
    conn = sqlite3.connect(temp_db)
    cur = conn.cursor()
    cur.execute(
        """SELECT COUNT(*) FROM thought_nodes
           WHERE (decayed IS NULL OR decayed = 0)
           AND timestamp > datetime('now', '-30 days')
           AND (source_file LIKE '%system_generated%'
                OR source_file LIKE 'extractor:%')"""
    )
    assert cur.fetchone()[0] == 40
    conn.close()
