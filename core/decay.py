#!/usr/bin/env python3
"""
Decay functionality for cashew thought graph.
Handles automatic aging/pruning of nodes that have shown no signal.

Decay signals (post-confidence-removal, post-node_type-removal):
  - access_count == 0   (never retrieved by a query/think cycle)
  - age >= min_age_days (had time to be useful)
  - edge_degree == 0    (orphaned: no derivation edges in or out)

These are pure deterministic graph facts. node_type is informational
metadata only and never participates in decay logic.

Cascade decay propagates only when the same gate holds on the child:
old, never-touched, and orphaned once the parent goes away.
"""

import sqlite3
from datetime import datetime, timezone, timedelta
from typing import Dict, Set


CASCADE_MIN_AGE_DAYS = 30


def _ensure_edges_table(cursor: sqlite3.Cursor) -> None:
    """Defensive: callers may pass a DB without the edges table (legacy
    test fixtures, partially-initialized brains). The degree gate references
    it from a subquery, so ensure it exists with at least the columns we read."""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS derivation_edges (
            parent_id TEXT,
            child_id TEXT,
            weight REAL,
            reasoning TEXT,
            timestamp TEXT,
            PRIMARY KEY (parent_id, child_id)
        )
    """)


def _edge_degree_excluding(cursor: sqlite3.Cursor, node_id: str,
                           exclude_parent: str = None) -> int:
    """Count live edges touching `node_id` (as src or dst), optionally
    pretending one parent edge is gone (used by cascade to test the
    "would-be orphan after parent decays" condition)."""
    if exclude_parent is None:
        cursor.execute("""
            SELECT COUNT(*) FROM derivation_edges de
            WHERE de.parent_id = ? OR de.child_id = ?
        """, (node_id, node_id))
    else:
        cursor.execute("""
            SELECT COUNT(*) FROM derivation_edges de
            WHERE (de.parent_id = ? OR de.child_id = ?)
            AND NOT (de.parent_id = ? AND de.child_id = ?)
        """, (node_id, node_id, exclude_parent, node_id))
    return cursor.fetchone()[0]


def _cascade_eligible(cursor: sqlite3.Cursor, child_id: str,
                      decayed_parent_id: str) -> bool:
    """Return True if `child_id` matches the cascade gate after
    `decayed_parent_id` is removed from the live graph.

    Gate: access_count == 0 AND age >= 30d AND edge_degree == 0
    (where edge_degree is computed excluding the parent that just decayed
    and ignoring all other already-decayed parents — see caller).
    Permanent or already-decayed nodes are filtered separately by callers.
    """
    cursor.execute("""
        SELECT access_count, timestamp
        FROM thought_nodes WHERE id = ?
    """, (child_id,))
    row = cursor.fetchone()
    if not row:
        return False
    access_count, timestamp = row
    if access_count and int(access_count) > 0:
        return False
    if not timestamp:
        return False
    try:
        ts = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
    except ValueError:
        return False
    age = datetime.now(timezone.utc) - ts.astimezone(timezone.utc)
    return age >= timedelta(days=CASCADE_MIN_AGE_DAYS)


def cascade_decay(db_path: str, decayed_node_id: str) -> Dict:
    """
    DFS down from a decayed node, decaying children that match the gate
    (access_count == 0 AND age >= 30d) AND whose only live parent was the
    just-decayed node (i.e. they become orphaned by its removal).

    Returns:
        Dict with cascading statistics
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    _ensure_edges_table(cursor)

    cascaded_count = 0
    nodes_to_process = [decayed_node_id]
    processed_nodes: Set[str] = set()

    while nodes_to_process:
        current_node_id = nodes_to_process.pop(0)

        if current_node_id in processed_nodes:
            continue
        processed_nodes.add(current_node_id)

        # Find all children of the current node
        cursor.execute("""
            SELECT child_id FROM derivation_edges
            WHERE parent_id = ?
        """, (current_node_id,))
        children = [row[0] for row in cursor.fetchall()]

        for child_id in children:
            # Skip children that have other live parents — they're anchored
            cursor.execute("""
                SELECT COUNT(*) FROM derivation_edges de
                JOIN thought_nodes tn ON de.parent_id = tn.id
                WHERE de.child_id = ?
                AND (tn.decayed IS NULL OR tn.decayed = 0)
                AND tn.id != ?
            """, (child_id, current_node_id))
            if cursor.fetchone()[0] > 0:
                continue

            # Skip if already decayed or permanent
            cursor.execute("""
                SELECT decayed, permanent FROM thought_nodes WHERE id = ?
            """, (child_id,))
            row = cursor.fetchone()
            if not row:
                continue
            is_decayed, is_permanent = row
            if is_decayed or is_permanent:
                continue

            # Apply degree+access+age gate
            if not _cascade_eligible(cursor, child_id, current_node_id):
                continue

            cursor.execute("""
                UPDATE thought_nodes
                SET decayed = 1, last_updated = ?
                WHERE id = ?
            """, (datetime.now(timezone.utc).isoformat(), child_id))

            cascaded_count += 1
            nodes_to_process.append(child_id)

    conn.commit()
    conn.close()

    return {"cascaded": cascaded_count}


def auto_decay(db_path: str, min_age_days: int = 14,
               enable_cascading: bool = True) -> Dict:
    """Decay orphaned, never-accessed nodes old enough to be safe to lose.

    Direct decay gate:
      - not already decayed, not permanent
      - access_count == 0
      - age >= min_age_days
      - edge_degree == 0 (no live derivation edges in or out)

    A "live" edge is one whose other endpoint is also not decayed. Already-
    decayed neighbors don't count — they're effectively gone.

    Cascade decay walks children that match the same gate (with the
    stricter 30d threshold) and have no other live parents.

    Args:
        db_path: Path to the SQLite database
        min_age_days: Minimum age in days for direct decay eligibility
        enable_cascading: Whether to enable cascading decay to children

    Returns:
        Dict with pruning statistics
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    _ensure_edges_table(cursor)

    cutoff = (datetime.now(timezone.utc) - timedelta(days=min_age_days)).isoformat()

    # Orphaned = no edges to any *live* (non-decayed) neighbor.
    direct_filter = """
        (decayed IS NULL OR decayed = 0)
        AND (permanent IS NULL OR permanent = 0)
        AND access_count = 0
        AND timestamp < ?
        AND NOT EXISTS (
            SELECT 1 FROM derivation_edges de
            JOIN thought_nodes other ON (
                (de.parent_id = thought_nodes.id AND other.id = de.child_id)
                OR (de.child_id = thought_nodes.id AND other.id = de.parent_id)
            )
            WHERE (other.decayed IS NULL OR other.decayed = 0)
        )
    """

    cursor.execute(f"""
        SELECT id FROM thought_nodes
        WHERE {direct_filter}
    """, (cutoff,))
    nodes_to_decay = [row[0] for row in cursor.fetchall()]

    cursor.execute(f"""
        UPDATE thought_nodes SET decayed = 1, last_updated = ?
        WHERE {direct_filter}
    """, (datetime.now(timezone.utc).isoformat(), cutoff))

    direct_count = cursor.rowcount
    conn.commit()
    conn.close()

    total_cascaded = 0
    if enable_cascading:
        for node_id in nodes_to_decay:
            cascade_result = cascade_decay(db_path, node_id)
            total_cascaded += cascade_result["cascaded"]

    return {
        "pruned": direct_count,
        "cascaded": total_cascaded,
        "total": direct_count + total_cascaded
    }


def get_decay_candidates(db_path: str, min_age_days: int = 14,
                        show_cascade_preview: bool = False) -> Dict:
    """Get statistics on nodes eligible for decay without actually decaying them."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    _ensure_edges_table(cursor)

    cutoff = (datetime.now(timezone.utc) - timedelta(days=min_age_days)).isoformat()

    direct_filter = """
        (decayed IS NULL OR decayed = 0)
        AND (permanent IS NULL OR permanent = 0)
        AND access_count = 0
        AND timestamp < ?
        AND NOT EXISTS (
            SELECT 1 FROM derivation_edges de
            JOIN thought_nodes other ON (
                (de.parent_id = thought_nodes.id AND other.id = de.child_id)
                OR (de.child_id = thought_nodes.id AND other.id = de.parent_id)
            )
            WHERE (other.decayed IS NULL OR other.decayed = 0)
        )
    """

    cursor.execute(f"""
        SELECT COUNT(*) FROM thought_nodes
        WHERE {direct_filter}
    """, (cutoff,))
    count = cursor.fetchone()[0]

    result_dict = {"candidates": count or 0}

    if show_cascade_preview and count and count > 0:
        cursor.execute(f"""
            SELECT id FROM thought_nodes
            WHERE {direct_filter}
        """, (cutoff,))
        candidate_ids = [row[0] for row in cursor.fetchall()]

        total_would_cascade = 0
        for node_id in candidate_ids:
            total_would_cascade += simulate_cascade_decay(db_path, node_id)

        result_dict["cascade_preview"] = total_would_cascade
        result_dict["total_preview"] = (count or 0) + total_would_cascade

    conn.close()
    return result_dict


def simulate_cascade_decay(db_path: str, decayed_node_id: str) -> int:
    """Simulate cascading decay without modifying the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    _ensure_edges_table(cursor)

    would_cascade = 0
    nodes_to_process = [decayed_node_id]
    processed_nodes: Set[str] = set()

    while nodes_to_process:
        current_node_id = nodes_to_process.pop(0)
        if current_node_id in processed_nodes:
            continue
        processed_nodes.add(current_node_id)

        cursor.execute("""
            SELECT child_id FROM derivation_edges WHERE parent_id = ?
        """, (current_node_id,))
        children = [row[0] for row in cursor.fetchall()]

        for child_id in children:
            cursor.execute("""
                SELECT COUNT(*) FROM derivation_edges de
                JOIN thought_nodes tn ON de.parent_id = tn.id
                WHERE de.child_id = ?
                AND (tn.decayed IS NULL OR tn.decayed = 0)
                AND tn.id != ?
            """, (child_id, current_node_id))
            if cursor.fetchone()[0] > 0:
                continue

            cursor.execute("""
                SELECT decayed, permanent FROM thought_nodes WHERE id = ?
            """, (child_id,))
            row = cursor.fetchone()
            if not row:
                continue
            is_decayed, is_permanent = row
            if is_decayed or is_permanent:
                continue

            if not _cascade_eligible(cursor, child_id, current_node_id):
                continue

            would_cascade += 1
            nodes_to_process.append(child_id)

    conn.close()
    return would_cascade
