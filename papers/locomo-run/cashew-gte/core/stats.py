#!/usr/bin/env python3
"""
Cashew Graph Statistics Module
Consolidated query functions for graph metrics. Single source of truth
for node counts, edge counts, embedding coverage, etc.

All functions accept a sqlite3.Cursor. Use get_connection() to obtain one.
"""

import sqlite3
from typing import Dict, Tuple, Union


def get_connection(db_path: str) -> sqlite3.Connection:
    """Single connection factory for the graph database."""
    return sqlite3.connect(db_path)


def get_active_node_count(cursor: sqlite3.Cursor) -> int:
    """Count non-decayed nodes."""
    cursor.execute(
        "SELECT COUNT(*) FROM thought_nodes WHERE decayed IS NULL OR decayed = 0"
    )
    return cursor.fetchone()[0]


def get_total_node_count(cursor: sqlite3.Cursor, include_decayed: bool = False) -> int:
    """Count nodes. If include_decayed is False (default), only active nodes."""
    if include_decayed:
        cursor.execute("SELECT COUNT(*) FROM thought_nodes")
    else:
        cursor.execute(
            "SELECT COUNT(*) FROM thought_nodes WHERE decayed IS NULL OR decayed = 0"
        )
    return cursor.fetchone()[0]


def get_edge_count(cursor: sqlite3.Cursor) -> int:
    """Count total derivation edges."""
    cursor.execute("SELECT COUNT(*) FROM derivation_edges")
    return cursor.fetchone()[0]


def get_embedding_coverage(cursor: sqlite3.Cursor) -> Tuple[int, int]:
    """Return (embedded_count, embeddable_count) for non-decayed nodes.

    Empty-content nodes are excluded from the denominator since they are
    intentionally not embedded — a zero-norm vector breaks sqlite-vec
    cosine distance queries.
    """
    cursor.execute("""
        SELECT COUNT(*) FROM embeddings e
        JOIN thought_nodes tn ON e.node_id = tn.id
        WHERE tn.decayed IS NULL OR tn.decayed = 0
    """)
    embedded = cursor.fetchone()[0]
    cursor.execute("""
        SELECT COUNT(*) FROM thought_nodes
        WHERE (decayed IS NULL OR decayed = 0)
        AND content IS NOT NULL AND TRIM(content) != ''
    """)
    total = cursor.fetchone()[0]
    return embedded, total


def get_orphan_count(cursor: sqlite3.Cursor) -> int:
    """Count active nodes with no edges (neither parent nor child)."""
    cursor.execute("""
        SELECT COUNT(*) FROM thought_nodes tn
        WHERE (tn.decayed IS NULL OR tn.decayed = 0)
        AND NOT EXISTS (
            SELECT 1 FROM derivation_edges de
            WHERE de.parent_id = tn.id OR de.child_id = tn.id
        )
    """)
    return cursor.fetchone()[0]


def get_node_edge_count(cursor: sqlite3.Cursor, node_id: str) -> int:
    """Count outgoing edges (children) for a specific node."""
    cursor.execute(
        "SELECT COUNT(*) FROM derivation_edges WHERE parent_id = ?", (node_id,)
    )
    return cursor.fetchone()[0]


def get_think_node_count(cursor: sqlite3.Cursor) -> int:
    """Count system-generated (think cycle) nodes."""
    cursor.execute(
        "SELECT COUNT(*) FROM thought_nodes WHERE source_file = 'system_generated'"
    )
    return cursor.fetchone()[0]


def get_domain_counts(cursor: sqlite3.Cursor) -> Dict[str, int]:
    """Return {domain: count} for active nodes."""
    cursor.execute("""
        SELECT COALESCE(domain, 'unknown'), COUNT(*)
        FROM thought_nodes
        WHERE decayed IS NULL OR decayed = 0
        GROUP BY domain
    """)
    return dict(cursor.fetchall())


def get_permanence_stats(cursor: sqlite3.Cursor) -> Dict:
    """Return permanence statistics for active nodes."""
    cursor.execute("""
        SELECT COUNT(*) FROM thought_nodes 
        WHERE (decayed IS NULL OR decayed = 0) AND permanent > 0
    """)
    permanent_total = cursor.fetchone()[0]
    
    cursor.execute("""
        SELECT permanent, COUNT(*) 
        FROM thought_nodes 
        WHERE (decayed IS NULL OR decayed = 0) AND permanent > 0
        GROUP BY permanent
    """)
    permanence_breakdown = dict(cursor.fetchall())
    
    return {
        "permanent_total": permanent_total,
        "auto_permanent": permanence_breakdown.get(1, 0),
        "manually_pinned": permanence_breakdown.get(2, 0),
    }


def get_graph_summary(db_path_or_cursor: Union[str, sqlite3.Cursor]) -> Dict:
    """
    Convenience function returning all key metrics as a dict.
    Accepts either a db_path string or an existing cursor.
    """
    own_conn = None
    if isinstance(db_path_or_cursor, str):
        own_conn = get_connection(db_path_or_cursor)
        cursor = own_conn.cursor()
    else:
        cursor = db_path_or_cursor

    try:
        active = get_active_node_count(cursor)
        total = get_total_node_count(cursor, include_decayed=True)
        edges = get_edge_count(cursor)
        embedded, _ = get_embedding_coverage(cursor)
        orphans = get_orphan_count(cursor)
        domains = get_domain_counts(cursor)
        think_nodes = get_think_node_count(cursor)
        permanence = get_permanence_stats(cursor)

        return {
            "active_nodes": active,
            "total_nodes": total,
            "decayed_nodes": total - active,
            "edges": edges,
            "edge_node_ratio": round(edges / max(1, active), 2),
            "embedded_nodes": embedded,
            "embedding_coverage": round(embedded / max(1, active), 4),
            "orphan_count": orphans,
            "orphan_pct": round(orphans / max(1, active) * 100, 1),
            "domains": domains,
            "think_node_count": think_nodes,
            "permanent_nodes": permanence["permanent_total"],
            "auto_permanent": permanence["auto_permanent"],
            "manually_pinned": permanence["manually_pinned"],
            "permanent_ratio": round(permanence["permanent_total"] / max(1, active), 4),
        }
    finally:
        if own_conn:
            own_conn.close()
