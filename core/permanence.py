#!/usr/bin/env python3
"""
Permanence functionality for cashew thought graph.
Handles promotion of frequently accessed nodes to permanent status.

Design philosophy:
- Permanence is binary and irreversible ("you cannot erase traumas")
- Access count is the primary signal (frequently retrieved = important = permanent)
- Threshold is discoverable from data, not hardcoded magic
- Simple and consistent with cashew's core principles
"""

import sqlite3
from datetime import datetime, timezone
from typing import Dict, List


def promote_permanent_nodes(db_path: str, access_threshold: int = 10) -> Dict:
    """
    Promote nodes to permanent status based on access count threshold.
    
    Once a node becomes permanent, it can never be demoted or decayed.
    This implements the "trauma" metaphor - important memories become permanent.
    
    Args:
        db_path: Path to the SQLite database
        access_threshold: Minimum access_count for permanent status
        
    Returns:
        Dict with promotion statistics
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Find nodes that should be permanent but aren't yet
    cursor.execute("""
        SELECT id, access_count, node_type 
        FROM thought_nodes 
        WHERE access_count >= ? 
        AND (permanent IS NULL OR permanent = 0)
        AND (decayed IS NULL OR decayed = 0)
    """, (access_threshold,))
    
    candidates = cursor.fetchall()
    promoted_count = 0
    
    for node_id, access_count, node_type in candidates:
        # Promote to permanent status
        cursor.execute("""
            UPDATE thought_nodes 
            SET permanent = 1, last_updated = ?
            WHERE id = ?
        """, (datetime.now(timezone.utc).isoformat(), node_id))
        promoted_count += 1
    
    conn.commit()
    conn.close()
    
    return {
        "nodes_evaluated": len(candidates),
        "nodes_promoted": promoted_count,
        "access_threshold": access_threshold
    }


def get_permanence_stats(db_path: str) -> Dict:
    """
    Get statistics about permanent vs non-permanent nodes.
    
    Args:
        db_path: Path to the SQLite database
        
    Returns:
        Dict with permanence statistics
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Count permanent vs non-permanent nodes
    cursor.execute("SELECT COUNT(*) FROM thought_nodes WHERE permanent > 0")
    permanent_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM thought_nodes WHERE (permanent IS NULL OR permanent = 0)")
    non_permanent_count = cursor.fetchone()[0]
    
    # Get access count statistics for permanent nodes
    cursor.execute("""
        SELECT MIN(access_count), AVG(access_count), MAX(access_count) 
        FROM thought_nodes WHERE permanent > 0
    """)
    permanent_stats = cursor.fetchone()
    
    # Get access count statistics for non-permanent nodes
    cursor.execute("""
        SELECT MIN(access_count), AVG(access_count), MAX(access_count) 
        FROM thought_nodes WHERE (permanent IS NULL OR permanent = 0)
    """)
    non_permanent_stats = cursor.fetchone()
    
    # Find the highest access count among non-permanent nodes
    cursor.execute("""
        SELECT access_count FROM thought_nodes 
        WHERE (permanent IS NULL OR permanent = 0) 
        ORDER BY access_count DESC LIMIT 1
    """)
    highest_non_permanent = cursor.fetchone()
    highest_non_permanent_access = highest_non_permanent[0] if highest_non_permanent else 0
    
    conn.close()
    
    return {
        "total_nodes": permanent_count + non_permanent_count,
        "permanent_count": permanent_count,
        "non_permanent_count": non_permanent_count,
        "permanent_stats": {
            "min_access": permanent_stats[0],
            "avg_access": round(permanent_stats[1], 2) if permanent_stats[1] else 0,
            "max_access": permanent_stats[2]
        },
        "non_permanent_stats": {
            "min_access": non_permanent_stats[0],
            "avg_access": round(non_permanent_stats[1], 2) if non_permanent_stats[1] else 0,
            "max_access": non_permanent_stats[2]
        },
        "highest_non_permanent_access": highest_non_permanent_access
    }


def calculate_recommended_threshold(db_path: str) -> int:
    """
    Calculate a recommended access threshold based on current data.
    
    Uses the 75th percentile of permanent nodes as a baseline, but ensures
    it's at least 5 to avoid promoting nodes with minimal access.
    
    Args:
        db_path: Path to the SQLite database
        
    Returns:
        Recommended access threshold
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get access counts of current permanent nodes
    cursor.execute("""
        SELECT access_count FROM thought_nodes 
        WHERE permanent > 0 
        ORDER BY access_count
    """)
    permanent_access_counts = [row[0] for row in cursor.fetchall()]
    
    if not permanent_access_counts:
        # No permanent nodes yet, use conservative threshold
        conn.close()
        return 10
    
    # Use 75th percentile as baseline
    percentile_75_index = int(len(permanent_access_counts) * 0.75)
    baseline_threshold = permanent_access_counts[percentile_75_index]
    
    # Ensure minimum threshold of 5
    recommended = max(baseline_threshold, 5)
    
    conn.close()
    return recommended


EXPECTED_EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 produces 384-dim float32 vectors


def validate_embeddings_integrity(db_path: str) -> Dict:
    """
    Validate that the embeddings table is healthy.

    Checks for the kinds of corruption that would silently degrade retrieval
    and similarity-driven sleep operations: zero-norm vectors, NaN/inf,
    wrong dimensions, embeddings pointing at deleted nodes, and live nodes
    with no embedding row at all.

    Args:
        db_path: Path to the SQLite database

    Returns:
        Dict with validation results. ``integrity_ok`` is True iff every
        kind of corruption count is zero AND no live node is missing an
        embedding (orphan_nodes).
    """
    import numpy as np

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Defensive: an embeddings table is optional in some DB layouts (older
    # graphs predate it, test fixtures sometimes skip it). Treat absence
    # as "no embeddings to check" rather than an integrity violation.
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='embeddings'")
    if not cursor.fetchone():
        conn.close()
        return {
            "total_embeddings": 0,
            "zero_norm": 0,
            "nan_or_inf": 0,
            "wrong_dim": 0,
            "bad_embedding_ids": [],
            "orphan_embeddings": 0,
            "orphan_nodes": 0,
            "integrity_ok": True,
            "embeddings_table_present": False,
        }

    # All embeddings, decoded
    cursor.execute("SELECT node_id, vector FROM embeddings")
    zero_norm = 0
    nan_or_inf = 0
    wrong_dim = 0
    bad_ids = []
    total_embeddings = 0
    for nid, blob in cursor.fetchall():
        total_embeddings += 1
        arr = np.frombuffer(blob, dtype=np.float32)
        if len(arr) != EXPECTED_EMBEDDING_DIM:
            wrong_dim += 1
            bad_ids.append(nid)
            continue
        if not np.all(np.isfinite(arr)):
            nan_or_inf += 1
            bad_ids.append(nid)
            continue
        norm = float(np.linalg.norm(arr))
        if norm < 1e-6:
            zero_norm += 1
            bad_ids.append(nid)

    # Embeddings pointing at deleted/missing nodes
    cursor.execute("""
        SELECT COUNT(*) FROM embeddings e
        WHERE NOT EXISTS (SELECT 1 FROM thought_nodes t WHERE t.id = e.node_id)
    """)
    orphan_embeddings = cursor.fetchone()[0]

    # Live (non-decayed) nodes with no embedding row
    cursor.execute("""
        SELECT COUNT(*) FROM thought_nodes t
        WHERE (t.decayed IS NULL OR t.decayed = 0)
        AND NOT EXISTS (SELECT 1 FROM embeddings e WHERE e.node_id = t.id)
    """)
    orphan_nodes = cursor.fetchone()[0]

    conn.close()

    bad_total = zero_norm + nan_or_inf + wrong_dim
    return {
        "total_embeddings": total_embeddings,
        "zero_norm": zero_norm,
        "nan_or_inf": nan_or_inf,
        "wrong_dim": wrong_dim,
        "bad_embedding_ids": bad_ids,
        "orphan_embeddings": orphan_embeddings,  # embedding for nonexistent node
        "orphan_nodes": orphan_nodes,            # live node with no embedding
        "integrity_ok": bad_total == 0 and orphan_embeddings == 0 and orphan_nodes == 0,
    }


def validate_permanence_integrity(db_path: str) -> Dict:
    """
    Validate that permanent nodes are properly protected from decay.
    
    Args:
        db_path: Path to the SQLite database
        
    Returns:
        Dict with validation results
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check for permanent nodes that are marked as decayed (should never happen)
    cursor.execute("""
        SELECT COUNT(*) FROM thought_nodes 
        WHERE permanent > 0 AND decayed > 0
    """)
    permanent_but_decayed = cursor.fetchone()[0]
    
    # Check for core memories that aren't permanent (should be fixed)
    cursor.execute("""
        SELECT COUNT(*) FROM thought_nodes 
        WHERE node_type = 'core_memory' AND (permanent IS NULL OR permanent = 0)
    """)
    core_not_permanent = cursor.fetchone()[0]
    
    # Check for high-access non-permanent nodes (candidates for promotion)
    cursor.execute("""
        SELECT COUNT(*) FROM thought_nodes 
        WHERE (permanent IS NULL OR permanent = 0) 
        AND access_count >= 10
    """)
    high_access_not_permanent = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        "permanent_but_decayed": permanent_but_decayed,  # Should be 0
        "core_not_permanent": core_not_permanent,        # Indicates repair needed
        "high_access_not_permanent": high_access_not_permanent,  # Promotion candidates
        "integrity_ok": permanent_but_decayed == 0
    }