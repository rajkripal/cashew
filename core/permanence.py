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