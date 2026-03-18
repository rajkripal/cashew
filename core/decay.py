#!/usr/bin/env python3
"""
Decay functionality for cashew thought graph.
Handles automatic aging/pruning of unused low-quality nodes.
"""

import sqlite3
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Set


def cascade_decay(db_path: str, decayed_node_id: str, decay_factor: float = 0.7, min_confidence: float = 0.3) -> Dict:
    """
    DFS down from a decayed node, reducing confidence of children.
    
    Args:
        db_path: Path to the SQLite database
        decayed_node_id: ID of the node that was just decayed
        decay_factor: Factor to reduce confidence by at each level
        min_confidence: Minimum confidence threshold below which nodes get decayed
        
    Returns:
        Dict with cascading statistics
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cascaded_count = 0
    nodes_to_process = [(decayed_node_id, 1)]  # (node_id, depth)
    processed_nodes = set()
    
    while nodes_to_process:
        current_node_id, depth = nodes_to_process.pop(0)
        
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
            # Check if this child has other live (non-decayed) parents
            cursor.execute("""
                SELECT COUNT(*) FROM derivation_edges de
                JOIN thought_nodes tn ON de.parent_id = tn.id
                WHERE de.child_id = ? 
                AND (tn.decayed IS NULL OR tn.decayed = 0)
                AND tn.id != ?
            """, (child_id, current_node_id))
            
            live_parent_count = cursor.fetchone()[0]
            
            # Skip children that have other live parents - they're anchored
            if live_parent_count > 0:
                continue
            
            # Get current child node info (including permanence status)
            cursor.execute("""
                SELECT confidence, decayed, permanent FROM thought_nodes 
                WHERE id = ?
            """, (child_id,))
            
            result = cursor.fetchone()
            if not result:
                continue
                
            current_confidence, is_decayed, is_permanent = result
            
            # Skip if already decayed or permanent
            if is_decayed or is_permanent:
                continue
            
            # Calculate new confidence with decay factor applied by depth
            new_confidence = current_confidence * (decay_factor ** depth)
            
            # Update the child's confidence
            cursor.execute("""
                UPDATE thought_nodes 
                SET confidence = ?, last_updated = ?
                WHERE id = ?
            """, (new_confidence, datetime.now(timezone.utc).isoformat(), child_id))
            
            # If confidence drops below threshold, mark as decayed and continue DFS
            if new_confidence < min_confidence:
                cursor.execute("""
                    UPDATE thought_nodes 
                    SET decayed = 1, last_updated = ?
                    WHERE id = ?
                """, (datetime.now(timezone.utc).isoformat(), child_id))
                
                cascaded_count += 1
                nodes_to_process.append((child_id, depth + 1))
    
    conn.commit()
    conn.close()
    
    return {"cascaded": cascaded_count}


def auto_decay(db_path: str, min_age_days: int = 14, max_confidence_for_decay: float = 0.85, 
               enable_cascading: bool = True, decay_factor: float = 0.7) -> Dict:
    """Decay nodes that have never been accessed and are old enough, with optional cascading
    
    Args:
        db_path: Path to the SQLite database
        min_age_days: Minimum age in days for a node to be eligible for decay
        max_confidence_for_decay: Maximum confidence for a node to be eligible for decay
        enable_cascading: Whether to enable cascading decay to children
        decay_factor: Factor for cascading decay (confidence *= decay_factor^depth)
        
    Returns:
        Dict with pruning statistics
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cutoff = (datetime.now(timezone.utc) - timedelta(days=min_age_days)).isoformat()
    
    # First, find nodes that will be decayed
    # Skip permanent nodes and hotspots (structural summary nodes)
    cursor.execute("""
        SELECT id FROM thought_nodes
        WHERE (decayed IS NULL OR decayed = 0)
        AND (permanent IS NULL OR permanent = 0)
        AND node_type != 'hotspot'
        AND access_count = 0
        AND confidence < ?
        AND timestamp < ?
    """, (max_confidence_for_decay, cutoff))
    
    nodes_to_decay = [row[0] for row in cursor.fetchall()]
    
    # Mark them as decayed (skip permanent nodes and hotspots)
    cursor.execute("""
        UPDATE thought_nodes SET decayed = 1, last_updated = ?
        WHERE (decayed IS NULL OR decayed = 0)
        AND (permanent IS NULL OR permanent = 0)
        AND node_type != 'hotspot'
        AND access_count = 0
        AND confidence < ?
        AND timestamp < ?
    """, (datetime.now(timezone.utc).isoformat(), max_confidence_for_decay, cutoff))
    
    direct_count = cursor.rowcount
    conn.commit()
    conn.close()
    
    # Run cascading decay for each newly decayed node
    total_cascaded = 0
    if enable_cascading:
        for node_id in nodes_to_decay:
            cascade_result = cascade_decay(db_path, node_id, decay_factor)
            total_cascaded += cascade_result["cascaded"]
    
    return {
        "pruned": direct_count, 
        "cascaded": total_cascaded,
        "total": direct_count + total_cascaded
    }


def get_decay_candidates(db_path: str, min_age_days: int = 14, max_confidence_for_decay: float = 0.85, 
                        show_cascade_preview: bool = False, decay_factor: float = 0.7) -> Dict:
    """Get statistics on nodes eligible for decay without actually decaying them
    
    Args:
        db_path: Path to the SQLite database
        min_age_days: Minimum age in days for a node to be eligible for decay
        max_confidence_for_decay: Maximum confidence for a node to be eligible for decay
        show_cascade_preview: Whether to simulate cascading effects
        decay_factor: Factor for cascading decay simulation
        
    Returns:
        Dict with candidate statistics
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cutoff = (datetime.now(timezone.utc) - timedelta(days=min_age_days)).isoformat()
    
    # Get direct candidates (including hotspots now, but excluding permanent nodes)
    cursor.execute("""
        SELECT COUNT(*), AVG(confidence), MIN(confidence), MAX(confidence)
        FROM thought_nodes 
        WHERE (decayed IS NULL OR decayed = 0)
        AND (permanent IS NULL OR permanent = 0)
        AND access_count = 0
        AND confidence < ?
        AND timestamp < ?
    """, (max_confidence_for_decay, cutoff))
    
    result = cursor.fetchone()
    count, avg_conf, min_conf, max_conf = result
    
    # Count hotspots specifically (excluding permanent ones)
    cursor.execute("""
        SELECT COUNT(*)
        FROM thought_nodes 
        WHERE (decayed IS NULL OR decayed = 0)
        AND (permanent IS NULL OR permanent = 0)
        AND access_count = 0
        AND confidence < ?
        AND timestamp < ?
        AND node_type = 'hotspot'
    """, (max_confidence_for_decay, cutoff))
    
    hotspot_count = cursor.fetchone()[0]
    
    result_dict = {
        "candidates": count or 0,
        "hotspot_candidates": hotspot_count or 0,
        "avg_confidence": round(avg_conf, 3) if avg_conf else 0,
        "min_confidence": round(min_conf, 3) if min_conf else 0,
        "max_confidence": round(max_conf, 3) if max_conf else 0
    }
    
    # Simulate cascade effects if requested
    if show_cascade_preview and count and count > 0:
        # Get the actual candidate node IDs (excluding permanent nodes)
        cursor.execute("""
            SELECT id FROM thought_nodes 
            WHERE (decayed IS NULL OR decayed = 0)
            AND (permanent IS NULL OR permanent = 0)
            AND access_count = 0
            AND confidence < ?
            AND timestamp < ?
        """, (max_confidence_for_decay, cutoff))
        
        candidate_ids = [row[0] for row in cursor.fetchall()]
        
        # Simulate cascade for each candidate
        total_would_cascade = 0
        for node_id in candidate_ids:
            would_cascade = simulate_cascade_decay(db_path, node_id, decay_factor)
            total_would_cascade += would_cascade
        
        result_dict["cascade_preview"] = total_would_cascade
        result_dict["total_preview"] = (count or 0) + total_would_cascade
    
    conn.close()
    
    return result_dict


def simulate_cascade_decay(db_path: str, decayed_node_id: str, decay_factor: float = 0.7, min_confidence: float = 0.3) -> int:
    """
    Simulate cascading decay without actually modifying the database.
    
    Args:
        db_path: Path to the SQLite database
        decayed_node_id: ID of the node that would be decayed
        decay_factor: Factor to reduce confidence by at each level
        min_confidence: Minimum confidence threshold below which nodes would get decayed
        
    Returns:
        Number of nodes that would cascade
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    would_cascade = 0
    nodes_to_process = [(decayed_node_id, 1)]  # (node_id, depth)
    processed_nodes = set()
    
    while nodes_to_process:
        current_node_id, depth = nodes_to_process.pop(0)
        
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
            # Check if this child has other live (non-decayed) parents
            cursor.execute("""
                SELECT COUNT(*) FROM derivation_edges de
                JOIN thought_nodes tn ON de.parent_id = tn.id
                WHERE de.child_id = ? 
                AND (tn.decayed IS NULL OR tn.decayed = 0)
                AND tn.id != ?
            """, (child_id, current_node_id))
            
            live_parent_count = cursor.fetchone()[0]
            
            # Skip children that have other live parents - they're anchored
            if live_parent_count > 0:
                continue
            
            # Get current child node info (including permanence status)
            cursor.execute("""
                SELECT confidence, decayed, permanent FROM thought_nodes 
                WHERE id = ?
            """, (child_id,))
            
            result = cursor.fetchone()
            if not result:
                continue
                
            current_confidence, is_decayed, is_permanent = result
            
            # Skip if already decayed or permanent
            if is_decayed or is_permanent:
                continue
            
            # Calculate what new confidence would be
            new_confidence = current_confidence * (decay_factor ** depth)
            
            # If confidence would drop below threshold, count it and continue simulation
            if new_confidence < min_confidence:
                would_cascade += 1
                nodes_to_process.append((child_id, depth + 1))
    
    conn.close()
    
    return would_cascade