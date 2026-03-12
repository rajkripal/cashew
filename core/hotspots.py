#!/usr/bin/env python3
"""
Cashew Hotspot System
Hierarchical summary nodes that act as authoritative indexes over clusters.

A hotspot is a node that:
- Summarizes a cluster of related nodes (the "current state")
- Points to files for full content
- Gets boosted in retrieval so it surfaces before detail nodes
- Is the SINGLE source of truth for "what is X right now?"

Detail nodes underneath remain for historical context but don't compete
with the hotspot for status queries.
"""

import sqlite3
import json
import hashlib
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass
import logging

from .embeddings import embed_text, embed_nodes

# Database path is now configurable via environment variable or CLI
from .config import get_db_path

HOTSPOT_TYPE = "hotspot"
HOTSPOT_BOOST = 3.0  # Multiplier applied to hotspot scores in retrieval

@dataclass
class Hotspot:
    """A hotspot node with its metadata"""
    id: str
    content: str  # Summary text
    status: str  # Current status (e.g., "draft_v1_complete")
    file_pointers: Dict[str, str]  # label -> path
    cluster_node_ids: List[str]  # IDs of nodes this hotspot summarizes
    domain: str
    last_updated: str
    confidence: float = 0.95  # Hotspots are high-confidence by design


def _get_connection(db_path: str = None) -> sqlite3.Connection:
    if db_path is None:
        db_path = get_db_path()
    return sqlite3.connect(db_path)


def create_hotspot(
    db_path: str,
    content: str,
    status: str,
    file_pointers: Dict[str, str],
    cluster_node_ids: List[str],
    domain: str = "bunny",
    tags: Optional[List[str]] = None
) -> str:
    """
    Create a new hotspot node and link it to its cluster.
    
    Args:
        db_path: Path to graph database
        content: Summary text for the hotspot
        status: Current status string
        file_pointers: Dict of label -> file path
        cluster_node_ids: Node IDs this hotspot summarizes
        domain: Domain (raj/bunny)
        tags: Optional search tags for better retrieval
        
    Returns:
        The hotspot node ID
    """
    conn = _get_connection(db_path)
    cursor = conn.cursor()
    
    now = datetime.now(timezone.utc).isoformat()
    
    # Generate deterministic ID from content
    node_id = hashlib.sha256(f"hotspot:{content[:100]}:{now}".encode()).hexdigest()[:12]
    
    # Build metadata
    metadata = {
        "status": status,
        "file_pointers": file_pointers,
        "cluster_size": len(cluster_node_ids),
        "is_hotspot": True,
    }
    if tags:
        metadata["tags"] = tags
    
    # Insert the hotspot node
    cursor.execute("""
        INSERT INTO thought_nodes (id, content, node_type, timestamp, confidence, 
                                    metadata, source_file, domain, last_updated)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        node_id,
        content,
        HOTSPOT_TYPE,
        now,
        0.95,  # High confidence
        json.dumps(metadata),
        "hotspot_system",
        domain,
        now
    ))
    
    # Create edges: hotspot -> each cluster node (relation: "summarizes")
    for cluster_id in cluster_node_ids:
        # Verify the cluster node exists
        cursor.execute("SELECT id FROM thought_nodes WHERE id = ?", (cluster_id,))
        if cursor.fetchone():
            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO derivation_edges (parent_id, child_id, relation, weight, reasoning)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    node_id,
                    cluster_id,
                    "summarizes",
                    0.9,
                    "Hotspot summary node for cluster"
                ))
            except sqlite3.IntegrityError:
                pass  # Edge already exists
    
    conn.commit()
    
    # NOTE: We no longer call embed_nodes() here to avoid O(N) re-embedding
    # storms when creating multiple hotspots. Callers should batch-embed
    # after all hotspot operations are complete.
    
    conn.close()
    logging.info(f"Created hotspot {node_id} summarizing {len(cluster_node_ids)} nodes")
    
    return node_id


def update_hotspot(
    db_path: str,
    hotspot_id: str,
    content: Optional[str] = None,
    status: Optional[str] = None,
    file_pointers: Optional[Dict[str, str]] = None,
    add_cluster_ids: Optional[List[str]] = None,
) -> bool:
    """
    Update an existing hotspot's content, status, or file pointers.
    
    Args:
        db_path: Path to graph database
        hotspot_id: ID of the hotspot to update
        content: New summary text (if changing)
        status: New status string (if changing)
        file_pointers: New file pointers dict (merged with existing)
        add_cluster_ids: Additional node IDs to link
        
    Returns:
        True if updated successfully
    """
    conn = _get_connection(db_path)
    cursor = conn.cursor()
    
    # Get current state
    cursor.execute("""
        SELECT content, metadata FROM thought_nodes 
        WHERE id = ? AND node_type = ?
    """, (hotspot_id, HOTSPOT_TYPE))
    
    row = cursor.fetchone()
    if not row:
        logging.error(f"Hotspot {hotspot_id} not found")
        conn.close()
        return False
    
    current_content, current_metadata_str = row
    current_metadata = json.loads(current_metadata_str) if current_metadata_str else {}
    
    now = datetime.now(timezone.utc).isoformat()
    
    # Merge updates
    new_content = content if content is not None else current_content
    
    if status is not None:
        current_metadata["status"] = status
    
    if file_pointers is not None:
        existing_fps = current_metadata.get("file_pointers", {})
        existing_fps.update(file_pointers)
        current_metadata["file_pointers"] = existing_fps
    
    # Update the node
    cursor.execute("""
        UPDATE thought_nodes 
        SET content = ?, metadata = ?, last_updated = ?
        WHERE id = ?
    """, (new_content, json.dumps(current_metadata), now, hotspot_id))
    
    # Add new cluster edges
    if add_cluster_ids:
        for cluster_id in add_cluster_ids:
            cursor.execute("SELECT id FROM thought_nodes WHERE id = ?", (cluster_id,))
            if cursor.fetchone():
                try:
                    cursor.execute("""
                        INSERT OR IGNORE INTO derivation_edges (parent_id, child_id, relation, weight, reasoning)
                        VALUES (?, ?, ?, ?, ?)
                    """, (hotspot_id, cluster_id, "summarizes", 0.9, "Hotspot summary node for cluster"))
                except sqlite3.IntegrityError:
                    pass
    
    conn.commit()
    
    # If content changed, delete old embedding so next embed_nodes() picks it up
    if content is not None:
        try:
            conn2 = _get_connection(db_path)
            conn2.execute("DELETE FROM embeddings WHERE node_id = ?", (hotspot_id,))
            conn2.commit()
            conn2.close()
        except Exception as e:
            logging.warning(f"Failed to clear embedding for hotspot {hotspot_id}: {e}")
    
    conn.close()
    logging.info(f"Updated hotspot {hotspot_id}")
    return True


def list_hotspots(db_path: str, domain: Optional[str] = None) -> List[Dict]:
    """List all hotspot nodes with their metadata"""
    conn = _get_connection(db_path)
    cursor = conn.cursor()
    
    if domain:
        cursor.execute("""
            SELECT id, content, metadata, domain, last_updated, confidence
            FROM thought_nodes 
            WHERE node_type = ? AND domain = ?
            AND (decayed IS NULL OR decayed = 0)
            ORDER BY last_updated DESC
        """, (HOTSPOT_TYPE, domain))
    else:
        cursor.execute("""
            SELECT id, content, metadata, domain, last_updated, confidence
            FROM thought_nodes 
            WHERE node_type = ?
            AND (decayed IS NULL OR decayed = 0)
            ORDER BY last_updated DESC
        """, (HOTSPOT_TYPE,))
    
    hotspots = []
    for row in cursor.fetchall():
        node_id, content, metadata_str, domain, last_updated, confidence = row
        metadata = json.loads(metadata_str) if metadata_str else {}
        
        # Count cluster members
        cursor.execute("""
            SELECT COUNT(*) FROM derivation_edges 
            WHERE parent_id = ? AND relation = 'summarizes'
        """, (node_id,))
        cluster_size = cursor.fetchone()[0]
        
        hotspots.append({
            "id": node_id,
            "content": content,
            "status": metadata.get("status", "unknown"),
            "file_pointers": metadata.get("file_pointers", {}),
            "cluster_size": cluster_size,
            "domain": domain,
            "last_updated": last_updated,
            "tags": metadata.get("tags", [])
        })
    
    conn.close()
    return hotspots


def get_hotspot(db_path: str, hotspot_id: str) -> Optional[Dict]:
    """Get a single hotspot with full details"""
    conn = _get_connection(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, content, metadata, domain, last_updated
        FROM thought_nodes WHERE id = ? AND node_type = ?
    """, (hotspot_id, HOTSPOT_TYPE))
    
    row = cursor.fetchone()
    if not row:
        conn.close()
        return None
    
    node_id, content, metadata_str, domain, last_updated = row
    metadata = json.loads(metadata_str) if metadata_str else {}
    
    # Get cluster members
    cursor.execute("""
        SELECT de.child_id, tn.content, tn.node_type
        FROM derivation_edges de
        JOIN thought_nodes tn ON de.child_id = tn.id
        WHERE de.parent_id = ? AND de.relation = 'summarizes'
    """, (node_id,))
    
    cluster = [{"id": r[0], "content": r[1], "type": r[2]} for r in cursor.fetchall()]
    
    conn.close()
    
    return {
        "id": node_id,
        "content": content,
        "status": metadata.get("status"),
        "file_pointers": metadata.get("file_pointers", {}),
        "tags": metadata.get("tags", []),
        "domain": domain,
        "last_updated": last_updated,
        "cluster": cluster
    }


def is_hotspot(node_type: str) -> bool:
    """Check if a node type is a hotspot"""
    return node_type == HOTSPOT_TYPE
