#!/usr/bin/env python3
"""
Graph utility functions extracted from clustering.py after hotspot removal.
Used by sleep.py for cross-linking and deduplication.
"""

import sqlite3
import logging
from typing import List, Dict, Tuple

import numpy as np

logger = logging.getLogger("cashew.graph_utils")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors"""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def load_embeddings(db_path: str) -> Tuple[List[str], np.ndarray, Dict[str, Dict]]:
    """
    Load all embeddings and node metadata from the database.
    
    Returns:
        node_ids: List of node IDs in order
        vectors: numpy array of shape (n_nodes, embedding_dim)
        node_meta: Dict of node_id -> {content, node_type, domain}
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT e.node_id, e.vector, tn.content, tn.node_type, 
               COALESCE(tn.domain, 'unknown') as domain
        FROM embeddings e
        JOIN thought_nodes tn ON e.node_id = tn.id
        WHERE (tn.decayed IS NULL OR tn.decayed = 0)
    """)
    
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return [], np.array([]), {}
    
    node_ids = []
    vectors = []
    node_meta = {}
    
    for node_id, vector_blob, content, node_type, domain in rows:
        try:
            vector = np.frombuffer(vector_blob, dtype=np.float32)
            
            if np.any(np.isnan(vector)) or np.any(np.isinf(vector)):
                logger.warning(f"Skipping node {node_id} due to NaN/inf values")
                continue
                
            if np.allclose(vector, 0):
                logger.warning(f"Skipping node {node_id} due to zero vector")
                continue
                
            node_ids.append(node_id)
            vectors.append(vector)
            node_meta[node_id] = {
                "content": content,
                "node_type": node_type,
                "domain": domain
            }
        except Exception as e:
            logger.warning(f"Failed to load embedding for {node_id}: {e}")
    
    vectors_array = np.array(vectors)
    
    if vectors_array.size > 0:
        if np.any(np.isnan(vectors_array)) or np.any(np.isinf(vectors_array)):
            logger.error("NaN or inf values detected in embedding matrix after filtering")
            valid_mask = ~(np.isnan(vectors_array).any(axis=1) | np.isinf(vectors_array).any(axis=1))
            vectors_array = vectors_array[valid_mask]
            node_ids = [node_ids[i] for i in range(len(node_ids)) if valid_mask[i]]
            node_meta = {nid: node_meta[nid] for nid in node_ids}
    
    return node_ids, vectors_array, node_meta
