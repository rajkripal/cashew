#!/usr/bin/env python3
"""
Cashew Clustering Module
Unsupervised clustering on embedding vectors using DBSCAN.
Used during sleep cycles to:
1. Detect natural clusters in the thought graph
2. Auto-generate/update hotspot nodes for each cluster
3. Detect stale hotspots that no longer match their cluster
"""

import sqlite3
import json
import hashlib
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from datetime import datetime, timezone
from dataclasses import dataclass
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_sim

from .hotspots import (
    create_hotspot, update_hotspot, list_hotspots,
    HOTSPOT_TYPE, get_hotspot
)

logger = logging.getLogger("cashew.clustering")

DB_PATH = "/Users/bunny/.openclaw/workspace/cashew/data/graph.db"

# Clustering parameters
DBSCAN_EPS = 0.35           # Max distance between samples (1 - cosine_sim). Lower = tighter clusters.
DBSCAN_MIN_SAMPLES = 3      # Min nodes to form a cluster
MIN_CLUSTER_SIZE = 5        # Don't create hotspots for tiny clusters
STALENESS_THRESHOLD = 0.65  # If hotspot-centroid similarity drops below this, regenerate
AUTO_HOTSPOT_CONFIDENCE = 0.90


@dataclass
class ClusterInfo:
    """Information about a detected cluster"""
    cluster_id: int
    node_ids: List[str]
    centroid: np.ndarray
    representative_content: List[str]  # Top-3 closest to centroid
    existing_hotspot_id: Optional[str]  # If a hotspot already covers this cluster


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
    
    # Get all non-decayed nodes with embeddings
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
            node_ids.append(node_id)
            vectors.append(vector)
            node_meta[node_id] = {
                "content": content,
                "node_type": node_type,
                "domain": domain
            }
        except Exception as e:
            logger.warning(f"Failed to load embedding for {node_id}: {e}")
    
    return node_ids, np.array(vectors), node_meta


def detect_clusters(
    db_path: str,
    eps: float = DBSCAN_EPS,
    min_samples: int = DBSCAN_MIN_SAMPLES
) -> List[ClusterInfo]:
    """
    Run DBSCAN on embedding vectors to find natural clusters.
    
    Uses cosine distance (1 - cosine_similarity) as the metric.
    
    Returns:
        List of ClusterInfo objects for clusters with 3+ members
    """
    node_ids, vectors, node_meta = load_embeddings(db_path)
    
    if len(node_ids) < min_samples:
        logger.info("Not enough nodes for clustering")
        return []
    
    logger.info(f"Clustering {len(node_ids)} nodes with DBSCAN (eps={eps}, min_samples={min_samples})")
    
    # Compute cosine distance matrix: distance = 1 - similarity
    sim_matrix = sklearn_cosine_sim(vectors)
    distance_matrix = 1.0 - sim_matrix
    # Clip to avoid tiny negative values from floating point
    distance_matrix = np.clip(distance_matrix, 0.0, 2.0)
    
    # Run DBSCAN
    clustering = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="precomputed"
    ).fit(distance_matrix)
    
    labels = clustering.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    logger.info(f"Found {n_clusters} clusters, {n_noise} noise points")
    
    # Get existing hotspot cluster memberships
    existing_hotspot_clusters = _get_existing_hotspot_clusters(db_path)
    
    # Build cluster info
    clusters = []
    for cluster_id in range(n_clusters):
        member_indices = np.where(labels == cluster_id)[0]
        member_ids = [node_ids[i] for i in member_indices]
        member_vectors = vectors[member_indices]
        
        if len(member_ids) < MIN_CLUSTER_SIZE:
            continue
        
        # Compute centroid
        centroid = member_vectors.mean(axis=0)
        
        # Find top-3 representative nodes (closest to centroid)
        similarities_to_centroid = [
            cosine_similarity(member_vectors[i], centroid)
            for i in range(len(member_ids))
        ]
        top_indices = np.argsort(similarities_to_centroid)[-3:][::-1]
        representative_content = [
            node_meta[member_ids[i]]["content"][:100]
            for i in top_indices
        ]
        
        # Check if an existing hotspot covers this cluster
        existing_hotspot = None
        for hotspot_id, hotspot_members in existing_hotspot_clusters.items():
            overlap = len(set(member_ids) & hotspot_members)
            if overlap >= len(member_ids) * 0.5:  # >50% overlap
                existing_hotspot = hotspot_id
                break
        
        clusters.append(ClusterInfo(
            cluster_id=cluster_id,
            node_ids=member_ids,
            centroid=centroid,
            representative_content=representative_content,
            existing_hotspot_id=existing_hotspot
        ))
    
    return clusters


def _get_existing_hotspot_clusters(db_path: str) -> Dict[str, Set[str]]:
    """Get existing hotspot -> cluster member mappings"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT de.parent_id, de.child_id
        FROM derivation_edges de
        JOIN thought_nodes tn ON de.parent_id = tn.id
        WHERE tn.node_type = ? AND de.relation = 'summarizes'
    """, (HOTSPOT_TYPE,))
    
    hotspot_clusters = {}
    for parent_id, child_id in cursor.fetchall():
        if parent_id not in hotspot_clusters:
            hotspot_clusters[parent_id] = set()
        hotspot_clusters[parent_id].add(child_id)
    
    conn.close()
    return hotspot_clusters


def check_hotspot_staleness(db_path: str) -> List[Dict]:
    """
    Check all hotspots for staleness.
    
    A hotspot is stale if:
    1. Its embedding is too dissimilar from its cluster's centroid
    2. New nodes have been added near the cluster but aren't in it
    
    Returns:
        List of {hotspot_id, staleness_score, reason, cluster_centroid_sim}
    """
    node_ids, vectors, node_meta = load_embeddings(db_path)
    
    if not node_ids:
        return []
    
    # Build lookup
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    
    # Get existing hotspot clusters
    hotspot_clusters = _get_existing_hotspot_clusters(db_path)
    
    stale_reports = []
    
    for hotspot_id, member_ids in hotspot_clusters.items():
        if hotspot_id not in id_to_idx:
            continue
        
        hotspot_vec = vectors[id_to_idx[hotspot_id]]
        
        # Get cluster member vectors (only those that still exist and have embeddings)
        member_indices = [id_to_idx[mid] for mid in member_ids if mid in id_to_idx]
        if not member_indices:
            continue
        
        member_vectors = vectors[member_indices]
        centroid = member_vectors.mean(axis=0)
        
        # Similarity between hotspot and its cluster centroid
        centroid_sim = cosine_similarity(hotspot_vec, centroid)
        
        is_stale = centroid_sim < STALENESS_THRESHOLD
        
        stale_reports.append({
            "hotspot_id": hotspot_id,
            "hotspot_content": node_meta.get(hotspot_id, {}).get("content", "")[:80],
            "cluster_size": len(member_indices),
            "centroid_similarity": float(centroid_sim),
            "is_stale": is_stale,
            "reason": f"Centroid similarity {centroid_sim:.3f} < {STALENESS_THRESHOLD}" if is_stale else "OK"
        })
    
    return stale_reports


def generate_hotspot_summary(representative_content: List[str], model_fn=None) -> str:
    """
    Generate a summary for a new hotspot.
    
    If model_fn is available, uses LLM. Otherwise, concatenates representative content.
    
    Args:
        representative_content: Top-3 representative node contents
        model_fn: Optional function(prompt) -> str for LLM summarization
        
    Returns:
        Summary text for the hotspot
    """
    if model_fn:
        prompt = f"""Summarize these related thoughts into a single concise status line (1-2 sentences max).
Focus on: what is this about, what's the current state, what's next.

Thoughts:
{chr(10).join(f'- {c}' for c in representative_content)}

Summary:"""
        try:
            return model_fn(prompt).strip()
        except Exception as e:
            logger.warning(f"LLM summary failed, falling back to concatenation: {e}")
    
    # Fallback: use the most representative node
    return representative_content[0] if representative_content else "Auto-generated cluster summary"


def run_clustering_cycle(
    db_path: str,
    model_fn=None,
    dry_run: bool = False
) -> Dict:
    """
    Full clustering cycle for sleep:
    1. Detect clusters
    2. Check existing hotspot staleness
    3. Create new hotspots for uncovered clusters
    4. Update stale hotspots
    
    Args:
        db_path: Path to graph database
        model_fn: Optional LLM function for summary generation
        dry_run: If True, don't modify database
        
    Returns:
        Summary dict of actions taken
    """
    results = {
        "clusters_found": 0,
        "new_hotspots_created": 0,
        "stale_hotspots_found": 0,
        "hotspots_updated": 0,
        "cluster_details": []
    }
    
    # Step 1: Detect clusters
    clusters = detect_clusters(db_path)
    results["clusters_found"] = len(clusters)
    
    for cluster in clusters:
        detail = {
            "cluster_id": cluster.cluster_id,
            "size": len(cluster.node_ids),
            "representative": cluster.representative_content[:2],
            "has_hotspot": cluster.existing_hotspot_id is not None,
            "action": "none"
        }
        
        if cluster.existing_hotspot_id is None:
            # New cluster without hotspot
            if not dry_run:
                # Determine domain from majority vote
                _, _, node_meta = load_embeddings(db_path)
                domains = [node_meta.get(nid, {}).get("domain", "unknown") for nid in cluster.node_ids]
                domain = max(set(domains), key=domains.count)
                
                summary = generate_hotspot_summary(cluster.representative_content, model_fn)
                
                hotspot_id = create_hotspot(
                    db_path=db_path,
                    content=summary,
                    status="auto_generated",
                    file_pointers={},
                    cluster_node_ids=cluster.node_ids,
                    domain=domain,
                    tags=["auto_cluster"]
                )
                detail["action"] = f"created_hotspot:{hotspot_id}"
                results["new_hotspots_created"] += 1
            else:
                detail["action"] = "would_create_hotspot"
        
        results["cluster_details"].append(detail)
    
    # Step 2: Check staleness
    stale_reports = check_hotspot_staleness(db_path)
    stale_hotspots = [r for r in stale_reports if r["is_stale"]]
    results["stale_hotspots_found"] = len(stale_hotspots)
    
    for report in stale_hotspots:
        if not dry_run and model_fn:
            # Get cluster members for regeneration
            hotspot = get_hotspot(db_path, report["hotspot_id"])
            if hotspot and hotspot.get("cluster"):
                representative = [n["content"][:100] for n in hotspot["cluster"][:5]]
                new_summary = generate_hotspot_summary(representative, model_fn)
                update_hotspot(
                    db_path=db_path,
                    hotspot_id=report["hotspot_id"],
                    content=new_summary,
                    status="auto_refreshed"
                )
                results["hotspots_updated"] += 1
    
    return results


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix.
    Exposed for sleep.py to use instead of word overlap.
    """
    return sklearn_cosine_sim(vectors)
