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

# Database path is now configurable via environment variable or CLI
from .config import get_db_path

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
    is_parent: bool = False  # True if this cluster has subclusters
    children: List['ClusterInfo'] = None  # List of sub-clusters
    parent_cluster: Optional['ClusterInfo'] = None  # Parent cluster if this is a sub-cluster
    
    def __post_init__(self):
        if self.children is None:
            self.children = []


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
            
            # Check for NaN or infinite values
            if np.any(np.isnan(vector)) or np.any(np.isinf(vector)):
                logger.warning(f"Skipping node {node_id} due to NaN/inf values in embedding")
                continue
                
            # Check for zero vector (might indicate embedding failure)
            if np.allclose(vector, 0):
                logger.warning(f"Skipping node {node_id} due to zero embedding vector")
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
    
    # Final check for the whole matrix
    if vectors_array.size > 0:
        if np.any(np.isnan(vectors_array)) or np.any(np.isinf(vectors_array)):
            logger.error("NaN or inf values detected in embedding matrix after filtering")
            # Remove problematic rows
            valid_mask = ~(np.isnan(vectors_array).any(axis=1) | np.isinf(vectors_array).any(axis=1))
            vectors_array = vectors_array[valid_mask]
            node_ids = [node_ids[i] for i in range(len(node_ids)) if valid_mask[i]]
            node_meta = {nid: node_meta[nid] for nid in node_ids}
    
    return node_ids, vectors_array, node_meta


def detect_clusters_recursive(
    db_path: str,
    eps: float = DBSCAN_EPS,
    min_samples: int = DBSCAN_MIN_SAMPLES,
    max_cluster_size: int = 15,
    parent_hotspot_id: Optional[str] = None,
    recursion_depth: int = 0,
    max_recursion_depth: int = 5
) -> List[ClusterInfo]:
    """
    Run recursive DBSCAN on embedding vectors to build hierarchical clusters.
    
    Uses cosine distance (1 - cosine_similarity) as the metric.
    If a cluster is larger than max_cluster_size, recursively splits it.
    
    Returns:
        List of ClusterInfo objects for final clusters with 3+ members
    """
    node_ids, vectors, node_meta = load_embeddings(db_path)
    
    if len(node_ids) < min_samples:
        logger.info("Not enough nodes for clustering")
        return []
    
    logger.info(f"Clustering {len(node_ids)} nodes with DBSCAN "
               f"(eps={eps}, min_samples={min_samples}, depth={recursion_depth})")
    
    # Compute cosine distance matrix: distance = 1 - similarity
    sim_matrix = sklearn_cosine_sim(vectors)
    
    # Check for issues with similarity matrix
    if np.any(np.isnan(sim_matrix)) or np.any(np.isinf(sim_matrix)):
        logger.error("NaN or inf values detected in similarity matrix")
        # Fallback: use manual cosine similarity computation
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized_vectors = vectors / norms
        sim_matrix = normalized_vectors @ normalized_vectors.T
        sim_matrix = np.clip(sim_matrix, -1.0, 1.0)  # Ensure valid range
    
    distance_matrix = 1.0 - sim_matrix
    # Clip to avoid tiny negative values from floating point
    distance_matrix = np.clip(distance_matrix, 0.0, 2.0)
    
    # Final check for distance matrix
    if np.any(np.isnan(distance_matrix)) or np.any(np.isinf(distance_matrix)):
        logger.error("NaN or inf values in distance matrix, cannot proceed with clustering")
        return []
    
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
    
    # Build cluster info and handle recursive splitting
    all_clusters = []
    
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
        
        # Check if this cluster is too large and needs recursive splitting
        if (len(member_ids) > max_cluster_size and 
            recursion_depth < max_recursion_depth):
            
            logger.info(f"Cluster {cluster_id} has {len(member_ids)} nodes > {max_cluster_size}, "
                       f"recursively splitting at depth {recursion_depth}")
            
            # Create temporary database subset for just this cluster
            temp_node_ids = member_ids
            temp_vectors = member_vectors
            temp_node_meta = {nid: node_meta[nid] for nid in temp_node_ids}
            
            # Recursively cluster this subset with tighter eps
            tighter_eps = eps * 0.7
            subclusters = _detect_subclusters(
                temp_node_ids, temp_vectors, temp_node_meta,
                eps=tighter_eps,
                min_samples=min_samples,
                max_cluster_size=max_cluster_size,
                recursion_depth=recursion_depth + 1,
                max_recursion_depth=max_recursion_depth
            )
            
            if len(subclusters) > 1:
                # Successfully split - create parent hotspot and add children
                parent_cluster = ClusterInfo(
                    cluster_id=cluster_id,
                    node_ids=member_ids,
                    centroid=centroid,
                    representative_content=representative_content,
                    existing_hotspot_id=None  # Will create new parent hotspot
                )
                parent_cluster.is_parent = True
                parent_cluster.children = subclusters
                
                # Mark each subcluster as having this parent
                for subcluster in subclusters:
                    subcluster.parent_cluster = parent_cluster
                    
                all_clusters.append(parent_cluster)
                all_clusters.extend(subclusters)
            else:
                # Couldn't split meaningfully, treat as regular cluster
                regular_cluster = _create_regular_cluster(
                    cluster_id, member_ids, centroid, representative_content,
                    existing_hotspot_clusters
                )
                all_clusters.append(regular_cluster)
        else:
            # Regular cluster (small enough or max recursion reached)
            regular_cluster = _create_regular_cluster(
                cluster_id, member_ids, centroid, representative_content,
                existing_hotspot_clusters
            )
            all_clusters.append(regular_cluster)
    
    return all_clusters


def _detect_subclusters(node_ids, vectors, node_meta, eps, min_samples, 
                       max_cluster_size, recursion_depth, max_recursion_depth):
    """Helper to recursively detect subclusters on a subset of nodes"""
    if len(node_ids) < min_samples:
        return []
    
    # Compute distance matrix for subset with proper NaN handling
    try:
        sim_matrix = sklearn_cosine_sim(vectors)
        
        # Check for issues with similarity matrix
        if np.any(np.isnan(sim_matrix)) or np.any(np.isinf(sim_matrix)):
            logger.warning("NaN or inf values detected in subcluster similarity matrix, using manual computation")
            # Fallback: use manual cosine similarity computation
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            normalized_vectors = vectors / norms
            sim_matrix = normalized_vectors @ normalized_vectors.T
            sim_matrix = np.clip(sim_matrix, -1.0, 1.0)  # Ensure valid range
        
        distance_matrix = 1.0 - sim_matrix
        distance_matrix = np.clip(distance_matrix, 0.0, 2.0)
        
        # Final check for distance matrix
        if np.any(np.isnan(distance_matrix)) or np.any(np.isinf(distance_matrix)):
            logger.warning("NaN or inf values in subcluster distance matrix, skipping subclustering")
            return []
            
    except Exception as e:
        logger.warning(f"Failed to compute distance matrix for subclustering: {e}")
        return []
    
    # Run DBSCAN on subset
    try:
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed").fit(distance_matrix)
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    except Exception as e:
        logger.warning(f"DBSCAN failed on subcluster: {e}")
        return []
    
    subclusters = []
    for sub_cluster_id in range(n_clusters):
        sub_member_indices = np.where(labels == sub_cluster_id)[0]
        sub_member_ids = [node_ids[i] for i in sub_member_indices]
        sub_member_vectors = vectors[sub_member_indices]
        
        if len(sub_member_ids) < MIN_CLUSTER_SIZE:
            continue
            
        sub_centroid = sub_member_vectors.mean(axis=0)
        
        sub_similarities_to_centroid = [
            cosine_similarity(sub_member_vectors[i], sub_centroid)
            for i in range(len(sub_member_ids))
        ]
        sub_top_indices = np.argsort(sub_similarities_to_centroid)[-3:][::-1]
        sub_representative_content = [
            node_meta[sub_member_ids[i]]["content"][:100]
            for i in sub_top_indices
        ]
        
        # Check if this subcluster needs further splitting
        if (len(sub_member_ids) > max_cluster_size and 
            recursion_depth < max_recursion_depth):
            
            # Recursively split again
            further_subclusters = _detect_subclusters(
                sub_member_ids, sub_member_vectors, node_meta,
                eps=eps * 0.7, min_samples=min_samples,
                max_cluster_size=max_cluster_size,
                recursion_depth=recursion_depth + 1,
                max_recursion_depth=max_recursion_depth
            )
            
            if len(further_subclusters) > 1:
                subclusters.extend(further_subclusters)
            else:
                # Couldn't split further
                subcluster = ClusterInfo(
                    cluster_id=f"sub_{sub_cluster_id}_d{recursion_depth}",
                    node_ids=sub_member_ids,
                    centroid=sub_centroid,
                    representative_content=sub_representative_content,
                    existing_hotspot_id=None
                )
                subclusters.append(subcluster)
        else:
            subcluster = ClusterInfo(
                cluster_id=f"sub_{sub_cluster_id}_d{recursion_depth}",
                node_ids=sub_member_ids,
                centroid=sub_centroid,
                representative_content=sub_representative_content,
                existing_hotspot_id=None
            )
            subclusters.append(subcluster)
    
    return subclusters


def _create_regular_cluster(cluster_id, member_ids, centroid, representative_content, existing_hotspot_clusters):
    """Helper to create a regular ClusterInfo with existing hotspot check"""
    # Check if an existing hotspot covers this cluster
    existing_hotspot = None
    for hotspot_id, hotspot_members in existing_hotspot_clusters.items():
        overlap = len(set(member_ids) & hotspot_members)
        if overlap >= len(member_ids) * 0.5:  # >50% overlap
            existing_hotspot = hotspot_id
            break
    
    return ClusterInfo(
        cluster_id=cluster_id,
        node_ids=member_ids,
        centroid=centroid,
        representative_content=representative_content,
        existing_hotspot_id=existing_hotspot
    )


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
    # For backward compatibility, call the recursive version with max size = infinity
    return detect_clusters_recursive(
        db_path=db_path,
        eps=eps,
        min_samples=min_samples,
        max_cluster_size=999,  # No splitting
        max_recursion_depth=0  # No recursion
    )


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
    dry_run: bool = False,
    max_cluster_size: int = 15
) -> Dict:
    """
    Full clustering cycle for sleep:
    1. Detect clusters recursively
    2. Check existing hotspot staleness  
    3. Create new hotspots for uncovered clusters
    4. Create hierarchical hotspot relationships
    5. Update stale hotspots
    
    Args:
        db_path: Path to graph database
        model_fn: Optional LLM function for summary generation
        dry_run: If True, don't modify database
        max_cluster_size: Maximum cluster size before recursive splitting
        
    Returns:
        Summary dict of actions taken
    """
    results = {
        "clusters_found": 0,
        "new_hotspots_created": 0,
        "stale_hotspots_found": 0,
        "hotspots_updated": 0,
        "parent_hotspots_created": 0,
        "hierarchical_edges_created": 0,
        "cluster_details": []
    }
    
    # Step 1: Detect clusters recursively
    clusters = detect_clusters_recursive(db_path, max_cluster_size=max_cluster_size)
    results["clusters_found"] = len(clusters)
    
    # Keep track of hotspots created for hierarchy building
    cluster_to_hotspot = {}
    
    for cluster in clusters:
        detail = {
            "cluster_id": cluster.cluster_id,
            "size": len(cluster.node_ids),
            "representative": cluster.representative_content[:2],
            "has_hotspot": cluster.existing_hotspot_id is not None,
            "is_parent": getattr(cluster, 'is_parent', False),
            "num_children": len(getattr(cluster, 'children', [])),
            "action": "none"
        }
        
        if cluster.existing_hotspot_id is None:
            # No matching hotspot — DO NOT create one here.
            # Hotspot creation is only allowed in placement_aware_extraction.py
            # which has MIN_CLUSTER_SIZE gates and novelty checks.
            # Creating hotspots in clustering causes runaway proliferation
            # because DBSCAN produces different clusters each run.
            detail["action"] = "skipped_no_hotspot_creation_in_clustering"
            results.setdefault("skipped_clusters", 0)
            results["skipped_clusters"] += 1
        else:
            cluster_to_hotspot[cluster.cluster_id] = cluster.existing_hotspot_id
        
        results["cluster_details"].append(detail)
    
    # Step 2: Create hierarchical relationships between hotspots
    if not dry_run:
        results["hierarchical_edges_created"] = _create_hierarchical_edges(
            db_path, clusters, cluster_to_hotspot
        )
    
    # Step 3: Check staleness
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


def _create_hierarchical_edges(db_path: str, clusters: List[ClusterInfo], 
                              cluster_to_hotspot: Dict) -> int:
    """Create hierarchical edges between parent and child hotspots"""
    import sqlite3
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    edges_created = 0
    
    for cluster in clusters:
        if not getattr(cluster, 'is_parent', False):
            continue
            
        parent_hotspot_id = cluster_to_hotspot.get(cluster.cluster_id)
        if not parent_hotspot_id:
            continue
            
        # Create edges from parent hotspot to child hotspots
        for child_cluster in getattr(cluster, 'children', []):
            child_hotspot_id = cluster_to_hotspot.get(child_cluster.cluster_id)
            if child_hotspot_id:
                try:
                    cursor.execute("""
                        INSERT OR IGNORE INTO derivation_edges 
                        (parent_id, child_id, relation, weight, reasoning)
                        VALUES (?, ?, 'summarizes', 0.9, 'Hierarchical clustering - parent to child hotspot')
                    """, (parent_hotspot_id, child_hotspot_id))
                    edges_created += 1
                except sqlite3.IntegrityError:
                    pass  # Edge already exists
    
    conn.commit()
    conn.close()
    return edges_created


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix.
    Exposed for sleep.py to use instead of word overlap.
    """
    return sklearn_cosine_sim(vectors)
