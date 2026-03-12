#!/usr/bin/env python3
"""
Cashew Complete Clustering Module
100% node coverage clustering that eliminates DBSCAN noise.

Two-phase approach:
1. DBSCAN finds natural dense clusters
2. Second pass assigns every noise point to nearest cluster centroid or creates micro-clusters

Every node belongs to exactly one cluster. Zero orphans.
"""

import os
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

from core.hotspots import (
    create_hotspot, update_hotspot, list_hotspots,
    HOTSPOT_TYPE, get_hotspot
)

logger = logging.getLogger("cashew.complete_clustering")

# Database path is now configurable via environment variable or CLI
from .config import get_db_path

# Clustering parameters (configurable via env vars for tuning)
DBSCAN_EPS = float(os.environ.get('CASHEW_CLUSTER_EPS', '0.35'))  # Max distance (1 - cosine_sim). Higher = looser clusters.
DBSCAN_MIN_SAMPLES = int(os.environ.get('CASHEW_CLUSTER_MIN_SAMPLES', '3'))  # Min nodes to form a cluster
MIN_CLUSTER_SIZE = 5        # Don't create hotspots for tiny clusters
NOISE_ASSIGNMENT_THRESHOLD = float(os.environ.get('CASHEW_NOISE_THRESHOLD', '0.3'))  # Max distance for orphan assignment
STALENESS_THRESHOLD = 0.65  # If hotspot-centroid similarity drops below this, regenerate
AUTO_HOTSPOT_CONFIDENCE = 0.90


@dataclass
class CompleteClusterInfo:
    """Information about a cluster with 100% coverage guarantee"""
    cluster_id: str
    node_ids: List[str]
    centroid: np.ndarray
    representative_content: List[str]  # Top-3 closest to centroid
    existing_hotspot_id: Optional[str]  # If a hotspot already covers this cluster
    is_parent: bool = False  # True if this cluster has subclusters
    children: List['CompleteClusterInfo'] = None  # List of sub-clusters
    parent_cluster: Optional['CompleteClusterInfo'] = None  # Parent cluster if this is a sub-cluster
    cluster_type: str = "natural"  # "natural", "noise_assigned", "micro"
    domain: str = "emergent"  # Domain emerges from content, not preset
    
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


def load_embeddings_with_metadata(db_path: str) -> Tuple[List[str], np.ndarray, Dict[str, Dict]]:
    """
    Load all embeddings and comprehensive node metadata from the database.
    
    Returns:
        node_ids: List of node IDs in order
        vectors: numpy array of shape (n_nodes, embedding_dim)
        node_meta: Dict of node_id -> {content, node_type, domain, timestamp, etc}
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all non-decayed nodes with embeddings - include ALL metadata for domain inference
    cursor.execute("""
        SELECT e.node_id, e.vector, tn.content, tn.node_type, 
               COALESCE(tn.domain, 'unknown') as domain, 
               tn.timestamp, tn.confidence, tn.source_file,
               COALESCE(tn.metadata, '{}') as metadata
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
    
    for node_id, vector_blob, content, node_type, domain, timestamp, confidence, source_file, metadata_str in rows:
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
            
            # Parse metadata
            try:
                metadata_dict = json.loads(metadata_str) if metadata_str else {}
            except (json.JSONDecodeError, TypeError):
                metadata_dict = {}
                
            node_ids.append(node_id)
            vectors.append(vector)
            node_meta[node_id] = {
                "content": content,
                "node_type": node_type,
                "domain": domain,
                "timestamp": timestamp,
                "confidence": confidence,
                "source_file": source_file,
                "metadata": metadata_dict
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


def infer_emergent_domains(node_meta: Dict[str, Dict]) -> Dict[str, str]:
    """
    Infer domains from node content and metadata. Domains emerge from the data.
    
    Returns:
        Dict mapping node_id -> emergent_domain
    """
    domain_mapping = {}
    
    # Keywords that suggest different domain areas
    domain_keywords = {
        "work": ["meta", "engineer", "promotion", "e5", "manager", "career", "job", "work", "company", "team"],
        "personal": ["family", "religion", "christian", "faith", "belief", "bunny", "personal", "identity"],
        "technical": ["embedding", "clustering", "algorithm", "code", "programming", "database", "api"],
        "projects": ["cashew", "blog", "project", "build", "implement", "dashboard", "brain"],
        "relationships": ["vinny", "cousin", "family", "friend", "relationship", "social"],
        "learning": ["insight", "pattern", "understanding", "realization", "discovery", "knowledge"],
        "decisions": ["decision", "choice", "plan", "will", "going to", "decided"],
        "reflections": ["think", "believe", "seems", "probably", "opinion", "reflection"]
    }
    
    for node_id, meta in node_meta.items():
        content_lower = meta["content"].lower()
        source_file = meta.get("source_file", "")
        
        # Check for domain keywords in content
        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for kw in keywords if kw in content_lower)
            if score > 0:
                domain_scores[domain] = score
        
        # If we found domain indicators, use the highest scoring one
        if domain_scores:
            best_domain = max(domain_scores, key=domain_scores.get)
            domain_mapping[node_id] = best_domain
        else:
            # Fallback: use node_type as domain indicator
            node_type = meta.get("node_type", "unknown")
            if node_type == "hotspot":
                domain_mapping[node_id] = "structure"  # hotspots are structural
            elif "system_generated" in source_file:
                domain_mapping[node_id] = "insights"  # think cycles, tensions
            else:
                domain_mapping[node_id] = "general"  # catch-all
    
    return domain_mapping


def detect_complete_clusters(
    db_path: str,
    eps: float = DBSCAN_EPS,
    min_samples: int = DBSCAN_MIN_SAMPLES,
    max_cluster_size: int = 15,
    max_recursion_depth: int = 5
) -> List[CompleteClusterInfo]:
    """
    Run complete clustering with 100% node coverage.
    
    Phase 1: DBSCAN finds natural dense clusters
    Phase 2: Every noise point gets assigned to nearest cluster or creates micro-cluster
    
    Returns:
        List of CompleteClusterInfo objects with ZERO orphans
    """
    node_ids, vectors, node_meta = load_embeddings_with_metadata(db_path)
    
    if len(node_ids) < min_samples:
        logger.info("Not enough nodes for clustering")
        return []
    
    logger.info(f"Complete clustering: {len(node_ids)} nodes with DBSCAN "
               f"(eps={eps}, min_samples={min_samples})")
    
    # Infer emergent domains
    domain_mapping = infer_emergent_domains(node_meta)
    
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
    
    # PHASE 1: Run DBSCAN
    clustering = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="precomputed"
    ).fit(distance_matrix)
    
    labels = clustering.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    logger.info(f"DBSCAN found {n_clusters} natural clusters, {n_noise} noise points")
    
    # Get existing hotspot cluster memberships
    existing_hotspot_clusters = _get_existing_hotspot_clusters(db_path)
    
    # Build natural clusters first
    natural_clusters = []
    cluster_centroids = []  # For noise assignment
    
    for cluster_id in range(n_clusters):
        member_indices = np.where(labels == cluster_id)[0]
        member_ids = [node_ids[i] for i in member_indices]
        member_vectors = vectors[member_indices]
        
        if len(member_ids) < MIN_CLUSTER_SIZE:
            continue
        
        # Compute centroid
        centroid = member_vectors.mean(axis=0)
        cluster_centroids.append((f"natural_{cluster_id}", centroid, member_ids))
        
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
        
        # Infer domain from cluster members
        member_domains = [domain_mapping.get(nid, "general") for nid in member_ids]
        cluster_domain = max(set(member_domains), key=member_domains.count)
        
        # Check if this cluster is too large and needs recursive splitting
        if len(member_ids) > max_cluster_size and max_recursion_depth > 0:
            # Recursively split large clusters
            subclusters = _split_large_cluster(
                member_ids, member_vectors, node_meta, domain_mapping,
                eps * 0.7, min_samples, max_cluster_size,
                existing_hotspot_clusters, max_recursion_depth - 1
            )
            
            if len(subclusters) > 1:
                # Successfully split - create parent cluster
                parent_cluster = CompleteClusterInfo(
                    cluster_id=f"parent_natural_{cluster_id}",
                    node_ids=member_ids,
                    centroid=centroid,
                    representative_content=representative_content,
                    existing_hotspot_id=None,
                    is_parent=True,
                    cluster_type="natural",
                    domain=cluster_domain
                )
                parent_cluster.children = subclusters
                for subcluster in subclusters:
                    subcluster.parent_cluster = parent_cluster
                
                natural_clusters.append(parent_cluster)
                natural_clusters.extend(subclusters)
            else:
                # Couldn't split meaningfully
                cluster = _create_regular_cluster(
                    f"natural_{cluster_id}", member_ids, centroid, 
                    representative_content, existing_hotspot_clusters,
                    cluster_domain, "natural"
                )
                natural_clusters.append(cluster)
        else:
            # Regular sized cluster
            cluster = _create_regular_cluster(
                f"natural_{cluster_id}", member_ids, centroid, 
                representative_content, existing_hotspot_clusters,
                cluster_domain, "natural"
            )
            natural_clusters.append(cluster)
    
    # PHASE 2: Assign ALL noise points to achieve 100% coverage
    noise_indices = np.where(labels == -1)[0]
    assigned_noise_clusters = []
    
    if n_noise > 0:
        logger.info(f"Phase 2: Assigning {n_noise} noise points to achieve 100% coverage")
        
        for noise_idx in noise_indices:
            noise_node_id = node_ids[noise_idx]
            noise_vector = vectors[noise_idx]
            
            # Find nearest cluster centroid
            best_cluster_id = None
            best_similarity = -1.0
            
            for cluster_name, centroid, cluster_members in cluster_centroids:
                sim = cosine_similarity(noise_vector, centroid)
                if sim > best_similarity:
                    best_similarity = sim
                    best_cluster_id = cluster_name
            
            if best_similarity >= NOISE_ASSIGNMENT_THRESHOLD and best_cluster_id:
                # Assign to nearest cluster
                # Find the cluster object and add this node
                for cluster in natural_clusters:
                    if cluster.cluster_id == best_cluster_id and not cluster.is_parent:
                        cluster.node_ids.append(noise_node_id)
                        cluster.cluster_type = "natural_expanded"
                        # Recompute centroid with new node
                        all_member_indices = [node_ids.index(nid) for nid in cluster.node_ids if nid in node_ids]
                        if all_member_indices:
                            cluster.centroid = vectors[all_member_indices].mean(axis=0)
                        break
            else:
                # Orphan nodes are a feature, not a bug.
                # Don't create micro-clusters of 1. Let them sit in the inbox
                # until the sleep cycle finds enough similar nodes to form a real cluster.
                logger.info(f"Node {noise_node_id} is an orphan — will be assigned to inbox, not micro-clustered")
    
    all_clusters = natural_clusters + assigned_noise_clusters
    
    # Verify 100% coverage
    all_clustered_nodes = set()
    for cluster in all_clusters:
        if not cluster.is_parent:  # Only count leaf clusters
            all_clustered_nodes.update(cluster.node_ids)
    
    coverage_percentage = len(all_clustered_nodes) / len(node_ids) * 100
    logger.info(f"Clustering coverage: {len(all_clustered_nodes)}/{len(node_ids)} nodes ({coverage_percentage:.1f}%)")
    
    if coverage_percentage < 100.0:
        missing_nodes = set(node_ids) - all_clustered_nodes
        logger.warning(f"Missing {len(missing_nodes)} nodes from clustering: {list(missing_nodes)[:5]}...")
        
        # Force assign missing nodes to micro-clusters
        for missing_node_id in missing_nodes:
            missing_idx = node_ids.index(missing_node_id)
            missing_vector = vectors[missing_idx]
            cluster_domain = domain_mapping.get(missing_node_id, "general")
            representative_content = [node_meta[missing_node_id]["content"][:100]]
            
            emergency_cluster = CompleteClusterInfo(
                cluster_id=f"emergency_{missing_node_id}",
                node_ids=[missing_node_id],
                centroid=missing_vector,
                representative_content=representative_content,
                existing_hotspot_id=None,
                cluster_type="micro",
                domain=cluster_domain
            )
            all_clusters.append(emergency_cluster)
    
    final_coverage = sum(len(c.node_ids) for c in all_clusters if not c.is_parent)
    logger.info(f"Final coverage: {final_coverage}/{len(node_ids)} nodes (100% guaranteed)")
    
    return all_clusters


def _split_large_cluster(member_ids, member_vectors, node_meta, domain_mapping, 
                        eps, min_samples, max_cluster_size, existing_hotspot_clusters, 
                        max_recursion_depth):
    """Helper to recursively split large clusters"""
    if len(member_ids) < min_samples or max_recursion_depth <= 0:
        return []
    
    # Run tighter DBSCAN on this subset
    try:
        sim_matrix = sklearn_cosine_sim(member_vectors)
        distance_matrix = 1.0 - sim_matrix
        distance_matrix = np.clip(distance_matrix, 0.0, 2.0)
        
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed").fit(distance_matrix)
        labels = clustering.labels_
        n_subclusters = len(set(labels)) - (1 if -1 in labels else 0)
    except Exception as e:
        logger.warning(f"Subcluster splitting failed: {e}")
        return []
    
    subclusters = []
    for sub_cluster_id in range(n_subclusters):
        sub_member_indices = np.where(labels == sub_cluster_id)[0]
        sub_member_ids = [member_ids[i] for i in sub_member_indices]
        sub_member_vectors = member_vectors[sub_member_indices]
        
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
        
        # Infer domain from subcluster members
        sub_member_domains = [domain_mapping.get(nid, "general") for nid in sub_member_ids]
        sub_cluster_domain = max(set(sub_member_domains), key=sub_member_domains.count)
        
        # Check if this subcluster needs further splitting
        if len(sub_member_ids) > max_cluster_size and max_recursion_depth > 0:
            further_subclusters = _split_large_cluster(
                sub_member_ids, sub_member_vectors, node_meta, domain_mapping,
                eps * 0.7, min_samples, max_cluster_size,
                existing_hotspot_clusters, max_recursion_depth - 1
            )
            
            if len(further_subclusters) > 1:
                subclusters.extend(further_subclusters)
            else:
                subcluster = CompleteClusterInfo(
                    cluster_id=f"sub_{sub_cluster_id}",
                    node_ids=sub_member_ids,
                    centroid=sub_centroid,
                    representative_content=sub_representative_content,
                    existing_hotspot_id=None,
                    cluster_type="natural",
                    domain=sub_cluster_domain
                )
                subclusters.append(subcluster)
        else:
            subcluster = CompleteClusterInfo(
                cluster_id=f"sub_{sub_cluster_id}",
                node_ids=sub_member_ids,
                centroid=sub_centroid,
                representative_content=sub_representative_content,
                existing_hotspot_id=None,
                cluster_type="natural",
                domain=sub_cluster_domain
            )
            subclusters.append(subcluster)
    
    return subclusters


def _create_regular_cluster(cluster_id, member_ids, centroid, representative_content, 
                          existing_hotspot_clusters, cluster_domain, cluster_type):
    """Helper to create a regular CompleteClusterInfo with existing hotspot check"""
    # Check if an existing hotspot covers this cluster
    existing_hotspot = None
    for hotspot_id, hotspot_members in existing_hotspot_clusters.items():
        overlap = len(set(member_ids) & hotspot_members)
        if overlap >= len(member_ids) * 0.5:  # >50% overlap
            existing_hotspot = hotspot_id
            break
    
    return CompleteClusterInfo(
        cluster_id=cluster_id,
        node_ids=member_ids,
        centroid=centroid,
        representative_content=representative_content,
        existing_hotspot_id=existing_hotspot,
        cluster_type=cluster_type,
        domain=cluster_domain
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


def check_complete_coverage(db_path: str) -> Dict:
    """
    Verify that every node in the graph belongs to exactly one cluster.
    
    Returns:
        Dict with coverage statistics and any orphaned nodes
    """
    node_ids, vectors, node_meta = load_embeddings_with_metadata(db_path)
    
    # Get all hotspot cluster memberships
    hotspot_clusters = _get_existing_hotspot_clusters(db_path)
    
    clustered_nodes = set()
    for hotspot_id, members in hotspot_clusters.items():
        clustered_nodes.update(members)
    
    all_nodes = set(node_ids)
    orphaned_nodes = all_nodes - clustered_nodes
    
    coverage_stats = {
        "total_nodes": len(all_nodes),
        "clustered_nodes": len(clustered_nodes),
        "orphaned_nodes": len(orphaned_nodes),
        "coverage_percentage": (len(clustered_nodes) / len(all_nodes)) * 100 if all_nodes else 100,
        "orphan_node_ids": list(orphaned_nodes)[:10],  # Show first 10
        "total_hotspots": len(hotspot_clusters)
    }
    
    return coverage_stats


def run_complete_clustering_cycle(db_path: str, model_fn=None, dry_run: bool = False) -> Dict:
    """
    Full complete clustering cycle with 100% coverage guarantee.
    
    1. Detect clusters with complete coverage (no noise orphans)
    2. Check existing hotspot staleness  
    3. Create new hotspots for uncovered clusters
    4. Create hierarchical hotspot relationships
    5. Update stale hotspots
    
    Args:
        db_path: Path to graph database
        model_fn: Optional LLM function for summary generation
        dry_run: If True, don't modify database
        
    Returns:
        Summary dict of actions taken
    """
    results = {
        "clusters_found": 0,
        "natural_clusters": 0,
        "micro_clusters": 0,
        "new_hotspots_created": 0,
        "coverage_percentage": 0,
        "orphaned_nodes_eliminated": 0,
        "stale_hotspots_found": 0,
        "hotspots_updated": 0,
        "parent_hotspots_created": 0,
        "hierarchical_edges_created": 0,
        "cluster_details": []
    }
    
    # Step 1: Detect complete clusters (100% coverage)
    clusters = detect_complete_clusters(db_path, max_cluster_size=15)
    results["clusters_found"] = len(clusters)
    
    # Count cluster types
    natural_count = len([c for c in clusters if c.cluster_type.startswith("natural") and not c.is_parent])
    micro_count = len([c for c in clusters if c.cluster_type == "micro"])
    results["natural_clusters"] = natural_count
    results["micro_clusters"] = micro_count
    
    # Calculate coverage
    total_nodes_in_clusters = sum(len(c.node_ids) for c in clusters if not c.is_parent)
    node_ids, _, _ = load_embeddings_with_metadata(db_path)
    results["coverage_percentage"] = (total_nodes_in_clusters / len(node_ids)) * 100 if node_ids else 100
    results["orphaned_nodes_eliminated"] = len(node_ids) - total_nodes_in_clusters
    
    # Keep track of hotspots created for hierarchy building
    cluster_to_hotspot = {}
    
    for cluster in clusters:
        if cluster.is_parent:
            continue  # Handle parent clusters separately
            
        detail = {
            "cluster_id": cluster.cluster_id,
            "size": len(cluster.node_ids),
            "type": cluster.cluster_type,
            "domain": cluster.domain,
            "representative": cluster.representative_content[:2],
            "has_hotspot": cluster.existing_hotspot_id is not None,
            "action": "none"
        }
        
        if cluster.existing_hotspot_id is None:
            # No matching hotspot — DO NOT create one here.
            # Hotspot creation is handled by placement_aware_extraction.py
            # which has proper MIN_CLUSTER_SIZE gates and novelty checks.
            # Creating hotspots in the sleep cycle causes runaway proliferation
            # because DBSCAN produces different clusters each run and member-overlap
            # matching fails to recognize them as duplicates.
            detail["action"] = "skipped_no_hotspot_creation_in_sleep"
            results.setdefault("skipped_clusters", 0)
            results["skipped_clusters"] += 1
        else:
            cluster_to_hotspot[cluster.cluster_id] = cluster.existing_hotspot_id
        
        results["cluster_details"].append(detail)
    
    # Handle parent clusters — same as above, no new hotspot creation in sleep
    for cluster in clusters:
        if cluster.is_parent:
            if cluster.existing_hotspot_id is not None:
                cluster_to_hotspot[cluster.cluster_id] = cluster.existing_hotspot_id
            else:
                results.setdefault("skipped_parent_clusters", 0)
                results["skipped_parent_clusters"] += 1
    
    # Step 2: Create hierarchical relationships between hotspots
    if not dry_run:
        results["hierarchical_edges_created"] = _create_hierarchical_edges(
            db_path, clusters, cluster_to_hotspot
        )
    
    # Step 3: Check staleness (using existing code from clustering.py)
    from core.clustering import check_hotspot_staleness, update_hotspot
    stale_reports = check_hotspot_staleness(db_path)
    stale_hotspots = [r for r in stale_reports if r["is_stale"]]
    results["stale_hotspots_found"] = len(stale_hotspots)
    
    for report in stale_hotspots:
        if not dry_run and model_fn:
            # Get cluster members for regeneration
            from core.hotspots import get_hotspot
            hotspot = get_hotspot(db_path, report["hotspot_id"])
            if hotspot and hotspot.get("cluster"):
                representative = [n["content"][:100] for n in hotspot["cluster"][:5]]
                new_summary = _generate_hotspot_summary_from_content(representative, model_fn)
                update_hotspot(
                    db_path=db_path,
                    hotspot_id=report["hotspot_id"],
                    content=new_summary,
                    status="auto_refreshed_complete"
                )
                results["hotspots_updated"] += 1
    
    return results


def _generate_hotspot_summary(cluster: CompleteClusterInfo, model_fn=None) -> str:
    """Generate a summary for a new hotspot from cluster info"""
    if model_fn:
        prompt = f"""Summarize this {cluster.cluster_type} cluster from the {cluster.domain} domain into a single concise status line (1-2 sentences max).
Focus on: what is this about, what's the current state, what's next.

Cluster type: {cluster.cluster_type}
Domain: {cluster.domain}  
Representative thoughts:
{chr(10).join(f'- {c}' for c in cluster.representative_content)}

Summary:"""
        try:
            return model_fn(prompt).strip()
        except Exception as e:
            logger.warning(f"LLM summary failed, falling back to heuristic: {e}")
    
    # Fallback: heuristic summary
    if cluster.cluster_type == "micro":
        return f"[MICRO-CLUSTER] {cluster.representative_content[0][:80]}"
    else:
        return f"[{cluster.domain.upper()}] {cluster.representative_content[0][:80]}"


def _generate_parent_hotspot_summary(cluster: CompleteClusterInfo, model_fn=None) -> str:
    """Generate summary for parent hotspot"""
    child_summaries = []
    for child in cluster.children:
        if child.representative_content:
            child_summaries.append(child.representative_content[0][:60])
    
    if model_fn and child_summaries:
        prompt = f"""Create a parent-level summary that captures the common theme across these sub-clusters in the {cluster.domain} domain.

Sub-clusters:
{chr(10).join(f'- {s}' for s in child_summaries)}

Parent summary (1-2 sentences):"""
        try:
            return model_fn(prompt).strip()
        except Exception as e:
            logger.warning(f"Parent summary failed: {e}")
    
    return f"[{cluster.domain.upper()} OVERVIEW] Multi-faceted area with {len(cluster.children)} sub-topics"


def _generate_hotspot_summary_from_content(representative_content, model_fn):
    """Generate hotspot summary from content list (for staleness updates)"""
    if model_fn:
        prompt = f"""Summarize these related thoughts into a single concise status line (1-2 sentences max).
Focus on: what is this about, what's the current state, what's next.

Thoughts:
{chr(10).join(f'- {c}' for c in representative_content)}

Summary:"""
        try:
            return model_fn(prompt).strip()
        except Exception as e:
            logger.warning(f"LLM summary failed: {e}")
    
    return representative_content[0] if representative_content else "Auto-refreshed cluster summary"


def _create_hierarchical_edges(db_path: str, clusters: List[CompleteClusterInfo], 
                              cluster_to_hotspot: Dict) -> int:
    """Create hierarchical edges between parent and child hotspots"""
    import sqlite3
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    edges_created = 0
    
    for cluster in clusters:
        if not cluster.is_parent:
            continue
            
        parent_hotspot_id = cluster_to_hotspot.get(cluster.cluster_id)
        if not parent_hotspot_id:
            continue
            
        # Create edges from parent hotspot to child hotspots
        for child_cluster in cluster.children:
            child_hotspot_id = cluster_to_hotspot.get(child_cluster.cluster_id)
            if child_hotspot_id:
                try:
                    cursor.execute("""
                        INSERT OR IGNORE INTO derivation_edges 
                        (parent_id, child_id, relation, weight, reasoning)
                        VALUES (?, ?, 'summarizes', 0.9, 'Complete clustering - parent to child hotspot')
                    """, (parent_hotspot_id, child_hotspot_id))
                    edges_created += 1
                except sqlite3.IntegrityError:
                    pass  # Edge already exists
    
    conn.commit()
    conn.close()
    return edges_created