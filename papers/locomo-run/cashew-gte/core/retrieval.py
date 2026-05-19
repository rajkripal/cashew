#!/usr/bin/env python3
"""
Cashew Hybrid Retrieval Module
Combines embedding search with graph traversal for context retrieval
"""

import sqlite3
import json
import time
from typing import List, Dict, Optional
from collections import defaultdict, deque
from dataclasses import dataclass
import sys
import argparse

import numpy as np
from .embeddings import search as embedding_search, embed_text
from .metrics import record_metric, is_metrics_enabled

@dataclass
class RetrievalResult:
    node_id: str
    content: str
    node_type: str
    domain: str
    score: float
    path: List[str]  # How this node was reached (for graph walk results)
    
    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "content": self.content,
            "node_type": self.node_type,
            "domain": self.domain,
            "score": self.score,
            "path": self.path
        }

def _get_connection(db_path: str) -> sqlite3.Connection:
    """Get database connection"""
    return sqlite3.connect(db_path)

def _load_node_details(db_path: str, node_ids: List[str], domain_filter: Optional[str] = None, tag_filter: Optional[List[str]] = None, exclude_tags: Optional[List[str]] = None) -> Dict[str, Dict]:
    """Load node details for multiple node IDs with optional domain, tag, and exclusion filtering"""
    if not node_ids:
        return {}
    
    conn = _get_connection(db_path)
    cursor = conn.cursor()
    
    # Check if domain and tags columns exist
    cursor.execute("PRAGMA table_info(thought_nodes)")
    columns = [row[1] for row in cursor.fetchall()]
    has_domain_column = 'domain' in columns
    has_tags_column = 'tags' in columns
    
    placeholders = ','.join(['?'] * len(node_ids))
    
    # Build tag filter SQL
    tag_conditions = ""
    tag_params = []
    if tag_filter and has_tags_column:
        tag_clauses = [f"tags LIKE ?" for _ in tag_filter]
        tag_conditions = " AND (" + " OR ".join(tag_clauses) + ")"
        tag_params = [f"%{tag}%" for tag in tag_filter]
    
    # Build tag exclusion SQL (vault:private etc)
    exclude_conditions = ""
    exclude_params = []
    if exclude_tags and has_tags_column:
        exclude_clauses = [f"(tags IS NULL OR tags NOT LIKE ?)" for _ in exclude_tags]
        exclude_conditions = " AND " + " AND ".join(exclude_clauses)
        exclude_params = [f"%{tag}%" for tag in exclude_tags]
    
    if has_domain_column:
        if domain_filter:
            cursor.execute(f"""
                SELECT id, content, node_type, COALESCE(metadata, '{{}}') as metadata, domain
                FROM thought_nodes 
                WHERE id IN ({placeholders})
                AND (decayed IS NULL OR decayed = 0)
                AND domain = ?
                {tag_conditions}
                {exclude_conditions}
            """, node_ids + [domain_filter] + tag_params + exclude_params)
        else:
            cursor.execute(f"""
                SELECT id, content, node_type, COALESCE(metadata, '{{}}') as metadata, COALESCE(domain, 'unknown') as domain
                FROM thought_nodes 
                WHERE id IN ({placeholders})
                AND (decayed IS NULL OR decayed = 0)
                {tag_conditions}
                {exclude_conditions}
            """, node_ids + tag_params + exclude_params)
    else:
        # Backwards compatibility: no domain column
        cursor.execute(f"""
            SELECT id, content, node_type, COALESCE(metadata, '{{}}') as metadata
            FROM thought_nodes 
            WHERE id IN ({placeholders})
            AND (decayed IS NULL OR decayed = 0)
            {exclude_conditions}
        """, node_ids + exclude_params)
    
    nodes = {}
    for row in cursor.fetchall():
        if has_domain_column:
            node_id, content, node_type, metadata, domain = row
        else:
            node_id, content, node_type, metadata = row
            domain = None
        
        try:
            metadata_dict = json.loads(metadata) if metadata else {}
        except (json.JSONDecodeError, TypeError):
            metadata_dict = {}
        
        # Get domain: prefer domain column, fall back to metadata, then 'unknown'
        if has_domain_column and domain:
            final_domain = domain
        else:
            final_domain = metadata_dict.get('domain', 'unknown')
        
        # If domain filtering is requested but this is an old DB without domain column,
        # filter based on metadata
        if domain_filter and not has_domain_column:
            if final_domain != domain_filter:
                continue
        
        nodes[node_id] = {
            "content": content,
            "node_type": node_type,
            "domain": final_domain
        }
    
    conn.close()
    return nodes

def _load_recency_clock(db_path: str, node_ids: List[str]) -> Dict[str, Optional[str]]:
    """Return node_id -> ISO8601 event clock for similarity recency weighting.

    Uses COALESCE(referent_time, timestamp): biographical / user-facing clock.
    Callers on the operational side (decay/GC/declassify/embeddings/recent
    activity) must NOT use this helper — read `timestamp` directly instead.
    """
    if not node_ids:
        return {}
    conn = _get_connection(db_path)
    cursor = conn.cursor()
    # Guard against older DBs that predate referent_time.
    cursor.execute("PRAGMA table_info(thought_nodes)")
    cols = {row[1] for row in cursor.fetchall()}
    clock_expr = "COALESCE(referent_time, timestamp)" if 'referent_time' in cols else "timestamp"
    placeholders = ','.join(['?'] * len(node_ids))
    cursor.execute(
        f"SELECT id, {clock_expr} FROM thought_nodes WHERE id IN ({placeholders})",
        node_ids,
    )
    out = {nid: ts for nid, ts in cursor.fetchall()}
    conn.close()
    return out


def _recency_weight(iso_ts: Optional[str]) -> float:
    """Gentle recency weight in [0.5, 1.0]. Halves over ~365 days of event age.

    Unknown/unparseable timestamps neutralize to 1.0 (no penalty) rather
    than dropping the node — we'd rather rank on similarity alone than
    silently bury a node with bad metadata.
    """
    if not iso_ts:
        return 1.0
    try:
        from datetime import datetime, timezone
        s = iso_ts.replace('Z', '+00:00') if iso_ts.endswith('Z') else iso_ts
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            return 1.0
        age_days = (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0
        if age_days <= 0:
            return 1.0
        # Exponential decay with half-life = 365 days, floored at 0.5.
        import math
        w = 0.5 + 0.5 * math.exp(-age_days / 365.0)
        return max(0.5, min(1.0, w))
    except Exception:
        return 1.0


def _graph_walk(db_path: str, entry_points: List[str], walk_depth: int = 2) -> Dict[str, List[str]]:
    """
    Walk the graph from entry points to find connected nodes
    
    Args:
        db_path: Path to database
        entry_points: List of node IDs to start walking from
        walk_depth: Maximum depth to walk
        
    Returns:
        Dict mapping node_id -> path (list of nodes showing how it was reached)
    """
    conn = _get_connection(db_path)
    cursor = conn.cursor()
    
    # Build adjacency lists for both directions
    cursor.execute("""
        SELECT parent_id, child_id FROM derivation_edges
        WHERE parent_id IN (
            SELECT id FROM thought_nodes WHERE decayed IS NULL OR decayed = 0
        ) AND child_id IN (
            SELECT id FROM thought_nodes WHERE decayed IS NULL OR decayed = 0
        )
    """)
    
    # Graph as adjacency lists (both directions)
    outgoing = defaultdict(list)  # parent -> [children]
    incoming = defaultdict(list)  # child -> [parents]
    
    for parent_id, child_id in cursor.fetchall():
        outgoing[parent_id].append(child_id)
        incoming[child_id].append(parent_id)
    
    conn.close()
    
    # BFS walk from each entry point
    found_nodes = {}  # node_id -> path
    
    for entry_point in entry_points:
        if entry_point not in found_nodes:
            found_nodes[entry_point] = [entry_point]
        
        # BFS queue: (node_id, path_to_node, depth)
        queue = deque([(entry_point, [entry_point], 0)])
        visited = {entry_point}
        
        while queue:
            current_node, path, depth = queue.popleft()
            
            if depth >= walk_depth:
                continue
            
            # Get neighbors in both directions
            neighbors = set()
            neighbors.update(outgoing.get(current_node, []))  # Children
            neighbors.update(incoming.get(current_node, []))  # Parents
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = path + [neighbor]
                    
                    # If we haven't seen this node or found it with a shorter path, update
                    if neighbor not in found_nodes or len(new_path) < len(found_nodes[neighbor]):
                        found_nodes[neighbor] = new_path
                    
                    queue.append((neighbor, new_path, depth + 1))
    
    return found_nodes

def retrieve(db_path: str, query: str, top_k: int = 5, walk_depth: int = 2, domain: Optional[str] = None, exclude_tags: Optional[List[str]] = None) -> List[RetrievalResult]:
    """
    Hybrid retrieval combining embeddings and graph walking
    
    Args:
        db_path: Path to SQLite database
        query: Search query
        top_k: Number of top results to return
        walk_depth: Graph walk depth from embedding entry points
        domain: Optional domain filter (user, ai, etc.). If None, returns all domains.
        exclude_tags: Optional list of tags to exclude from results (e.g. ['vault:private'])
        
    Returns:
        List of RetrievalResult objects ranked by hybrid score
    """
    if not query or not query.strip():
        return []
    
    # Step 1: Embedding search to find entry points
    embedding_results = embedding_search(db_path, query, top_k * 2)  # Get more candidates
    
    if not embedding_results:
        return []
    
    entry_point_ids = [node_id for node_id, _ in embedding_results]
    embedding_scores = {node_id: score for node_id, score in embedding_results}
    
    # Step 2: Graph walk from entry points
    walked_nodes = _graph_walk(db_path, entry_point_ids, walk_depth)
    
    # Step 3: Collect all unique nodes
    all_node_ids = set(embedding_scores.keys()) | set(walked_nodes.keys())
    
    # Load node details (with domain filtering if specified)
    node_details = _load_node_details(db_path, list(all_node_ids), domain, exclude_tags=exclude_tags)

    # Similarity recency weighting uses the EVENT clock (referent_time) when
    # available, falling back to ingestion time. This is the user-facing /
    # biographical reader — a 2019 WhatsApp note imported today should read
    # as 2019-old, not fresh. Operational readers (decay/GC/declassify/
    # embeddings maintenance/recent-activity) deliberately stay on
    # `timestamp` — do NOT copy this COALESCE pattern there.
    recency_by_id = _load_recency_clock(db_path, list(all_node_ids))

    # Step 4: Calculate hybrid scores
    results = []

    for node_id in all_node_ids:
        if node_id not in node_details:
            continue

        details = node_details[node_id]

        # Embedding score (0.0 if not found in embedding search)
        embedding_score = embedding_scores.get(node_id, 0.0)

        # Graph proximity score (inverse of path length)
        if node_id in walked_nodes:
            path_length = len(walked_nodes[node_id])
            # Score: 1.0 for direct hits, decreasing with path length
            graph_score = 1.0 / path_length if path_length > 0 else 1.0
            path = walked_nodes[node_id]
        else:
            graph_score = 0.0
            path = [node_id] if node_id in embedding_scores else []

        # Hybrid score: weighted combination, multiplied by a gentle recency
        # factor on the event clock. Recency factor in [0.5, 1.0]: halves
        # over ~1 year of event-time age. Pure metadata — no graph semantics.
        recency_factor = _recency_weight(recency_by_id.get(node_id))
        hybrid_score = (embedding_score * 0.5 + graph_score * 0.5) * recency_factor

        result = RetrievalResult(
            node_id=node_id,
            content=details["content"],
            node_type=details["node_type"],
            domain=details["domain"],
            score=hybrid_score,
            path=path
        )
        
        results.append(result)
    
    # Step 5: Rank and return top results
    results.sort(key=lambda r: r.score, reverse=True)

    return results[:top_k]

def retrieve_recursive_bfs(db_path: str, query: str, top_k: int = 10, n_seeds: int = 5,
                           picks_per_hop: int = 3, max_depth: int = 3,
                           domain: Optional[str] = None, tags: Optional[List[str]] = None,
                           exclude_tags: Optional[List[str]] = None) -> List[RetrievalResult]:
    """
    Recursive BFS retrieval: seed by embedding similarity, then explore neighbors
    guided by embedding scores, picking the best at each hop. All traversed nodes
    become candidates; final ranking is by cosine similarity to query.

    Args:
        db_path: Path to SQLite database
        query: Search query
        top_k: Number of top results to return
        n_seeds: Number of initial seed nodes from embedding search
        picks_per_hop: How many neighbors to pick per frontier node per depth level
        max_depth: Maximum BFS depth
        domain: Optional domain filter
        tags: Optional tag filter (include nodes matching any of these tags)
        exclude_tags: Optional tags to exclude

    Returns:
        List of RetrievalResult objects ranked by cosine similarity
    """
    if not query or not query.strip():
        return []

    # Start timing if metrics are enabled
    start_time = time.perf_counter() if is_metrics_enabled() else None
    embed_start = time.perf_counter() if is_metrics_enabled() else None

    # Step 1: Seed — find top N entry points via embedding search (O(log N) with sqlite-vec)
    seed_results = embedding_search(db_path, query, top_k=n_seeds)
    if not seed_results:
        return []
    
    seeds = [nid for nid, _ in seed_results]
    seed_scores = {nid: score for nid, score in seed_results}
    candidates = set(seeds)
    
    search_time = (time.perf_counter() - embed_start) * 1000 if is_metrics_enabled() else 0

    # Embed the query for BFS neighbor scoring
    embed_start2 = time.perf_counter() if is_metrics_enabled() else None
    query_vec = np.array(embed_text(query), dtype=np.float32)
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        return []
    
    embed_time = (time.perf_counter() - embed_start2) * 1000 if is_metrics_enabled() else 0

    conn = _get_connection(db_path)
    cursor = conn.cursor()

    # Preload adjacency list (both directions)
    neighbors = defaultdict(set)
    cursor.execute("SELECT parent_id, child_id FROM derivation_edges")
    for parent, child in cursor.fetchall():
        neighbors[parent].add(child)
        neighbors[child].add(parent)

    # Cache for embeddings loaded on-demand during BFS
    _vec_cache = {}

    def cosine_sim(node_id: str) -> float:
        if node_id in seed_scores:
            return seed_scores[node_id]
        if node_id in _vec_cache:
            vec = _vec_cache[node_id]
        else:
            row = cursor.execute("SELECT vector FROM embeddings WHERE node_id = ?", (node_id,)).fetchone()
            if row is None:
                _vec_cache[node_id] = None
                return 0.0
            vec = np.frombuffer(row[0], dtype=np.float32)
            _vec_cache[node_id] = vec
        if vec is None:
            return 0.0
        nv = np.linalg.norm(vec)
        if nv == 0:
            return 0.0
        return float(np.dot(query_vec, vec) / (query_norm * nv))

    # Step 2: Recursive BFS — explore from seeds
    bfs_start = time.perf_counter() if is_metrics_enabled() else None
    initial_seed_count = len(seeds)
    
    frontier = list(seeds)
    for _depth in range(max_depth):
        next_frontier = []
        for node_id in frontier:
            nbrs = neighbors.get(node_id, set())
            if not nbrs:
                continue
            scored = [(nid, cosine_sim(nid)) for nid in nbrs if nid not in candidates]
            scored.sort(key=lambda x: x[1], reverse=True)
            for nid, _sim in scored[:picks_per_hop]:
                candidates.add(nid)
                next_frontier.append(nid)
        frontier = next_frontier
        if not frontier:
            break

    bfs_time = (time.perf_counter() - bfs_start) * 1000 if is_metrics_enabled() else 0
    bfs_explored = len(candidates)

    conn.close()

    # Step 3: Final ranking — all candidates scored by cosine similarity
    final = [(nid, cosine_sim(nid)) for nid in candidates]
    final.sort(key=lambda x: x[1], reverse=True)

    # Take more than top_k initially so filtering doesn't starve us
    candidate_ids = [nid for nid, _ in final[:top_k * 3]]

    # Load node details with domain/tag filtering
    node_details = _load_node_details(db_path, candidate_ids, domain, tag_filter=tags, exclude_tags=exclude_tags)

    # Build results for nodes that passed filtering
    score_map = dict(final)
    results = []
    for node_id in candidate_ids:
        if node_id not in node_details:
            continue
        details = node_details[node_id]
        results.append(RetrievalResult(
            node_id=node_id,
            content=details["content"],
            node_type=details["node_type"],
            domain=details["domain"],
            score=score_map[node_id],
            path=[node_id]
        ))
        if len(results) >= top_k:
            break

    # Record metrics if enabled
    if is_metrics_enabled() and start_time is not None:
        total_time = (time.perf_counter() - start_time) * 1000
        
        # Calculate overlap ratio: how many results were seeds vs BFS-discovered
        seed_results_count = len([r for r in results if r.node_id in seeds])
        overlap_ratio = seed_results_count / max(len(results), 1) if results else 0
        
        record_metric(db_path, 'retrieval', total_time,
                      embed_time_ms=embed_time,
                      search_time_ms=search_time,
                      bfs_time_ms=bfs_time,
                      seeds_found=initial_seed_count,
                      bfs_explored=bfs_explored,
                      results_returned=len(results),
                      overlap_ratio=overlap_ratio)

    return results



def retrieve_bfs_streaming(db_path: str, query: str, n_seeds: int = 5,
                           picks_per_hop: int = 3, max_depth: int = 3):
    """Streaming variant of retrieve_recursive_bfs that yields events as the
    walk progresses. Used by the dashboard to animate "LLM-query-style" node
    reveal: seeds first, then each hop.

    Yields dicts shaped like:
      {"event": "seeds", "nodes": [{"id", "score"}, ...]}
      {"event": "hop", "level": int, "nodes": [{"id", "score", "from": parent_id}, ...]}
      {"event": "done", "ranked": [{"id", "score"}, ...]}
    """
    if not query or not query.strip():
        yield {"event": "done", "ranked": []}
        return

    seed_results = embedding_search(db_path, query, top_k=n_seeds)
    if not seed_results:
        yield {"event": "done", "ranked": []}
        return

    seed_scores = {nid: score for nid, score in seed_results}
    seeds = [nid for nid, _ in seed_results]
    candidates = set(seeds)
    yield {"event": "seeds",
           "nodes": [{"id": nid, "score": float(s)} for nid, s in seed_results]}

    query_vec = np.array(embed_text(query), dtype=np.float32)
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        yield {"event": "done", "ranked": []}
        return

    conn = _get_connection(db_path)
    cursor = conn.cursor()
    neighbors = defaultdict(set)
    cursor.execute("SELECT parent_id, child_id FROM derivation_edges")
    for parent, child in cursor.fetchall():
        neighbors[parent].add(child)
        neighbors[child].add(parent)

    _vec_cache = {}
    def cosine_sim(node_id: str) -> float:
        if node_id in seed_scores:
            return seed_scores[node_id]
        if node_id in _vec_cache:
            vec = _vec_cache[node_id]
        else:
            row = cursor.execute("SELECT vector FROM embeddings WHERE node_id = ?", (node_id,)).fetchone()
            if row is None:
                _vec_cache[node_id] = None
                return 0.0
            vec = np.frombuffer(row[0], dtype=np.float32)
            _vec_cache[node_id] = vec
        if vec is None:
            return 0.0
        nv = np.linalg.norm(vec)
        if nv == 0:
            return 0.0
        return float(np.dot(query_vec, vec) / (query_norm * nv))

    frontier = list(seeds)
    for depth in range(max_depth):
        next_frontier = []
        hop_nodes = []
        for node_id in frontier:
            nbrs = neighbors.get(node_id, set())
            if not nbrs:
                continue
            scored = [(nid, cosine_sim(nid)) for nid in nbrs if nid not in candidates]
            scored.sort(key=lambda x: x[1], reverse=True)
            for nid, sim in scored[:picks_per_hop]:
                candidates.add(nid)
                next_frontier.append(nid)
                hop_nodes.append({"id": nid, "score": float(sim), "from": node_id})
        if hop_nodes:
            yield {"event": "hop", "level": depth + 1, "nodes": hop_nodes}
        frontier = next_frontier
        if not frontier:
            break

    conn.close()
    final = sorted(((nid, cosine_sim(nid)) for nid in candidates), key=lambda x: x[1], reverse=True)
    yield {"event": "done",
           "ranked": [{"id": nid, "score": float(s)} for nid, s in final]}


def format_context(results: List[RetrievalResult], include_paths: bool = False) -> str:
    """
    Format retrieval results into a prompt-ready context string
    
    Args:
        results: List of RetrievalResult objects
        include_paths: Whether to include derivation paths in output
        
    Returns:
        Formatted context string
    """
    if not results:
        return "No relevant context found."
    
    context_lines = ["=== RELEVANT CONTEXT ==="]
    
    for i, result in enumerate(results, 1):
        # Basic info line with domain inline
        domain_str = f" (Domain: {result.domain})" if result.domain and result.domain != "unknown" else ""
        lines = [f"{i}. [{result.node_type.upper()}] {result.content}{domain_str}"]
        
        # Add path if requested and meaningful
        if include_paths and len(result.path) > 1:
            path_display = " → ".join(result.path[-3:])  # Show last 3 nodes in path
            if len(result.path) > 3:
                path_display = "... → " + path_display
            lines.append(f"   Path: {path_display}")
        
        # Add relevance score for debugging
        lines.append(f"   Relevance: {result.score:.3f}")
        
        context_lines.extend(lines)
        context_lines.append("")  # Blank line between results
    
    return "\n".join(context_lines)

def explain_retrieval(db_path: str, query: str, top_k: int = 5, walk_depth: int = 2) -> Dict:
    """
    Detailed explanation of the retrieval process for debugging
    
    Args:
        db_path: Path to SQLite database
        query: Search query
        top_k: Number of results to return
        walk_depth: Graph walk depth
        
    Returns:
        Dictionary with detailed breakdown of retrieval process
    """
    # Step 1: Embedding search
    embedding_results = embedding_search(db_path, query, top_k * 2)
    entry_points = [node_id for node_id, _ in embedding_results]
    
    # Step 2: Graph walk
    walked_nodes = _graph_walk(db_path, entry_points, walk_depth)
    
    # Step 3: Get final results
    final_results = retrieve(db_path, query, top_k, walk_depth)
    
    # Load node details for explanation
    all_node_ids = list(set(entry_points) | set(walked_nodes.keys()))
    node_details = _load_node_details(db_path, all_node_ids)
    
    explanation = {
        "query": query,
        "embedding_search": {
            "num_results": len(embedding_results),
            "entry_points": [
                {
                    "node_id": node_id,
                    "content": node_details.get(node_id, {}).get("content", "Unknown")[:60] + "...",
                    "embedding_score": score
                }
                for node_id, score in embedding_results[:5]  # Show top 5
            ]
        },
        "graph_walk": {
            "walk_depth": walk_depth,
            "nodes_discovered": len(walked_nodes),
            "paths": {
                node_id: path for node_id, path in list(walked_nodes.items())[:10]  # Show first 10
            }
        },
        "final_results": [result.to_dict() for result in final_results],
        "summary": {
            "embedding_hits": len(embedding_results),
            "graph_expansion": len(walked_nodes) - len(entry_points),
            "final_results": len(final_results)
        }
    }
    
    return explanation

def main():
    """CLI interface for retrieval module"""
    parser = argparse.ArgumentParser(description="Cashew Hybrid Retrieval")
    parser.add_argument("command", choices=["retrieve", "explain"], help="Command to run")
    parser.add_argument("--db", default="./data/graph.db", help="Database path")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    parser.add_argument("--walk-depth", type=int, default=2, help="Graph walk depth")
    parser.add_argument("--domain", help="Filter results to specific domain (user, ai, etc.)")
    parser.add_argument("--include-paths", action="store_true", help="Include paths in output")
    
    args = parser.parse_args()
    
    if args.command == "retrieve":
        print(f"🔍 Hybrid retrieval for: {args.query}")
        print()
        
        results = retrieve(args.db, args.query, args.top_k, args.walk_depth, args.domain)
        
        if results:
            context = format_context(results, args.include_paths)
            print(context)
        else:
            print("No results found.")
    
    elif args.command == "explain":
        print(f"🔍 Retrieval explanation for: {args.query}")
        print()
        
        explanation = explain_retrieval(args.db, args.query, args.top_k, args.walk_depth)
        print(json.dumps(explanation, indent=2))
    
    return 0

if __name__ == "__main__":
    sys.exit(main())