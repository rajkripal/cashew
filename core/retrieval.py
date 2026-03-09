#!/usr/bin/env python3
"""
Cashew Hybrid Retrieval Module
Combines embedding search with graph traversal for context retrieval
"""

import sqlite3
import json
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, deque
from dataclasses import dataclass
import sys
import argparse

from .embeddings import search as embedding_search

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

def _load_node_details(db_path: str, node_ids: List[str]) -> Dict[str, Dict]:
    """Load node details for multiple node IDs"""
    if not node_ids:
        return {}
    
    conn = _get_connection(db_path)
    cursor = conn.cursor()
    
    placeholders = ','.join(['?'] * len(node_ids))
    cursor.execute(f"""
        SELECT id, content, node_type, COALESCE(metadata, '{{}}') as metadata
        FROM thought_nodes 
        WHERE id IN ({placeholders})
        AND (decayed IS NULL OR decayed = 0)
    """, node_ids)
    
    nodes = {}
    for row in cursor.fetchall():
        node_id, content, node_type, metadata = row
        try:
            metadata_dict = json.loads(metadata) if metadata else {}
        except (json.JSONDecodeError, TypeError):
            metadata_dict = {}
        
        domain = metadata_dict.get('domain', 'unknown')
        
        nodes[node_id] = {
            "content": content,
            "node_type": node_type,
            "domain": domain
        }
    
    conn.close()
    return nodes

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

def retrieve(db_path: str, query: str, top_k: int = 5, walk_depth: int = 2) -> List[RetrievalResult]:
    """
    Hybrid retrieval combining embeddings and graph walking
    
    Args:
        db_path: Path to SQLite database
        query: Search query
        top_k: Number of top results to return
        walk_depth: Graph walk depth from embedding entry points
        
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
    
    # Load node details
    node_details = _load_node_details(db_path, list(all_node_ids))
    
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
        
        # Hybrid score: weighted combination
        hybrid_score = embedding_score * 0.5 + graph_score * 0.5
        
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
        # Basic info line
        lines = [f"{i}. [{result.node_type.upper()}] {result.content}"]
        
        # Add domain if available and not unknown
        if result.domain and result.domain != "unknown":
            lines.append(f"   Domain: {result.domain}")
        
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
    parser.add_argument("--db", default="/Users/bunny/.openclaw/workspace/cashew/data/graph.db", help="Database path")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    parser.add_argument("--walk-depth", type=int, default=2, help="Graph walk depth")
    parser.add_argument("--include-paths", action="store_true", help="Include paths in output")
    
    args = parser.parse_args()
    
    if args.command == "retrieve":
        print(f"🔍 Hybrid retrieval for: {args.query}")
        print()
        
        results = retrieve(args.db, args.query, args.top_k, args.walk_depth)
        
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