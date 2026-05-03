#!/usr/bin/env python3
"""
Cashew Traversal Engine
Provides graph navigation functions: why(), how(), audit()
"""

import sqlite3
import json
from typing import List, Dict, Optional, Tuple, Set
from collections import deque, defaultdict
from dataclasses import dataclass
import sys
import argparse

# Database path is now configurable via environment variable or CLI
from .config import get_db_path

@dataclass
class ThoughtNode:
    id: str
    content: str
    node_type: str
    mood_state: str
    metadata: dict
    source_file: str
    timestamp: str

@dataclass
class DerivationEdge:
    parent_id: str
    child_id: str
    weight: float
    reasoning: str

@dataclass
class AuditReport:
    cycles: List[List[str]]
    contradictions: List[Tuple[str, str, str]]  # (node1, node2, reasoning)
    orphan_nodes: List[str]  # nodes with no parents that aren't seeds
    weak_chains: List[Tuple[str, float]]  # (node_id, avg_chain_weight)

class TraversalEngine:
    """Graph traversal and audit functionality"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = get_db_path()
        self.db_path = db_path
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def _load_node(self, node_id: str) -> Optional[ThoughtNode]:
        """Load a single node from database"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, content, node_type, mood_state, metadata, source_file, timestamp
            FROM thought_nodes
            WHERE id = ?
        """, (node_id,))

        row = cursor.fetchone()
        conn.close()

        if row:
            return ThoughtNode(
                id=row[0],
                content=row[1],
                node_type=row[2],
                mood_state=row[3],
                metadata=json.loads(row[4]) if row[4] else {},
                source_file=row[5],
                timestamp=row[6]
            )
        return None
    
    def _get_parents(self, node_id: str) -> List[Tuple[str, DerivationEdge]]:
        """Get parent nodes with edge information"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT de.parent_id, de.weight, de.reasoning,
                   tn.content, tn.node_type, tn.mood_state,
                   tn.metadata, tn.source_file, tn.timestamp
            FROM derivation_edges de
            JOIN thought_nodes tn ON de.parent_id = tn.id
            WHERE de.child_id = ?
            ORDER BY de.weight DESC
        """, (node_id,))

        results = []
        for row in cursor.fetchall():
            edge = DerivationEdge(
                parent_id=row[0],
                child_id=node_id,
                weight=row[1],
                reasoning=row[2]
            )
            parent = ThoughtNode(
                id=row[0],
                content=row[3],
                node_type=row[4],
                mood_state=row[5],
                metadata=json.loads(row[6]) if row[6] else {},
                source_file=row[7],
                timestamp=row[8]
            )
            results.append((parent, edge))
        
        conn.close()
        return results
    
    def _get_children(self, node_id: str) -> List[str]:
        """Get child node IDs"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT child_id FROM derivation_edges WHERE parent_id = ?
        """, (node_id,))
        
        results = [row[0] for row in cursor.fetchall()]
        conn.close()
        return results
    
    def why(self, node_id: str, max_depth: int = 50) -> List[Dict]:
        """
        Returns the full derivation chain back to seed nodes
        
        Args:
            node_id: The node to trace back from
            max_depth: Maximum traversal depth to prevent infinite loops
            
        Returns:
            List of derivation steps, each containing node and edge info
        """
        visited = set()
        derivation_chain = []
        
        def traverse(current_id: str, depth: int = 0, path: List[str] = None) -> List[Dict]:
            if path is None:
                path = []
            
            if depth > max_depth:
                return [{"error": f"Max depth {max_depth} exceeded", "path": path}]
            
            if current_id in visited:
                return [{"cycle_detected": current_id, "path": path}]
            
            if current_id in path:
                return [{"immediate_cycle": current_id, "path": path}]
            
            visited.add(current_id)
            new_path = path + [current_id]
            
            # Load current node
            current_node = self._load_node(current_id)
            if not current_node:
                return [{"error": f"Node not found: {current_id}"}]
            
            # Get parents
            parents = self._get_parents(current_id)
            
            if not parents:
                # This is a root node (check if it's actually a seed)
                is_actual_seed = current_node.node_type == "seed"
                return [{
                    "node": {
                        "id": current_node.id,
                        "content": current_node.content,
                        "type": current_node.node_type,
                        "source": current_node.source_file
                    },
                    "depth": depth,
                    "is_seed": is_actual_seed
                }]
            
            # Traverse parents
            chain = [{
                "node": {
                    "id": current_node.id,
                    "content": current_node.content,
                    "type": current_node.node_type,
                    "source": current_node.source_file
                },
                "depth": depth,
                "is_seed": False,
                "derived_from": []
            }]
            
            for parent_node, edge in parents:
                parent_chain = traverse(parent_node.id, depth + 1, new_path)
                chain[0]["derived_from"].append({
                    "weight": edge.weight,
                    "reasoning": edge.reasoning,
                    "parent_chain": parent_chain
                })
            
            return chain
        
        return traverse(node_id)
    
    def how(self, node_a: str, node_b: str) -> Optional[List[Dict]]:
        """
        Find shortest path between two nodes
        
        Args:
            node_a: Starting node ID
            node_b: Target node ID
            
        Returns:
            Shortest path as list of nodes and edges, or None if no path exists
        """
        if node_a == node_b:
            node = self._load_node(node_a)
            return [{
                "node": {
                    "id": node.id,
                    "content": node.content,
                    "type": node.node_type,
                },
                "distance": 0
            }] if node else None
        
        # BFS to find shortest path
        queue = deque([(node_a, [node_a])])
        visited = {node_a}
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        while queue:
            current_id, path = queue.popleft()
            
            # Get all connected nodes (both directions)
            cursor.execute("""
                SELECT child_id as connected_id, 'outgoing' as direction, weight, reasoning
                FROM derivation_edges WHERE parent_id = ?
                UNION
                SELECT parent_id as connected_id, 'incoming' as direction, weight, reasoning
                FROM derivation_edges WHERE child_id = ?
            """, (current_id, current_id))
            
            for row in cursor.fetchall():
                connected_id, direction, weight, reasoning = row
                
                if connected_id == node_b:
                    # Found target - build path
                    full_path = path + [connected_id]
                    result = []
                    
                    for i, node_id in enumerate(full_path):
                        node = self._load_node(node_id)
                        step = {
                            "node": {
                                "id": node.id,
                                "content": node.content,
                                "type": node.node_type,
                            },
                            "distance": i
                        }
                        
                        if i > 0:
                            step["connection"] = {
                                "weight": weight,
                                "reasoning": reasoning,
                                "direction": direction
                            }
                        
                        result.append(step)
                    
                    conn.close()
                    return result
                
                if connected_id not in visited:
                    visited.add(connected_id)
                    queue.append((connected_id, path + [connected_id]))
        
        conn.close()
        return None  # No path found
    
    def audit(self) -> AuditReport:
        """
        Detect cycles, contradictions, orphan nodes, and weak chains
        
        Returns:
            AuditReport with all detected issues
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get all nodes and edges
        cursor.execute("SELECT id, node_type FROM thought_nodes")
        nodes = {row[0]: row[1] for row in cursor.fetchall()}
        
        cursor.execute("SELECT parent_id, child_id, weight, reasoning FROM derivation_edges")
        edges = [(row[0], row[1], row[2], row[3]) for row in cursor.fetchall()]
        
        # Build adjacency lists
        graph = defaultdict(list)
        reverse_graph = defaultdict(list)
        
        for parent, child, weight, reasoning in edges:
            graph[parent].append((child, reasoning, weight))
            reverse_graph[child].append((parent, reasoning, weight))
        
        # 1. Detect cycles using DFS
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs_cycle(node: str, path: List[str]):
            if node in rec_stack:
                # Found cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycles.append(cycle)
                return
            
            if node in visited:
                return
            
            visited.add(node)
            rec_stack.add(node)
            
            for child, _, _ in graph.get(node, []):
                dfs_cycle(child, path + [node])
            
            rec_stack.remove(node)
        
        for node_id in nodes:
            if node_id not in visited:
                dfs_cycle(node_id, [])
        
        # 2. Find contradictions by checking reasoning text
        contradictions = []
        for parent, child, weight, reasoning in edges:
            if any(word in reasoning.lower() for word in ['contradict', 'conflict', 'oppose']):
                parent_node = self._load_node(parent)
                child_node = self._load_node(child)
                contradictions.append((
                    parent_node.content if parent_node else parent,
                    child_node.content if child_node else child,
                    f"Contradiction detected: {reasoning} (weight: {weight})"
                ))
        
        # 3. Find orphan nodes (no parents, not seeds)
        orphan_nodes = []
        for node_id, node_type in nodes.items():
            if node_id not in reverse_graph and node_type != "seed":
                orphan_nodes.append(node_id)
        
        # 4. Find weak chains (low average weight to roots)
        weak_chains = []
        for node_id in nodes:
            if nodes[node_id] != "seed":  # Skip seeds
                # Calculate average weight to all seed nodes
                chain = self.why(node_id)
                if chain and not any("error" in step or "cycle_detected" in step for step in chain):
                    weights = []
                    self._collect_weights(chain[0], weights)
                    if weights:
                        avg_weight = sum(weights) / len(weights)
                        if avg_weight < 0.5:  # Threshold for "weak"
                            weak_chains.append((node_id, avg_weight))
        
        # Sort weak chains by weight
        weak_chains.sort(key=lambda x: x[1])
        
        conn.close()
        return AuditReport(
            cycles=cycles,
            contradictions=contradictions,
            orphan_nodes=orphan_nodes,
            weak_chains=weak_chains
        )
    
    def _collect_weights(self, chain_step: Dict, weights: List[float]):
        """Recursively collect weights from derivation chain"""
        if "derived_from" in chain_step:
            for derivation in chain_step["derived_from"]:
                weights.append(derivation["weight"])
                if "parent_chain" in derivation:
                    for parent_step in derivation["parent_chain"]:
                        self._collect_weights(parent_step, weights)


def main():
    """CLI interface for traversal functions"""
    parser = argparse.ArgumentParser(description="Cashew Traversal Engine")
    parser.add_argument("command", choices=["why", "how", "audit"], help="Command to run")
    parser.add_argument("--node", help="Node ID for why command")
    parser.add_argument("--node-a", help="First node ID for how command")
    parser.add_argument("--node-b", help="Second node ID for how command")
    parser.add_argument("--max-depth", type=int, default=50, help="Maximum traversal depth")
    
    args = parser.parse_args()
    
    engine = TraversalEngine()
    
    if args.command == "why":
        if not args.node:
            print("Error: --node required for why command")
            return 1
        
        chain = engine.why(args.node, args.max_depth)
        print(f"\n🤔 WHY: {args.node[:12]}...")
        print("=" * 50)
        print(json.dumps(chain, indent=2))
    
    elif args.command == "how":
        if not args.node_a or not args.node_b:
            print("Error: --node-a and --node-b required for how command")
            return 1
        
        path = engine.how(args.node_a, args.node_b)
        if path:
            print(f"\n🛤️  HOW: {args.node_a[:12]}... → {args.node_b[:12]}...")
            print("=" * 50)
            print(json.dumps(path, indent=2))
        else:
            print(f"No path found between {args.node_a} and {args.node_b}")
    
    elif args.command == "audit":
        report = engine.audit()
        print("\n🔍 AUDIT REPORT")
        print("=" * 50)
        
        if report.cycles:
            print(f"\n⚠️  CYCLES DETECTED ({len(report.cycles)}):")
            for i, cycle in enumerate(report.cycles):
                print(f"  {i+1}. {' → '.join(node[:12] for node in cycle)}")
        
        if report.contradictions:
            print(f"\n💥 CONTRADICTIONS DETECTED ({len(report.contradictions)}):")
            for i, (node1, node2, reasoning) in enumerate(report.contradictions):
                print(f"  {i+1}. {node1[:60]}...")
                print(f"     ⟷ {node2[:60]}...")
                print(f"     ({reasoning})")
        
        if report.orphan_nodes:
            print(f"\n🏝️  ORPHAN NODES ({len(report.orphan_nodes)}):")
            for node_id in report.orphan_nodes:
                node = engine._load_node(node_id)
                print(f"  - {node.content[:60]}..." if node else f"  - {node_id}")
        
        if report.weak_chains:
            print(f"\n🔗 WEAK CHAINS ({len(report.weak_chains)}):")
            for node_id, avg_weight in report.weak_chains[:10]:  # Show top 10
                node = engine._load_node(node_id)
                print(f"  [{avg_weight:.2f}] {node.content[:60]}..." if node else f"  [{avg_weight:.2f}] {node_id}")
        
        if not any([report.cycles, report.contradictions, report.orphan_nodes, report.weak_chains]):
            print("\n✅ No issues detected!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())