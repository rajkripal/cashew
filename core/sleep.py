#!/usr/bin/env python3
"""
Cashew Sleep Protocol
Memory consolidation, cross-linking, garbage collection, and core memory promotion
"""

import sqlite3
import json
import math
import random
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
from dataclasses import dataclass, asdict
import argparse
import sys
from datetime import datetime

DB_PATH = "/Users/bunny/.openclaw/workspace/cashew/data/graph.db"
SLEEP_LOG_PATH = "/Users/bunny/.openclaw/workspace/cashew/data/sleep_log.json"

@dataclass
class SleepEvent:
    timestamp: str
    event_type: str  # "cross_link", "dedup", "dream", "gc_decay", "core_promotion", "core_demotion"
    details: dict

@dataclass
class CrossLinkCandidate:
    node1_id: str
    node2_id: str
    similarity: float
    action: str  # "dedup", "cross_link", "contradiction"

@dataclass
class NodeMetrics:
    node_id: str
    branching_factor: int  # number of children
    cross_links: int
    retrieval_frequency: int  # how often referenced in derivations
    derivation_depth: int  # max depth from seeds
    composite_fitness: float

class SleepProtocol:
    """Handles memory consolidation during sleep cycles"""
    
    def __init__(self, db_path: str = DB_PATH, sleep_log_path: str = SLEEP_LOG_PATH):
        self.db_path = db_path
        self.sleep_log_path = sleep_log_path
        self.sleep_frequency = 10  # Sleep every N thoughts (tunable)
        self.dedup_threshold = 0.9
        self.cross_link_threshold = 0.7
        self.gc_threshold = 0.3  # Below this fitness score → decay
        self.events: List[SleepEvent] = []
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def _log_event(self, event_type: str, details: dict):
        """Log sleep event"""
        event = SleepEvent(
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            details=details
        )
        self.events.append(event)
    
    def _ensure_decayed_column(self):
        """Ensure decayed column exists in thought_nodes table"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Check if column exists
        cursor.execute("PRAGMA table_info(thought_nodes)")
        columns = [row[1] for row in cursor.fetchall()]
        
        if 'decayed' not in columns:
            cursor.execute("ALTER TABLE thought_nodes ADD COLUMN decayed INTEGER DEFAULT 0")
            conn.commit()
        
        conn.close()
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Simple text similarity based on word overlap
        TODO: Replace with embeddings for better similarity
        """
        # Tokenize and normalize
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Remove common stop words
        stop_words = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
                     'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
                     'to', 'was', 'were', 'will', 'with', 'i', 'you', 'they', 'we'}
        
        words1 = words1 - stop_words
        words2 = words2 - stop_words
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def find_cross_link_candidates(self) -> List[CrossLinkCandidate]:
        """Find nodes that should be cross-linked or deduplicated"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get all non-decayed nodes
        cursor.execute("""
            SELECT id, content, node_type 
            FROM thought_nodes 
            WHERE decayed = 0 OR decayed IS NULL
        """)
        nodes = cursor.fetchall()
        conn.close()
        
        candidates = []
        
        # Compare all pairs of nodes
        for i, (id1, content1, type1) in enumerate(nodes):
            for j, (id2, content2, type2) in enumerate(nodes):
                if i >= j:  # Avoid duplicate comparisons
                    continue
                
                similarity = self._text_similarity(content1, content2)
                
                if similarity >= self.dedup_threshold:
                    # High similarity - deduplication candidate
                    candidates.append(CrossLinkCandidate(
                        node1_id=id1,
                        node2_id=id2,
                        similarity=similarity,
                        action="dedup"
                    ))
                elif similarity >= self.cross_link_threshold:
                    # Medium similarity - cross-link candidate
                    candidates.append(CrossLinkCandidate(
                        node1_id=id1,
                        node2_id=id2,
                        similarity=similarity,
                        action="cross_link"
                    ))
        
        return candidates
    
    def cross_link_nodes(self, node1_id: str, node2_id: str, similarity: float, reasoning: str = ""):
        """Create cross-link edge between similar nodes"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Check if edge already exists
        cursor.execute("""
            SELECT COUNT(*) FROM derivation_edges 
            WHERE (parent_id = ? AND child_id = ?) OR (parent_id = ? AND child_id = ?)
        """, (node1_id, node2_id, node2_id, node1_id))
        
        if cursor.fetchone()[0] > 0:
            conn.close()
            return  # Edge already exists
        
        # Create bidirectional cross-link
        cursor.execute("""
            INSERT INTO derivation_edges (parent_id, child_id, relation, weight, reasoning)
            VALUES (?, ?, 'cross_link', ?, ?)
        """, (node1_id, node2_id, similarity, reasoning or f"Semantic similarity: {similarity:.2f}"))
        
        cursor.execute("""
            INSERT INTO derivation_edges (parent_id, child_id, relation, weight, reasoning)
            VALUES (?, ?, 'cross_link', ?, ?)
        """, (node2_id, node1_id, similarity, reasoning or f"Semantic similarity: {similarity:.2f}"))
        
        conn.commit()
        conn.close()
        
        self._log_event("cross_link", {
            "node1_id": node1_id,
            "node2_id": node2_id,
            "similarity": similarity,
            "reasoning": reasoning
        })
    
    def deduplicate_nodes(self, node1_id: str, node2_id: str, similarity: float):
        """Merge nearly identical nodes"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get both nodes
        cursor.execute("SELECT * FROM thought_nodes WHERE id IN (?, ?)", (node1_id, node2_id))
        nodes = cursor.fetchall()
        
        if len(nodes) != 2:
            conn.close()
            return
        
        # Keep the node with higher confidence
        node1, node2 = nodes
        keep_node = node1 if node1[4] >= node2[4] else node2  # confidence is index 4
        remove_node = node2 if node1[4] >= node2[4] else node1
        
        keep_id = keep_node[0]
        remove_id = remove_node[0]
        
        # Redirect all edges from remove_node to keep_node (skip if would create duplicate)
        cursor.execute("""
            UPDATE OR IGNORE derivation_edges 
            SET parent_id = ? 
            WHERE parent_id = ? AND child_id != ?
        """, (keep_id, remove_id, keep_id))
        
        cursor.execute("""
            UPDATE OR IGNORE derivation_edges 
            SET child_id = ? 
            WHERE child_id = ? AND parent_id != ?
        """, (keep_id, remove_id, keep_id))
        
        # Delete any remaining edges pointing to removed node
        cursor.execute("""
            DELETE FROM derivation_edges WHERE parent_id = ? OR child_id = ?
        """, (remove_id, remove_id))
        
        # Remove self-loops
        cursor.execute("""
            DELETE FROM derivation_edges 
            WHERE parent_id = ? AND child_id = ?
        """, (keep_id, keep_id))
        
        # Mark the duplicate as decayed instead of deleting
        cursor.execute("""
            UPDATE thought_nodes SET decayed = 1 WHERE id = ?
        """, (remove_id,))
        
        conn.commit()
        conn.close()
        
        self._log_event("dedup", {
            "kept_node": keep_id,
            "removed_node": remove_id,
            "similarity": similarity
        })
    
    def generate_dream_node(self, cross_links: List[CrossLinkCandidate]) -> Optional[str]:
        """Generate dream nodes connecting separate thought chains"""
        if not cross_links:
            return None
        
        # Find cross-links that bridge different chains
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Look for cross-links between nodes with different source files (indicating different chains)
        bridge_candidates = []
        for candidate in cross_links:
            cursor.execute("""
                SELECT source_file FROM thought_nodes 
                WHERE id IN (?, ?)
            """, (candidate.node1_id, candidate.node2_id))
            sources = [row[0] for row in cursor.fetchall()]
            
            if len(set(sources)) > 1:  # Different sources
                bridge_candidates.append(candidate)
        
        if not bridge_candidates:
            conn.close()
            return None
        
        # Pick the strongest bridge
        best_bridge = max(bridge_candidates, key=lambda c: c.similarity)
        
        # Get the connected nodes
        cursor.execute("""
            SELECT content, node_type FROM thought_nodes 
            WHERE id IN (?, ?)
        """, (best_bridge.node1_id, best_bridge.node2_id))
        
        nodes = cursor.fetchall()
        if len(nodes) != 2:
            conn.close()
            return None
        
        # Generate dream content
        content1, type1 = nodes[0]
        content2, type2 = nodes[1]
        
        dream_content = f"Connection discovered: '{content1[:50]}...' relates to '{content2[:50]}...'"
        
        # Create dream node
        import hashlib
        dream_id = hashlib.sha256(dream_content.encode()).hexdigest()[:12]
        
        cursor.execute("""
            INSERT OR REPLACE INTO thought_nodes 
            (id, content, node_type, timestamp, confidence, mood_state, metadata, source_file)
            VALUES (?, ?, 'dream', ?, 0.7, 'dreamy', '{}', 'sleep_protocol')
        """, (dream_id, dream_content, datetime.now().isoformat()))
        
        # Connect dream to both nodes
        cursor.execute("""
            INSERT INTO derivation_edges (parent_id, child_id, relation, weight, reasoning)
            VALUES (?, ?, 'derived_from', ?, 'Dream synthesis')
        """, (best_bridge.node1_id, dream_id, best_bridge.similarity))
        
        cursor.execute("""
            INSERT INTO derivation_edges (parent_id, child_id, relation, weight, reasoning)
            VALUES (?, ?, 'derived_from', ?, 'Dream synthesis')
        """, (best_bridge.node2_id, dream_id, best_bridge.similarity))
        
        conn.commit()
        conn.close()
        
        self._log_event("dream", {
            "dream_id": dream_id,
            "dream_content": dream_content,
            "bridged_nodes": [best_bridge.node1_id, best_bridge.node2_id],
            "similarity": best_bridge.similarity
        })
        
        return dream_id
    
    def calculate_node_metrics(self) -> Dict[str, NodeMetrics]:
        """Calculate fitness metrics for all nodes"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get all non-decayed nodes
        cursor.execute("""
            SELECT id, node_type FROM thought_nodes 
            WHERE decayed = 0 OR decayed IS NULL
        """)
        nodes = {row[0]: row[1] for row in cursor.fetchall()}
        
        metrics = {}
        
        for node_id, node_type in nodes.items():
            # Branching factor (outgoing edges)
            cursor.execute("""
                SELECT COUNT(*) FROM derivation_edges WHERE parent_id = ?
            """, (node_id,))
            branching_factor = cursor.fetchone()[0]
            
            # Cross-links (bidirectional cross_link edges)
            cursor.execute("""
                SELECT COUNT(*) FROM derivation_edges 
                WHERE (parent_id = ? OR child_id = ?) AND relation = 'cross_link'
            """, (node_id, node_id))
            cross_links = cursor.fetchone()[0]
            
            # Retrieval frequency (how often this node appears as parent in derivations)
            cursor.execute("""
                SELECT COUNT(*) FROM derivation_edges WHERE parent_id = ?
            """, (node_id,))
            retrieval_frequency = cursor.fetchone()[0]
            
            # Derivation depth (max depth from seeds)
            derivation_depth = self._calculate_depth_from_seeds(node_id)
            
            # Composite fitness score
            # Seeds get bonus, core memories get bonus, depth adds value
            base_score = branching_factor + cross_links * 0.5 + derivation_depth * 0.1
            
            if node_type == "seed":
                base_score *= 2.0  # Seeds are important
            elif node_type == "core_memory":
                base_score *= 1.5  # Core memories are valuable
            
            composite_fitness = base_score
            
            metrics[node_id] = NodeMetrics(
                node_id=node_id,
                branching_factor=branching_factor,
                cross_links=cross_links,
                retrieval_frequency=retrieval_frequency,
                derivation_depth=derivation_depth,
                composite_fitness=composite_fitness
            )
        
        conn.close()
        return metrics
    
    def _calculate_depth_from_seeds(self, node_id: str) -> int:
        """Calculate max depth from seed nodes"""
        from .traversal import TraversalEngine
        
        engine = TraversalEngine(self.db_path)
        chain = engine.why(node_id, max_depth=20)
        
        if not chain or any("error" in step or "cycle_detected" in step for step in chain):
            return 0
        
        def get_max_depth(step: dict, current_depth: int = 0) -> int:
            max_d = current_depth
            if "derived_from" in step:
                for derivation in step["derived_from"]:
                    if "parent_chain" in derivation:
                        for parent_step in derivation["parent_chain"]:
                            depth = get_max_depth(parent_step, current_depth + 1)
                            max_d = max(max_d, depth)
            return max_d
        
        return get_max_depth(chain[0]) if chain else 0
    
    def garbage_collect(self, metrics: Dict[str, NodeMetrics], k_nodes: int = 20) -> List[str]:
        """
        Randomly select K nodes and decay those below fitness threshold
        Random selection introduces noise that forces rederivation through novel paths
        """
        if len(metrics) <= k_nodes:
            return []  # Don't GC if we have too few nodes
        
        # Randomly select K nodes
        node_ids = list(metrics.keys())
        selected = random.sample(node_ids, min(k_nodes, len(node_ids)))
        
        decayed_nodes = []
        conn = self._get_connection()
        cursor = conn.cursor()
        
        for node_id in selected:
            metric = metrics[node_id]
            
            # Don't decay seeds or core memories
            cursor.execute("SELECT node_type FROM thought_nodes WHERE id = ?", (node_id,))
            node_type = cursor.fetchone()
            if node_type and node_type[0] in ("seed", "core_memory"):
                continue
            
            if metric.composite_fitness < self.gc_threshold:
                # Mark as decayed
                cursor.execute("UPDATE thought_nodes SET decayed = 1 WHERE id = ?", (node_id,))
                decayed_nodes.append(node_id)
                
                self._log_event("gc_decay", {
                    "node_id": node_id,
                    "fitness_score": metric.composite_fitness,
                    "threshold": self.gc_threshold,
                    "metrics": asdict(metric)
                })
        
        conn.commit()
        conn.close()
        
        return decayed_nodes
    
    def promote_core_memories(self, metrics: Dict[str, NodeMetrics]) -> Tuple[List[str], List[str]]:
        """
        Promote/demote nodes based on network metrics
        Top √(total_nodes) nodes get core_memory status
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Calculate target number of core memories
        total_nodes = len(metrics)
        target_core_memories = int(math.sqrt(total_nodes))
        
        # Get current core memories
        cursor.execute("SELECT id FROM thought_nodes WHERE node_type = 'core_memory'")
        current_core = set(row[0] for row in cursor.fetchall())
        
        # Rank all nodes by composite fitness
        ranked_nodes = sorted(metrics.values(), key=lambda m: m.composite_fitness, reverse=True)
        
        # Top nodes should be core memories
        should_be_core = set(m.node_id for m in ranked_nodes[:target_core_memories])
        
        # Promote new core memories
        promotions = should_be_core - current_core
        for node_id in promotions:
            cursor.execute("UPDATE thought_nodes SET node_type = 'core_memory' WHERE id = ?", (node_id,))
            self._log_event("core_promotion", {
                "node_id": node_id,
                "fitness_score": metrics[node_id].composite_fitness,
                "rank": next(i for i, m in enumerate(ranked_nodes) if m.node_id == node_id) + 1
            })
        
        # Demote old core memories
        demotions = current_core - should_be_core
        for node_id in demotions:
            # Demote to 'derived' unless it's a seed
            cursor.execute("SELECT node_type FROM thought_nodes WHERE id = ?", (node_id,))
            current_type = cursor.fetchone()[0]
            
            if current_type != "seed":
                cursor.execute("UPDATE thought_nodes SET node_type = 'derived' WHERE id = ?", (node_id,))
                self._log_event("core_demotion", {
                    "node_id": node_id,
                    "fitness_score": metrics.get(node_id, NodeMetrics("", 0, 0, 0, 0, 0)).composite_fitness,
                    "reason": "Below core memory threshold"
                })
        
        conn.commit()
        conn.close()
        
        return list(promotions), list(demotions)
    
    def run_sleep_cycle(self) -> Dict:
        """Run a complete sleep cycle"""
        print("💤 Starting sleep cycle...")
        
        # Ensure schema is up to date
        self._ensure_decayed_column()
        
        # 1. Cross-linking phase
        print("🔗 Finding cross-link candidates...")
        candidates = self.find_cross_link_candidates()
        
        cross_links_created = 0
        dedups_performed = 0
        
        for candidate in candidates:
            if candidate.action == "dedup":
                self.deduplicate_nodes(candidate.node1_id, candidate.node2_id, candidate.similarity)
                dedups_performed += 1
            elif candidate.action == "cross_link":
                self.cross_link_nodes(candidate.node1_id, candidate.node2_id, candidate.similarity)
                cross_links_created += 1
        
        # 2. Dream generation
        print("💭 Generating dream nodes...")
        cross_link_candidates = [c for c in candidates if c.action == "cross_link"]
        dream_id = self.generate_dream_node(cross_link_candidates)
        
        # 3. Calculate metrics
        print("📊 Calculating node metrics...")
        metrics = self.calculate_node_metrics()
        
        # 4. Garbage collection
        print("🗑️  Running garbage collection...")
        decayed_nodes = self.garbage_collect(metrics)
        
        # 5. Core memory promotion/demotion
        print("⭐ Updating core memories...")
        promotions, demotions = self.promote_core_memories(metrics)
        
        # 6. Build summary before saving (save clears events)
        events_count = len(self.events)
        self.save_sleep_log()
        
        summary = {
            "cross_links_created": cross_links_created,
            "deduplications": dedups_performed,
            "dream_nodes_created": 1 if dream_id else 0,
            "nodes_decayed": len(decayed_nodes),
            "core_promotions": len(promotions),
            "core_demotions": len(demotions),
            "total_nodes": len(metrics),
            "events_logged": events_count
        }
        
        print(f"✅ Sleep cycle complete: {summary}")
        return summary
    
    def save_sleep_log(self):
        """Save sleep events to log file"""
        try:
            # Load existing log
            existing_log = []
            try:
                with open(self.sleep_log_path, 'r') as f:
                    existing_log = json.load(f)
            except FileNotFoundError:
                pass
            
            # Add new events
            new_events = [asdict(event) for event in self.events]
            existing_log.extend(new_events)
            
            # Save back
            with open(self.sleep_log_path, 'w') as f:
                json.dump(existing_log, f, indent=2)
            
            self.events = []  # Clear processed events
            
        except Exception as e:
            print(f"Warning: Could not save sleep log: {e}")


def main():
    """CLI interface for sleep protocol"""
    parser = argparse.ArgumentParser(description="Cashew Sleep Protocol")
    parser.add_argument("command", choices=["run", "status"], help="Command to run")
    parser.add_argument("--frequency", type=int, default=10, help="Sleep every N thoughts")
    parser.add_argument("--gc-nodes", type=int, default=20, help="Number of nodes to consider for GC")
    
    args = parser.parse_args()
    
    protocol = SleepProtocol()
    protocol.sleep_frequency = args.frequency
    
    if args.command == "run":
        summary = protocol.run_sleep_cycle()
        print(f"\n💤 Sleep cycle completed:")
        for key, value in summary.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
    
    elif args.command == "status":
        # Show current sleep statistics
        try:
            with open(protocol.sleep_log_path, 'r') as f:
                events = json.load(f)
            
            print(f"\n📊 Sleep Protocol Status:")
            print(f"Total sleep events: {len(events)}")
            
            event_counts = defaultdict(int)
            for event in events:
                event_counts[event['event_type']] += 1
            
            for event_type, count in event_counts.items():
                print(f"  {event_type.replace('_', ' ').title()}: {count}")
            
        except FileNotFoundError:
            print("No sleep log found. Run a sleep cycle first.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())