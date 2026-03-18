#!/usr/bin/env python3
"""
Cashew Sleep Protocol
Memory consolidation, cross-linking, garbage collection, and core memory promotion
"""

import sqlite3
import json
import math
import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
from dataclasses import dataclass, asdict
import argparse
import sys
from datetime import datetime
import logging

logger = logging.getLogger("cashew.sleep")

# Database path is now configurable via environment variable or CLI
from .config import get_db_path
DEFAULT_SLEEP_LOG_PATH = "./data/sleep_log.json"

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
    
    def __init__(self, db_path: str = None, sleep_log_path: str = None):
        if db_path is None:
            db_path = get_db_path()
        if sleep_log_path is None:
            sleep_log_path = DEFAULT_SLEEP_LOG_PATH
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
    
    def _load_embedding_sim_cache(self):
        """Load all embeddings into a similarity cache for fast pairwise lookups"""
        if hasattr(self, '_sim_cache'):
            return
        
        from .clustering import load_embeddings, cosine_similarity
        
        node_ids, vectors, _ = load_embeddings(self.db_path)
        self._embed_ids = node_ids
        self._embed_vectors = vectors
        self._embed_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        self._cosine_similarity = cosine_similarity
    
    def _text_similarity(self, text1: str, text2: str, node1_id: str = None, node2_id: str = None) -> float:
        """
        Similarity between two nodes. Uses cosine similarity on embeddings 
        when node IDs are provided, falls back to Jaccard on text.
        """
        # Try embedding-based similarity first
        if node1_id and node2_id:
            try:
                self._load_embedding_sim_cache()
                idx1 = self._embed_id_to_idx.get(node1_id)
                idx2 = self._embed_id_to_idx.get(node2_id)
                if idx1 is not None and idx2 is not None:
                    return self._cosine_similarity(
                        self._embed_vectors[idx1], 
                        self._embed_vectors[idx2]
                    )
            except Exception as e:
                logger.debug(f"Embedding similarity failed, falling back to text: {e}")
        
        # Fallback: Jaccard on words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        stop_words = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
                     'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
                     'to', 'was', 'were', 'will', 'with', 'i', 'you', 'they', 'we'}
        
        words1 = words1 - stop_words
        words2 = words2 - stop_words
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def find_cross_link_candidates(self) -> List[CrossLinkCandidate]:
        """
        Find nodes that should be cross-linked or deduplicated.
        Uses embedding cosine similarity for accurate comparison.
        Optimized: computes full similarity matrix once, then filters.
        """
        try:
            from .clustering import load_embeddings
            from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_sim
            
            node_ids, vectors, node_meta = load_embeddings(self.db_path)
            if len(node_ids) < 2:
                return []
            
            # Compute full pairwise similarity matrix (N x N)
            sim_matrix = sklearn_cosine_sim(vectors)
            
            candidates = []
            
            # Only check upper triangle (avoid duplicates)
            for i in range(len(node_ids)):
                for j in range(i + 1, len(node_ids)):
                    similarity = float(sim_matrix[i, j])
                    
                    if similarity >= self.dedup_threshold:
                        candidates.append(CrossLinkCandidate(
                            node1_id=node_ids[i],
                            node2_id=node_ids[j],
                            similarity=similarity,
                            action="dedup"
                        ))
                    elif similarity >= self.cross_link_threshold:
                        candidates.append(CrossLinkCandidate(
                            node1_id=node_ids[i],
                            node2_id=node_ids[j],
                            similarity=similarity,
                            action="cross_link"
                        ))
            
            logger.info(f"Found {len(candidates)} cross-link candidates from {len(node_ids)} nodes")
            return candidates
            
        except Exception as e:
            logger.warning(f"Embedding-based cross-link failed, falling back to text: {e}")
            return self._find_cross_link_candidates_text_fallback()
    
    def _find_cross_link_candidates_text_fallback(self) -> List[CrossLinkCandidate]:
        """Fallback: text-based cross-link detection"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, content, node_type 
            FROM thought_nodes 
            WHERE decayed = 0 OR decayed IS NULL
        """)
        nodes = cursor.fetchall()
        conn.close()
        
        candidates = []
        for i, (id1, content1, type1) in enumerate(nodes):
            for j, (id2, content2, type2) in enumerate(nodes):
                if i >= j:
                    continue
                similarity = self._text_similarity(content1, content2)
                if similarity >= self.dedup_threshold:
                    candidates.append(CrossLinkCandidate(id1, id2, similarity, "dedup"))
                elif similarity >= self.cross_link_threshold:
                    candidates.append(CrossLinkCandidate(id1, id2, similarity, "cross_link"))
        
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
            INSERT OR IGNORE INTO derivation_edges (parent_id, child_id, weight, reasoning)
            VALUES (?, ?, ?, ?)
        """, (node1_id, node2_id, similarity, f"cross_link - {reasoning or f'Semantic similarity: {similarity:.2f}'}"))
        
        cursor.execute("""
            INSERT OR IGNORE INTO derivation_edges (parent_id, child_id, weight, reasoning)
            VALUES (?, ?, ?, ?)
        """, (node2_id, node1_id, similarity, f"cross_link - {reasoning or f'Semantic similarity: {similarity:.2f}'}"))
        
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
            INSERT OR IGNORE INTO derivation_edges (parent_id, child_id, weight, reasoning)
            VALUES (?, ?, ?, ?)
        """, (best_bridge.node1_id, dream_id, best_bridge.similarity, 'derived_from - Dream synthesis'))
        
        cursor.execute("""
            INSERT OR IGNORE INTO derivation_edges (parent_id, child_id, weight, reasoning)
            VALUES (?, ?, ?, ?)
        """, (best_bridge.node2_id, dream_id, best_bridge.similarity, 'derived_from - Dream synthesis'))
        
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
                WHERE (parent_id = ? OR child_id = ?) AND reasoning LIKE '%cross_link%'
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
            cursor.execute("SELECT node_type, source_file FROM thought_nodes WHERE id = ?", (node_id,))
            row = cursor.fetchone()
            if not row:
                continue
            node_type, source_file = row
            if node_type in ("seed", "core_memory"):
                continue
            
            # Think cycle nodes get a higher decay threshold — 
            # they need to earn their place more than human-extracted knowledge
            is_think_cycle = source_file and "think_cycle" in str(source_file)
            effective_threshold = self.gc_threshold * 1.5 if is_think_cycle else self.gc_threshold
            
            if metric.composite_fitness < effective_threshold:
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
    
    def run_sleep_cycle(self, model_fn=None) -> Dict:
        """
        Run a complete sleep cycle
        
        Args:
            model_fn: Optional model function for LLM-powered operations (hotspot summaries).
                     If None, LLM-dependent features will use fallbacks or be skipped.
        """
        print("💤 Starting sleep cycle...")
        
        if not model_fn:
            print("   ⚠️  No LLM access - hotspot summaries will use fallbacks")
        
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
        
        # 4.5. Permanence evaluation (after decay, before core memory operations)
        print("🔒 Evaluating node permanence...")
        permanence_stats = self.evaluate_permanence()
        
        # 5. Core memory promotion/demotion
        print("⭐ Updating core memories...")
        promotions, demotions = self.promote_core_memories(metrics)
        
        # 6. Clustering & hotspot maintenance
        print("📍 Running cluster detection & hotspot maintenance...")
        clustering_results = self._run_clustering_phase(model_fn)
        
        # 7. Build summary before saving (save clears events)
        events_count = len(self.events)
        self.save_sleep_log()
        
        summary = {
            "cross_links_created": cross_links_created,
            "deduplications": dedups_performed,
            "permanence_stats": permanence_stats,
            "dream_nodes_created": 1 if dream_id else 0,
            "nodes_decayed": len(decayed_nodes),
            "core_promotions": len(promotions),
            "core_demotions": len(demotions),
            "clusters_found": clustering_results.get("clusters_found", 0),
            "new_hotspots": clustering_results.get("new_hotspots_created", 0),
            "stale_hotspots": clustering_results.get("stale_hotspots_found", 0),
            "total_nodes": len(metrics),
            "events_logged": events_count
        }
        
        print(f"✅ Sleep cycle complete: {summary}")
        return summary
    
    def _run_clustering_phase(self, model_fn=None) -> Dict:
        """Run cluster detection and hotspot maintenance as part of sleep"""
        try:
            from .clustering import run_clustering_cycle
            
            if not model_fn:
                logger.debug("No model function provided - hotspot summaries will use fallbacks")
            
            # Use recursive clustering with max cluster size = 15
            results = run_clustering_cycle(self.db_path, model_fn=model_fn, max_cluster_size=15)
            
            # Log events
            if results["new_hotspots_created"] > 0:
                self._log_event("clustering", {
                    "action": "new_hotspots",
                    "count": results["new_hotspots_created"],
                    "clusters_found": results["clusters_found"]
                })
            if results.get("parent_hotspots_created", 0) > 0:
                self._log_event("clustering", {
                    "action": "parent_hotspots",
                    "count": results["parent_hotspots_created"],
                    "hierarchical_edges": results.get("hierarchical_edges_created", 0)
                })
            if results["stale_hotspots_found"] > 0:
                self._log_event("clustering", {
                    "action": "stale_detected",
                    "count": results["stale_hotspots_found"]
                })
            
            return results
            
        except Exception as e:
            logger.warning(f"Clustering phase failed (non-fatal): {e}")
            return {"clusters_found": 0, "new_hotspots_created": 0, "stale_hotspots_found": 0}

    def evaluate_permanence(self) -> Dict:
        """
        Evaluate and update permanence for all nodes in the graph.
        This runs during sleep cycle after decay to mark valuable nodes permanent.
        
        Returns:
            Dict with permanence evaluation statistics
        """
        try:
            # Import the permanence module
            from .permanence import evaluate_all_permanence
            
            # Run the evaluation
            stats = evaluate_all_permanence(self.db_path)
            
            # Log the permanence evaluation event
            self._log_event("permanence_evaluation", {
                "nodes_evaluated": stats.get('nodes_evaluated', 0),
                "nodes_made_permanent": stats.get('nodes_made_permanent', 0),
                "nodes_lost_permanence": stats.get('nodes_lost_permanence', 0),
                "hotspots_made_permanent": stats.get('hotspots_made_permanent', 0),
                "hotspots_lost_permanence": stats.get('hotspots_lost_permanence', 0)
            })
            
            return stats
            
        except Exception as e:
            logger.warning(f"Permanence evaluation failed (non-fatal): {e}")
            return {
                'nodes_evaluated': 0,
                'nodes_made_permanent': 0,
                'nodes_lost_permanence': 0,
                'hotspots_made_permanent': 0,
                'hotspots_lost_permanence': 0
            }

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


def run_sleep_cycle(db_path: str = None, model_fn = None) -> Dict:
    """
    Public function to run a sleep cycle on the graph database.
    
    Args:
        db_path: Path to the SQLite database. Uses config default if None.
        model_fn: Optional model function for LLM-powered operations. If None, some features will be skipped.
        
    Returns:
        Dict with sleep cycle statistics
    """
    if db_path is None:
        db_path = get_db_path()
    
    protocol = SleepProtocol(db_path)
    return protocol.run_sleep_cycle(model_fn)


if __name__ == "__main__":
    sys.exit(main())