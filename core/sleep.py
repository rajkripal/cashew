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
from .config import get_db_path, config
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
        self.dedup_threshold = 0.82
        self.cross_link_threshold = 0.7
        # GC settings from config
        self.gc_mode = getattr(config, 'gc_mode', 'soft')
        self.gc_threshold = getattr(config, 'gc_threshold', 0.05)
        self.gc_grace_days = getattr(config, 'gc_grace_days', 7)
        self.gc_protect_types = getattr(config, 'gc_protect_types', ['seed', 'core_memory'])
        self.gc_think_cycle_penalty = getattr(config, 'gc_think_cycle_penalty', 1.5)
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
        
        from .graph_utils import load_embeddings, cosine_similarity
        
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
            from .graph_utils import load_embeddings
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
        
        # Mark the duplicate as decayed instead of deleting.
        # Skip if the loser was promoted to permanent — permanent nodes must
        # never end up with decayed=1 (violates integrity check).
        cursor.execute("""
            UPDATE thought_nodes SET decayed = 1
            WHERE id = ? AND (permanent IS NULL OR permanent = 0)
        """, (remove_id,))
        
        conn.commit()
        conn.close()
        
        self._log_event("dedup", {
            "kept_node": keep_id,
            "removed_node": remove_id,
            "similarity": similarity
        })
    
    def find_merge_clusters(self, candidates: List[CrossLinkCandidate]) -> List[List[str]]:
        """
        Find maximal cliques among dedup candidates. A cluster is a set of node ids
        where every pair is in the candidate set above dedup_threshold.

        Uses Bron-Kerbosch (without pivoting). Cluster sizes are tiny (2-5) in
        practice so the simple recursion is fine. Connected-component clustering
        is deliberately avoided to prevent chaining unrelated nodes.
        """
        # Build adjacency from dedup candidates only
        adj: Dict[str, Set[str]] = defaultdict(set)
        for c in candidates:
            if c.action != "dedup":
                continue
            adj[c.node1_id].add(c.node2_id)
            adj[c.node2_id].add(c.node1_id)

        if not adj:
            return []

        cliques: List[Set[str]] = []

        def bron_kerbosch(R: Set[str], P: Set[str], X: Set[str]):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(set(R))
                return
            for v in list(P):
                neighbors = adj[v]
                bron_kerbosch(R | {v}, P & neighbors, X & neighbors)
                P = P - {v}
                X = X | {v}

        bron_kerbosch(set(), set(adj.keys()), set())

        # Greedy maximal-clique cover: sort cliques by size desc, take any clique
        # that introduces at least one un-covered node. Each node ends up in
        # exactly one cluster, biased toward the largest clique it belongs to.
        cliques.sort(key=len, reverse=True)
        used: Set[str] = set()
        result: List[List[str]] = []
        for cl in cliques:
            # Take only the unused portion; require size >= 2 after subtraction.
            remaining = [n for n in cl if n not in used]
            if len(remaining) < 2:
                continue
            # All pairs in `remaining` are still a clique (subset of a clique).
            result.append(sorted(remaining))
            used.update(remaining)
        return result

    def _synthesize_cluster_content(
        self,
        cluster_contents: List[str],
        cluster_types: List[str],
        model_fn=None,
    ) -> str:
        """Produce one representative statement for a cluster of near-duplicates.

        With model_fn, asks the LLM for a single-line synthesis. Falls back to
        the longest member's content on None or failure (degraded but not blank).
        """
        longest = max(cluster_contents, key=lambda s: len(s or "")) if cluster_contents else ""
        if model_fn is None:
            return longest

        snippet_block = "\n\n".join(
            f"SNIPPET {i+1} ({t}):\n{c}"
            for i, (c, t) in enumerate(zip(cluster_contents, cluster_types))
        )
        prompt = (
            "The following thought-snippets are near-duplicates from the same "
            "thought-graph: they say substantially the same thing. Produce ONE "
            "consolidated statement, in plain prose, that captures what they all "
            "express. Preserve the strongest, most specific phrasing. Do not "
            "average or hedge. If they disagree on detail, keep the version that "
            "is most concrete.\n\n"
            "Rules: no preamble, no headers, no markdown. Output only the "
            "consolidated statement, on a single line.\n\n"
            f"{snippet_block}\n"
        )
        try:
            response = model_fn(prompt)
            if response:
                candidate = response.strip().splitlines()[0].strip()
                if len(candidate) >= 10:
                    return candidate
        except Exception as e:
            logger.warning(f"Cluster LLM synthesis failed, falling back: {e}")
        return longest

    def merge_cluster(self, node_ids: List[str], model_fn=None) -> Optional[str]:
        """
        Merge a cluster of near-duplicate nodes into a single representative node.

        - Synthesizes content via model_fn (fallback: longest member).
        - permanent = OR across cluster, confidence = max, access_count = sum,
          timestamp = earliest, last_accessed = most recent (NULL-safe),
          source_file/tags = unioned, metadata = merged (keeper wins on shared keys),
          node_type = mode (fallback "derived"), mood_state = mode (fallback NULL).
        - Rewires every edge touching any cluster member to the new merged id;
          drops self-loops; INSERT OR IGNORE semantics avoid PK violations.
        - Deletes the original rows from thought_nodes and their embeddings.
        - Logs a `merge_cluster` event with full audit trail.

        Returns the new merged node id, or None if invalid (size < 2, missing nodes).
        """
        if not node_ids or len(node_ids) < 2:
            return None

        node_ids = list(dict.fromkeys(node_ids))  # de-dup, preserve order
        if len(node_ids) < 2:
            return None

        conn = self._get_connection()
        cursor = conn.cursor()

        # Fetch all cluster rows generically (column list varies across schemas).
        cursor.execute("PRAGMA table_info(thought_nodes)")
        columns = [row[1] for row in cursor.fetchall()]

        placeholders = ",".join("?" for _ in node_ids)
        cursor.execute(
            f"SELECT {', '.join(columns)} FROM thought_nodes WHERE id IN ({placeholders})",
            node_ids,
        )
        rows = cursor.fetchall()
        if len(rows) != len(node_ids):
            conn.close()
            return None  # missing node(s) — bail

        members = [dict(zip(columns, r)) for r in rows]

        def col(m, name, default=None):
            return m.get(name, default) if name in m else default

        # --- merged properties ---
        contents = [m["content"] or "" for m in members]
        types = [m.get("node_type") or "derived" for m in members]

        synthesized = self._synthesize_cluster_content(contents, types, model_fn=model_fn)

        had_permanent = any(bool(col(m, "permanent")) for m in members)
        merged_permanent = 1 if had_permanent else 0

        merged_confidence = max(float(col(m, "confidence") or 0.0) for m in members)

        merged_access_count = sum(int(col(m, "access_count") or 0) for m in members)

        timestamps = [col(m, "timestamp") for m in members if col(m, "timestamp")]
        merged_timestamp = min(timestamps) if timestamps else datetime.now().isoformat()

        last_accesses = [col(m, "last_accessed") for m in members if col(m, "last_accessed")]
        merged_last_accessed = max(last_accesses) if last_accesses else None

        # source_file: union, semicolon-joined, drop dup/NULL
        sources: List[str] = []
        for m in members:
            sf = col(m, "source_file")
            if sf:
                for piece in str(sf).split(";"):
                    piece = piece.strip()
                    if piece and piece not in sources:
                        sources.append(piece)
        merged_source = ";".join(sources) if sources else None

        # tags: union, comma-joined
        tag_set: List[str] = []
        for m in members:
            t = col(m, "tags")
            if t:
                for piece in str(t).split(","):
                    piece = piece.strip()
                    if piece and piece not in tag_set:
                        tag_set.append(piece)
        merged_tags = ",".join(tag_set) if tag_set else None

        # metadata: dict-merge; keeper (highest confidence) wins on shared keys
        keeper_idx = max(range(len(members)),
                         key=lambda i: float(col(members[i], "confidence") or 0.0))
        merged_metadata: dict = {}
        # First, fold in non-keeper entries; keeper goes last so it overrides.
        order = [i for i in range(len(members)) if i != keeper_idx] + [keeper_idx]
        for i in order:
            raw = col(members[i], "metadata")
            if not raw:
                continue
            try:
                d = json.loads(raw) if isinstance(raw, str) else dict(raw)
                if isinstance(d, dict):
                    merged_metadata.update(d)
            except (ValueError, TypeError):
                continue
        merged_metadata_json = json.dumps(merged_metadata) if merged_metadata else "{}"

        # node_type: mode, fallback "derived"
        def mode_or(vals: List, fallback):
            counts: Dict = defaultdict(int)
            for v in vals:
                if v is None:
                    continue
                counts[v] += 1
            if not counts:
                return fallback
            top = max(counts.values())
            top_vals = [k for k, v in counts.items() if v == top]
            if len(top_vals) == 1:
                return top_vals[0]
            return fallback

        merged_node_type = mode_or([col(m, "node_type") for m in members], "derived")
        merged_mood = mode_or([col(m, "mood_state") for m in members], None)

        # domain: prefer keeper's domain (not in spec; pick deterministic default)
        merged_domain = col(members[keeper_idx], "domain")

        # --- new id ---
        import hashlib
        merged_id = hashlib.sha256(synthesized.encode("utf-8")).hexdigest()[:12]

        # If merged_id collides with one of the cluster members, we'll do an
        # in-place upsert via INSERT OR REPLACE rather than insert+delete.
        cluster_set = set(node_ids)

        # --- write merged node ---
        # Build column list dynamically based on what the schema actually has.
        write_cols = ["id", "content", "node_type", "timestamp", "confidence"]
        write_vals = [merged_id, synthesized, merged_node_type, merged_timestamp, merged_confidence]
        if "mood_state" in columns:
            write_cols.append("mood_state"); write_vals.append(merged_mood)
        if "metadata" in columns:
            write_cols.append("metadata"); write_vals.append(merged_metadata_json)
        if "source_file" in columns:
            write_cols.append("source_file"); write_vals.append(merged_source)
        if "decayed" in columns:
            write_cols.append("decayed"); write_vals.append(0)
        if "permanent" in columns:
            write_cols.append("permanent"); write_vals.append(merged_permanent)
        if "domain" in columns:
            write_cols.append("domain"); write_vals.append(merged_domain)
        if "access_count" in columns:
            write_cols.append("access_count"); write_vals.append(merged_access_count)
        if "last_accessed" in columns:
            write_cols.append("last_accessed"); write_vals.append(merged_last_accessed)
        if "tags" in columns:
            write_cols.append("tags"); write_vals.append(merged_tags)
        if "last_updated" in columns:
            write_cols.append("last_updated"); write_vals.append(datetime.now().isoformat())

        cursor.execute(
            f"INSERT OR REPLACE INTO thought_nodes ({', '.join(write_cols)}) "
            f"VALUES ({', '.join('?' for _ in write_cols)})",
            write_vals,
        )

        # --- rewire edges ---
        # parent_id sweep
        cursor.execute(
            f"SELECT parent_id, child_id, weight, reasoning FROM derivation_edges "
            f"WHERE parent_id IN ({placeholders}) OR child_id IN ({placeholders})",
            node_ids + node_ids,
        )
        edges = cursor.fetchall()

        # Delete all existing edges touching cluster members; we'll re-insert rewired versions.
        cursor.execute(
            f"DELETE FROM derivation_edges "
            f"WHERE parent_id IN ({placeholders}) OR child_id IN ({placeholders})",
            node_ids + node_ids,
        )

        for parent_id, child_id, weight, reasoning in edges:
            new_parent = merged_id if parent_id in cluster_set else parent_id
            new_child = merged_id if child_id in cluster_set else child_id
            if new_parent == new_child:
                continue  # drop self-loop
            cursor.execute(
                "INSERT OR IGNORE INTO derivation_edges (parent_id, child_id, weight, reasoning) "
                "VALUES (?, ?, ?, ?)",
                (new_parent, new_child, weight, reasoning),
            )

        # --- delete embeddings for old ids (merged node will be re-embedded downstream) ---
        # Some test schemas don't have embeddings/vec_embeddings; guard with try.
        for old_id in node_ids:
            if old_id == merged_id:
                continue
            try:
                cursor.execute("DELETE FROM embeddings WHERE node_id = ?", (old_id,))
            except sqlite3.OperationalError:
                pass
            try:
                cursor.execute("DELETE FROM vec_embeddings WHERE node_id = ?", (old_id,))
            except sqlite3.OperationalError:
                pass

        # --- delete old rows ---
        ids_to_delete = [nid for nid in node_ids if nid != merged_id]
        if ids_to_delete:
            del_placeholders = ",".join("?" for _ in ids_to_delete)
            cursor.execute(
                f"DELETE FROM thought_nodes WHERE id IN ({del_placeholders})",
                ids_to_delete,
            )

        conn.commit()
        conn.close()

        # Invalidate cached embeddings since rows changed.
        if hasattr(self, "_embed_id_to_idx"):
            for attr in ("_embed_ids", "_embed_vectors", "_embed_id_to_idx", "_cosine_similarity"):
                if hasattr(self, attr):
                    delattr(self, attr)

        self._log_event("merge_cluster", {
            "cluster_node_ids": list(node_ids),
            "cluster_contents": contents,
            "merged_node_id": merged_id,
            "merged_content": synthesized,
            "had_permanent": had_permanent,
            "size": len(node_ids),
        })

        return merged_id

    def generate_dream_node(self, cross_links: List[CrossLinkCandidate],
                            model_fn=None) -> Optional[str]:
        """Generate a dream node by synthesizing the strongest cross-source bridge.

        With model_fn, the LLM is asked to surface the underlying pattern,
        assumption, or risk that the two snippets jointly point at — the dream
        is a synthesis, not a templated restatement. Without model_fn, falls
        back to a stub template so the node still exists in the graph.
        """
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

        dream_content = None
        if model_fn is not None:
            prompt = (
                "Two thought-snippets surfaced from the same body of work. They were "
                "embedded close in vector space, suggesting they share something. Read "
                "them and find what they JOINTLY point at: a shared assumption, a hidden "
                "invariant, a recurring failure mode, a deeper principle, or a contradiction. "
                "Output ONE statement, in plain prose, that captures the synthesis. "
                "Be specific. Name the concrete thing they share. If they don't share "
                "anything meaningful, output a one-line note about WHY the embedding "
                "linked them anyway (lexical overlap, structural similarity, etc.).\n\n"
                "Rules: no preamble, no headers, no markdown. Output only the synthesis "
                "statement, on a single line.\n\n"
                f"SNIPPET A ({type1}):\n{content1}\n\n"
                f"SNIPPET B ({type2}):\n{content2}\n"
            )
            try:
                response = model_fn(prompt)
                if response:
                    candidate = response.strip().splitlines()[0].strip()
                    if len(candidate) > 20:
                        dream_content = candidate
            except Exception as e:
                logging.warning(f"Dream LLM synthesis failed, falling back: {e}")

        if not dream_content:
            dream_content = (
                f"Connection discovered: '{content1[:50]}...' relates to "
                f"'{content2[:50]}...'"
            )
        
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
        
        # Get all non-decayed nodes (with confidence)
        cursor.execute("""
            SELECT id, node_type, COALESCE(confidence, 0.5) FROM thought_nodes 
            WHERE decayed = 0 OR decayed IS NULL
        """)
        rows = cursor.fetchall()
        nodes = {row[0]: row[1] for row in rows}
        node_confidence = {row[0]: row[2] for row in rows}
        
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
            # Confidence prevents high-quality orphans from being GC'd before sleep links them
            confidence = node_confidence.get(node_id, 0.5)
            base_score = branching_factor + cross_links * 0.5 + derivation_depth * 0.1 + confidence * 0.5
            
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
        Randomly select K nodes and GC those below fitness threshold.
        Random selection introduces noise that forces rederivation through novel paths.

        GC mode is read from config:
          - soft: set decayed=1 (default, existing behavior)
          - hard: DELETE the node + its edges
          - off: skip GC entirely
        """
        gc_mode = config.gc_mode
        gc_threshold = config.gc_threshold
        gc_grace_days = config.gc_grace_days
        gc_protect_types = set(config.gc_protect_types)
        gc_think_cycle_penalty = config.gc_think_cycle_penalty

        if gc_mode == "off":
            logger.info("GC mode is off — skipping garbage collection")
            return []

        if len(metrics) <= k_nodes:
            return []  # Don't GC if we have too few nodes

        # Randomly select K nodes
        node_ids = list(metrics.keys())
        selected = random.sample(node_ids, min(k_nodes, len(node_ids)))

        collected_nodes = []
        conn = self._get_connection()
        cursor = conn.cursor()

        for node_id in selected:
            metric = metrics[node_id]

            cursor.execute(
                "SELECT node_type, source_file, last_accessed, permanent FROM thought_nodes WHERE id = ?",
                (node_id,),
            )
            row = cursor.fetchone()
            if not row:
                continue
            node_type, source_file, last_accessed, is_permanent = row

            # Permanent nodes are protected regardless of node_type.
            # gc_protect_types only covers a hardcoded subset (seed, core_memory)
            # and misses derived/fact/insight/etc. that sleep promoted to permanent.
            if is_permanent:
                continue

            # Protect configured types
            if node_type in gc_protect_types:
                continue

            # Grace period — use last_accessed (NOT created_at)
            if last_accessed and gc_grace_days > 0:
                try:
                    accessed_dt = datetime.fromisoformat(last_accessed.replace("Z", "+00:00"))
                    age_days = (datetime.now(accessed_dt.tzinfo or None) - accessed_dt).days
                    if age_days < gc_grace_days:
                        continue
                except (ValueError, TypeError):
                    pass  # can't parse — don't block GC on bad data

            # Think cycle nodes get a higher threshold (penalty multiplier)
            is_think_cycle = source_file and "think_cycle" in str(source_file)
            effective_threshold = gc_threshold * gc_think_cycle_penalty if is_think_cycle else gc_threshold

            if metric.composite_fitness < effective_threshold:
                if gc_mode == "hard":
                    # DELETE the node and its edges
                    cursor.execute("DELETE FROM derivation_edges WHERE parent_id = ? OR child_id = ?",
                                   (node_id, node_id))
                    cursor.execute("DELETE FROM embeddings WHERE node_id = ?", (node_id,))
                    cursor.execute("DELETE FROM thought_nodes WHERE id = ?", (node_id,))
                    logger.info(f"Hard-deleted node {node_id} (fitness={metric.composite_fitness:.3f})")
                else:
                    # soft mode (default): mark as decayed
                    cursor.execute("UPDATE thought_nodes SET decayed = 1 WHERE id = ?", (node_id,))

                collected_nodes.append(node_id)

                self._log_event("gc_decay", {
                    "node_id": node_id,
                    "mode": gc_mode,
                    "fitness_score": metric.composite_fitness,
                    "threshold": effective_threshold,
                    "metrics": asdict(metric)
                })

        conn.commit()
        conn.close()

        return collected_nodes
    
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
        
        # Promote new core memories (and make them permanent)
        promotions = should_be_core - current_core
        for node_id in promotions:
            cursor.execute(
                "UPDATE thought_nodes SET node_type = 'core_memory', permanent = 1 WHERE id = ?", 
                (node_id,)
            )
            self._log_event("core_promotion", {
                "node_id": node_id,
                "fitness_score": metrics[node_id].composite_fitness,
                "rank": next(i for i, m in enumerate(ranked_nodes) if m.node_id == node_id) + 1,
                "set_permanent": True
            })
        
        # Ensure all existing core memories are also permanent (repair any inconsistencies)
        cursor.execute("UPDATE thought_nodes SET permanent = 1 WHERE node_type = 'core_memory' AND (permanent IS NULL OR permanent = 0)")
        existing_core_repairs = cursor.rowcount
        if existing_core_repairs > 0:
            self._log_event("core_memory_repair", {
                "nodes_repaired": existing_core_repairs,
                "action": "set_permanent_for_existing_core_memories"
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
            model_fn: Optional model function for LLM-powered operations.
                     If None, LLM-dependent features will use fallbacks or be skipped.
        """
        print("💤 Starting sleep cycle...")
        
        if not model_fn:
            print("   ⚠️  No LLM access - some operations will use fallbacks")
        
        # Ensure schema is up to date
        self._ensure_decayed_column()
        
        # 1. Cross-linking phase
        print("🔗 Finding cross-link candidates...")
        candidates = self.find_cross_link_candidates()
        
        cross_links_created = 0
        dedups_performed = 0

        # Cross-links first (these don't mutate node identity)
        for candidate in candidates:
            if candidate.action == "cross_link":
                self.cross_link_nodes(candidate.node1_id, candidate.node2_id, candidate.similarity)
                cross_links_created += 1

        # N-plicate cluster merge: clique-detect over dedup candidates and
        # consolidate each clique into a single LLM-synthesized node.
        dedup_candidates = [c for c in candidates if c.action == "dedup"]
        clusters = self.find_merge_clusters(dedup_candidates)
        for cluster in clusters:
            if self.merge_cluster(cluster, model_fn=model_fn):
                dedups_performed += 1
        
        # 2. Dream generation
        print("💭 Generating dream nodes...")
        cross_link_candidates = [c for c in candidates if c.action == "cross_link"]
        dream_id = self.generate_dream_node(cross_link_candidates, model_fn=model_fn)
        
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
        
        # 6. Clustering (hotspots removed — cluster detection only for metrics)
        print("📍 Running cluster detection...")
        clustering_results = {"clusters_found": 0, "new_hotspots_created": 0, "stale_hotspots_found": 0}
        
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
    
    def evaluate_permanence(self) -> Dict:
        """
        Evaluate node permanence based on access_count threshold.
        Implements simple, data-driven permanence promotion.
        """
        from .permanence import promote_permanent_nodes, calculate_recommended_threshold, validate_permanence_integrity
        
        # Calculate recommended threshold based on current data
        recommended_threshold = calculate_recommended_threshold(self.db_path)
        
        # Promote nodes that meet the threshold
        promotion_stats = promote_permanent_nodes(self.db_path, recommended_threshold)
        
        # Validate integrity
        integrity_stats = validate_permanence_integrity(self.db_path)
        
        # Log promotion events
        if promotion_stats["nodes_promoted"] > 0:
            self._log_event("permanence_promotion", {
                "nodes_promoted": promotion_stats["nodes_promoted"],
                "access_threshold": promotion_stats["access_threshold"],
                "integrity_check": integrity_stats
            })
        
        return {
            'nodes_evaluated': promotion_stats["nodes_evaluated"],
            'nodes_made_permanent': promotion_stats["nodes_promoted"],
            'access_threshold': promotion_stats["access_threshold"],
            'integrity_ok': integrity_stats["integrity_ok"]
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