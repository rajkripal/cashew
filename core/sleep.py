#!/usr/bin/env python3
"""
Cashew Sleep Protocol — memory consolidation, cross-linking, GC, and core memory.

Two layers:

1. **Vectorized pipeline** (free functions) — work-capped, batched, Numpy-based.
   Designed for lifecycle hooks where latency matters.  Configurable via the
   module-level constants at the top of this file.

2. **SleepProtocol class** — backward-compatible wrapper that delegates to the
   free functions for the heavy work but preserves the old method signatures so
   existing callers (tests, scripts, downstream integrations) keep working.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import re
import sqlite3
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger("cashew.sleep")

from .config import get_db_path, config
from .decay_audit import log_decay_event, gc_decay_audit

# ── module-level defaults (tunable) ──────────────────────────────────────

# Similarity thresholds are model-specific and resolve from the active embedding
# model's calibrated profile (see core/model_profiles.py). Hardcoding them broke
# after the all-MiniLM -> gte-large migration: the old 0.70 cross-link threshold
# matched 96% of all pairs and saturated the graph with ~15.6M edges.
from .model_profiles import get_active_profile as _get_active_profile

_profile = _get_active_profile()
CROSS_LINK_THRESHOLD = _profile.cross_link_threshold  # cosine ≥ this → cross-link edge
DEDUP_THRESHOLD       = _profile.dedup_threshold       # cosine ≥ this → dedup candidate
MAX_NODES_PER_CYCLE   = 2000   # work cap: process at most N oldest nodes
MAX_EDGES_PER_CYCLE   = 100_000  # hard cap on cross-links per cycle
EDGES_PER_BATCH       = 500    # commit watermark for batched inserts
GC_K_NODES            = 50     # random sample size for garbage collection
GC_THRESHOLD          = 0.0    # fitness below this → collectable (config overrides)
DEFAULT_SLEEP_LOG_PATH = "./data/sleep_log.json"

# ── Temporal-anchor detection (preserved from upstream) ──────────────────

_MONTHS = (
    "january|february|march|april|may|june|july|august|"
    "september|october|november|december|"
    "jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)
_WEEKDAYS = "monday|tuesday|wednesday|thursday|friday|saturday|sunday"
_RELATIVE = (
    r"yesterday|today|tonight|tomorrow|"
    r"(?:last|next|this|past|coming)\s+(?:week|month|year|"
    + _WEEKDAYS + r")|"
    r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|few|several|couple\s+of)\s+"
    r"(?:minute|hour|day|week|month|year)s?\s+ago|"
    r"in\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(?:minute|hour|day|week|month|year)s?"
)
_TEMPORAL_PATTERNS = [
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
    re.compile(rf"\b(?:{_MONTHS})\b\.?\s*\d{{1,2}}(?:,\s*\d{{4}})?", re.IGNORECASE),
    re.compile(rf"\b\d{{1,2}}\s+(?:{_MONTHS})\b", re.IGNORECASE),
    re.compile(rf"\b(?:{_MONTHS})\s+\d{{4}}\b", re.IGNORECASE),
    re.compile(rf"\b(?:{_WEEKDAYS})\b", re.IGNORECASE),
    re.compile(r"\b(?:19|20)\d{2}\b"),
    re.compile(rf"\b(?:{_RELATIVE})\b", re.IGNORECASE),
]


def _collect_temporal_anchors(snippets: List[str]) -> List[str]:
    """Return distinct lowercase temporal anchor strings found across snippets."""
    seen: Set[str] = set()
    for s in snippets or ():
        if not s:
            continue
        for pat in _TEMPORAL_PATTERNS:
            for m in pat.findall(s):
                tok = m.lower().strip()
                if tok:
                    seen.add(tok)
    return list(seen)


def _has_any_anchor(text: str, anchors: List[str]) -> bool:
    """True if *text* (case-insensitive) contains at least one anchor string."""
    if not text or not anchors:
        return False
    low = text.lower()
    return any(a in low for a in anchors)


# ── free-function helpers (vectorized pipeline) ──────────────────────────


def _set_wal(conn: sqlite3.Connection) -> None:
    """Enable WAL mode if not already active."""
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    if mode.lower() != "wal":
        logger.info("sleep: switching journal_mode %s → wal", mode)
        conn.execute("PRAGMA journal_mode=WAL")


def _load_embedding_matrix(
    conn: sqlite3.Connection, node_ids: List[str],
) -> Tuple[List[str], np.ndarray]:
    """Load embeddings for *node_ids* from the ``embeddings`` table.

    Returns (valid_ids, matrix) where *matrix* has shape (N, embedding_dim).
    Filters NaN, inf, and zero vectors.
    """
    if not node_ids:
        return [], np.array([])

    placeholders = ",".join("?" * len(node_ids))
    rows = conn.execute(
        f"SELECT e.node_id, e.vector FROM embeddings e "
        f"WHERE e.node_id IN ({placeholders})",
        node_ids,
    ).fetchall()

    vectors: List[np.ndarray] = []
    valid_ids: List[str] = []
    bad = 0
    for nid, blob in rows:
        try:
            vec = np.frombuffer(blob, dtype=np.float32)
            if np.any(np.isnan(vec)) or np.any(np.isinf(vec)):
                bad += 1
                continue
            if np.allclose(vec, 0):
                bad += 1
                continue
            valid_ids.append(nid)
            vectors.append(vec)
        except Exception:
            bad += 1

    if bad:
        logger.warning("sleep: skipped %d bad embeddings", bad)
    if not vectors:
        return [], np.array([])
    return valid_ids, np.array(vectors)


# ── Phase 1: candidate discovery (vectorized) ────────────────────────────


def _find_pairs(
    ids: List[str], matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (cross_link_pairs, dedup_pairs, similarity_matrix).

    Each pair array has shape (K, 2) of indices into *ids*.
    """
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_sim

    t0 = time.perf_counter()
    sim = sklearn_cosine_sim(matrix)
    logger.debug(
        "sleep: similarity matrix %d×%d computed in %.1fs (%.0f MB)",
        len(ids), len(ids), time.perf_counter() - t0,
        sim.nbytes / 1024**2,
    )

    upper = np.triu(sim, k=1)
    cross_mask = (upper >= CROSS_LINK_THRESHOLD) & (upper < DEDUP_THRESHOLD)
    dedup_mask = upper >= DEDUP_THRESHOLD

    cross_pairs = np.argwhere(cross_mask)
    dedup_pairs = np.argwhere(dedup_mask)

    logger.info(
        "sleep: %d cross-link + %d dedup candidates (%d total / %d pairs)",
        len(cross_pairs), len(dedup_pairs),
        len(cross_pairs) + len(dedup_pairs),
        len(ids) * (len(ids) - 1) // 2,
    )
    return cross_pairs, dedup_pairs, sim


# ── Phase 2: batched cross-linking ───────────────────────────────────────


def _batch_cross_links(
    conn: sqlite3.Connection,
    ids: List[str],
    cross_pairs: np.ndarray,
    sim: np.ndarray,
    source_files: Optional[Dict[str, str]] = None,
    max_edges: Optional[int] = None,
) -> dict:
    """Insert cross-link edges in batches. Returns stats dict.

    When *source_files* is provided, pairs whose nodes share the same
    ``source_file`` are skipped (counted in ``same_source_skipped``).
    When *max_edges* is set, stops after reaching the cap.
    """
    stats = {
        "candidates": len(cross_pairs),
        "created": 0,
        "skipped": 0,
        "same_source_skipped": 0,
        "capped": False,
    }
    pending: List[Tuple[str, str, float]] = []
    t0 = time.perf_counter()

    for batch_start in range(0, len(cross_pairs), EDGES_PER_BATCH):
        batch = cross_pairs[batch_start:batch_start + EDGES_PER_BATCH]
        for i, j in batch:
            if max_edges is not None and stats["created"] >= max_edges:
                stats["capped"] = True
                break
            n1 = ids[int(i)]
            n2 = ids[int(j)]
            # Same-source check
            if source_files is not None:
                sf1 = source_files.get(n1, "")
                sf2 = source_files.get(n2, "")
                if sf1 and sf2 and sf1 == sf2:
                    stats["same_source_skipped"] += 1
                    continue
            row = conn.execute(
                "SELECT COUNT(*) FROM derivation_edges "
                "WHERE (parent_id=? AND child_id=?) OR (parent_id=? AND child_id=?)",
                (n1, n2, n2, n1),
            ).fetchone()
            if row[0] > 0:
                stats["skipped"] += 1
                continue
            sim_val = float(sim[int(i), int(j)])
            pending.append((n1, n2, sim_val))
            pending.append((n2, n1, sim_val))
            stats["created"] += 1

        if max_edges is not None and stats["created"] >= max_edges:
            stats["capped"] = True
            break

        if pending:
            conn.executemany(
                "INSERT OR IGNORE INTO derivation_edges "
                "(parent_id, child_id, weight, reasoning) VALUES (?, ?, ?, ?)",
                [
                    (p, c, w, f"cross_link - similarity={w:.3f}")
                    for p, c, w in pending
                ],
            )
            conn.commit()
        pending.clear()

    elapsed = time.perf_counter() - t0
    logger.info(
        "sleep: cross-links %d created, %d skipped in %.1fs",
        stats["created"], stats["skipped"], elapsed,
    )
    return stats


# ── Phase 3: dedup via Bron-Kerbosch maximal cliques ───────────────────


def _merge_cluster(
    conn: sqlite3.Connection, cluster_ids: List[str],
) -> Optional[str]:
    """Merge a cluster of near-duplicate nodes into the keeper.

    Keeper: highest access_count (tiebreak oldest timestamp).
    Rewires edges via read→delete→reinsert, decays losers.
    """
    if len(cluster_ids) < 2:
        return None

    keeper = conn.execute(
        "SELECT id FROM thought_nodes WHERE id IN ({}) "
        "ORDER BY COALESCE(access_count, 0) DESC, "
        "COALESCE(timestamp, '9999') ASC LIMIT 1".format(
            ",".join("?" * len(cluster_ids))
        ),
        cluster_ids,
    ).fetchone()
    if not keeper:
        return None

    keeper_id = keeper[0]
    losers = [n for n in cluster_ids if n != keeper_id]
    cluster_set = set(cluster_ids)
    all_p = ",".join("?" * len(cluster_ids))

    # Read all edges touching any cluster member
    edges = conn.execute(
        "SELECT parent_id, child_id, weight, reasoning "
        "FROM derivation_edges "
        "WHERE parent_id IN ({}) OR child_id IN ({})".format(all_p, all_p),
        cluster_ids + cluster_ids,
    ).fetchall()

    # Delete all edges touching cluster members
    conn.execute(
        "DELETE FROM derivation_edges "
        "WHERE parent_id IN ({}) OR child_id IN ({})".format(all_p, all_p),
        cluster_ids + cluster_ids,
    )

    # Re-insert rewired (skip self-loops)
    for parent_id, child_id, weight, reasoning in edges:
        new_parent = keeper_id if parent_id in cluster_set else parent_id
        new_child = keeper_id if child_id in cluster_set else child_id
        if new_parent == new_child:
            continue
        conn.execute(
            "INSERT OR IGNORE INTO derivation_edges "
            "(parent_id, child_id, weight, reasoning) VALUES (?, ?, ?, ?)",
            (new_parent, new_child, weight, reasoning),
        )

    # Decay losers (soft-delete)
    if losers:
        lp = ",".join("?" * len(losers))
        conn.execute(
            "UPDATE thought_nodes SET decayed=1 WHERE id IN ({})".format(lp),
            losers,
        )
    return keeper_id


def _run_dedup(
    conn: sqlite3.Connection, ids: List[str], dedup_pairs: np.ndarray,
) -> dict:
    """Build dedup graph, extract connected components, merge each."""
    stats = {"components": 0, "nodes_merged": 0}
    if len(dedup_pairs) == 0:
        return stats

    # Build adjacency
    adj: Dict[str, Set[str]] = defaultdict(set)
    for i, j in dedup_pairs:
        n1, n2 = ids[int(i)], ids[int(j)]
        adj[n1].add(n2)
        adj[n2].add(n1)

    # Bron-Kerbosch maximal clique enumeration
    # Connected-components clustering is deliberately avoided here —
    # cosine similarity is not transitive: sim(A,B) > θ and sim(B,C) > θ
    # does not imply sim(A,C) > θ.  Bron-Kerbosch enumerates strict
    # cliques where every pair is above the dedup threshold.
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

    # Greedy cluster assignment: largest clique first, disallow reuse
    cliques.sort(key=len, reverse=True)
    used: Set[str] = set()
    components: List[List[str]] = []
    for cl in cliques:
        remaining = [n for n in cl if n not in used]
        if len(remaining) < 2:
            continue
        components.append(remaining)
        used.update(remaining)

    logger.info("sleep: %d dedup components to merge", len(components))

    for comp in components:
        result = _merge_cluster(conn, comp)
        if result:
            stats["components"] += 1
            stats["nodes_merged"] += len(comp) - 1

    if stats["components"] > 0:
        conn.commit()

    logger.info(
        "sleep: dedup %d components merged, %d nodes decayed",
        stats["components"], stats["nodes_merged"],
    )
    return stats


# ── Phase 4: node metrics ────────────────────────────────────────────────


def _compute_metrics(conn: sqlite3.Connection) -> Dict[str, dict]:
    """Compute branching factor + cross-link count for all active nodes."""
    t0 = time.perf_counter()
    rows = conn.execute(
        "SELECT tn.id, "
        "  (SELECT COUNT(*) FROM derivation_edges "
        "   WHERE parent_id = tn.id) AS branching, "
        "  (SELECT COUNT(*) FROM derivation_edges "
        "   WHERE (parent_id = tn.id OR child_id = tn.id)"
        "     AND reasoning LIKE '%cross_link%') AS cross_links "
        "FROM thought_nodes tn "
        "WHERE (tn.decayed IS NULL OR tn.decayed = 0)"
    ).fetchall()

    metrics: Dict[str, dict] = {}
    for nid, branching, cross_links in rows:
        metrics[nid] = {
            "branching_factor": branching or 0,
            "cross_links": cross_links or 0,
            "fitness": float((branching or 0) + (cross_links or 0) * 0.5),
        }

    logger.debug(
        "sleep: metrics computed for %d nodes in %.1fs",
        len(metrics), time.perf_counter() - t0,
    )
    return metrics


# ── Phase 5: garbage collection ──────────────────────────────────────────


def _garbage_collect(
    conn: sqlite3.Connection,
    metrics: Dict[str, dict],
    *,
    threshold: float = GC_THRESHOLD,
    sample_k: int = GC_K_NODES,
    grace_days: int = 7,
    think_cycle_penalty: float = 1.5,
    mode: str = "soft",
) -> List[str]:
    """Randomly sample non-permanent nodes, decay those below threshold.

    Parameters
    ----------
    threshold : float
        Fitness below which a node is collectable.
    sample_k : int
        Number of nodes to probe this cycle (random sample).
    grace_days : int
        Nodes accessed within this many days are exempt.
    think_cycle_penalty : float
        Multiplier applied to *threshold* for think-cycle-generated nodes.
    mode : "soft" | "hard" | "off"
        "soft" → set decayed=1; "hard" → DELETE; "off" → skip.
    """
    if mode == "off":
        logger.info("GC mode is off — skipping")
        return []
    if not metrics:
        return []

    perm_ids = {
        r[0] for r in conn.execute(
            "SELECT id FROM thought_nodes "
            "WHERE permanent=1 AND (decayed IS NULL OR decayed=0)"
        ).fetchall()
    }

    now = datetime.now()

    candidates: List[Tuple[str, float, Optional[str]]] = []
    for nid, m in metrics.items():
        if nid in perm_ids:
            continue
        fitness = m["fitness"]

        # Look up last_accessed and source_file for grace/penalty checks
        row = conn.execute(
            "SELECT last_accessed, source_file FROM thought_nodes WHERE id = ?",
            (nid,),
        ).fetchone()
        if not row:
            continue
        last_accessed, source_file = row

        # Grace period
        if grace_days > 0 and last_accessed:
            try:
                la = datetime.fromisoformat(last_accessed.replace("Z", "+00:00"))
                age = (now - la).total_seconds() / 86400
                if age < grace_days:
                    continue
            except (ValueError, TypeError):
                pass

        # Think-cycle penalty
        effective_threshold = threshold
        if source_file and "think_cycle" in str(source_file):
            effective_threshold *= think_cycle_penalty

        if fitness < effective_threshold:
            candidates.append((nid, fitness, source_file))

    if not candidates:
        return []

    sample = (
        candidates
        if len(candidates) <= sample_k
        else random.sample(candidates, sample_k)
    )

    collected: List[str] = []
    for nid, fitness, src in sample:
        if mode == "hard":
            conn.execute(
                "DELETE FROM derivation_edges WHERE parent_id = ? OR child_id = ?",
                (nid, nid),
            )
            conn.execute("DELETE FROM embeddings WHERE node_id = ?", (nid,))
            conn.execute("DELETE FROM thought_nodes WHERE id = ?", (nid,))
        else:
            conn.execute("UPDATE thought_nodes SET decayed = 1 WHERE id = ?", (nid,))
        collected.append(nid)

    conn.commit()
    logger.info("sleep: GC %s-decayed %d low-fitness nodes", mode, len(collected))
    return collected


# ── Phase 6: permanence evaluation ───────────────────────────────────────


def _evaluate_permanence(conn: sqlite3.Connection, access_threshold: int = 10) -> dict:
    """Promote nodes with *access_count* ≥ *access_threshold* to permanent."""
    try:
        from core.permanence import promote_permanent_nodes
        db_path = conn.execute("PRAGMA database_list").fetchone()[2]
        if db_path:
            stats = promote_permanent_nodes(db_path, access_threshold=access_threshold)
            logger.info(
                "sleep: permanence promoted %d nodes (threshold=%d)",
                stats.get("nodes_promoted", 0), access_threshold,
            )
            return stats
    except (ImportError, Exception):
        pass

    # Fallback direct SQL
    cur = conn.execute(
        "UPDATE thought_nodes SET permanent=1 "
        "WHERE access_count >= ? "
        "AND (permanent IS NULL OR permanent = 0) "
        "AND (decayed IS NULL OR decayed = 0)",
        (access_threshold,),
    )
    count = cur.rowcount
    conn.commit()
    logger.info("sleep: permanence promoted %d nodes (fallback, threshold=%d)", count, access_threshold)
    return {"nodes_promoted": count, "nodes_evaluated": count, "access_threshold": access_threshold}


# ── Phase 7: core memory promotion ───────────────────────────────────────


def _promote_core_memories(conn: sqlite3.Connection, metrics: Dict[str, dict]) -> dict:
    """Top √N nodes by fitness become core_memory + permanent."""
    if not metrics:
        return {"promoted": 0, "demoted": 0}

    curr = {
        r[0] for r in conn.execute(
            "SELECT id FROM thought_nodes WHERE node_type='core_memory'"
        ).fetchall()
    }

    ranked = sorted(metrics.items(), key=lambda x: x[1]["fitness"], reverse=True)
    target = int(math.sqrt(len(metrics)))
    should_be = {nid for nid, _ in ranked[:target]}
    promoted = should_be - curr
    demoted = curr - should_be

    if promoted:
        pp = ",".join("?" * len(promoted))
        conn.execute(
            f"UPDATE thought_nodes SET node_type='core_memory', permanent=1 "
            f"WHERE id IN ({pp})",
            list(promoted),
        )

    conn.execute(
        "UPDATE thought_nodes SET permanent=1 "
        "WHERE node_type='core_memory' AND (permanent IS NULL OR permanent = 0)"
    )

    if demoted:
        dp = ",".join("?" * len(demoted))
        conn.execute(
            f"UPDATE thought_nodes SET node_type='derived' "
            f"WHERE id IN ({dp}) AND node_type != 'seed'",
            list(demoted),
        )

    conn.commit()
    logger.info(
        "sleep: core memory %d promoted, %d demoted (target=%d)",
        len(promoted), len(demoted), target,
    )
    return {"promoted": len(promoted), "demoted": len(demoted), "target": target}


# ── Phase 8: dream generation ────────────────────────────────────────────


def _generate_dream(
    conn: sqlite3.Connection,
    cross_link_tuples: List[Tuple[str, str, float]],
    model_fn=None,
) -> Optional[str]:
    """LLM-powered dream node bridging the strongest cross-source pair."""
    if not cross_link_tuples or model_fn is None:
        return None

    # Find cross-links bridging different source files
    bridge_candidates = []
    for n1, n2, sim in cross_link_tuples:
        sources = conn.execute(
            "SELECT source_file FROM thought_nodes WHERE id IN (?, ?)",
            (n1, n2),
        ).fetchall()
        srcs = [r[0] for r in sources]
        if len({s for s in srcs if s}) > 1:
            bridge_candidates.append((n1, n2, sim))

    if not bridge_candidates:
        return None

    best = max(bridge_candidates, key=lambda x: x[2])
    n1, n2, sim = best

    nodes = conn.execute(
        "SELECT content, node_type FROM thought_nodes WHERE id IN (?, ?)",
        (n1, n2),
    ).fetchall()
    if len(nodes) != 2:
        return None

    content1, type1 = nodes[0]
    content2, type2 = nodes[1]

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
        if not response:
            return None
        dream_content = response.strip().splitlines()[0].strip()
        if len(dream_content) < 20:
            return None
    except Exception:
        logger.warning("sleep: dream LLM synthesis failed", exc_info=True)
        return None

    dream_id = hashlib.sha256(dream_content.encode()).hexdigest()[:12]

    conn.execute(
        "INSERT OR REPLACE INTO thought_nodes "
        "(id, content, node_type, timestamp, mood_state, metadata, source_file) "
        "VALUES (?, ?, 'dream', datetime('now'), 'dreamy', '{}', 'sleep_protocol')",
        (dream_id, dream_content),
    )
    conn.execute(
        "INSERT OR IGNORE INTO derivation_edges "
        "(parent_id, child_id, weight, reasoning) "
        "VALUES (?, ?, ?, 'derived_from - Dream synthesis')",
        (n1, dream_id, sim),
    )
    conn.execute(
        "INSERT OR IGNORE INTO derivation_edges "
        "(parent_id, child_id, weight, reasoning) "
        "VALUES (?, ?, ?, 'derived_from - Dream synthesis')",
        (n2, dream_id, sim),
    )
    conn.commit()

    logger.info(
        "sleep: dream node %s bridging %s… ↔ %s…",
        dream_id, n1[:8], n2[:8],
    )
    return dream_id


# ── Phase 9: orphan embedding ────────────────────────────────────────────


def _embed_orphans(conn: sqlite3.Connection) -> int:
    """Embed any active nodes lacking an embedding row. Returns count."""
    rows = conn.execute(
        "SELECT tn.id, tn.content FROM thought_nodes tn "
        "LEFT JOIN embeddings e ON tn.id = e.node_id "
        "WHERE e.node_id IS NULL "
        "AND (tn.decayed IS NULL OR tn.decayed = 0) "
        "AND tn.content IS NOT NULL AND TRIM(tn.content) != ''"
    ).fetchall()

    if not rows:
        return 0

    logger.info("sleep: embedding %d orphaned nodes…", len(rows))

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    embedded = 0
    for nid, content in rows:
        try:
            vec = model.encode(content, normalize_embeddings=True)
            blob = vec.astype(np.float32).tobytes()
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO embeddings "
                    "(node_id, vector, model, updated_at) "
                    "VALUES (?, ?, ?, datetime('now'))",
                    (nid, blob, "all-MiniLM-L6-v2"),
                )
            except sqlite3.OperationalError:
                conn.execute(
                    "INSERT OR REPLACE INTO embeddings (node_id, vector) VALUES (?, ?)",
                    (nid, blob),
                )
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO vec_embeddings "
                    "(node_id, embedding) VALUES (?, ?)",
                    (nid, vec.astype(np.float32).tolist()),
                )
            except sqlite3.OperationalError:
                pass
            embedded += 1
        except Exception as e:
            logger.warning("sleep: failed to embed node %s: %s", nid[:8], e)

    conn.commit()
    logger.info("sleep: embedded %d orphaned nodes", embedded)
    return embedded


# ── Background dream threading ───────────────────────────────────────────


def _run_dream_async(
    db_path: str,
    cross_link_tuples: List[Tuple[str, str, float]],
    model_fn,
) -> None:
    """Run Phase 8 (dream) + Phase 9 (orphan embedding) in a daemon thread.

    Opens its own SQLite connection — WAL mode handles concurrency with
    the new session's sync worker writes.
    """
    def _task():
        try:
            conn = sqlite3.connect(db_path)
            _set_wal(conn)
            dream_id = _generate_dream(conn, cross_link_tuples, model_fn=model_fn)
            orphans = _embed_orphans(conn)
            conn.close()
            logger.info(
                "sleep: background dream complete (id=%s, orphans=%d)",
                dream_id or "none", orphans,
            )
        except Exception:
            logger.warning("sleep: background dream failed", exc_info=True)

    t = threading.Thread(target=_task, daemon=True)
    t.start()
    logger.debug("sleep: background dream thread spawned")


# ── main entry point (free function) ─────────────────────────────────────


def run_sleep_cycle(
    db_path: str = None,
    limit: Optional[int] = None,
    model_fn=None,
    background_dream: bool = False,
    max_edges: int = MAX_EDGES_PER_CYCLE,
    cross_source_only: bool = False,
) -> dict:
    """Run one complete refactored sleep cycle.

    This is the **primary entry point** for lifecycle hooks.  It is work-
    capped, batched, and optionally runs the LLM dream phase in a background
    thread so the caller can return promptly.

    Parameters
    ----------
    db_path : str or None
        Path to Cashew SQLite database.  Uses config default when ``None``.
    limit : Optional[int]
        Max nodes to process this cycle (oldest-first ordering).
        ``None`` (default) = process all active nodes in one full pass.
        Pass an int (e.g. ``2000``) to work-cap for bounded-latency
        lifecycle hooks.
    model_fn : callable or None
        LLM callable for dream generation.  ``None`` = skip dreams.
    background_dream : bool
        When True, Phase 8 (dream) and Phase 9 (orphan embedding) run in a
        daemon thread instead of blocking the caller.
    max_edges : int
        Hard cap on cross-link edges created per cycle.
    cross_source_only : bool
        When True, only cross-link pairs from different ``source_file``
        values (reduces same-source noise).

    Returns
    -------
    dict
        Statistics for each phase.
    """
    if db_path is None:
        db_path = get_db_path()

    t_start = time.perf_counter()
    conn = sqlite3.connect(db_path)
    _set_wal(conn)

    # Check if embeddings table exists — required for vectorized pipeline
    table_check = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='embeddings'"
    ).fetchone()
    if not table_check:
        logger.warning("sleep: no embeddings table — aborting (run cashew init first)")
        conn.close()
        return {"error": "no embeddings table", "nodes_selected": 0}

    # ── Select nodes for this cycle (oldest-first) ──
    if limit is None:
        rows = conn.execute(
            "SELECT e.node_id FROM embeddings e "
            "JOIN thought_nodes tn ON e.node_id = tn.id "
            "WHERE (tn.decayed IS NULL OR tn.decayed = 0) "
            "ORDER BY tn.timestamp ASC"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT e.node_id FROM embeddings e "
            "JOIN thought_nodes tn ON e.node_id = tn.id "
            "WHERE (tn.decayed IS NULL OR tn.decayed = 0) "
            "ORDER BY tn.timestamp ASC "
            "LIMIT ?",
            (limit,),
        ).fetchall()

    ids = [r[0] for r in rows]
    logger.info("sleep: selected %d nodes (limit=%s)", len(ids), limit)

    valid_ids, matrix = _load_embedding_matrix(conn, ids)
    if len(valid_ids) < 2:
        logger.warning("sleep: too few valid embeddings — aborting")
        conn.close()
        return {"error": "too few nodes", "nodes_selected": len(ids)}

    # Phase 1: candidate discovery
    cross_pairs, dedup_pairs, sim = _find_pairs(valid_ids, matrix)

    # Build source_file map for cross-source filtering
    source_files: Optional[Dict[str, str]] = None
    if cross_source_only and len(cross_pairs) > 0:
        sf_rows = conn.execute(
            "SELECT id, COALESCE(source_file, '') FROM thought_nodes "
            "WHERE id IN ({})".format(
                ",".join("?" * len(valid_ids))
            ),
            valid_ids,
        ).fetchall()
        source_files = {r[0]: r[1] for r in sf_rows}

    # Phase 2: cross-linking
    cross_stats = {"created": 0, "skipped": 0}
    cross_link_tuples: List[Tuple[str, str, float]] = []
    if len(cross_pairs) > 0:
        cross_stats = _batch_cross_links(
            conn, valid_ids, cross_pairs, sim,
            source_files=source_files if cross_source_only else None,
            max_edges=max_edges,
        )
        if model_fn is not None:
            for i, j in cross_pairs:
                cross_link_tuples.append((
                    valid_ids[int(i)], valid_ids[int(j)],
                    float(sim[int(i), int(j)]),
                ))

    # Phase 3: dedup
    dedup_stats = {"components": 0, "nodes_merged": 0}
    if len(dedup_pairs) > 0:
        dedup_stats = _run_dedup(conn, valid_ids, dedup_pairs)

    # Phase 4: metrics
    metrics = _compute_metrics(conn)

    # Phase 5: garbage collection (config-driven)
    gc_mode = getattr(config, 'gc_mode', 'soft')
    gc_threshold = getattr(config, 'gc_threshold', 0.05)
    gc_grace_days = getattr(config, 'gc_grace_days', 7)
    gc_think_cycle_penalty_val = getattr(config, 'gc_think_cycle_penalty', 1.5)
    gc_count = len(_garbage_collect(
        conn, metrics,
        threshold=gc_threshold,
        sample_k=GC_K_NODES,
        grace_days=gc_grace_days,
        think_cycle_penalty=gc_think_cycle_penalty_val,
        mode=gc_mode,
    ))

    # Phase 6: permanence
    perm_stats = _evaluate_permanence(conn)

    # Phase 7: core memory
    core_stats = _promote_core_memories(conn, metrics)

    # Phase 8: dream generation
    dream_id = None
    dream_pending = False
    if model_fn is not None and cross_link_tuples:
        if background_dream:
            _run_dream_async(
                db_path=db_path,
                cross_link_tuples=cross_link_tuples,
                model_fn=model_fn,
            )
            dream_pending = True
        else:
            dream_id = _generate_dream(conn, cross_link_tuples, model_fn=model_fn)

    # Phase 9: embed orphans
    if background_dream:
        orphans = 0  # handled by background dream thread
    else:
        orphans = _embed_orphans(conn)

    conn.close()
    elapsed = round(time.perf_counter() - t_start, 1)

    # Decay-audit GC (one-shot per cycle)
    try:
        audit_conn = sqlite3.connect(db_path)
        audit_pruned = gc_decay_audit(audit_conn, retention_days=7)
        audit_conn.commit()
        audit_conn.close()
        if audit_pruned:
            logger.info("sleep: decay-audit GC pruned %d rows", audit_pruned)
    except Exception as e:
        logger.warning("sleep: decay-audit GC failed: %s", e)

    summary = {
        "nodes_selected": len(ids),
        "nodes_with_embeddings": len(valid_ids),
        "cross_link_candidates": len(cross_pairs),
        "dedup_candidates": len(dedup_pairs),
        "cross_links_created": cross_stats["created"],
        "cross_links_skipped": cross_stats["skipped"],
        "cross_link_same_source_skipped": cross_stats.get("same_source_skipped", 0),
        "cross_link_capped": cross_stats.get("capped", False),
        "dedup_components": dedup_stats["components"],
        "dedup_nodes_merged": dedup_stats["nodes_merged"],
        "nodes_gc_decayed": gc_count,
        "nodes_made_permanent": perm_stats.get("nodes_promoted", 0),
        "core_promoted": core_stats.get("promoted", 0),
        "core_demoted": core_stats.get("demoted", 0),
        "dream_id": dream_id,
        "dream_pending": dream_pending,
        "orphans_embedded": orphans,
        "total_nodes": len(metrics),
        "elapsed_s": elapsed,
    }

    if dream_pending:
        logger.info(
            "sleep: sync phases complete in %.1fs — %d nodes, %d cross-links, "
            "%d dedups, %d GC, %d permanent, %d core (dream pending)",
            elapsed, summary["total_nodes"],
            summary["cross_links_created"], summary["dedup_nodes_merged"],
            summary["nodes_gc_decayed"], summary["nodes_made_permanent"],
            summary["core_promoted"],
        )
    else:
        logger.info(
            "sleep: cycle complete in %.1fs — %d nodes, %d cross-links, "
            "%d dedups, %d GC, %d permanent, %d core, %s dream, %d embedded",
            elapsed, summary["total_nodes"],
            summary["cross_links_created"], summary["dedup_nodes_merged"],
            summary["nodes_gc_decayed"], summary["nodes_made_permanent"],
            summary["core_promoted"],
            "1" if dream_id else "0", orphans,
        )

    return summary


# ── backward-compatible SleepProtocol class ──────────────────────────────

@dataclass
class CrossLinkCandidate:
    node1_id: str
    node2_id: str
    similarity: float
    action: str  # "dedup", "cross_link", "contradiction"


@dataclass
class NodeMetrics:
    node_id: str
    branching_factor: int
    cross_links: int
    retrieval_frequency: int
    derivation_depth: int
    composite_fitness: float


@dataclass
class SleepEvent:
    timestamp: str
    event_type: str
    details: dict


class SleepProtocol:
    """Backward-compatible sleep protocol.

    All existing public methods are preserved so downstream callers (tests,
    scripts, integrations) continue to work.  The orchestration method
    ``run_sleep_cycle()`` delegates to the vectorized pipeline above.
    """

    def __init__(self, db_path: str = None, sleep_log_path: str = None):
        if db_path is None:
            db_path = get_db_path()
        if sleep_log_path is None:
            sleep_log_path = DEFAULT_SLEEP_LOG_PATH
        self.db_path = db_path
        self.sleep_log_path = sleep_log_path
        self.sleep_frequency = 10
        self.dedup_threshold = DEDUP_THRESHOLD
        self.cross_link_threshold = CROSS_LINK_THRESHOLD
        self.gc_mode = getattr(config, 'gc_mode', 'soft')
        self.gc_threshold = getattr(config, 'gc_threshold', 0.05)
        self.gc_grace_days = getattr(config, 'gc_grace_days', 7)
        self.gc_think_cycle_penalty = getattr(config, 'gc_think_cycle_penalty', 1.5)
        self.events: List[SleepEvent] = []

    # ── internal helpers ──────────────────────────────────────────────────

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _log_event(self, event_type: str, details: dict):
        event = SleepEvent(
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            details=details,
        )
        self.events.append(event)

    def _ensure_decayed_column(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(thought_nodes)")
        columns = [row[1] for row in cursor.fetchall()]
        if 'decayed' not in columns:
            cursor.execute("ALTER TABLE thought_nodes ADD COLUMN decayed INTEGER DEFAULT 0")
            conn.commit()
        conn.close()

    def _load_embedding_sim_cache(self):
        if hasattr(self, '_sim_cache'):
            return
        from .graph_utils import load_embeddings, cosine_similarity
        node_ids, vectors, _ = load_embeddings(self.db_path)
        self._embed_ids = node_ids
        self._embed_vectors = vectors
        self._embed_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        self._cosine_similarity = cosine_similarity

    # ── text similarity (preserved from upstream) ─────────────────────────

    def _text_similarity(self, text1: str, text2: str,
                         node1_id: str = None, node2_id: str = None) -> float:
        if node1_id and node2_id:
            try:
                self._load_embedding_sim_cache()
                idx1 = self._embed_id_to_idx.get(node1_id)
                idx2 = self._embed_id_to_idx.get(node2_id)
                if idx1 is not None and idx2 is not None:
                    return self._cosine_similarity(
                        self._embed_vectors[idx1],
                        self._embed_vectors[idx2],
                    )
            except Exception as e:
                logger.debug(f"Embedding similarity failed, falling back to text: {e}")

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

    # ── cross-link detection ──────────────────────────────────────────────

    def find_cross_link_candidates(self) -> List[CrossLinkCandidate]:
        try:
            from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_sim
            from .graph_utils import load_embeddings

            node_ids, vectors, node_meta = load_embeddings(self.db_path)
            if len(node_ids) < 2:
                return []

            sim_matrix = sklearn_cosine_sim(vectors)
            candidates = []
            for i in range(len(node_ids)):
                for j in range(i + 1, len(node_ids)):
                    similarity = float(sim_matrix[i, j])
                    if similarity >= self.dedup_threshold:
                        candidates.append(CrossLinkCandidate(
                            node_ids[i], node_ids[j], similarity, "dedup",
                        ))
                    elif similarity >= self.cross_link_threshold:
                        candidates.append(CrossLinkCandidate(
                            node_ids[i], node_ids[j], similarity, "cross_link",
                        ))
            logger.info("Found %d cross-link candidates from %d nodes",
                        len(candidates), len(node_ids))
            return candidates

        except Exception as e:
            logger.warning(f"Embedding-based cross-link failed, falling back to text: {e}")
            return self._find_cross_link_candidates_text_fallback()

    def _find_cross_link_candidates_text_fallback(self) -> List[CrossLinkCandidate]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, content, node_type FROM thought_nodes "
            "WHERE decayed = 0 OR decayed IS NULL"
        )
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

    # ── individual node operations (preserved for backward compat) ────────

    def cross_link_nodes(self, node1_id: str, node2_id: str,
                         similarity: float, reasoning: str = ""):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM derivation_edges "
            "WHERE (parent_id = ? AND child_id = ?) OR (parent_id = ? AND child_id = ?)",
            (node1_id, node2_id, node2_id, node1_id),
        )
        if cursor.fetchone()[0] > 0:
            conn.close()
            return

        cursor.execute(
            "INSERT OR IGNORE INTO derivation_edges "
            "(parent_id, child_id, weight, reasoning) VALUES (?, ?, ?, ?)",
            (node1_id, node2_id, similarity,
             f"cross_link - {reasoning or f'Semantic similarity: {similarity:.2f}'}"),
        )
        cursor.execute(
            "INSERT OR IGNORE INTO derivation_edges "
            "(parent_id, child_id, weight, reasoning) VALUES (?, ?, ?, ?)",
            (node2_id, node1_id, similarity,
             f"cross_link - {reasoning or f'Semantic similarity: {similarity:.2f}'}"),
        )
        conn.commit()
        conn.close()
        self._log_event("cross_link", {
            "node1_id": node1_id, "node2_id": node2_id,
            "similarity": similarity, "reasoning": reasoning,
        })

    def deduplicate_nodes(self, node1_id: str, node2_id: str, similarity: float):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, COALESCE(access_count, 0), COALESCE(timestamp, '') "
            "FROM thought_nodes WHERE id IN (?, ?)",
            (node1_id, node2_id),
        )
        nodes = cursor.fetchall()
        if len(nodes) != 2:
            conn.close()
            return

        node1, node2 = nodes
        if node1[1] != node2[1]:
            keep_node, remove_node = (node1, node2) if node1[1] > node2[1] else (node2, node1)
        else:
            ts1 = node1[2] or "9999"
            ts2 = node2[2] or "9999"
            keep_node, remove_node = (node1, node2) if ts1 <= ts2 else (node2, node1)

        keep_id, remove_id = keep_node[0], remove_node[0]

        cursor.execute(
            "UPDATE OR IGNORE derivation_edges SET parent_id = ? "
            "WHERE parent_id = ? AND child_id != ?",
            (keep_id, remove_id, keep_id),
        )
        cursor.execute(
            "UPDATE OR IGNORE derivation_edges SET child_id = ? "
            "WHERE child_id = ? AND parent_id != ?",
            (keep_id, remove_id, keep_id),
        )
        cursor.execute(
            "DELETE FROM derivation_edges WHERE parent_id = ? OR child_id = ?",
            (remove_id, remove_id),
        )
        cursor.execute(
            "DELETE FROM derivation_edges WHERE parent_id = ? AND child_id = ?",
            (keep_id, keep_id),
        )
        cursor.execute(
            "UPDATE thought_nodes SET decayed = 1 "
            "WHERE id = ? AND (permanent IS NULL OR permanent = 0)",
            (remove_id,),
        )
        if cursor.rowcount > 0:
            log_decay_event(
                conn, remove_id, "dedup_loser",
                related_nodes={"keeper_id": keep_id},
                metadata={"similarity": similarity},
            )
        conn.commit()
        conn.close()
        self._log_event("dedup", {
            "kept_node": keep_id, "removed_node": remove_id,
            "similarity": similarity,
        })

    # ── cluster merge (Bron-Kerbosch, preserved) ──────────────────────────

    def find_merge_clusters(self, candidates: List[CrossLinkCandidate]) -> List[List[str]]:
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

        cliques.sort(key=len, reverse=True)
        used: Set[str] = set()
        result: List[List[str]] = []
        for cl in cliques:
            remaining = [n for n in cl if n not in used]
            if len(remaining) < 2:
                continue
            result.append(sorted(remaining))
            used.update(remaining)
        return result

    def _synthesize_cluster_content(
        self, cluster_contents: List[str], cluster_types: List[str],
        model_fn=None,
    ) -> str:
        longest = max(cluster_contents, key=lambda s: len(s or "")) if cluster_contents else ""
        if model_fn is None:
            return longest

        source_anchors = _collect_temporal_anchors(cluster_contents)

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
            "Critical: preserve every temporal anchor (specific dates, weekdays, "
            "months, years, relative times like 'last Tuesday' or 'two weeks ago') "
            "that appears in any snippet. Temporal context is load-bearing — "
            "never drop it for brevity.\n\n"
            "Rules: no preamble, no headers, no markdown. Output only the "
            "consolidated statement, on a single line.\n\n"
            f"{snippet_block}\n"
        )
        try:
            response = model_fn(prompt)
            if response:
                candidate = response.strip().splitlines()[0].strip()
                if len(candidate) >= 10:
                    if source_anchors and not _has_any_anchor(candidate, source_anchors):
                        logger.warning(
                            "Cluster synthesis dropped all temporal anchors; "
                            "falling back to longest source."
                        )
                        return longest
                    return candidate
        except Exception as e:
            logger.warning(f"Cluster LLM synthesis failed, falling back: {e}")
        return longest

    def merge_cluster(self, node_ids: List[str], model_fn=None) -> Optional[str]:
        if not node_ids or len(node_ids) < 2:
            return None
        node_ids = list(dict.fromkeys(node_ids))
        if len(node_ids) < 2:
            return None

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("PRAGMA table_info(thought_nodes)")
        columns = [row[1] for row in cursor.fetchall()]

        placeholders = ",".join("?" for _ in node_ids)
        cursor.execute(
            f"SELECT {', '.join(columns)} FROM thought_nodes "
            f"WHERE id IN ({placeholders})",
            node_ids,
        )
        rows = cursor.fetchall()
        if len(rows) != len(node_ids):
            conn.close()
            return None

        def col(m, name, default=None):
            return m.get(name, default) if name in m else default

        members = [dict(zip(columns, r)) for r in rows]

        contents = [m["content"] or "" for m in members]
        types = [m.get("node_type") or "derived" for m in members]
        synthesized = self._synthesize_cluster_content(contents, types, model_fn=model_fn)

        had_permanent = any(bool(col(m, "permanent")) for m in members)
        merged_permanent = 1 if had_permanent else 0
        merged_access_count = sum(int(col(m, "access_count") or 0) for m in members)

        timestamps = [col(m, "timestamp") for m in members if col(m, "timestamp")]
        merged_timestamp = min(timestamps) if timestamps else datetime.now().isoformat()

        last_accesses = [col(m, "last_accessed") for m in members if col(m, "last_accessed")]
        merged_last_accessed = max(last_accesses) if last_accesses else None

        sources: List[str] = []
        for m in members:
            sf = col(m, "source_file")
            if sf:
                for piece in str(sf).split(";"):
                    piece = piece.strip()
                    if piece and piece not in sources:
                        sources.append(piece)
        merged_source = ";".join(sources) if sources else None

        tag_set: List[str] = []
        for m in members:
            t = col(m, "tags")
            if t:
                for piece in str(t).split(","):
                    piece = piece.strip()
                    if piece and piece not in tag_set:
                        tag_set.append(piece)
        merged_tags = ",".join(tag_set) if tag_set else None

        def _keeper_rank(i):
            ac = int(col(members[i], "access_count") or 0)
            ts = col(members[i], "timestamp") or "9999"
            return (-ac, ts)

        keeper_idx = min(range(len(members)), key=_keeper_rank)
        merged_metadata: dict = {}
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

        def mode_or(vals, fallback):
            counts = defaultdict(int)
            for v in vals:
                if v is None:
                    continue
                counts[v] += 1
            if not counts:
                return fallback
            top = max(counts.values())
            top_vals = [k for k, v in counts.items() if v == top]
            return top_vals[0] if len(top_vals) == 1 else fallback

        merged_node_type = mode_or([col(m, "node_type") for m in members], "derived")
        merged_mood = mode_or([col(m, "mood_state") for m in members], None)
        merged_domain = col(members[keeper_idx], "domain")

        merged_id = hashlib.sha256(synthesized.encode("utf-8")).hexdigest()[:12]
        cluster_set = set(node_ids)

        write_cols = ["id", "content", "node_type", "timestamp"]
        write_vals = [merged_id, synthesized, merged_node_type, merged_timestamp]
        if "mood_state" in columns:
            write_cols.append("mood_state")
            write_vals.append(merged_mood)
        if "metadata" in columns:
            write_cols.append("metadata")
            write_vals.append(merged_metadata_json)
        if "source_file" in columns:
            write_cols.append("source_file")
            write_vals.append(merged_source)
        if "decayed" in columns:
            write_cols.append("decayed")
            write_vals.append(0)
        if "permanent" in columns:
            write_cols.append("permanent")
            write_vals.append(merged_permanent)
        if "domain" in columns:
            write_cols.append("domain")
            write_vals.append(merged_domain)
        if "access_count" in columns:
            write_cols.append("access_count")
            write_vals.append(merged_access_count)
        if "last_accessed" in columns:
            write_cols.append("last_accessed")
            write_vals.append(merged_last_accessed)
        if "tags" in columns:
            write_cols.append("tags")
            write_vals.append(merged_tags)
        if "last_updated" in columns:
            write_cols.append("last_updated")
            write_vals.append(datetime.now().isoformat())

        cursor.execute(
            f"INSERT OR REPLACE INTO thought_nodes ({', '.join(write_cols)}) "
            f"VALUES ({', '.join('?' for _ in write_cols)})",
            write_vals,
        )

        cursor.execute(
            f"SELECT parent_id, child_id, weight, reasoning FROM derivation_edges "
            f"WHERE parent_id IN ({placeholders}) OR child_id IN ({placeholders})",
            node_ids + node_ids,
        )
        edges = cursor.fetchall()
        cursor.execute(
            f"DELETE FROM derivation_edges "
            f"WHERE parent_id IN ({placeholders}) OR child_id IN ({placeholders})",
            node_ids + node_ids,
        )

        for parent_id, child_id, weight, reasoning in edges:
            new_parent = merged_id if parent_id in cluster_set else parent_id
            new_child = merged_id if child_id in cluster_set else child_id
            if new_parent == new_child:
                continue
            cursor.execute(
                "INSERT OR IGNORE INTO derivation_edges "
                "(parent_id, child_id, weight, reasoning) VALUES (?, ?, ?, ?)",
                (new_parent, new_child, weight, reasoning),
            )

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

        ids_to_delete = [nid for nid in node_ids if nid != merged_id]
        if ids_to_delete:
            del_placeholders = ",".join("?" for _ in ids_to_delete)
            cursor.execute(
                f"DELETE FROM thought_nodes WHERE id IN ({del_placeholders})",
                ids_to_delete,
            )

        conn.commit()
        conn.close()

        if hasattr(self, "_embed_id_to_idx"):
            for attr in ("_embed_ids", "_embed_vectors", "_embed_id_to_idx", "_cosine_similarity"):
                if hasattr(self, attr):
                    delattr(self, attr)

        self._log_event("merge_cluster", {
            "cluster_node_ids": list(node_ids),
            "merged_node_id": merged_id,
            "merged_content": synthesized,
            "had_permanent": had_permanent,
            "size": len(node_ids),
        })
        return merged_id

    # ── dream generation (backward compat) ────────────────────────────────

    def generate_dream_node(self, cross_links: List[CrossLinkCandidate],
                            model_fn=None) -> Optional[str]:
        if not cross_links:
            return None

        conn = self._get_connection()
        cursor = conn.cursor()

        bridge_candidates = []
        for candidate in cross_links:
            cursor.execute(
                "SELECT source_file FROM thought_nodes WHERE id IN (?, ?)",
                (candidate.node1_id, candidate.node2_id),
            )
            sources = [row[0] for row in cursor.fetchall()]
            if len(set(sources)) > 1:
                bridge_candidates.append(candidate)

        if not bridge_candidates:
            conn.close()
            return None

        best_bridge = max(bridge_candidates, key=lambda c: c.similarity)

        cursor.execute(
            "SELECT content, node_type FROM thought_nodes WHERE id IN (?, ?)",
            (best_bridge.node1_id, best_bridge.node2_id),
        )
        nodes = cursor.fetchall()
        if len(nodes) != 2:
            conn.close()
            return None

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

        dream_id = hashlib.sha256(dream_content.encode()).hexdigest()[:12]
        cursor.execute(
            "INSERT OR REPLACE INTO thought_nodes "
            "(id, content, node_type, timestamp, mood_state, metadata, source_file) "
            "VALUES (?, ?, 'dream', ?, 'dreamy', '{}', 'sleep_protocol')",
            (dream_id, dream_content, datetime.now().isoformat()),
        )
        cursor.execute(
            "INSERT OR IGNORE INTO derivation_edges "
            "(parent_id, child_id, weight, reasoning) "
            "VALUES (?, ?, ?, 'derived_from - Dream synthesis')",
            (best_bridge.node1_id, dream_id, best_bridge.similarity),
        )
        cursor.execute(
            "INSERT OR IGNORE INTO derivation_edges "
            "(parent_id, child_id, weight, reasoning) "
            "VALUES (?, ?, ?, 'derived_from - Dream synthesis')",
            (best_bridge.node2_id, dream_id, best_bridge.similarity),
        )
        conn.commit()
        conn.close()

        self._log_event("dream", {
            "dream_id": dream_id,
            "dream_content": dream_content,
            "bridged_nodes": [best_bridge.node1_id, best_bridge.node2_id],
            "similarity": best_bridge.similarity,
        })
        return dream_id

    # ── node metrics (backward compat) ────────────────────────────────────

    def calculate_node_metrics(self) -> Dict[str, NodeMetrics]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM thought_nodes "
            "WHERE decayed = 0 OR decayed IS NULL"
        )
        rows = cursor.fetchall()
        nodes = {row[0]: None for row in rows}

        metrics = {}
        for node_id in nodes:
            cursor.execute(
                "SELECT COUNT(*) FROM derivation_edges WHERE parent_id = ?",
                (node_id,),
            )
            branching_factor = cursor.fetchone()[0]

            cursor.execute(
                "SELECT COUNT(*) FROM derivation_edges "
                "WHERE (parent_id = ? OR child_id = ?) AND reasoning LIKE '%cross_link%'",
                (node_id, node_id),
            )
            cross_links = cursor.fetchone()[0]

            cursor.execute(
                "SELECT COUNT(*) FROM derivation_edges WHERE parent_id = ?",
                (node_id,),
            )
            retrieval_frequency = cursor.fetchone()[0]

            derivation_depth = self._calculate_depth_from_seeds(node_id)

            composite_fitness = (
                branching_factor + cross_links * 0.5 + derivation_depth * 0.1
            )

            metrics[node_id] = NodeMetrics(
                node_id=node_id,
                branching_factor=branching_factor,
                cross_links=cross_links,
                retrieval_frequency=retrieval_frequency,
                derivation_depth=derivation_depth,
                composite_fitness=composite_fitness,
            )

        conn.close()
        return metrics

    def _calculate_depth_from_seeds(self, node_id: str) -> int:
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

    # ── garbage collection (backward compat) ──────────────────────────────

    def garbage_collect(self, metrics: Dict[str, NodeMetrics],
                        k_nodes: int = 20) -> List[str]:
        gc_mode = config.gc_mode
        gc_threshold = config.gc_threshold
        gc_grace_days = config.gc_grace_days
        gc_think_cycle_penalty = config.gc_think_cycle_penalty

        if gc_mode == "off":
            logger.info("GC mode is off — skipping garbage collection")
            return []

        if len(metrics) <= k_nodes:
            return []

        node_ids = list(metrics.keys())
        selected = random.sample(node_ids, min(k_nodes, len(node_ids)))

        collected_nodes = []
        conn = self._get_connection()
        cursor = conn.cursor()

        for node_id in selected:
            metric = metrics[node_id]

            cursor.execute(
                "SELECT source_file, last_accessed, permanent "
                "FROM thought_nodes WHERE id = ?",
                (node_id,),
            )
            row = cursor.fetchone()
            if not row:
                continue
            source_file, last_accessed, is_permanent = row

            if is_permanent:
                continue

            if last_accessed and gc_grace_days > 0:
                try:
                    accessed_dt = datetime.fromisoformat(
                        last_accessed.replace("Z", "+00:00")
                    )
                    age_days = (datetime.now(accessed_dt.tzinfo or None) - accessed_dt).days
                    if age_days < gc_grace_days:
                        continue
                except (ValueError, TypeError):
                    pass

            is_think_cycle = source_file and "think_cycle" in str(source_file)
            effective_threshold = (
                gc_threshold * gc_think_cycle_penalty if is_think_cycle else gc_threshold
            )

            if metric.composite_fitness < effective_threshold:
                log_decay_event(
                    conn, node_id, "gc_fitness",
                    related_nodes={},
                    metadata={
                        "fitness_score": metric.composite_fitness,
                        "threshold": effective_threshold,
                        "mode": gc_mode,
                    },
                )

                if gc_mode == "hard":
                    cursor.execute(
                        "DELETE FROM derivation_edges "
                        "WHERE parent_id = ? OR child_id = ?",
                        (node_id, node_id),
                    )
                    cursor.execute(
                        "DELETE FROM embeddings WHERE node_id = ?", (node_id,)
                    )
                    cursor.execute(
                        "DELETE FROM thought_nodes WHERE id = ?", (node_id,)
                    )
                else:
                    cursor.execute(
                        "UPDATE thought_nodes SET decayed = 1 WHERE id = ?",
                        (node_id,),
                    )
                collected_nodes.append(node_id)

                self._log_event("gc_decay", {
                    "node_id": node_id,
                    "mode": gc_mode,
                    "fitness_score": metric.composite_fitness,
                    "threshold": effective_threshold,
                    "metrics": asdict(metric),
                })

        conn.commit()
        conn.close()
        return collected_nodes

    # ── core memory promotion (backward compat) ───────────────────────────

    def promote_core_memories(self, metrics: Dict[str, NodeMetrics]) -> Tuple[List[str], List[str]]:
        conn = self._get_connection()
        cursor = conn.cursor()

        total_nodes = len(metrics)
        target_core_memories = int(math.sqrt(total_nodes))

        cursor.execute("SELECT id FROM thought_nodes WHERE node_type = 'core_memory'")
        current_core = set(row[0] for row in cursor.fetchall())

        ranked_nodes = sorted(
            metrics.values(), key=lambda m: m.composite_fitness, reverse=True
        )
        should_be_core = set(m.node_id for m in ranked_nodes[:target_core_memories])

        promotions = should_be_core - current_core
        for node_id in promotions:
            cursor.execute(
                "UPDATE thought_nodes SET node_type = 'core_memory', permanent = 1 WHERE id = ?",
                (node_id,),
            )
            self._log_event("core_promotion", {
                "node_id": node_id,
                "fitness_score": metrics[node_id].composite_fitness,
                "set_permanent": True,
            })

        cursor.execute(
            "UPDATE thought_nodes SET permanent = 1 "
            "WHERE node_type = 'core_memory' AND (permanent IS NULL OR permanent = 0)"
        )

        demotions = current_core - should_be_core
        for node_id in demotions:
            cursor.execute("SELECT node_type FROM thought_nodes WHERE id = ?", (node_id,))
            current_type = cursor.fetchone()[0]
            if current_type != "seed":
                cursor.execute(
                    "UPDATE thought_nodes SET node_type = 'derived' WHERE id = ?",
                    (node_id,),
                )
                self._log_event("core_demotion", {
                    "node_id": node_id,
                    "fitness_score": metrics.get(
                        node_id, NodeMetrics("", 0, 0, 0, 0, 0)
                    ).composite_fitness,
                    "reason": "Below core memory threshold",
                })

        conn.commit()
        conn.close()
        return list(promotions), list(demotions)

    # ── permanence evaluation (backward compat) ───────────────────────────

    def evaluate_permanence(self) -> Dict:
        from .permanence import (
            promote_permanent_nodes,
            calculate_recommended_threshold,
            validate_permanence_integrity,
            validate_embeddings_integrity,
        )

        recommended_threshold = calculate_recommended_threshold(self.db_path)
        promotion_stats = promote_permanent_nodes(self.db_path, recommended_threshold)
        integrity_stats = validate_permanence_integrity(self.db_path)
        embedding_stats = validate_embeddings_integrity(self.db_path)

        if promotion_stats["nodes_promoted"] > 0:
            self._log_event("permanence_promotion", {
                "nodes_promoted": promotion_stats["nodes_promoted"],
                "access_threshold": promotion_stats["access_threshold"],
                "integrity_check": integrity_stats,
                "embedding_check": embedding_stats,
            })

        if not embedding_stats["integrity_ok"]:
            self._log_event("embedding_integrity_violation", {
                "zero_norm": embedding_stats["zero_norm"],
                "nan_or_inf": embedding_stats["nan_or_inf"],
                "wrong_dim": embedding_stats["wrong_dim"],
                "orphan_embeddings": embedding_stats["orphan_embeddings"],
                "orphan_nodes": embedding_stats["orphan_nodes"],
                "bad_embedding_ids": embedding_stats["bad_embedding_ids"][:50],
            })

        return {
            'nodes_evaluated': promotion_stats["nodes_evaluated"],
            'nodes_made_permanent': promotion_stats["nodes_promoted"],
            'access_threshold': promotion_stats["access_threshold"],
            'integrity_ok': integrity_stats["integrity_ok"] and embedding_stats["integrity_ok"],
            'permanence_integrity': integrity_stats,
            'embedding_integrity': embedding_stats,
        }

    # ── fallback: legacy per-method orchestration (no embeddings table) ───

    def _run_sleep_cycle_legacy(self, model_fn=None) -> Dict:
        """Fallback sleep cycle using individual per-method calls (text-based cross-link).

        Used when the ``embeddings`` table doesn't exist in the database.
        Preserves the original upstream behavior exactly.
        """
        self._ensure_decayed_column()

        candidates = self.find_cross_link_candidates()

        cross_links_created = 0
        dedups_performed = 0

        for candidate in candidates:
            if candidate.action == "cross_link":
                self.cross_link_nodes(candidate.node1_id, candidate.node2_id, candidate.similarity)
                cross_links_created += 1

        dedup_candidates = [c for c in candidates if c.action == "dedup"]
        clusters = self.find_merge_clusters(dedup_candidates)
        for cluster in clusters:
            if self.merge_cluster(cluster, model_fn=model_fn):
                dedups_performed += 1

        cross_link_candidates_list = [c for c in candidates if c.action == "cross_link"]
        dream_id = self.generate_dream_node(cross_link_candidates_list, model_fn=model_fn)

        metrics = self.calculate_node_metrics()
        decayed_nodes = self.garbage_collect(metrics)
        permanence_stats = self.evaluate_permanence()
        promotions, demotions = self.promote_core_memories(metrics)

        try:
            audit_conn = self._get_connection()
            audit_pruned = gc_decay_audit(audit_conn, retention_days=7)
            audit_conn.commit()
            audit_conn.close()
            if audit_pruned:
                self._log_event("decay_audit_gc", {"rows_pruned": audit_pruned})
        except Exception as e:
            logger.warning(f"decay_audit GC failed: {e}")

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
            "clusters_found": 0,
            "new_hotspots": 0,
            "stale_hotspots": 0,
            "total_nodes": len(metrics),
            "events_logged": events_count,
        }
        logger.info("Sleep cycle complete (legacy fallback): %s", summary)
        return summary

    # ── main orchestration (now delegates to vectorized pipeline) ────────

    def run_sleep_cycle(self, model_fn=None, **kwargs) -> Dict:
        """Run a complete sleep cycle.

        This method now delegates to the vectorized free-function pipeline
        for the heavy cross-linking, dedup, GC, and core-memory phases.
        Backward-compatible — returns a summary dict with the same keys as
        the original implementation.

        Parameters
        ----------
        model_fn : callable or None
            LLM callable for dream generation.
        **kwargs
            Passed through to :func:`run_sleep_cycle`: ``limit``,
            ``background_dream``, ``max_edges``, ``cross_source_only``.
        """
        # Preserve backward-compat: old run_sleep_cycle didn't have a limit param.
        # Check if embeddings table exists — if not, fall back to old per-method path.
        conn = self._get_connection()
        has_embeddings = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='embeddings'"
        ).fetchone() is not None
        active_count = conn.execute(
            "SELECT COUNT(*) FROM thought_nodes WHERE decayed IS NULL OR decayed = 0"
        ).fetchone()[0]
        conn.close()

        if not has_embeddings:
            # Old path: use individual methods which fall back to text similarity
            return self._run_sleep_cycle_legacy(model_fn=model_fn)

        result = run_sleep_cycle(
            db_path=self.db_path,
            limit=kwargs.get("limit", active_count),
            model_fn=model_fn,
            background_dream=kwargs.get("background_dream", False),
            max_edges=kwargs.get("max_edges", MAX_EDGES_PER_CYCLE),
            cross_source_only=kwargs.get("cross_source_only", False),
        )

        # Map vectorized result back to old-style summary keys for compat
        summary = {
            "cross_links_created": result.get("cross_links_created", 0),
            "deduplications": result.get("dedup_nodes_merged", 0),
            "permanence_stats": {},
            "dream_nodes_created": 1 if result.get("dream_id") else 0,
            "nodes_decayed": result.get("nodes_gc_decayed", 0),
            "core_promotions": result.get("core_promoted", 0),
            "core_demotions": result.get("core_demoted", 0),
            "clusters_found": 0,
            "new_hotspots": 0,
            "stale_hotspots": 0,
            "total_nodes": result.get("total_nodes", 0),
            "events_logged": len(self.events),
        }

        logger.info("Sleep cycle complete: %s", summary)
        return summary

    # ── sleep log ─────────────────────────────────────────────────────────

    def save_sleep_log(self):
        try:
            existing_log = []
            try:
                with open(self.sleep_log_path, 'r') as f:
                    existing_log = json.load(f)
            except FileNotFoundError:
                pass

            new_events = [asdict(event) for event in self.events]
            existing_log.extend(new_events)

            with open(self.sleep_log_path, 'w') as f:
                json.dump(existing_log, f, indent=2)

            self.events = []
        except Exception as e:
            print(f"Warning: Could not save sleep log: {e}")


# ── CLI ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Cashew Sleep Protocol")
    parser.add_argument("command", choices=["run", "status"], help="Command to run")
    parser.add_argument("--frequency", type=int, default=10, help="Sleep every N thoughts")
    parser.add_argument("--gc-nodes", type=int, default=20, help="Nodes to consider for GC")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max nodes to process (work cap). Default: process all.")
    parser.add_argument("--background-dream", action="store_true",
                        help="Run dream phase in daemon thread")

    args = parser.parse_args()

    protocol = SleepProtocol()
    protocol.sleep_frequency = args.frequency

    if args.command == "run":
        summary = protocol.run_sleep_cycle(limit=args.limit,
                                            background_dream=args.background_dream)
        print(f"\n Sleep cycle completed:")
        for key, value in summary.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")

    elif args.command == "status":
        try:
            with open(protocol.sleep_log_path, 'r') as f:
                events = json.load(f)

            print(f"\n Sleep Protocol Status:")
            print(f"Total sleep events: {len(events)}")
            event_counts = defaultdict(int)
            for event in events:
                event_counts[event['event_type']] += 1
            for event_type, count in event_counts.items():
                print(f"  {event_type.replace('_', ' ').title()}: {count}")

        except FileNotFoundError:
            print("No sleep log found. Run a sleep cycle first.")

    return 0


# ── convenience entry point (backward compatible) ────────────────────────


# This is also exposed as a public function (the old signature).
# New callers should use the free-function ``run_sleep_cycle`` at module
# level, which has the full signature with all parameters.


if __name__ == "__main__":
    sys.exit(main())
