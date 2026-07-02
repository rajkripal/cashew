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
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger("cashew.sleep")

from .config import get_db_path, config, DEFAULT_EMBEDDING_MODEL
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
GC_ACCESS_FLOOR       = 3      # nodes retrieved at least this often are never GC'd
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


def _parse_ts(value: Optional[str]) -> Optional[datetime]:
    """Parse a stored timestamp into a tz-aware datetime (UTC), or None.

    Timestamps are written tz-aware, but legacy rows may be naive; those are
    read as UTC so comparisons against a tz-aware ``now`` never raise.
    """
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _resolve_expected_dim() -> int:
    """Configured embedding model's dim, with a MiniLM fallback if the
    embedding service cannot be constructed (e.g. in CI without the model
    cached)."""
    try:
        from .embedding_service import get_default_service
        return get_default_service().dim
    except Exception:
        return 384


def _load_embedding_matrix(
    conn: sqlite3.Connection, node_ids: List[str],
) -> Tuple[List[str], np.ndarray]:
    """Load embeddings for *node_ids* from the ``embeddings`` table.

    Returns (valid_ids, matrix) where *matrix* has shape (N, embedding_dim).
    Filters NaN, inf, zero, and wrong-dim vectors. Wrong-dim filtering is
    keyed off the configured model's dim so a partially-migrated brain
    cannot crash sleep at the ``np.array(vectors)`` call.
    """
    if not node_ids:
        return [], np.array([])

    placeholders = ",".join("?" * len(node_ids))
    rows = conn.execute(
        f"SELECT e.node_id, e.vector FROM embeddings e "
        f"WHERE e.node_id IN ({placeholders})",
        node_ids,
    ).fetchall()

    expected_dim = _resolve_expected_dim()

    vectors: List[np.ndarray] = []
    valid_ids: List[str] = []
    bad = 0
    wrong_dim = 0
    for nid, blob in rows:
        try:
            vec = np.frombuffer(blob, dtype=np.float32)
            if np.any(np.isnan(vec)) or np.any(np.isinf(vec)):
                bad += 1
                continue
            if np.allclose(vec, 0):
                bad += 1
                continue
            if len(vec) != expected_dim:
                wrong_dim += 1
                continue
            valid_ids.append(nid)
            vectors.append(vec)
        except Exception:
            bad += 1

    if bad:
        logger.warning("sleep: skipped %d bad embeddings", bad)
    if wrong_dim:
        logger.warning(
            "sleep: skipped %d embeddings with dim != %d (configured model). "
            "Run `cashew migrate-embeddings -y` to re-embed under the current model.",
            wrong_dim, expected_dim,
        )
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

    Keeper: a permanent node always wins so duplicates collapse into it;
    otherwise highest access_count, tiebreak oldest timestamp. Rewires loser
    edges onto the keeper, then decays the losers.

    A permanent loser (only possible when the keeper is itself permanent, i.e.
    two permanent duplicates) is absorbed: its permanent flag is cleared before
    it is decayed, so the memory survives in the permanent keeper and no
    permanent node is ever left decayed. The graph keeps no duplicate branches.
    """
    if len(cluster_ids) < 2:
        return None

    # permanent >= 1 covers both auto-promoted (1) and manually-pinned (2).
    perm = {
        r[0] for r in conn.execute(
            "SELECT id FROM thought_nodes WHERE id IN ({}) AND permanent >= 1".format(
                ",".join("?" * len(cluster_ids))
            ),
            cluster_ids,
        ).fetchall()
    }

    keeper = conn.execute(
        "SELECT id FROM thought_nodes WHERE id IN ({}) "
        "ORDER BY COALESCE(permanent, 0) DESC, COALESCE(access_count, 0) DESC, "
        "COALESCE(timestamp, '9999') ASC LIMIT 1".format(
            ",".join("?" * len(cluster_ids))
        ),
        cluster_ids,
    ).fetchone()
    if not keeper:
        return None

    keeper_id = keeper[0]
    losers = [n for n in cluster_ids if n != keeper_id]
    if not losers:
        return None
    loser_set = set(losers)
    lp = ",".join("?" * len(losers))

    # Read all edges touching a loser, rewire the loser endpoint to the keeper.
    edges = conn.execute(
        "SELECT parent_id, child_id, weight, reasoning "
        "FROM derivation_edges "
        "WHERE parent_id IN ({0}) OR child_id IN ({0})".format(lp),
        losers + losers,
    ).fetchall()
    conn.execute(
        "DELETE FROM derivation_edges "
        "WHERE parent_id IN ({0}) OR child_id IN ({0})".format(lp),
        losers + losers,
    )
    for parent_id, child_id, weight, reasoning in edges:
        new_parent = keeper_id if parent_id in loser_set else parent_id
        new_child = keeper_id if child_id in loser_set else child_id
        if new_parent == new_child:
            continue
        conn.execute(
            "INSERT OR IGNORE INTO derivation_edges "
            "(parent_id, child_id, weight, reasoning) VALUES (?, ?, ?, ?)",
            (new_parent, new_child, weight, reasoning),
        )

    # A permanent loser is absorbed into the permanent keeper: clear its flag
    # first so the "no permanent node is decayed" invariant holds.
    perm_losers = [n for n in losers if n in perm]
    if perm_losers:
        conn.execute(
            "UPDATE thought_nodes SET permanent=0 WHERE id IN ({})".format(
                ",".join("?" * len(perm_losers))
            ),
            perm_losers,
        )

    # Audit each loser before decaying so the row records why it went.
    for lid in losers:
        log_decay_event(
            conn, lid, "dedup_loser",
            related_nodes={"keeper": keeper_id},
            metadata={"absorbed_permanent": True} if lid in perm else None,
        )
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
    # Don't prune a graph no larger than one sample: too little signal, and a
    # young brain needs its sparse nodes to accrue edges first.
    if len(metrics) <= sample_k:
        return []

    now = datetime.now(timezone.utc)

    candidates: List[Tuple[str, float, Optional[str]]] = []
    for nid, m in metrics.items():
        fitness = m["fitness"]

        # One read per candidate covers every protection check.
        row = conn.execute(
            "SELECT permanent, access_count, last_accessed, source_file "
            "FROM thought_nodes WHERE id = ?",
            (nid,),
        ).fetchone()
        if not row:
            continue
        permanent, access_count, last_accessed, source_file = row

        # Permanent nodes can never be collected (>=1 covers pinned=2 too).
        if (permanent or 0) >= 1:
            continue

        # Frequently-retrieved nodes are load-bearing even when orphaned;
        # this is the orphan protection that replaces the dropped confidence
        # bonus (confidence was removed in the v2 migration).
        if (access_count or 0) >= GC_ACCESS_FLOOR:
            continue

        # Grace period: recently-accessed nodes are exempt. last_accessed is
        # written tz-aware; a naive value (legacy rows) is read as UTC. Both
        # `now` and `la` are tz-aware here, so the subtraction can't raise the
        # way the old naive `datetime.now()` did.
        if grace_days > 0 and last_accessed:
            la = _parse_ts(last_accessed)
            if la is not None and (now - la).total_seconds() / 86400 < grace_days:
                continue

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
        # Audit before mutating so the row can still read the node's fields.
        log_decay_event(conn, nid, "gc_fitness", metadata={"fitness": fitness})
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
    model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)

    embedded = 0
    for nid, content in rows:
        try:
            vec = model.encode(content, normalize_embeddings=True)
            blob = vec.astype(np.float32).tobytes()

            if not blob:
                logger.warning(
                    "sleep: skipping node %s — embedding produced empty bytes", nid[:8]
                )
                continue

            try:
                conn.execute(
                    "INSERT OR REPLACE INTO embeddings "
                    "(node_id, vector, model, updated_at) "
                    "VALUES (?, ?, ?, datetime('now'))",
                    (nid, blob, DEFAULT_EMBEDDING_MODEL),
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
            conn.execute("PRAGMA busy_timeout = 5000")
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
    conn.execute("PRAGMA busy_timeout = 5000")
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
        audit_conn.execute("PRAGMA busy_timeout = 5000")
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
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    def run_sleep_cycle(self, model_fn=None, **kwargs) -> Dict:
        """Run a complete sleep cycle.

        Delegates to the vectorized free-function pipeline (cross-linking,
        dedup, GC, core-memory) and maps the result back to the historical
        summary keys so existing callers keep working. The free function
        handles the no-embeddings case itself.

        Parameters
        ----------
        model_fn : callable or None
            LLM callable for dream generation.
        **kwargs
            Passed through to :func:`run_sleep_cycle`: ``limit``,
            ``background_dream``, ``max_edges``, ``cross_source_only``.
        """
        conn = self._get_connection()
        active_count = conn.execute(
            "SELECT COUNT(*) FROM thought_nodes WHERE decayed IS NULL OR decayed = 0"
        ).fetchone()[0]
        conn.close()

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
