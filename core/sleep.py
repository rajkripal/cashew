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
from .decay_audit import log_decay_event, gc_decay_audit, ensure_decay_audit_schema

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
ORPHANS_PER_BATCH     = 100    # conservative caller-facing encode batch
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
    """Configured profile dimension without constructing an embedding model."""
    try:
        return _get_active_profile().dim
    except Exception:
        return 384


def _load_embedding_matrix(
    conn: sqlite3.Connection, node_ids: List[str],
    expected_dimension: Optional[int] = None,
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

    expected_dim = (
        int(expected_dimension)
        if expected_dimension is not None else _resolve_expected_dim()
    )

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
            if not np.any(vec):
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
        return [], np.array([], dtype=np.float64)
    # float64: cosine matmul on float32 raises spurious FPE warnings (divide-by-
    # zero / overflow / invalid) under Apple's Accelerate BLAS even though the
    # results are finite and correct (float32 vs float64 differ by ~1e-6).
    return valid_ids, np.array(vectors, dtype=np.float64)


_SLEEP_CURSOR_NAME = "candidate_cursor_v1"
_ORPHAN_MISSING_CURSOR_NAME = "orphan_missing_cursor_v1"
_ORPHAN_REPAIR_CURSOR_NAME = "orphan_repair_cursor_v1"
_ORPHAN_PHASE_CURSOR_NAME = "orphan_phase_cursor_v1"
_SLEEP_CURSOR_NAMES = {
    _SLEEP_CURSOR_NAME,
    _ORPHAN_MISSING_CURSOR_NAME,
    _ORPHAN_REPAIR_CURSOR_NAME,
    _ORPHAN_PHASE_CURSOR_NAME,
}


def _create_sleep_state_table(conn: sqlite3.Connection) -> None:
    """Create the private cursor table in its current on-disk shape."""
    conn.execute(
        "CREATE TABLE _cashew_sleep_state ("
        "name TEXT PRIMARY KEY, cursor_timestamp TEXT NOT NULL, "
        "cursor_node_id TEXT NOT NULL, epoch INTEGER NOT NULL)"
    )


def _valid_sleep_cursor_row(row: tuple) -> bool:
    """Return whether one private cursor row has canonical SQLite types."""
    return (
        len(row) == 3
        and isinstance(row[0], str)
        and isinstance(row[1], str)
        and isinstance(row[2], int)
        and not isinstance(row[2], bool)
        and row[2] >= 0
    )


def _ensure_sleep_state_schema(conn: sqlite3.Connection) -> None:
    """Validate or transactionally migrate the private cursor table.

    The table was introduced as private state and has no user-owned columns.
    Rebuilding a partial shape is safer than letting a later cursor query fail
    halfway through sleep.  A usable legacy cursor is retained; otherwise the
    next page starts at the deterministic beginning.
    """
    entry = conn.execute(
        "SELECT type FROM sqlite_master WHERE name='_cashew_sleep_state'"
    ).fetchone()
    if entry is None:
        _create_sleep_state_table(conn)
        return
    if entry[0] != "table":
        raise sqlite3.DatabaseError("sleep_state_schema_invalid")

    info = conn.execute("PRAGMA table_info(_cashew_sleep_state)").fetchall()
    shape = [(row[1], (row[2] or "").upper(), row[3], row[5]) for row in info]
    expected = [
        ("name", "TEXT", 0, 1),
        ("cursor_timestamp", "TEXT", 1, 0),
        ("cursor_node_id", "TEXT", 1, 0),
        ("epoch", "INTEGER", 1, 0),
    ]
    if shape == expected:
        for name in sorted(_SLEEP_CURSOR_NAMES):
            row = conn.execute(
                "SELECT cursor_timestamp, cursor_node_id, epoch "
                "FROM _cashew_sleep_state WHERE name=?",
                (name,),
            ).fetchone()
            if row is not None and not _valid_sleep_cursor_row(row):
                conn.execute(
                    "DELETE FROM _cashew_sleep_state WHERE name=?", (name,)
                )
        return

    columns = {row[1] for row in info}
    saved: List[Tuple[str, str, str, int]] = []
    cursor_columns = {"name", "cursor_timestamp", "cursor_node_id"}
    if cursor_columns.issubset(columns):
        epoch_expr = "epoch" if "epoch" in columns else "0"
        for name in sorted(_SLEEP_CURSOR_NAMES):
            rows = conn.execute(
                "SELECT cursor_timestamp, cursor_node_id, " + epoch_expr + " "
                "FROM _cashew_sleep_state WHERE name=?",
                (name,),
            ).fetchall()
            # A malformed table may contain duplicate or type-confused rows.
            # There is no canonical owner in that case, so reset this private
            # cursor to its deterministic origin rather than preserving an
            # arbitrary SQLite row.
            if len(rows) != 1:
                continue
            if not _valid_sleep_cursor_row(rows[0]):
                continue
            timestamp, node_id, epoch_value = rows[0]
            saved.append((name, timestamp, node_id, epoch_value))

    conn.execute("DROP TABLE _cashew_sleep_state")
    _create_sleep_state_table(conn)
    if saved:
        conn.executemany(
            "INSERT INTO _cashew_sleep_state "
            "(name, cursor_timestamp, cursor_node_id, epoch) VALUES (?, ?, ?, ?)",
            saved,
        )


def _select_cycle_node_ids(
    conn: sqlite3.Connection, limit: Optional[int],
) -> List[str]:
    """Select one deterministic page and durably advance its private cursor.

    Uncapped callers retain the historical full oldest-first pass and do not
    create cursor state.  Capped callers rotate through ``(timestamp, id)`` so
    an already-saturated or unproductive oldest page cannot monopolize every
    cycle.  The cursor claim commits before expensive work: a crash can defer a
    page until the next wrap, but cannot corrupt the cursor or starve later
    pages forever.
    """
    thought_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(thought_nodes)")
    }
    timestamp_expr = (
        "COALESCE(tn.timestamp, '')" if "timestamp" in thought_columns else "''"
    )
    base = (
        f"SELECT e.node_id, {timestamp_expr} AS sleep_ts "
        "FROM embeddings e JOIN thought_nodes tn ON e.node_id = tn.id "
        "WHERE (tn.decayed IS NULL OR tn.decayed = 0) "
    )
    order = "ORDER BY sleep_ts ASC, e.node_id ASC "
    if limit is None:
        return [row[0] for row in conn.execute(base + order).fetchall()]
    if limit == 0:
        return []

    try:
        conn.execute("BEGIN IMMEDIATE")
        _ensure_sleep_state_schema(conn)
        state = conn.execute(
            "SELECT cursor_timestamp, cursor_node_id, epoch "
            "FROM _cashew_sleep_state WHERE name = ?",
            (_SLEEP_CURSOR_NAME,),
        ).fetchone()
        rows: List[tuple] = []
        epoch = 0
        if state is not None:
            cursor_timestamp, cursor_node_id, epoch = state
            rows.extend(
                conn.execute(
                    base
                    + f"AND ({timestamp_expr} > ? OR "
                    f"({timestamp_expr} = ? AND e.node_id > ?)) "
                    + order
                    + "LIMIT ?",
                    (cursor_timestamp, cursor_timestamp, cursor_node_id, limit),
                ).fetchall()
            )
        if len(rows) < limit:
            remaining = limit - len(rows)
            if state is None:
                rows.extend(
                    conn.execute(base + order + "LIMIT ?", (remaining,)).fetchall()
                )
            else:
                cursor_timestamp, cursor_node_id, _ = state
                rows.extend(
                    conn.execute(
                        base
                        + f"AND ({timestamp_expr} < ? OR "
                        f"({timestamp_expr} = ? AND e.node_id <= ?)) "
                        + order
                        + "LIMIT ?",
                        (cursor_timestamp, cursor_timestamp, cursor_node_id, remaining),
                    ).fetchall()
                )
        if rows:
            last_id, last_timestamp = rows[-1]
            conn.execute(
                "INSERT OR REPLACE INTO _cashew_sleep_state "
                "(name, cursor_timestamp, cursor_node_id, epoch) VALUES (?, ?, ?, ?)",
                (_SLEEP_CURSOR_NAME, last_timestamp, last_id, int(epoch) + 1),
            )
        conn.commit()
        return [row[0] for row in rows]
    except Exception:
        conn.rollback()
        raise


# ── Phase 1: candidate discovery (vectorized) ────────────────────────────


def _find_pairs(
    ids: List[str], matrix: np.ndarray,
    cross_threshold: Optional[float] = None,
    dedup_threshold: Optional[float] = None,
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
    cross_threshold = (
        CROSS_LINK_THRESHOLD if cross_threshold is None else cross_threshold
    )
    dedup_threshold = DEDUP_THRESHOLD if dedup_threshold is None else dedup_threshold
    cross_mask = (upper >= cross_threshold) & (upper < dedup_threshold)
    dedup_mask = upper >= dedup_threshold

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
    progress: Optional[dict] = None,
) -> dict:
    """Insert cross-link pairs atomically, with pair-budget accounting.

    Each candidate is isolated in a savepoint and verified in both directions.
    Successful pairs are committed in bounded batches so a trigger,
    constraint, or interrupted write cannot produce a claimed half-pair while
    a large cycle still avoids one transaction per edge.
    """
    stats = {
        "candidates": len(cross_pairs), "created": 0, "repaired": 0,
        "skipped": 0, "failed": 0, "directed_rows": 0,
        "same_source_skipped": 0, "capped": False,
        # Private handoff to dream generation.  Entries are added only after
        # the transaction containing both directed rows commits.
        "dream_pairs": [],
    }
    t0 = time.perf_counter()
    dirty = False
    pending_dream_pairs: List[Tuple[str, str, float]] = []
    pending_created = 0
    pending_repaired = 0
    pending_directed_rows = 0
    pending_pairs: List[Tuple[str, str, bool]] = []

    def publish_progress(*, uncertain: bool = False) -> None:
        if progress is None:
            return
        progress.update({
            "cross_links_created": stats["created"],
            "cross_links_repaired": stats["repaired"],
            "cross_links_skipped": stats["skipped"],
            "cross_link_directed_rows": stats["directed_rows"],
            "cross_link_same_source_skipped": stats["same_source_skipped"],
            "cross_link_capped": stats["capped"],
        })
        if uncertain:
            progress["_outcome_uncertain"] = True

    def apply_pending() -> None:
        nonlocal dirty, pending_created, pending_repaired, pending_directed_rows
        if not dirty:
            return
        stats["created"] += pending_created
        stats["repaired"] += pending_repaired
        stats["directed_rows"] += pending_directed_rows
        stats["dream_pairs"].extend(pending_dream_pairs)
        pending_dream_pairs.clear()
        pending_pairs.clear()
        pending_created = 0
        pending_repaired = 0
        pending_directed_rows = 0
        dirty = False
        publish_progress()

    def commit_pending() -> bool:
        if not dirty:
            return True
        try:
            conn.commit()
        except Exception:
            try:
                conn.rollback()
            except sqlite3.Error:
                pass
            # Verify the attempted transaction after rollback.  SQLite commit
            # errors normally leave every pending pair at its original shape;
            # a wrapper can also raise after a successful commit.  Only those
            # two all-or-nothing outcomes are knowable.  A failed verification
            # or mixed state is genuinely uncertain and must not invent public
            # counters or dream inputs.
            try:
                final_counts = []
                for n1, n2, _was_repair in pending_pairs:
                    final_counts.append(conn.execute(
                        "SELECT count(*) FROM derivation_edges "
                        "WHERE (parent_id=? AND child_id=?) OR "
                        "(parent_id=? AND child_id=?)",
                        (n1, n2, n2, n1),
                    ).fetchone()[0])
            except Exception:
                publish_progress(uncertain=True)
                raise
            original_counts = [1 if repaired else 0 for _, _, repaired in pending_pairs]
            if final_counts == [2] * len(pending_pairs):
                apply_pending()
                return True
            elif final_counts != original_counts:
                publish_progress(uncertain=True)
                raise
            else:
                pending_dream_pairs.clear()
                pending_pairs.clear()
                # The whole attempted transaction rolled back.  The already
                # committed prefix remains exact and the failed suffix does
                # not contribute to pair or directed-row counters.
                nonlocal_reset_pending()
                publish_progress()
            stats["failed"] += 1
            return False
        apply_pending()
        return True

    def nonlocal_reset_pending() -> None:
        nonlocal dirty, pending_created, pending_repaired, pending_directed_rows
        pending_created = 0
        pending_repaired = 0
        pending_directed_rows = 0
        dirty = False

    for pair_no, (i, j) in enumerate(cross_pairs):
        budget = (
            stats["created"] + stats["repaired"]
            + pending_created + pending_repaired
        )
        if max_edges is not None and budget >= max_edges:
            stats["capped"] = True
            publish_progress()
            break
        n1, n2 = ids[int(i)], ids[int(j)]
        if source_files is not None:
            sf1, sf2 = source_files.get(n1, ""), source_files.get(n2, "")
            if sf1 and sf2 and sf1 == sf2:
                stats["same_source_skipped"] += 1
                continue
        present = {(p, c) for p, c in conn.execute(
            "SELECT parent_id, child_id FROM derivation_edges "
            "WHERE (parent_id=? AND child_id=?) OR (parent_id=? AND child_id=?)",
            (n1, n2, n2, n1),
        ).fetchall()}
        if len(present) == 2:
            stats["skipped"] += 1
            continue
        missing = [(n1, n2), (n2, n1)]
        missing = [(p, c) for p, c in missing if (p, c) not in present]
        name = f"cross_pair_{pair_no}"
        started_batch = not dirty
        try:
            if started_batch:
                conn.execute("BEGIN IMMEDIATE")
            conn.execute(f"SAVEPOINT {name}")
            weight = float(sim[int(i), int(j)])
            conn.executemany(
                "INSERT OR IGNORE INTO derivation_edges "
                "(parent_id, child_id, weight, reasoning) VALUES (?, ?, ?, ?)",
                [(p, c, weight, f"cross_link - similarity={weight:.3f}")
                 for p, c in missing],
            )
            final = {(p, c) for p, c in conn.execute(
                "SELECT parent_id, child_id FROM derivation_edges "
                "WHERE (parent_id=? AND child_id=?) OR (parent_id=? AND child_id=?)",
                (n1, n2, n2, n1),
            ).fetchall()}
            if len(final) != 2:
                raise sqlite3.IntegrityError("cross_link_pair_not_persisted")
            conn.execute(f"RELEASE SAVEPOINT {name}")
            dirty = True
        except Exception as exc:
            try:
                conn.execute(f"ROLLBACK TO SAVEPOINT {name}")
                conn.execute(f"RELEASE SAVEPOINT {name}")
            except sqlite3.Error:
                pass
            if started_batch:
                try:
                    conn.rollback()
                except sqlite3.Error:
                    pass
            stats["failed"] += 1
            logger.warning("sleep: cross-link pair failed: %s", type(exc).__name__)
            continue
        if len(present) == 1:
            pending_repaired += 1
        else:
            pending_created += 1
        pending_directed_rows += len(missing)
        pending_dream_pairs.append((n1, n2, weight))
        pending_pairs.append((n1, n2, len(present) == 1))
        if dirty and len(pending_dream_pairs) >= EDGES_PER_BATCH:
            if not commit_pending():
                break
    if dirty:
        commit_pending()
    publish_progress()
    elapsed = time.perf_counter() - t0
    logger.info(
        "sleep: cross-links %d created, %d repaired, %d skipped in %.1fs",
        stats["created"], stats["repaired"], stats["skipped"], elapsed,
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
            # Never promote a decayed node: it can be in the top-√N by fitness
            # (metrics are computed pre-GC, and Phase 5 may have just decayed it),
            # and permanent=1 on a decayed node breaks the permanence invariant
            # (validate_permanence_integrity: permanent_but_decayed must be 0).
            f"UPDATE thought_nodes SET node_type='core_memory', permanent=1 "
            f"WHERE id IN ({pp}) AND (decayed IS NULL OR decayed = 0)",
            list(promoted),
        )

    conn.execute(
        "UPDATE thought_nodes SET permanent=1 "
        "WHERE node_type='core_memory' AND (permanent IS NULL OR permanent = 0) "
        "AND (decayed IS NULL OR decayed = 0)"
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


def _vec_write_capability(conn: sqlite3.Connection) -> bool:
    """Return whether this connection can actually read/write the vec table.

    A virtual ``vec0`` table requires loading sqlite-vec on *this* connection;
    merely finding its name in sqlite_master is insufficient.  A missing
    extension/table is a genuine ordinary-only capability loss.  Once the
    virtual table is readable, later write errors remain hard dual-write
    failures and are handled by the per-node savepoint.
    """
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='vec_embeddings'"
    ).fetchone()
    if not row:
        return False
    ddl = (row[0] or "").lower()
    if (
        re.match(r"\s*create\s+virtual\s+table\b", ddl) is None
        or re.search(r"\busing\s+vec0\s*\(", ddl) is None
    ):
        return False
    try:
        import sqlite_vec
    except ImportError:
        return False
    try:
        try:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
        except (OSError, sqlite3.Error):
            # Enable/load failure is the only ordinary-only capability loss.
            # Errors after this point prove that the extension loaded and must
            # remain visible as a hard dual-write/schema failure.
            return False
        conn.execute("SELECT count(*) FROM vec_embeddings").fetchone()
        return True
    finally:
        try:
            conn.enable_load_extension(False)
        except sqlite3.Error:
            pass


def _claim_orphan_phase_order(
    conn: sqlite3.Connection, *, capped: bool, vec_available: bool,
) -> Tuple[str, ...]:
    """Alternate capped ordinary and vec-repair admission across cycles."""
    if not capped or not vec_available:
        return ("ordinary", "repair") if vec_available else ("ordinary",)
    try:
        conn.execute("BEGIN IMMEDIATE")
        _ensure_sleep_state_schema(conn)
        row = conn.execute(
            "SELECT epoch FROM _cashew_sleep_state WHERE name=?",
            (_ORPHAN_PHASE_CURSOR_NAME,),
        ).fetchone()
        try:
            epoch = int(row[0]) if row is not None else 0
        except (TypeError, ValueError, OverflowError):
            epoch = 0
        epoch = max(0, epoch)
        conn.execute(
            "INSERT OR REPLACE INTO _cashew_sleep_state "
            "(name, cursor_timestamp, cursor_node_id, epoch) VALUES (?, '', '', ?)",
            (_ORPHAN_PHASE_CURSOR_NAME, epoch + 1),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return ("ordinary", "repair") if epoch % 2 == 0 else ("repair", "ordinary")


def _select_orphan_page(
    conn: sqlite3.Connection,
    *,
    kind: str,
    take: int,
    timestamp_expr: str,
    embedding_columns: Set[str],
) -> Tuple[List[tuple], bool]:
    """Claim one durable capped orphan page before inference or DML.

    The boolean is true at a deterministic ordering boundary.  Callers stop
    this phase there so a short tail is not immediately followed by the same
    oldest failing rows in one cycle.
    """
    if kind == "ordinary":
        state_name = _ORPHAN_MISSING_CURSOR_NAME
        select = f"SELECT tn.id, tn.content, {timestamp_expr} "
        source = (
            "FROM thought_nodes tn "
            "LEFT JOIN embeddings e ON tn.id = e.node_id "
            "WHERE e.node_id IS NULL AND (tn.decayed IS NULL OR tn.decayed = 0) "
            "AND tn.content IS NOT NULL AND TRIM(tn.content) != '' "
        )
        id_expr = "tn.id"
    elif kind == "repair":
        state_name = _ORPHAN_REPAIR_CURSOR_NAME
        select = (
            "SELECT e.node_id, e.vector, "
            + ("COALESCE(e.model, '')" if "model" in embedding_columns else "''")
            + f", {timestamp_expr} "
        )
        source = (
            "FROM embeddings e "
            "LEFT JOIN vec_embeddings v ON v.node_id=e.node_id "
            "JOIN thought_nodes tn ON tn.id=e.node_id "
            "WHERE v.node_id IS NULL AND (tn.decayed IS NULL OR tn.decayed=0) "
        )
        id_expr = "e.node_id"
    else:
        raise ValueError("invalid_orphan_phase")

    order = f"ORDER BY {timestamp_expr}, {id_expr} LIMIT ?"
    try:
        conn.execute("BEGIN IMMEDIATE")
        _ensure_sleep_state_schema(conn)
        state = conn.execute(
            "SELECT cursor_timestamp, cursor_node_id, epoch "
            "FROM _cashew_sleep_state WHERE name=?",
            (state_name,),
        ).fetchone()
        rows: List[tuple] = []
        wrapped = False
        epoch = 0
        if (
            state is not None
            and isinstance(state[0], str)
            and isinstance(state[1], str)
        ):
            cursor_timestamp, cursor_node_id, epoch_value = state
            try:
                epoch = max(0, int(epoch_value))
            except (TypeError, ValueError, OverflowError):
                epoch = 0
            rows = conn.execute(
                select
                + source
                + f"AND ({timestamp_expr} > ? OR "
                + f"({timestamp_expr} = ? AND {id_expr} > ?)) "
                + order,
                (cursor_timestamp, cursor_timestamp, cursor_node_id, take),
            ).fetchall()
            if not rows:
                wrapped = True
                rows = conn.execute(select + source + order, (take,)).fetchall()
        else:
            rows = conn.execute(select + source + order, (take,)).fetchall()
        if rows:
            last_id = rows[-1][0]
            last_timestamp = rows[-1][-1]
            conn.execute(
                "INSERT OR REPLACE INTO _cashew_sleep_state "
                "(name, cursor_timestamp, cursor_node_id, epoch) VALUES (?, ?, ?, ?)",
                (state_name, last_timestamp, last_id, epoch + 1),
            )
        conn.commit()
        return rows, wrapped or len(rows) < take
    except Exception:
        conn.rollback()
        raise


def _embed_orphans(
    conn: sqlite3.Connection,
    *,
    embedding_client=None,
    embedding_model: Optional[str] = None,
    expected_dimension: Optional[int] = None,
    limit: Optional[int] = None,
    batch_size: int = ORPHANS_PER_BATCH,
    stats: Optional[dict] = None,
) -> int:
    """Embed active orphans through the caller-owned ``encode`` client.

    This helper never constructs a model or selects a device; the entry point
    supplies the default backend or injected client. Inference happens one
    bounded batch at a time before its write
    transaction. Each node is dual-written under a savepoint; a loaded vec
    index that rejects a write rolls back the ordinary row as well.
    """
    stats = stats if stats is not None else {}
    stats.setdefault("orphan_write_failed", 0)
    stats.setdefault("orphan_vec_unavailable", 0)
    stats.setdefault("orphan_ordinary_written", 0)
    stats.setdefault("orphan_examined", 0)
    if limit is not None and (
        not isinstance(limit, int) or isinstance(limit, bool) or limit < 0
    ):
        raise ValueError("invalid_orphan_limit")
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
        or batch_size > ORPHANS_PER_BATCH
    ):
        raise ValueError("invalid_orphan_batch_size")
    if limit == 0:
        return 0

    vec_available = _vec_write_capability(conn)
    ordinary_exists = conn.execute(
        "SELECT 1 FROM thought_nodes tn LEFT JOIN embeddings e ON tn.id=e.node_id "
        "WHERE e.node_id IS NULL AND (tn.decayed IS NULL OR tn.decayed=0) "
        "AND tn.content IS NOT NULL AND TRIM(tn.content) != '' LIMIT 1"
    ).fetchone()
    repair_exists = None
    if vec_available:
        repair_exists = conn.execute(
            "SELECT 1 FROM embeddings e LEFT JOIN vec_embeddings v ON v.node_id=e.node_id "
            "JOIN thought_nodes tn ON tn.id=e.node_id "
            "WHERE v.node_id IS NULL AND (tn.decayed IS NULL OR tn.decayed=0) LIMIT 1"
        ).fetchone()
    if not ordinary_exists and not repair_exists:
        return 0
    if (
        embedding_client is None or not embedding_model or
        expected_dimension is None or int(expected_dimension) <= 0
    ):
        stats["capability_missing"] = True
        return 0

    expected = int(expected_dimension) if expected_dimension is not None else 0
    embedded = 0
    remaining = limit
    columns = {r[1] for r in conn.execute("PRAGMA table_info(embeddings)")}
    thought_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(thought_nodes)")
    }
    timestamp_expr = (
        "COALESCE(tn.timestamp, '')" if "timestamp" in thought_columns else "''"
    )

    def write_pair(nid: str, blob: bytes, model_name: str) -> bool:
        savepoint = "orphan_" + re.sub(r"[^A-Za-z0-9_]", "_", nid)[:40]
        try:
            conn.execute(f"SAVEPOINT {savepoint}")
            if {"model", "updated_at"}.issubset(columns):
                conn.execute(
                    "INSERT OR REPLACE INTO embeddings "
                    "(node_id, vector, model, updated_at) "
                    "VALUES (?, ?, ?, datetime('now'))",
                    (nid, blob, model_name),
                )
            else:
                conn.execute(
                    "INSERT OR REPLACE INTO embeddings (node_id, vector) VALUES (?, ?)",
                    (nid, blob),
                )
            if vec_available:
                conn.execute(
                    "INSERT OR REPLACE INTO vec_embeddings "
                    "(node_id, embedding) VALUES (?, ?)",
                    (nid, blob),
                )
            conn.execute(f"RELEASE SAVEPOINT {savepoint}")
            return True
        except Exception as exc:
            try:
                conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                conn.execute(f"RELEASE SAVEPOINT {savepoint}")
            except sqlite3.Error:
                pass
            logger.warning(
                "sleep: orphan %s dual-write failed: %s", nid[:8], type(exc).__name__
            )
            return False

    def commit_batch(items: List[Tuple[str, bytes, str]]) -> bool:
        nonlocal embedded
        if not items:
            return True
        try:
            conn.execute("BEGIN IMMEDIATE")
        except Exception as exc:
            try:
                conn.rollback()
            except sqlite3.Error:
                pass
            stats["orphan_write_failed"] += len(items)
            logger.warning(
                "sleep: orphan batch admission failed: %s", type(exc).__name__
            )
            return False
        written = 0
        failed = 0
        for nid, blob, model_name in items:
            if write_pair(nid, blob, model_name):
                written += 1
            else:
                failed += 1
        try:
            conn.commit()
        except Exception as exc:
            conn.rollback()
            stats["orphan_write_failed"] += len(items)
            logger.warning(
                "sleep: orphan batch commit failed: %s", type(exc).__name__
            )
            return False
        stats["orphan_write_failed"] += failed
        stats["orphan_ordinary_written"] += written
        if vec_available:
            embedded += written
        else:
            stats["orphan_vec_unavailable"] += written
        return True

    ordinary_cursor: Optional[Tuple[str, str]] = None
    repair_cursor: Optional[Tuple[str, str]] = None

    def ordinary_page(take: int) -> Tuple[List[tuple], bool]:
        nonlocal ordinary_cursor
        if limit is not None:
            return _select_orphan_page(
                conn, kind="ordinary", take=take,
                timestamp_expr=timestamp_expr, embedding_columns=columns,
            )
        cursor_sql = ""
        params: List[object] = []
        if ordinary_cursor is not None:
            cursor_sql = (
                f"AND ({timestamp_expr} > ? OR "
                f"({timestamp_expr} = ? AND tn.id > ?)) "
            )
            params.extend(
                [ordinary_cursor[0], ordinary_cursor[0], ordinary_cursor[1]]
            )
        params.append(take)
        rows = conn.execute(
            f"SELECT tn.id, tn.content, {timestamp_expr} FROM thought_nodes tn "
            "LEFT JOIN embeddings e ON tn.id = e.node_id "
            "WHERE e.node_id IS NULL AND (tn.decayed IS NULL OR tn.decayed = 0) "
            "AND tn.content IS NOT NULL AND TRIM(tn.content) != '' "
            + cursor_sql
            + f"ORDER BY {timestamp_expr}, tn.id LIMIT ?",
            params,
        ).fetchall()
        if rows:
            ordinary_cursor = (rows[-1][2], rows[-1][0])
        return rows, False

    def repair_page(take: int) -> Tuple[List[tuple], bool]:
        nonlocal repair_cursor
        if limit is not None:
            return _select_orphan_page(
                conn, kind="repair", take=take,
                timestamp_expr=timestamp_expr, embedding_columns=columns,
            )
        cursor_sql = ""
        params: List[object] = []
        if repair_cursor is not None:
            cursor_sql = (
                f"AND ({timestamp_expr} > ? OR "
                f"({timestamp_expr} = ? AND e.node_id > ?)) "
            )
            params.extend([repair_cursor[0], repair_cursor[0], repair_cursor[1]])
        params.append(take)
        rows = conn.execute(
            "SELECT e.node_id, e.vector, "
            + ("COALESCE(e.model, '')" if "model" in columns else "''")
            + f", {timestamp_expr} FROM embeddings e "
            "LEFT JOIN vec_embeddings v ON v.node_id=e.node_id "
            "JOIN thought_nodes tn ON tn.id=e.node_id "
            "WHERE v.node_id IS NULL AND (tn.decayed IS NULL OR tn.decayed=0) "
            + cursor_sql
            + f"ORDER BY {timestamp_expr}, e.node_id LIMIT ?",
            params,
        ).fetchall()
        if rows:
            repair_cursor = (rows[-1][3], rows[-1][0])
        return rows, False

    phase_order = _claim_orphan_phase_order(
        conn, capped=limit is not None, vec_available=vec_available,
    )
    for phase in phase_order:
        while remaining is None or remaining > 0:
            take = batch_size if remaining is None else min(batch_size, remaining)
            if phase == "ordinary":
                rows, at_boundary = ordinary_page(take)
                if not rows:
                    break
                stats["orphan_examined"] += len(rows)
                if remaining is not None:
                    remaining -= len(rows)
                try:
                    raw = embedding_client.encode([content for _, content, _ in rows])
                    array = np.asarray(raw)
                    if array.shape != (len(rows), expected):
                        raise ValueError("embedding_shape_mismatch")
                    vectors = [np.asarray(item, dtype=np.float32) for item in array]
                    if any(
                        not np.all(np.isfinite(vec)) or not np.any(vec)
                        for vec in vectors
                    ):
                        raise ValueError("embedding_invalid")
                except Exception as exc:
                    logger.warning(
                        "sleep: orphan embedding batch rejected: %s",
                        type(exc).__name__,
                    )
                    stats["orphan_write_failed"] += len(rows)
                    break
                if not commit_batch([
                    (nid, vec.tobytes(), str(embedding_model))
                    for (nid, _content, _timestamp), vec in zip(rows, vectors)
                ]):
                    break
            else:
                rows, at_boundary = repair_page(take)
                if not rows:
                    break
                stats["orphan_examined"] += len(rows)
                if remaining is not None:
                    remaining -= len(rows)
                items: List[Tuple[str, bytes, str]] = []
                for nid, blob, stored_model, _timestamp in rows:
                    try:
                        vec = np.frombuffer(blob, dtype=np.float32)
                        if (
                            embedding_model
                            and stored_model
                            and stored_model != embedding_model
                        ):
                            raise ValueError("embedding_model_mismatch")
                        if (
                            len(vec) != expected
                            or not np.all(np.isfinite(vec))
                            or not np.any(vec)
                        ):
                            raise ValueError("embedding_invalid")
                    except (TypeError, ValueError, BufferError):
                        stats["orphan_write_failed"] += 1
                        continue
                    items.append((nid, bytes(blob), stored_model or str(embedding_model)))
                if not commit_batch(items):
                    break
            if at_boundary:
                break

    logger.info("sleep: embedded/repaired %d orphaned nodes", embedded)
    return embedded


# ── Background dream threading ───────────────────────────────────────────


def _run_dream_async(
    db_path: str,
    cross_link_tuples: List[Tuple[str, str, float]],
    model_fn,
    embedding_client=None,
    embedding_model: Optional[str] = None,
    expected_dimension: Optional[int] = None,
    journal_policy: str = "manage",
    orphan_limit: Optional[int] = None,
    orphan_batch_size: int = ORPHANS_PER_BATCH,
) -> None:
    """Run Phase 8 (dream) + Phase 9 (orphan embedding) in a daemon thread.

    Opens its own SQLite connection — WAL mode handles concurrency with
    the new session's sync worker writes.
    """
    def _task():
        conn = None
        try:
            conn = sqlite3.connect(db_path)
            conn.execute("PRAGMA busy_timeout = 5000")
            if journal_policy == "manage":
                _set_wal(conn)
            dream_id = _generate_dream(conn, cross_link_tuples, model_fn=model_fn)
            orphans = _embed_orphans(
                conn, embedding_client=embedding_client,
                embedding_model=embedding_model, expected_dimension=expected_dimension,
                limit=orphan_limit, batch_size=orphan_batch_size,
            )
            logger.info(
                "sleep: background dream complete (id=%s, orphans=%d)",
                dream_id or "none", orphans,
            )
        except Exception:
            logger.warning("sleep: background dream failed", exc_info=True)
        finally:
            if conn is not None:
                conn.close()

    t = threading.Thread(target=_task, daemon=True)
    t.start()
    logger.debug("sleep: background dream thread spawned")


# ── main entry point (free function) ─────────────────────────────────────


def _empty_sleep_result(
    status: str, error: Optional[str], elapsed_s: float = 0.0
) -> dict:
    """Return the stable JSON shape for preflight/capability outcomes."""
    result = {
        "status": status, "error": error,
        "nodes_selected": 0, "nodes_with_embeddings": 0, "total_nodes": 0,
        "cross_link_candidates": 0, "cross_links_created": 0,
        "cross_links_repaired": 0, "cross_links_skipped": 0,
        "cross_link_directed_rows": 0, "cross_link_capped": False,
        "cross_link_same_source_skipped": 0, "dedup_candidates": 0,
        "dedup_components": 0, "dedup_nodes_merged": 0,
        "nodes_gc_decayed": 0, "nodes_made_permanent": 0,
        "core_promoted": 0, "core_demoted": 0, "orphans_embedded": 0,
        "orphan_write_failed": 0, "orphan_vec_unavailable": 0,
        "vec_rows_compacted": 0, "dream_id": None, "dream_pending": False,
        "dream_generation": "skipped", "elapsed_s": max(0.0, float(elapsed_s)),
    }
    return result


def _public_sleep_result(result: dict) -> dict:
    """Project internal phase bookkeeping onto the stable public schema."""
    public = _empty_sleep_result(
        result.get("status", "failed"), result.get("error"), result.get("elapsed_s", 0.0)
    )
    public.update({key: value for key, value in result.items() if key in public})
    return public


def run_sleep_cycle(
    db_path: Optional[str] = None,
    limit: Optional[int] = None,
    model_fn=None,
    background_dream: bool = False,
    max_edges: int = MAX_EDGES_PER_CYCLE,
    cross_source_only: bool = False,
    *,
    embedding_client=None,
    embedding_model: Optional[str] = None,
    expected_dimension: Optional[int] = None,
    auto_embed: bool = True,
    journal_policy: str = "manage",
    orphan_limit: Optional[int] = None,
    orphan_batch_size: int = ORPHANS_PER_BATCH,
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
    auto_embed : bool
        Lazily use the configured local encoder when no client is supplied.
        Set False to forbid automatic model loading. An injected client never
        falls back to a local encoder, even when it fails.
    orphan_limit : Optional[int]
        Maximum orphan rows examined across both repair passes. ``None`` keeps
        the historical behavior of repairing every eligible orphan.
    orphan_batch_size : int
        Encode and commit at most this many orphans at once. Values from 1 to
        100 are accepted so bounded embedding clients are never overfilled.

    Returns
    -------
    dict
        Statistics for each phase.
    """
    conn = None
    t_start = time.perf_counter()
    progress = _empty_sleep_result("failed", "sleep_cycle_failed")
    try:
        if db_path is None:
            db_path = get_db_path()

        if not isinstance(auto_embed, bool):
            return _empty_sleep_result("rejected", "invalid_auto_embed")
        if journal_policy not in {"manage", "preserve"}:
            return _empty_sleep_result("rejected", "invalid_journal_policy")
        if limit is not None and (not isinstance(limit, int) or limit < 0):
            return _empty_sleep_result("rejected", "invalid_limit")
        if not isinstance(max_edges, int) or max_edges < 0:
            return _empty_sleep_result("rejected", "invalid_max_edges")
        if orphan_limit is not None and (
            not isinstance(orphan_limit, int)
            or isinstance(orphan_limit, bool)
            or orphan_limit < 0
        ):
            return _empty_sleep_result("rejected", "invalid_orphan_limit")
        if (
            not isinstance(orphan_batch_size, int)
            or isinstance(orphan_batch_size, bool)
            or orphan_batch_size <= 0
            or orphan_batch_size > ORPHANS_PER_BATCH
        ):
            return _empty_sleep_result("rejected", "invalid_orphan_batch_size")
        supplied_embedding = (
            embedding_client is not None,
            bool(embedding_model),
            expected_dimension is not None,
        )
        if any(supplied_embedding) and not all(supplied_embedding):
            return _empty_sleep_result("rejected", "invalid_embedding_contract")
        if all(supplied_embedding) and (
            not callable(getattr(embedding_client, "encode", None))
            or not isinstance(expected_dimension, int)
            or expected_dimension <= 0
        ):
            return _empty_sleep_result("rejected", "invalid_embedding_contract")
        try:
            profile = _get_active_profile(embedding_model)
        except Exception:
            if embedding_model:
                return _empty_sleep_result("rejected", "uncalibrated_embedding_model")
            return _empty_sleep_result("unavailable", "uncalibrated_embedding_model")
        if all(supplied_embedding) and profile is not None:
            if profile.dim != expected_dimension:
                return _empty_sleep_result("rejected", "embedding_dimension_mismatch")
        if not any(supplied_embedding) and auto_embed:
            from .config import get_embedding_model
            from .embedding_service import LocalBackend

            embedding_model = get_embedding_model()
            expected_dimension = profile.dim
            # LocalBackend loads its model only on encode: empty cycles and
            # vector-index repairs from stored rows do not load a model.
            embedding_client = LocalBackend(embedding_model)
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA busy_timeout = 5000")
        if journal_policy == "manage":
            _set_wal(conn)
        ensure_decay_audit_schema(conn)

        # Check if embeddings table exists — required for vectorized pipeline
        table_check = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='embeddings'"
        ).fetchone()
        if not table_check:
            logger.warning("sleep: no embeddings table — aborting (run cashew init first)")
            conn.commit()
            conn.close()
            return _empty_sleep_result("unavailable", "no_embeddings_table",
                                       time.perf_counter() - t_start)

        # ── Select one fair deterministic page for this cycle ──
        ids = _select_cycle_node_ids(conn, limit)
        logger.info("sleep: selected %d nodes (limit=%s)", len(ids), limit)

        valid_ids, matrix = _load_embedding_matrix(conn, ids, expected_dimension)
        orphan_stats: dict = {}
        orphans = 0
        if len(valid_ids) < 2:
            logger.warning("sleep: too few valid embeddings — aborting")
            orphans = _embed_orphans(
                conn,
                embedding_client=embedding_client,
                embedding_model=embedding_model,
                expected_dimension=expected_dimension,
                limit=orphan_limit,
                batch_size=orphan_batch_size,
                stats=orphan_stats,
            )
            ids = _select_cycle_node_ids(conn, limit)
            valid_ids, matrix = _load_embedding_matrix(conn, ids, expected_dimension)
            progress.update({
                "nodes_selected": len(ids),
                "nodes_with_embeddings": len(valid_ids),
                "orphans_embedded": orphans,
                "orphan_ordinary_written": orphan_stats.get(
                    "orphan_ordinary_written", 0
                ),
            })
            if len(valid_ids) < 2:
                conn.close()
                durable = bool(
                    orphans or orphan_stats.get("orphan_ordinary_written", 0)
                )
                error = (
                    "orphan_write_failed" if orphan_stats.get("orphan_write_failed")
                    else "vec_capability_unavailable"
                    if orphan_stats.get("orphan_vec_unavailable")
                    else "too_few_embeddings"
                )
                result = _empty_sleep_result(
                    "partial" if durable else "unavailable", error,
                                             time.perf_counter() - t_start)
                result["nodes_selected"] = len(ids)
                result["nodes_with_embeddings"] = len(valid_ids)
                result["orphans_embedded"] = orphans
                result["orphan_write_failed"] = orphan_stats.get("orphan_write_failed", 0)
                result["orphan_vec_unavailable"] = orphan_stats.get(
                    "orphan_vec_unavailable", 0
                )
                return result

        # Phase 1: candidate discovery
        cross_pairs, dedup_pairs, sim = _find_pairs(
            valid_ids, matrix,
            cross_threshold=profile.cross_link_threshold if profile else None,
            dedup_threshold=profile.dedup_threshold if profile else None,
        )

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
            progress.update({
                "nodes_selected": len(ids),
                "nodes_with_embeddings": len(valid_ids),
                "cross_link_candidates": len(cross_pairs),
                "dedup_candidates": len(dedup_pairs),
            })
            cross_stats = _batch_cross_links(
                conn, valid_ids, cross_pairs, sim,
                source_files=source_files if cross_source_only else None,
                max_edges=max_edges,
                progress=progress,
            )
            if model_fn is not None:
                cross_link_tuples = list(cross_stats.get("dream_pairs", ()))

        # Preserve committed work if a later phase fails.  The outer failure
        # boundary below returns this snapshot instead of erasing persisted
        # cross-link progress.
        progress.update({
            "nodes_selected": len(ids),
            "nodes_with_embeddings": len(valid_ids),
            "cross_link_candidates": len(cross_pairs),
            "cross_links_created": cross_stats.get("created", 0),
            "cross_links_repaired": cross_stats.get("repaired", 0),
            "cross_links_skipped": cross_stats.get("skipped", 0),
            "cross_link_directed_rows": cross_stats.get("directed_rows", 0),
            "cross_link_same_source_skipped": cross_stats.get("same_source_skipped", 0),
            "cross_link_capped": cross_stats.get("capped", False),
            "dedup_candidates": len(dedup_pairs),
        })

        # Phase 3: dedup
        dedup_stats = {"components": 0, "nodes_merged": 0}
        if len(dedup_pairs) > 0:
            dedup_stats = _run_dedup(conn, valid_ids, dedup_pairs)
        progress.update({
            "dedup_components": dedup_stats.get("components", 0),
            "dedup_nodes_merged": dedup_stats.get("nodes_merged", 0),
        })

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
        progress["nodes_gc_decayed"] = gc_count

        # Phase 6: permanence
        perm_stats = _evaluate_permanence(conn)
        progress["nodes_made_permanent"] = perm_stats.get("nodes_promoted", 0)

        # Phase 7: core memory
        core_stats = _promote_core_memories(conn, metrics)
        progress["core_promoted"] = core_stats.get("promoted", 0)
        progress["core_demoted"] = core_stats.get("demoted", 0)

        # Phase 8: dream generation
        dream_id = None
        dream_pending = False
        remaining_orphan_limit = (
            None
            if orphan_limit is None
            else max(0, orphan_limit - orphan_stats.get("orphan_examined", 0))
        )
        if model_fn is not None and cross_link_tuples:
            if background_dream:
                _run_dream_async(
                    db_path=db_path,
                    cross_link_tuples=cross_link_tuples,
                    model_fn=model_fn,
                    embedding_client=embedding_client, embedding_model=embedding_model,
                    expected_dimension=expected_dimension, journal_policy=journal_policy,
                    orphan_limit=remaining_orphan_limit,
                    orphan_batch_size=orphan_batch_size,
                )
                dream_pending = True
            else:
                dream_id = _generate_dream(conn, cross_link_tuples, model_fn=model_fn)
        progress["dream_id"] = dream_id
        progress["dream_generation"] = (
            "pending" if dream_pending else ("ran" if dream_id else "skipped")
        )
        if model_fn is not None and cross_link_tuples and not background_dream and dream_id is None:
            progress["dream_generation"] = "failed"

        # Phase 9: embed orphans
        if background_dream:
            # Preserve any synchronous prefix needed to establish two anchors;
            # late background repairs are reported only in the worker log.
            pass
        else:
            later_orphans = _embed_orphans(
                conn, embedding_client=embedding_client, embedding_model=embedding_model,
                expected_dimension=expected_dimension,
                limit=remaining_orphan_limit,
                batch_size=orphan_batch_size,
                stats=orphan_stats,
            )
            orphans += later_orphans
        progress["orphans_embedded"] = orphans
        progress["orphan_ordinary_written"] = orphan_stats.get(
            "orphan_ordinary_written", 0
        )

        conn.close()
        elapsed = round(time.perf_counter() - t_start, 1)

        # Decay-audit GC (one-shot per cycle)
        audit_conn = None
        try:
            audit_conn = sqlite3.connect(db_path)
            audit_conn.execute("PRAGMA busy_timeout = 5000")
            if journal_policy == "manage":
                _set_wal(audit_conn)
            ensure_decay_audit_schema(audit_conn)
            audit_pruned = gc_decay_audit(audit_conn, retention_days=7)
            audit_conn.commit()
            if audit_pruned:
                logger.info("sleep: decay-audit GC pruned %d rows", audit_pruned)
        except Exception as e:
            logger.warning("sleep: decay-audit GC failed: %s", e)
        finally:
            if audit_conn is not None:
                audit_conn.close()

        # Vec-index compaction (one-shot per cycle): decay never touched the vec
        # index, so prune rows for nodes decayed this cycle (and any backlog) to
        # keep the fast search path in sync with the live graph.
        vec_compacted = 0
        try:
            from .embeddings import compact_vec_index
            vec_compacted = compact_vec_index(db_path)
            if vec_compacted:
                logger.info("sleep: vec-index compaction pruned %d stale rows", vec_compacted)
        except Exception as e:
            logger.warning("sleep: vec-index compaction failed: %s", e)

        dream_generation = (
            "pending" if dream_pending else ("ran" if dream_id else "skipped")
        )
        if (model_fn is not None and cross_link_tuples and not background_dream
                and dream_id is None):
            dream_generation = "failed"
        phase_committed = bool(
            cross_stats.get("created", 0) or cross_stats.get("repaired", 0)
            or dedup_stats.get("nodes_merged", 0) or gc_count
            or perm_stats.get("nodes_promoted", 0) or core_stats.get("promoted", 0)
            or core_stats.get("demoted", 0)
            or orphans
            or orphan_stats.get("orphan_ordinary_written", 0)
            or dream_id
        )
        if cross_stats.get("failed"):
            result_status = "partial" if phase_committed else "failed"
            result_error = "cross_link_failed"
        elif orphan_stats.get("capability_missing"):
            result_status = "partial" if phase_committed else "unavailable"
            result_error = "embedding_capability_unavailable"
        elif orphan_stats.get("orphan_vec_unavailable"):
            result_status = "partial" if phase_committed else "unavailable"
            result_error = "vec_capability_unavailable"
        elif orphan_stats.get("orphan_write_failed"):
            result_status = "partial" if phase_committed else "failed"
            result_error = "orphan_write_failed"
        else:
            result_status, result_error = "completed", None
        summary = {
            "status": result_status,
            "error": result_error,
            "nodes_selected": len(ids),
            "nodes_with_embeddings": len(valid_ids),
            "cross_link_candidates": len(cross_pairs),
            "dedup_candidates": len(dedup_pairs),
            "cross_links_created": cross_stats["created"],
            "cross_links_repaired": cross_stats.get("repaired", 0),
            "cross_links_skipped": cross_stats["skipped"],
            "cross_link_directed_rows": cross_stats.get("directed_rows", 0),
            "cross_link_same_source_skipped": cross_stats.get("same_source_skipped", 0),
            "cross_link_capped": cross_stats.get("capped", False),
            "dedup_components": dedup_stats["components"],
            "dedup_nodes_merged": dedup_stats["nodes_merged"],
            "nodes_gc_decayed": gc_count,
            "vec_rows_compacted": vec_compacted,
            "nodes_made_permanent": perm_stats.get("nodes_promoted", 0),
            "core_promoted": core_stats.get("promoted", 0),
            "core_demoted": core_stats.get("demoted", 0),
            "dream_id": dream_id,
            "dream_pending": dream_pending,
            "dream_generation": dream_generation,
            "orphans_embedded": orphans,
            "orphan_write_failed": orphan_stats.get("orphan_write_failed", 0),
            "orphan_vec_unavailable": orphan_stats.get("orphan_vec_unavailable", 0),
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

        if dream_generation == "failed" and summary["status"] == "completed":
            summary["status"] = "partial" if phase_committed else "failed"
            summary["error"] = "dream_failed"
        return summary

    except Exception as exc:
        logger.warning("sleep: cycle failed: %s", type(exc).__name__)
        if progress.get("_outcome_uncertain"):
            progress["status"] = "uncertain"
            progress["error"] = "cross_link_commit_uncertain"
            progress["elapsed_s"] = round(time.perf_counter() - t_start, 1)
            return _public_sleep_result(progress)
        persisted = progress.get("cross_links_created", 0) + progress.get(
            "cross_links_repaired", 0
        ) + progress.get("dedup_nodes_merged", 0) + progress.get(
            "nodes_gc_decayed", 0
        ) + progress.get("nodes_made_permanent", 0) + progress.get(
            "core_promoted", 0
        ) + progress.get("core_demoted", 0) + progress.get(
            "orphan_ordinary_written", 0
        ) + progress.get("orphans_embedded", 0) + bool(progress.get("dream_id"))
        if persisted:
            progress["status"] = "partial"
            progress["error"] = "sleep_cycle_failed"
            progress["elapsed_s"] = round(time.perf_counter() - t_start, 1)
            return _public_sleep_result(progress)
        return _empty_sleep_result("failed", "sleep_cycle_failed",
                                   time.perf_counter() - t_start)
    finally:
        if conn is not None:
            conn.close()

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

    def __init__(
        self, db_path: Optional[str] = None, sleep_log_path: Optional[str] = None
    ):
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
            ``background_dream``, ``max_edges``, ``cross_source_only``,
            ``orphan_limit``, and ``orphan_batch_size``.
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
            embedding_client=kwargs.get("embedding_client"),
            embedding_model=kwargs.get("embedding_model"),
            expected_dimension=kwargs.get("expected_dimension"),
            auto_embed=kwargs.get("auto_embed", True),
            journal_policy=kwargs.get("journal_policy", "manage"),
            orphan_limit=kwargs.get("orphan_limit"),
            orphan_batch_size=kwargs.get("orphan_batch_size", ORPHANS_PER_BATCH),
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
