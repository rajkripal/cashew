"""
Decay audit logging.

Single helper that all decay execution sites call so the `decay_audit` table
records every transition from live → decayed. Decay decision logic lives in
the call sites; this module only writes audit rows.

Schema (decay_audit) is created elsewhere; we only INSERT here.
"""

import json
import sqlite3
from datetime import datetime, timezone
from typing import Optional


# Allowed decay reasons. Keep in sync with brain-metrics.py distribution chart.
DECAY_REASONS = (
    "organic_age_no_access",  # core/decay.py auto_decay direct prune
    "organic_cascade",        # core/decay.py cascade_decay child propagation
    "dedup_loser",            # core/sleep.py deduplicate_nodes loser
    "gc_fitness",             # core/sleep.py garbage_collect fitness prune
)

_SUMMARY_LEN = 80


def _summary(content: Optional[str]) -> str:
    if not content:
        return ""
    s = content.strip().replace("\n", " ")
    return s[:_SUMMARY_LEN]


def log_decay_event(
    conn: sqlite3.Connection,
    node_id: str,
    reason: str,
    related_nodes: Optional[dict] = None,
    metadata: Optional[dict] = None,
) -> None:
    """Insert one decay_audit row for `node_id`.

    Reads node fields (content, source_file, domain, node_type,
    access_count, last_accessed) from `thought_nodes` on the same connection
    so callers don't have to re-fetch. Caller is responsible for committing.

    `related_nodes` and `metadata` are stored as JSON strings if provided.
    """
    if reason not in DECAY_REASONS:
        # Don't silently accept typos — they break the metrics distribution.
        raise ValueError(
            f"Unknown decay_reason {reason!r}; expected one of {DECAY_REASONS}"
        )

    cursor = conn.cursor()
    # Defensive: if decay_audit table is absent (legacy fixtures, half-init
    # brains), skip silently. The decision to decay already happened; missing
    # an audit row is preferable to crashing the cycle.
    has_audit = cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='decay_audit'"
    ).fetchone()
    if not has_audit:
        return
    cursor.execute(
        "SELECT content, source_file, domain, node_type, access_count, last_accessed "
        "FROM thought_nodes WHERE id = ?",
        (node_id,),
    )
    row = cursor.fetchone()
    if row is None:
        # Node was deleted (hard-GC path) before we could audit. Log a stub
        # row so the event isn't lost.
        content = source_file = domain = node_type = last_accessed = None
        access_count = None
    else:
        content, source_file, domain, node_type, access_count, last_accessed = row

    cursor.execute(
        """
        INSERT INTO decay_audit (
            node_id, content_summary, decay_reason,
            confidence_at_decay, access_count_at_decay, last_access_date,
            related_nodes, source_file, domain, node_type,
            decay_timestamp, metadata
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            node_id,
            _summary(content),
            reason,
            None,  # confidence field is dead in cashew; kept for schema compat
            access_count,
            last_accessed,
            json.dumps(related_nodes) if related_nodes else None,
            source_file,
            domain,
            node_type,
            datetime.now(timezone.utc).isoformat(),
            json.dumps(metadata) if metadata else None,
        ),
    )


def gc_decay_audit(conn: sqlite3.Connection, retention_days: int = 7) -> int:
    """Delete audit rows older than `retention_days`. Returns rows deleted.

    Safe to call repeatedly. Caller commits.
    """
    cursor = conn.cursor()
    cursor.execute(
        f"DELETE FROM decay_audit "
        f"WHERE decay_timestamp < datetime('now', '-{int(retention_days)} days')"
    )
    return cursor.rowcount or 0
