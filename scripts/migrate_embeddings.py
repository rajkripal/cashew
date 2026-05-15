"""Re-embed every node in a cashew brain under the currently configured
embedding model.

Used when an existing brain was built under a different embedding model
(e.g., upgrading from MiniLM-384 to gte-large-1024). Wipes the existing
``embeddings`` rows and the ``vec_embeddings`` virtual table, then re-runs
``embed_nodes`` so the brain is consistent with the active model.

Public entry point: ``migrate_embeddings(db_path, *, confirm=False) -> dict``.

The cashew CLI exposes this as ``cashew migrate-embeddings``.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.embeddings import _load_vec, embed_nodes
from core.embedding_service import resolve_embedding_dim


def _stored_embedding_dim(conn: sqlite3.Connection) -> Optional[int]:
    """Inspect the embeddings table and return the dim of stored vectors,
    or None if no embeddings exist yet."""
    cur = conn.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='embeddings'"
    )
    if cur.fetchone() is None:
        return None
    row = cur.execute(
        "SELECT LENGTH(vector) FROM embeddings WHERE vector IS NOT NULL LIMIT 1"
    ).fetchone()
    if row is None or row[0] is None:
        return None
    # vectors are stored as float32 (4 bytes per dim)
    return row[0] // 4


def detect_mismatch(db_path: str) -> Optional[Dict]:
    """Return a description of the dim mismatch if one is present, else None.

    Result dict (when mismatched):
      - ``db_path``: input path
      - ``stored_dim``: dim of vectors currently in the db
      - ``configured_dim``: dim of the currently configured embedding model
      - ``configured_model``: model name from current config
    """
    conn = sqlite3.connect(db_path)
    try:
        stored = _stored_embedding_dim(conn)
    finally:
        conn.close()

    if stored is None:
        return None

    configured = resolve_embedding_dim()
    if stored == configured:
        return None

    from core.embedding_service import _resolve_default_model
    return {
        "db_path": db_path,
        "stored_dim": stored,
        "configured_dim": configured,
        "configured_model": _resolve_default_model(),
    }


def migrate_embeddings(
    db_path: str,
    *,
    confirm: bool = False,
    quiet: bool = False,
) -> Dict:
    """Wipe and re-embed every node under the currently configured model.

    Args:
        db_path: Path to the cashew SQLite DB.
        confirm: If True, skip the interactive confirmation prompt.
        quiet:   If True, suppress progress output.

    Returns:
        Dict with: nodes_embedded, wall_seconds, stored_dim_before,
        stored_dim_after, configured_model.

    Raises:
        FileNotFoundError if db_path does not exist.
        RuntimeError if the user declines confirmation.
    """
    if not Path(db_path).exists():
        raise FileNotFoundError(f"db not found: {db_path}")

    mismatch = detect_mismatch(db_path)
    stored_before = mismatch["stored_dim"] if mismatch else None
    configured_dim = resolve_embedding_dim()
    from core.embedding_service import _resolve_default_model
    configured_model = _resolve_default_model()

    if not quiet:
        if mismatch:
            print(
                f"Detected dim mismatch: brain has {mismatch['stored_dim']}-dim "
                f"vectors, configured model {configured_model} produces "
                f"{configured_dim}-dim."
            )
        else:
            print(
                f"No dim mismatch detected (existing dim matches configured "
                f"model {configured_model}). Re-embedding anyway."
            )

    if not confirm and not quiet:
        ans = input("Wipe existing embeddings and re-embed all nodes? [y/N]: ").strip().lower()
        if ans not in ("y", "yes"):
            raise RuntimeError("migration cancelled by user")

    t0 = time.time()
    conn = sqlite3.connect(db_path)
    _load_vec(conn)
    conn.execute("DELETE FROM embeddings")
    try:
        conn.execute("DROP TABLE IF EXISTS vec_embeddings")
    except sqlite3.OperationalError:
        # Some sqlite-vec versions reject DROP via the virtual-table path;
        # the shadow tables go with it once the index regenerates.
        pass
    conn.commit()
    conn.close()

    result = embed_nodes(db_path)
    wall = time.time() - t0

    # Confirm post-state
    conn = sqlite3.connect(db_path)
    try:
        stored_after = _stored_embedding_dim(conn)
    finally:
        conn.close()

    summary = {
        "nodes_embedded": result.get("embedded", 0),
        "wall_seconds": round(wall, 1),
        "stored_dim_before": stored_before,
        "stored_dim_after": stored_after,
        "configured_model": configured_model,
        "configured_dim": configured_dim,
    }
    if not quiet:
        print(
            f"Migrated {summary['nodes_embedded']} nodes in "
            f"{summary['wall_seconds']}s. Stored dim: "
            f"{stored_before} -> {stored_after}."
        )
    return summary


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True, help="Path to the cashew SQLite DB")
    ap.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt",
    )
    ap.add_argument("--quiet", action="store_true", help="Suppress progress output")
    a = ap.parse_args()
    try:
        migrate_embeddings(a.db, confirm=a.yes, quiet=a.quiet)
    except RuntimeError as e:
        print(f"aborted: {e}", file=sys.stderr)
        sys.exit(1)
