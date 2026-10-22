#!/usr/bin/env python3
"""Re-embed nodes whose stored vector dimension doesn't match the currently
configured embedding model.

Background
----------
PR #130 added a guard so retrieval skips vectors whose dim doesn't match the
expected dim (e.g. legacy 384-dim MiniLM rows sitting alongside current
1024-dim gte-large rows) instead of crashing on
``shapes (1024,) and (384,) not aligned``. That guard only masks the
symptom: the stale rows are silently excluded from every search, so those
nodes are permanently unsearchable even though their content is fine.

This script fixes the underlying data: it finds every row in ``embeddings``
whose vector length doesn't match ``resolve_embedding_dim()``, re-embeds
that node's content through the current embedding pipeline (the same
``embed_text``/service machinery ``embed_nodes`` uses), and replaces the
stored vector + model + updated_at in place. Nodes already at the correct
dim are left untouched — this is a targeted repair, not a full
``migrate_embeddings`` wipe-and-rebuild.

Usage
-----
    python3 scripts/reembed_stale_dims.py --db data/graph.db --dry-run
    python3 scripts/reembed_stale_dims.py --db data/graph.db
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.embeddings import _has_vec_table, _load_vec, _vec_table_dim_matches  # noqa: E402
from core.embedding_service import resolve_embedding_dim  # noqa: E402

DEFAULT_DB_PATH = str(ROOT / "data" / "graph.db")


def find_stale_nodes(db_path: str, expected_dim: int) -> list[tuple[str, str, int]]:
    """Return (node_id, content, stored_dim) for every embedded node whose
    stored vector dim != expected_dim. Nodes missing from thought_nodes
    (orphaned embedding rows) or with empty content are skipped — there's
    nothing to re-embed them from."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA busy_timeout = 5000")
    try:
        rows = conn.execute(
            """
            SELECT e.node_id, tn.content, LENGTH(e.vector) / 4 AS dim
            FROM embeddings e
            JOIN thought_nodes tn ON e.node_id = tn.id
            WHERE e.vector IS NOT NULL
            """
        ).fetchall()
    finally:
        conn.close()

    stale = [
        (node_id, content, dim)
        for node_id, content, dim in rows
        if dim != expected_dim and content and content.strip()
    ]
    return stale


def reembed_stale(db_path: str, dry_run: bool = False) -> dict:
    """Find and re-embed all stale-dimension vectors in-place.

    Returns a summary dict: expected_dim, stale_found, reembedded, failed,
    failures (list of (node_id, error)).
    """
    expected_dim = resolve_embedding_dim()
    stale = find_stale_nodes(db_path, expected_dim)

    summary = {
        "expected_dim": expected_dim,
        "stale_found": len(stale),
        "reembedded": 0,
        "failed": 0,
        "failures": [],
    }

    if not stale:
        return summary

    if dry_run:
        for node_id, content, dim in stale:
            preview = content[:60].replace("\n", " ")
            print(f"[dry-run] would re-embed {node_id} (dim {dim} -> {expected_dim}): {preview!r}")
        return summary

    from core.embedding_service import get_default_service

    service = get_default_service()

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA busy_timeout = 5000")
    _load_vec(conn)
    cursor = conn.cursor()

    has_vec = _has_vec_table(conn) and _vec_table_dim_matches(conn, service.dim)

    for node_id, content, old_dim in stale:
        try:
            vector = service.embed_np([content])[0]
            vector_bytes = vector.astype("float32").tobytes()
            if not vector_bytes:
                raise ValueError("embedding produced empty bytes")

            cursor.execute(
                """
                UPDATE embeddings
                SET vector = ?, model = ?, updated_at = ?
                WHERE node_id = ?
                """,
                (vector_bytes, service.model, datetime.now().isoformat(), node_id),
            )

            if has_vec:
                cursor.execute("DELETE FROM vec_embeddings WHERE node_id = ?", (node_id,))
                cursor.execute(
                    "INSERT INTO vec_embeddings(node_id, embedding) VALUES (?, ?)",
                    (node_id, vector_bytes),
                )

            conn.commit()
            summary["reembedded"] += 1
            print(f"re-embedded {node_id} (dim {old_dim} -> {expected_dim})")
        except Exception as e:  # noqa: BLE001 — one bad node shouldn't abort the run
            conn.rollback()
            summary["failed"] += 1
            summary["failures"].append((node_id, str(e)))
            print(f"FAILED to re-embed {node_id}: {e}", file=sys.stderr)

    conn.close()
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="Path to the cashew SQLite DB")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be re-embedded without writing anything",
    )
    args = parser.parse_args()

    if not Path(args.db).exists():
        print(f"error: db not found: {args.db}", file=sys.stderr)
        return 1

    summary = reembed_stale(args.db, dry_run=args.dry_run)

    print()
    print(f"expected dim: {summary['expected_dim']}")
    print(f"stale nodes found: {summary['stale_found']}")
    if args.dry_run:
        print("(dry run — nothing written)")
    else:
        print(f"re-embedded: {summary['reembedded']}")
        print(f"failed: {summary['failed']}")
        if summary["failures"]:
            for node_id, err in summary["failures"]:
                print(f"  {node_id}: {err}")

    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
