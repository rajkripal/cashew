#!/usr/bin/env python3
"""
Declassification script for vault:private nodes.
Reviews nodes older than 7 days for potential declassification.

DB access goes through core.db (the shared chokepoint) — no raw sqlite3.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
import argparse

# Resolve repo root for imports.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from core import db as cdb  # noqa: E402


def get_candidates(db_path="data/graph.db", days_old=7):
    """Get vault:private nodes older than specified days."""
    cutoff_date = (datetime.now() - timedelta(days=days_old)).isoformat()

    with cdb.transaction(db_path) as conn:
        cursor = cdb.execute(
            conn,
            f"""
            SELECT id, content, timestamp, tags, domain
            FROM {cdb.NODES_TABLE}
            WHERE tags LIKE '%vault:private%'
            AND timestamp < ?
            AND (decayed IS NULL OR decayed = 0)
            ORDER BY timestamp
            """,
            (cutoff_date,),
        )
        candidates = cursor.fetchall()

    return candidates


def declassify_nodes(db_path="data/graph.db", node_ids=None):
    """Remove vault:private tag from specified nodes."""
    if not node_ids:
        print("No nodes specified for declassification")
        return

    with cdb.transaction(db_path) as conn:
        for node_id in node_ids:
            cdb.execute(
                conn,
                f"UPDATE {cdb.NODES_TABLE} SET tags = REPLACE(tags, 'vault:private,', '') WHERE id = ?",
                (node_id,),
            )
            cdb.execute(
                conn,
                f"UPDATE {cdb.NODES_TABLE} SET tags = REPLACE(tags, ',vault:private', '') WHERE id = ?",
                (node_id,),
            )
            cdb.execute(
                conn,
                f"UPDATE {cdb.NODES_TABLE} SET tags = REPLACE(tags, 'vault:private', '') WHERE id = ?",
                (node_id,),
            )
            print(f"Declassified node: {node_id}")


def main():
    parser = argparse.ArgumentParser(description="Declassify vault:private nodes")
    parser.add_argument("--candidates", action="store_true", help="List declassification candidates")
    parser.add_argument("--declassify-ids", nargs="+", help="Declassify specific node IDs")
    parser.add_argument("--days", type=int, default=7, help="Minimum age in days for candidates")

    args = parser.parse_args()

    if args.candidates:
        candidates = get_candidates(days_old=args.days)
        print(f"\n=== DECLASSIFICATION CANDIDATES (older than {args.days} days) ===")

        if not candidates:
            print("No vault:private nodes found older than specified days")
            return

        for node_id, content, timestamp, tags, domain in candidates:
            print(f"\nID: {node_id}")
            print(f"Domain: {domain}")
            print(f"Timestamp: {timestamp}")
            print(f"Tags: {tags}")
            print(f"Content: {content[:200]}...")
            print("-" * 80)

    elif args.declassify_ids:
        declassify_nodes(node_ids=args.declassify_ids)
        print(f"Declassified {len(args.declassify_ids)} nodes")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
