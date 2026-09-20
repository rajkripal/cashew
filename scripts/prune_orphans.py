#!/usr/bin/env python3
"""
Orphan node pruning for the cashew thought-graph.

An orphan is an active node (decayed=0) that appears in neither parent_id nor
child_id of any derivation edge. These accumulate from extraction runs that
don't link new nodes to existing ones.

Usage:
  python3 scripts/prune_orphans.py              # dry-run report (default)
  python3 scripts/prune_orphans.py --dry-run    # explicit dry-run
  python3 scripts/prune_orphans.py --execute    # decay all orphan nodes

Run from the cashew directory.
"""

import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
import argparse
import sqlite3
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

try:
    from core.config import get_db_path
    _config_db_path = get_db_path()
except Exception:
    _config_db_path = None


ORPHAN_QUERY = """
    SELECT id, domain, node_type, content
    FROM thought_nodes
    WHERE decayed = 0
      AND id NOT IN (
          SELECT DISTINCT parent_id FROM derivation_edges
          UNION
          SELECT DISTINCT child_id FROM derivation_edges
      )
    ORDER BY domain, node_type, id
"""

DECAY_QUERY = """
    UPDATE thought_nodes
    SET decayed = 1
    WHERE decayed = 0
      AND id NOT IN (
          SELECT DISTINCT parent_id FROM derivation_edges
          UNION
          SELECT DISTINCT child_id FROM derivation_edges
      )
"""

TOTAL_ACTIVE_QUERY = "SELECT COUNT(*) FROM thought_nodes WHERE decayed = 0"


def resolve_db_path(cli_db_path):
    """Resolve DB path: CLI arg > config > fallback relative path."""
    if cli_db_path:
        return cli_db_path
    if _config_db_path:
        return _config_db_path
    # Fallback: data/graph.db relative to cashew root (parent of scripts/)
    cashew_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(cashew_root, 'data', 'graph.db')


def fetch_orphans(conn):
    """Return list of (id, domain, node_type, content) for all orphan nodes."""
    cursor = conn.cursor()
    cursor.execute(ORPHAN_QUERY)
    return cursor.fetchall()


def fetch_total_active(conn):
    cursor = conn.cursor()
    cursor.execute(TOTAL_ACTIVE_QUERY)
    return cursor.fetchone()[0]


def print_report(orphans, total_active, execute_mode):
    orphan_count = len(orphans)
    pct = (orphan_count / total_active * 100) if total_active > 0 else 0

    print("=== Orphan Node Report ===")
    print(f"  Total active nodes : {total_active:,}")
    print(f"  Orphan nodes       : {orphan_count:,}  ({pct:.1f}% of active)")
    print()

    # Breakdown by domain
    domain_counts = Counter(row[1] or '(none)' for row in orphans)
    print("  By domain:")
    for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
        print(f"    {domain:<20} {count:>5}")
    print()

    # Breakdown by node_type
    type_counts = Counter(row[2] or '(none)' for row in orphans)
    print("  By node_type:")
    for ntype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {ntype:<20} {count:>5}")
    print()

    # 5 sample nodes
    print("  Sample orphans (first 5):")
    for row in orphans[:5]:
        node_id, domain, node_type, content = row
        snippet = (content or '')[:80].replace('\n', ' ')
        if len(content or '') > 80:
            snippet += '...'
        print(f"    [{node_id}] ({domain}/{node_type}) {snippet}")
    print()

    if execute_mode:
        print(f"  ACTION: decayed {orphan_count:,} orphan nodes (decayed=1)")
    else:
        print("  (dry-run: no changes made)")
        if orphan_count > 0:
            print("  Re-run with --execute to decay these nodes.")

    print("=" * 26)


def main():
    parser = argparse.ArgumentParser(
        description='Report and optionally decay orphan nodes in the cashew brain graph.'
    )
    parser.add_argument(
        '--db-path', default=None,
        help='Path to graph.db (default: from config or data/graph.db)'
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        '--dry-run', action='store_true', default=True,
        help='Report orphans without making changes (default)'
    )
    mode.add_argument(
        '--execute', action='store_true', default=False,
        help='Set decayed=1 on all orphan nodes'
    )
    args = parser.parse_args()

    execute_mode = args.execute

    db_path = resolve_db_path(args.db_path)

    if not os.path.exists(db_path):
        print(f"Error: database not found at {db_path}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(db_path)
    try:
        orphans = fetch_orphans(conn)
        total_active = fetch_total_active(conn)

        if execute_mode and orphans:
            cursor = conn.cursor()
            cursor.execute(DECAY_QUERY)
            conn.commit()

        print_report(orphans, total_active, execute_mode)
    finally:
        conn.close()

    return 0


if __name__ == '__main__':
    sys.exit(main())
