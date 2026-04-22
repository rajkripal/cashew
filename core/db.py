"""
core/db — single chokepoint for graph-DB access.

Rule: maintenance scripts and the CLI go through this module instead of
calling sqlite3.connect directly. The cashew CLI remains the only supported
user-facing surface; internally both it and the utility scripts share this
helper so that schema knowledge (table names, column names, index names)
lives in exactly one place.

This module intentionally stays thin. It is NOT an ORM. It exposes:

- Schema constants          (SCHEMA, NODES_TABLE, ...)
- Connection management     (resolve_db_path, connect, transaction)
- Small query primitives    (get_node, update_node_tags, iter_nodes, ...)
- Introspection             (pragma_columns, table_exists)
- Migration runner          (execute_migration)

core.session already exposes `_get_connection` and `_ensure_schema`; this
module wraps them so the migration path stays authoritative.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# Schema constants — the single source of truth for identifier strings.
# --------------------------------------------------------------------------

NODES_TABLE = "thought_nodes"
EDGES_TABLE = "derivation_edges"
EMBEDDINGS_TABLE = "embeddings"
METRICS_TABLE = "metrics"

# Columns on thought_nodes that callers in this repo reference directly.
# (Not exhaustive of every column — just the ones used by scripts we own.)
NODE_COLUMNS = (
    "id",
    "content",
    "node_type",
    "domain",
    "timestamp",
    "access_count",
    "last_accessed",
    "confidence",
    "source_file",
    "decayed",
    "metadata",
    "last_updated",
    "mood_state",
    "permanent",
    "tags",
    "referent_time",
)

SCHEMA = {
    "nodes_table": NODES_TABLE,
    "edges_table": EDGES_TABLE,
    "embeddings_table": EMBEDDINGS_TABLE,
    "metrics_table": METRICS_TABLE,
    "node_columns": NODE_COLUMNS,
}


# --------------------------------------------------------------------------
# Path resolution.
# --------------------------------------------------------------------------

def resolve_db_path(override: Optional[str] = None) -> str:
    """Resolve the DB path callers should use.

    Precedence:
    1. Explicit override (e.g. CLI --db flag passed through)
    2. CASHEW_DB_PATH environment variable
    3. core.config.get_db_path() — config file value
    """
    if override:
        return str(override)
    env = os.environ.get("CASHEW_DB_PATH")
    if env:
        return env
    # Lazy import to avoid circulars during test collection.
    from core.config import get_db_path  # noqa: WPS433
    return get_db_path()


# --------------------------------------------------------------------------
# Connection management.
# --------------------------------------------------------------------------

def connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Open a connection. Thin wrapper so scripts never call sqlite3 directly.

    This delegates to core.session._get_connection so the "how we open the
    DB" policy stays in one place.
    """
    from core.session import _get_connection  # noqa: WPS433
    path = resolve_db_path(db_path)
    return _get_connection(path)


@contextmanager
def transaction(db_path: Optional[str] = None) -> Iterator[sqlite3.Connection]:
    """Context manager yielding a connection.

    Commits on clean exit, rolls back on exception, always closes.
    """
    conn = connect(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# --------------------------------------------------------------------------
# Schema introspection / migration.
# --------------------------------------------------------------------------

def pragma_columns(conn: sqlite3.Connection, table: str = NODES_TABLE) -> list[str]:
    """Return the list of column names on `table`."""
    cur = conn.execute(f"PRAGMA table_info({table})")
    return [row[1] for row in cur.fetchall()]


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    )
    return cur.fetchone() is not None


def list_tables(conn: sqlite3.Connection) -> list[str]:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return [row[0] for row in cur.fetchall()]


def ensure_schema(db_path: Optional[str] = None) -> None:
    """Public, idempotent schema init / migration entrypoint.

    Creates cashew's canonical tables from scratch if they don't exist,
    applies additive column migrations to legacy databases, and stamps
    `PRAGMA user_version` with the current schema version.

    Safe for downstream library consumers to call before layering their
    own schema additions. See DESIGN.md for the ownership contract.
    """
    from core.session import _ensure_schema  # noqa: WPS433
    _ensure_schema(resolve_db_path(db_path))


def get_schema_version(db_path: Optional[str] = None) -> int:
    """Return the cashew schema version applied to this database.

    Reads `PRAGMA user_version`. Returns 0 for databases that have never
    been through `ensure_schema`. Downstream consumers can branch on this
    to decide whether to apply their own layered migrations.
    """
    from core.session import get_schema_version as _get  # noqa: WPS433
    return _get(resolve_db_path(db_path))


def schema_version() -> int:
    """The schema version this build of cashew knows how to produce."""
    from core.session import SCHEMA_VERSION  # noqa: WPS433
    return SCHEMA_VERSION


def execute_migration(sql: str, db_path: Optional[str] = None) -> None:
    """Run a multi-statement migration script inside a single transaction."""
    with transaction(db_path) as conn:
        conn.executescript(sql)


# --------------------------------------------------------------------------
# Query primitives.
#
# These are the common shapes the 8 scripts needed. Keep them boring — they
# exist so nobody has to type `SELECT ... FROM thought_nodes WHERE id = ?`
# again.
# --------------------------------------------------------------------------

def get_node(conn: sqlite3.Connection, node_id: str,
             columns: Sequence[str] = NODE_COLUMNS) -> Optional[tuple]:
    """Fetch one node row by id. Returns None if missing."""
    cols = ", ".join(columns)
    cur = conn.execute(
        f"SELECT {cols} FROM {NODES_TABLE} WHERE id = ?", (node_id,)
    )
    return cur.fetchone()


def get_node_tags(conn: sqlite3.Connection, node_id: str) -> Optional[str]:
    """Return the tags column (raw comma-delimited string) for a node."""
    cur = conn.execute(
        f"SELECT tags FROM {NODES_TABLE} WHERE id = ?", (node_id,)
    )
    row = cur.fetchone()
    return row[0] if row else None


def set_node_tags(conn: sqlite3.Connection, node_id: str, tags: str) -> None:
    """Overwrite the tags column for a node. Caller provides the final string."""
    conn.execute(
        f"UPDATE {NODES_TABLE} SET tags = ? WHERE id = ?", (tags, node_id)
    )


def merge_node_tags(conn: sqlite3.Connection, node_id: str,
                    add_tags: Iterable[str]) -> None:
    """Union the existing tag set with `add_tags`, write back sorted+joined."""
    existing = get_node_tags(conn, node_id) or ""
    tag_set = {t.strip() for t in existing.split(",") if t.strip()}
    tag_set.update(t.strip() for t in add_tags if t and t.strip())
    set_node_tags(conn, node_id, ",".join(sorted(tag_set)))


def iter_nodes(conn: sqlite3.Connection,
               where: Optional[str] = None,
               params: Sequence[Any] = (),
               columns: Sequence[str] = NODE_COLUMNS,
               batch_size: int = 500) -> Iterator[tuple]:
    """Stream node rows matching an optional WHERE clause, in batches.

    Avoids fetchall() for callers that may scan the whole graph.
    """
    cols = ", ".join(columns)
    sql = f"SELECT {cols} FROM {NODES_TABLE}"
    if where:
        sql += f" WHERE {where}"
    cur = conn.execute(sql, tuple(params))
    while True:
        rows = cur.fetchmany(batch_size)
        if not rows:
            return
        for row in rows:
            yield row


def execute(conn: sqlite3.Connection, sql: str,
            params: Sequence[Any] = ()) -> sqlite3.Cursor:
    """Pass-through for arbitrary queries against the graph DB.

    Exposed so callers can run ad-hoc SELECTs without reaching for raw
    sqlite3. Callers should still use constants from this module rather
    than hardcoded table/column strings where possible.
    """
    return conn.execute(sql, tuple(params))


def executemany(conn: sqlite3.Connection, sql: str,
                seq: Iterable[Sequence[Any]]) -> sqlite3.Cursor:
    return conn.executemany(sql, seq)


__all__ = [
    "NODES_TABLE",
    "EDGES_TABLE",
    "EMBEDDINGS_TABLE",
    "METRICS_TABLE",
    "NODE_COLUMNS",
    "SCHEMA",
    "resolve_db_path",
    "connect",
    "transaction",
    "pragma_columns",
    "table_exists",
    "list_tables",
    "ensure_schema",
    "execute_migration",
    "get_node",
    "get_node_tags",
    "set_node_tags",
    "merge_node_tags",
    "iter_nodes",
    "execute",
    "executemany",
]
