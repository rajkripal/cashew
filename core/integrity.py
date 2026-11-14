"""Connection-owned integrity inspection and targeted repair.

Cashew stores ordinary embeddings, a derived sqlite-vec table, and graph
relationships in one SQLite database.  This module exposes the small repair
surface that downstream adapters need without taking ownership of the
connection lifecycle.

The caller owns the connection, outer transaction, journal policy, backup,
and any process-level maintenance lock.  These functions never open or close
connections, commit, roll back an outer transaction, or change journal mode.
Each individual repair is protected by a savepoint so a failed repair cannot
undo unrelated work in the caller's transaction.

Only information-preserving repairs are enabled by default.  The API never
tries to reconstruct merged or decayed memories.  Permanence conflicts and
self-edges are report-only unless an explicit policy/action is supplied.
"""

from __future__ import annotations

import math
import re
import sqlite3
import struct
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime, timezone
from typing import Any

import numpy as np


_REQUIRED_TABLES = {
    "thought_nodes",
    "embeddings",
    "derivation_edges",
}
_VEC_DDL = re.compile(
    r"^\s*CREATE\s+VIRTUAL\s+TABLE\s+"
    r"(?:(?:IF\s+NOT\s+EXISTS)\s+)?"
    r"(?:vec_embeddings|\"vec_embeddings\"|`vec_embeddings`|\[vec_embeddings\])\s+"
    r"USING\s+vec0(?:\s|\()",
    re.IGNORECASE,
)
_VEC_DIM = re.compile(r"(?:float|int8)\s*\[\s*(\d+)\s*\]", re.IGNORECASE)
_DEFAULT_ACTIONS = frozenset(
    {
        "repair_vec",
        "repair_embeddings",
        "remove_orphan_embeddings",
        "remove_orphan_edges",
    }
)
_KNOWN_ACTIONS = _DEFAULT_ACTIONS | {
    "remove_self_edges",
    "promote_core_memories",
}
_SAVEPOINT_PREFIX = "cashew_integrity"


EmbeddingFn = Callable[[Sequence[str]], Any]


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }


def _vec_schema(conn: sqlite3.Connection) -> tuple[str, int | None, str | None]:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='vec_embeddings'"
    ).fetchone()
    if row is None:
        return "missing", None, "vec_index_missing"
    ddl = str(row[0] or "")
    if not _VEC_DDL.match(ddl):
        return "invalid", None, "vec_schema_invalid"
    match = _VEC_DIM.search(ddl)
    if match is None:
        return "invalid", None, "vec_dimension_unavailable"
    return "present", int(match.group(1)), None


def _load_vec(conn: sqlite3.Connection) -> bool:
    """Register sqlite-vec on this connection without changing the database."""
    try:
        import sqlite_vec

        conn.enable_load_extension(True)
        try:
            sqlite_vec.load(conn)
        finally:
            conn.enable_load_extension(False)
        conn.execute("SELECT count(*) FROM vec_embeddings").fetchone()
        return True
    except Exception:
        try:
            conn.enable_load_extension(False)
        except sqlite3.Error:
            pass
        return False


def _decode_vector(blob: object, expected_dimension: int | None) -> str | None:
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        return "embedding_blob_invalid"
    raw = bytes(blob)
    if not raw or len(raw) % 4:
        return "embedding_blob_invalid"
    dimension = len(raw) // 4
    if expected_dimension is not None and dimension != expected_dimension:
        return "embedding_dimension_mismatch"
    try:
        values = struct.unpack(f"<{dimension}f", raw)
    except (struct.error, ValueError):
        return "embedding_blob_invalid"
    if not all(math.isfinite(value) for value in values):
        return "embedding_nonfinite"
    if math.sqrt(sum(value * value for value in values)) <= 1e-12:
        return "embedding_zero_norm"
    return None


def _normalise_vector(value: object, expected_dimension: int) -> bytes:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (expected_dimension,):
        raise ValueError("embedding_dimension_mismatch")
    if not np.all(np.isfinite(array)):
        raise ValueError("embedding_nonfinite")
    if float(np.linalg.norm(array)) <= 1e-12:
        raise ValueError("embedding_zero_norm")
    return array.tobytes()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _savepoint(conn: sqlite3.Connection, index: int) -> str:
    name = f"{_SAVEPOINT_PREFIX}_{index}"
    conn.execute(f"SAVEPOINT {name}")
    return name


def _rollback_savepoint(conn: sqlite3.Connection, name: str) -> None:
    try:
        conn.execute(f"ROLLBACK TO SAVEPOINT {name}")
    finally:
        conn.execute(f"RELEASE SAVEPOINT {name}")


def _release_savepoint(conn: sqlite3.Connection, name: str) -> None:
    conn.execute(f"RELEASE SAVEPOINT {name}")


def _base_report(
    *,
    expected_model: str | None,
    expected_dimension: int | None,
    vec_status: str,
    vec_dimension: int | None,
    vec_reason: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "transaction_owner": "caller",
        "committed": False,
        "mutated": False,
        "mutation_uncertain": False,
        "expected_model": expected_model,
        "expected_dimension": expected_dimension,
        "vec": {
            "status": vec_status,
            "dimension": vec_dimension,
        },
        "counts": {
            "embeddings": 0,
            "missing_embeddings": 0,
            "invalid_embeddings": 0,
            "orphan_embeddings": 0,
            "missing_vec": 0,
            "stale_vec": 0,
            "invalid_vec": 0,
            "vec_mismatched": 0,
            "orphan_edges": 0,
            "self_edges": 0,
            "permanent_and_decayed": 0,
            "core_memory_not_permanent": 0,
        },
        "repairs": {
            "embeddings_repaired": 0,
            "vec_rows_inserted": 0,
            "vec_rows_removed": 0,
            "orphan_embeddings_removed": 0,
            "orphan_edges_removed": 0,
            "self_edges_removed": 0,
            "core_memories_promoted": 0,
            "permanence_conflicts_resolved": 0,
        },
        "skipped": {},
        "failures": {},
        "reasons": [vec_reason] if vec_reason else [],
        "reasons_by_kind": {},
        "limits": {
            "max_items": None,
            "items_considered": 0,
            "bounded": True,
        },
        "permanence_policy": "report",
    }


def _record(mapping: dict[str, int], reason: str, amount: int = 1) -> None:
    mapping[reason] = mapping.get(reason, 0) + amount


def _vectors_match(left: object, right: object, dimension: int) -> bool:
    try:
        left_values = np.frombuffer(bytes(left), dtype="<f4")
        right_values = np.frombuffer(bytes(right), dtype="<f4")
    except (TypeError, ValueError):
        return False
    if left_values.shape != (dimension,) or right_values.shape != (dimension,):
        return False
    return bool(np.array_equal(left_values, right_values))


def _vec_issues(
    conn: sqlite3.Connection,
    *,
    vec_dimension: int,
    expected_dimension: int | None,
) -> tuple[set[str], dict[str, int]]:
    """Return vec node ids and per-row integrity findings.

    The query is deliberately capped.  A caller that needs a complete audit
    can repeat it with a profile-specific maintenance policy; repairs use the
    same cap through ``batch_size``.
    """
    vec_ids: set[str] = set()
    issues: dict[str, int] = {}
    rows = conn.execute(
        "SELECT v.node_id, v.embedding, e.vector, n.decayed "
        "FROM vec_embeddings v "
        "LEFT JOIN embeddings e ON e.node_id=v.node_id "
        "LEFT JOIN thought_nodes n ON n.id=v.node_id "
        "ORDER BY v.node_id LIMIT 1001"
    ).fetchall()
    for node_id, vector, ordinary, decayed in rows:
        if node_id is not None:
            node_id = str(node_id)
            vec_ids.add(node_id)
        if node_id is None or ordinary is None:
            _record(issues, "vec_orphan")
            continue
        if decayed not in (None, 0):
            _record(issues, "vec_stale_decayed")
        reason = _decode_vector(vector, vec_dimension)
        if reason is not None:
            _record(issues, f"vec_{reason}")
            continue
        compare_dimension = expected_dimension or vec_dimension
        if _decode_vector(ordinary, compare_dimension) is None and not _vectors_match(
            vector, ordinary, vec_dimension
        ):
            _record(issues, "vec_value_mismatch")
    return vec_ids, issues


def inspect_integrity(
    conn: sqlite3.Connection,
    *,
    expected_model: str | None = None,
    expected_dimension: int | None = None,
) -> dict[str, Any]:
    """Inspect a database through a supplied connection without mutation.

    The connection remains open and no transaction is committed, rolled back,
    or otherwise finalized.  The report contains counts only; callers that
    need privacy-preserving identifiers can add their own redaction boundary.
    """
    if not isinstance(conn, sqlite3.Connection):
        raise TypeError("conn must be a sqlite3.Connection")
    if expected_dimension is not None and expected_dimension <= 0:
        raise ValueError("expected_dimension must be positive")

    tables = _tables(conn)
    missing_tables = sorted(_REQUIRED_TABLES - tables)
    if missing_tables:
        return {
            "schema_version": 1,
            "status": "unavailable",
            "mutated": False,
            "transaction_owner": "caller",
            "committed": False,
            "missing_tables": missing_tables,
            "reasons": ["schema_tables_missing"],
        }

    vec_status, vec_dimension, vec_reason = _vec_schema(conn)
    report = _base_report(
        expected_model=expected_model,
        expected_dimension=expected_dimension,
        vec_status=vec_status,
        vec_dimension=vec_dimension,
        vec_reason=vec_reason,
    )
    counts = report["counts"]
    rows = conn.execute(
        "SELECT node_id, vector, model FROM embeddings "
        "ORDER BY node_id LIMIT 1001"
    ).fetchall()
    counts["embeddings"] = len(rows)
    if len(rows) > 1000:
        report["reasons"].append("audit_row_cap")
        rows = rows[:1000]
    active_nodes = {
        str(row[0])
        for row in conn.execute(
            "SELECT id FROM thought_nodes "
            "WHERE decayed IS NULL OR decayed = 0 LIMIT 1001"
        ).fetchall()
    }
    embedding_ids = {str(row[0]) for row in rows if row[0] is not None}
    for node_id, blob, model in rows:
        reason = _decode_vector(blob, expected_dimension)
        if reason is not None:
            counts["invalid_embeddings"] += 1
            _record(report["reasons_by_kind"], reason)
        if expected_model is not None and model != expected_model:
            counts["invalid_embeddings"] += 1
            _record(report["reasons_by_kind"], "embedding_model_mismatch")
    orphan_count = conn.execute(
        "SELECT COUNT(*) FROM embeddings e "
        "LEFT JOIN thought_nodes n ON n.id=e.node_id WHERE n.id IS NULL"
    ).fetchone()[0]
    counts["orphan_embeddings"] = int(orphan_count)
    missing_count = conn.execute(
        "SELECT COUNT(*) FROM thought_nodes n "
        "LEFT JOIN embeddings e ON e.node_id=n.id "
        "WHERE (n.decayed IS NULL OR n.decayed=0) AND e.node_id IS NULL"
    ).fetchone()[0]
    counts["missing_embeddings"] = int(missing_count)
    if vec_status == "present" and _load_vec(conn):
        vec_ids, vec_issues = _vec_issues(
            conn,
            vec_dimension=vec_dimension or 0,
            expected_dimension=expected_dimension,
        )
        active_embedding_ids = embedding_ids & set(active_nodes)
        counts["missing_vec"] = len(active_embedding_ids - vec_ids)
        counts["stale_vec"] = len(vec_ids - active_embedding_ids)
        counts["invalid_vec"] = sum(
            value
            for reason, value in vec_issues.items()
            if reason.startswith("vec_embedding_")
        )
        counts["vec_mismatched"] = vec_issues.get("vec_value_mismatch", 0)
        for reason, value in vec_issues.items():
            _record(report["reasons_by_kind"], reason, value)
        if expected_dimension is not None and vec_dimension != expected_dimension:
            report["reasons"].append("vec_dimension_mismatch")
            _record(report["reasons_by_kind"], "vec_dimension_mismatch")
            counts["invalid_vec"] += 1
    elif vec_status == "present":
        report["reasons"].append("vec_index_unverifiable")
    counts["orphan_edges"] = int(
        conn.execute(
            "SELECT COUNT(*) FROM derivation_edges e "
            "LEFT JOIN thought_nodes p ON p.id=e.parent_id "
            "LEFT JOIN thought_nodes c ON c.id=e.child_id "
            "WHERE p.id IS NULL OR c.id IS NULL"
        ).fetchone()[0]
    )
    counts["self_edges"] = int(
        conn.execute(
            "SELECT COUNT(*) FROM derivation_edges WHERE parent_id=child_id"
        ).fetchone()[0]
    )
    counts["permanent_and_decayed"] = int(
        conn.execute(
            "SELECT COUNT(*) FROM thought_nodes "
            "WHERE COALESCE(permanent,0) != 0 AND COALESCE(decayed,0) != 0"
        ).fetchone()[0]
    )
    counts["core_memory_not_permanent"] = int(
        conn.execute(
            "SELECT COUNT(*) FROM thought_nodes "
            "WHERE node_type='core_memory' AND COALESCE(permanent,0)=0"
        ).fetchone()[0]
    )
    anomaly_keys = {
        key for key in counts if key not in {"embeddings"}
    }
    report["status"] = "findings" if any(counts[key] for key in anomaly_keys) else "ok"
    report["uncertainty"] = ["historical_consolidation"]
    return report


def repair_integrity(
    conn: sqlite3.Connection,
    *,
    embedding_fn: EmbeddingFn | None = None,
    embedding_model: str | None = None,
    expected_dimension: int | None = None,
    require_vec_parity: bool = True,
    actions: Iterable[str] | None = None,
    permanence_policy: str = "report",
    batch_size: int = 100,
    max_items: int = 1000,
) -> dict[str, Any]:
    """Apply bounded, targeted repairs using only ``conn``.

    ``conn`` remains open and the caller retains transaction ownership.  This
    function never commits, rolls back an outer transaction, closes the
    connection, or changes journal mode.  The caller should take a backup and
    an exclusive maintenance lease before invoking it, then commit and run a
    fresh integrity audit afterward.

    ``embedding_fn`` must accept a sequence of node contents and return one
    vector per input.  It is never constructed or selected by Cashew.  Without
    it, invalid or missing embeddings are reported as skipped.

    ``permanence_policy`` is report-only by default.  ``preserve_permanent``
    clears ``decayed`` for permanent nodes; ``preserve_decay`` clears
    ``permanent``.  The choice is intentionally explicit because the original
    intended state cannot be reconstructed from contradictory flags.
    """
    if not isinstance(conn, sqlite3.Connection):
        raise TypeError("conn must be a sqlite3.Connection")
    if expected_dimension is not None and expected_dimension <= 0:
        return {"status": "rejected", "reason": "invalid_embedding_dimension"}
    if batch_size <= 0 or max_items <= 0:
        return {"status": "rejected", "reason": "invalid_repair_bounds"}
    if permanence_policy not in {"report", "preserve_permanent", "preserve_decay"}:
        return {"status": "rejected", "reason": "invalid_permanence_policy"}
    selected = set(_DEFAULT_ACTIONS if actions is None else actions)
    unknown = sorted(selected - _KNOWN_ACTIONS)
    if unknown:
        return {
            "status": "rejected",
            "reason": "unknown_repair_action",
            "actions": unknown,
        }
    tables = _tables(conn)
    if not _REQUIRED_TABLES.issubset(tables):
        return {
            "status": "unavailable",
            "reason": "schema_tables_missing",
            "missing_tables": sorted(_REQUIRED_TABLES - tables),
        }
    if not conn.in_transaction:
        return {
            "status": "rejected",
            "reason": "outer_transaction_required",
            "transaction_owner": "caller",
            "committed": False,
        }

    vec_status, vec_dimension, vec_reason = _vec_schema(conn)
    report = _base_report(
        expected_model=embedding_model,
        expected_dimension=expected_dimension,
        vec_status=vec_status,
        vec_dimension=vec_dimension,
        vec_reason=vec_reason,
    )
    report["limits"].update({"max_items": max_items, "batch_size": batch_size})
    report["permanence_policy"] = permanence_policy
    counts = report["counts"]
    repairs = report["repairs"]
    skipped = report["skipped"]
    failures = report["failures"]
    counts["orphan_embeddings"] = int(
        conn.execute(
            "SELECT COUNT(*) FROM embeddings e "
            "LEFT JOIN thought_nodes n ON n.id=e.node_id WHERE n.id IS NULL"
        ).fetchone()[0]
    )
    counts["missing_embeddings"] = int(
        conn.execute(
            "SELECT COUNT(*) FROM thought_nodes n "
            "LEFT JOIN embeddings e ON e.node_id=n.id "
            "WHERE (n.decayed IS NULL OR n.decayed=0) AND e.node_id IS NULL"
        ).fetchone()[0]
    )
    counts["orphan_edges"] = int(
        conn.execute(
            "SELECT COUNT(*) FROM derivation_edges e "
            "LEFT JOIN thought_nodes p ON p.id=e.parent_id "
            "LEFT JOIN thought_nodes c ON c.id=e.child_id "
            "WHERE p.id IS NULL OR c.id IS NULL"
        ).fetchone()[0]
    )
    counts["self_edges"] = int(
        conn.execute(
            "SELECT COUNT(*) FROM derivation_edges WHERE parent_id=child_id"
        ).fetchone()[0]
    )
    counts["permanent_and_decayed"] = int(
        conn.execute(
            "SELECT COUNT(*) FROM thought_nodes "
            "WHERE COALESCE(permanent,0)!=0 AND COALESCE(decayed,0)!=0"
        ).fetchone()[0]
    )
    counts["core_memory_not_permanent"] = int(
        conn.execute(
            "SELECT COUNT(*) FROM thought_nodes "
            "WHERE node_type='core_memory' AND COALESCE(permanent,0)=0"
        ).fetchone()[0]
    )
    vec_operational = vec_status == "present" and _load_vec(conn)
    vec_ready = (
        vec_operational
        and (expected_dimension is None or vec_dimension == expected_dimension)
    )
    if vec_operational:
        vec_ids, vec_issues = _vec_issues(
            conn,
            vec_dimension=vec_dimension or 0,
            expected_dimension=expected_dimension,
        )
        active_embedding_ids = {
            str(row[0])
            for row in conn.execute(
                "SELECT e.node_id FROM embeddings e "
                "JOIN thought_nodes n ON n.id=e.node_id "
                "WHERE n.decayed IS NULL OR n.decayed=0 LIMIT 1001"
            ).fetchall()
        }
        counts["missing_vec"] = len(active_embedding_ids - vec_ids)
        counts["stale_vec"] = len(vec_ids - active_embedding_ids)
        counts["invalid_vec"] = sum(
            value
            for reason, value in vec_issues.items()
            if reason.startswith("vec_embedding_")
        )
        counts["vec_mismatched"] = vec_issues.get("vec_value_mismatch", 0)
        for reason, value in vec_issues.items():
            _record(report["reasons_by_kind"], reason, value)
        if expected_dimension is not None and vec_dimension != expected_dimension:
            report["reasons"].append("vec_dimension_mismatch")
            _record(report["reasons_by_kind"], "vec_dimension_mismatch")
            counts["invalid_vec"] += 1
    vec_needed = bool(selected & {"repair_vec", "repair_embeddings"})
    if vec_needed and require_vec_parity and not vec_ready:
        if vec_reason:
            report["reasons"].append(vec_reason)
        elif expected_dimension is not None and vec_dimension not in (None, expected_dimension):
            report["reasons"].append("vec_dimension_mismatch")
        else:
            report["reasons"].append("vec_index_unverifiable")
        skipped["vec_repairs"] = max_items

    def item_budget() -> bool:
        report["limits"]["items_considered"] += 1
        return report["limits"]["items_considered"] <= max_items

    def run_item(callback: Callable[[], None]) -> bool:
        index = report["limits"]["items_considered"]
        try:
            name = _savepoint(conn, index)
        except (OSError, RuntimeError, sqlite3.Error, TypeError, ValueError) as exc:
            _record(failures, "savepoint_begin_failed")
            _record(failures, type(exc).__name__)
            return False
        try:
            callback()
        except (OSError, RuntimeError, sqlite3.Error, TypeError, ValueError) as exc:
            try:
                _rollback_savepoint(conn, name)
            except (OSError, RuntimeError, sqlite3.Error, TypeError, ValueError):
                _record(failures, "savepoint_rollback_failed")
                report["mutated"] = True
                report["mutation_uncertain"] = True
            _record(failures, type(exc).__name__)
            return False
        try:
            _release_savepoint(conn, name)
        except (OSError, RuntimeError, sqlite3.Error, TypeError, ValueError) as exc:
            _record(failures, "savepoint_release_failed")
            _record(failures, type(exc).__name__)
            try:
                _rollback_savepoint(conn, name)
            except (OSError, RuntimeError, sqlite3.Error, TypeError, ValueError):
                _record(failures, "savepoint_rollback_failed")
                report["mutated"] = True
                report["mutation_uncertain"] = True
            return False
        report["mutated"] = True
        return True

    if "remove_orphan_embeddings" in selected:
        orphan_rows = conn.execute(
            "SELECT e.rowid, e.node_id FROM embeddings e "
            "LEFT JOIN thought_nodes n ON n.id=e.node_id "
            "WHERE n.id IS NULL ORDER BY e.node_id LIMIT ?",
            (min(batch_size, max_items),),
        ).fetchall()
        counts["orphan_embeddings"] = len(orphan_rows)
        for rowid, node_id in orphan_rows:
            if not item_budget():
                break

            def remove_orphan(rowid=rowid, node_id=node_id) -> None:
                conn.execute("DELETE FROM embeddings WHERE rowid=?", (rowid,))
                if conn.execute(
                    "SELECT 1 FROM embeddings WHERE rowid=?", (rowid,)
                ).fetchone() is not None:
                    raise RuntimeError("orphan_embedding_not_removed")
                if vec_ready and node_id is not None:
                    conn.execute("DELETE FROM vec_embeddings WHERE node_id=?", (node_id,))

            if run_item(remove_orphan):
                repairs["orphan_embeddings_removed"] += 1

    if "remove_orphan_edges" in selected:
        edge_rows = conn.execute(
            "SELECT e.rowid, e.parent_id, e.child_id FROM derivation_edges e "
            "LEFT JOIN thought_nodes p ON p.id=e.parent_id "
            "LEFT JOIN thought_nodes c ON c.id=e.child_id "
            "WHERE p.id IS NULL OR c.id IS NULL "
            "ORDER BY e.parent_id, e.child_id LIMIT ?",
            (min(batch_size, max_items),),
        ).fetchall()
        counts["orphan_edges"] = len(edge_rows)
        for rowid, _parent_id, _child_id in edge_rows:
            if not item_budget():
                break

            def remove_edge(rowid=rowid) -> None:
                conn.execute("DELETE FROM derivation_edges WHERE rowid=?", (rowid,))
                if conn.execute(
                    "SELECT 1 FROM derivation_edges WHERE rowid=?", (rowid,)
                ).fetchone() is not None:
                    raise RuntimeError("orphan_edge_not_removed")

            if run_item(remove_edge):
                repairs["orphan_edges_removed"] += 1

    if "remove_self_edges" in selected:
        self_rows = conn.execute(
            "SELECT parent_id FROM derivation_edges WHERE parent_id=child_id LIMIT ?",
            (min(batch_size, max_items),),
        ).fetchall()
        counts["self_edges"] = len(self_rows)
        for (node_id,) in self_rows:
            if not item_budget():
                break

            def remove_self(node_id=node_id) -> None:
                conn.execute(
                    "DELETE FROM derivation_edges WHERE parent_id=? AND child_id=?",
                    (node_id, node_id),
                )

            if run_item(remove_self):
                repairs["self_edges_removed"] += 1

    if vec_operational and "repair_vec" in selected:
        rows = conn.execute(
            "SELECT v.node_id, v.embedding, e.vector, e.model, n.decayed "
            "FROM vec_embeddings v "
            "LEFT JOIN embeddings e ON e.node_id=v.node_id "
            "LEFT JOIN thought_nodes n ON n.id=v.node_id "
            "ORDER BY v.node_id LIMIT ?",
            (min(batch_size, max_items),),
        ).fetchall()
        for node_id, vector, ordinary, ordinary_model, decayed in rows:
            if not item_budget():
                break
            vector_reason = _decode_vector(vector, vec_dimension)
            ordinary_reason = (
                _decode_vector(ordinary, expected_dimension or vec_dimension)
                if ordinary is not None
                else "embedding_missing"
            )
            stale = node_id is None or ordinary is None or decayed not in (None, 0)
            model_compatible = embedding_model is None or ordinary_model == embedding_model
            mismatched = (
                not stale
                and vector_reason is None
                and ordinary_reason is None
                and not _vectors_match(vector, ordinary, vec_dimension)
            )
            invalid = vector_reason is not None or mismatched
            if not stale and not invalid:
                continue
            if invalid and not model_compatible:
                _record(skipped, "embedding_model_mismatch")
                continue
            replacement = (
                ordinary
                if not stale
                and ordinary_reason is None
                and vec_ready
                and expected_dimension in (None, vec_dimension)
                else None
            )

            def replace_vec(node_id=node_id, replacement=replacement) -> None:
                if node_id is None:
                    conn.execute("DELETE FROM vec_embeddings WHERE node_id IS NULL")
                    remaining = conn.execute(
                        "SELECT 1 FROM vec_embeddings WHERE node_id IS NULL"
                    ).fetchone()
                else:
                    conn.execute(
                        "DELETE FROM vec_embeddings WHERE node_id=?", (node_id,)
                    )
                    remaining = conn.execute(
                        "SELECT 1 FROM vec_embeddings WHERE node_id=?", (node_id,)
                    ).fetchone()
                if remaining is not None:
                    raise RuntimeError("vec_row_not_removed")
                if replacement is not None:
                    conn.execute(
                        "INSERT INTO vec_embeddings(node_id, embedding) "
                        "SELECT node_id, ? FROM embeddings WHERE node_id=?",
                        (replacement, node_id),
                    )

            if run_item(replace_vec):
                repairs["vec_rows_removed"] += 1
                if replacement is not None:
                    repairs["vec_rows_inserted"] += 1

        valid_rows = conn.execute(
            "SELECT e.node_id, e.vector, e.model FROM embeddings e "
            "JOIN thought_nodes n ON n.id=e.node_id "
            "LEFT JOIN vec_embeddings v ON v.node_id=e.node_id "
            "WHERE (n.decayed IS NULL OR n.decayed=0) AND v.node_id IS NULL "
            "ORDER BY e.node_id LIMIT ?",
            (min(batch_size, max_items),),
        ).fetchall()
        for node_id, blob, model in valid_rows:
            if not item_budget():
                break
            reason = _decode_vector(blob, expected_dimension or vec_dimension)
            if (
                reason is not None
                or not vec_ready
                or (embedding_model is not None and model != embedding_model)
            ):
                _record(skipped, reason or "vec_dimension_mismatch")
                continue

            def insert_vec(node_id=node_id, blob=blob) -> None:
                conn.execute(
                    "INSERT INTO vec_embeddings(node_id, embedding) VALUES (?, ?)",
                    (node_id, blob),
                )

            if run_item(insert_vec):
                repairs["vec_rows_inserted"] += 1

    # Re-embed only rows that can be reconstructed from current node content.
    if "repair_embeddings" in selected:
        if embedding_fn is None:
            skipped["embedding_client_unavailable"] = 1
        elif expected_dimension is None or not embedding_model:
            skipped["embedding_identity_unavailable"] = 1
        elif require_vec_parity and not vec_ready:
            skipped["vec_parity_unavailable"] = 1
        else:
            candidates = conn.execute(
                "SELECT n.id, n.content, e.vector, e.model "
                "FROM thought_nodes n LEFT JOIN embeddings e ON e.node_id=n.id "
                "WHERE (n.decayed IS NULL OR n.decayed=0) "
                "AND n.content IS NOT NULL AND TRIM(n.content) != '' "
                "ORDER BY n.id LIMIT ?",
                (min(batch_size, max_items),),
            ).fetchall()
            for node_id, content, _old_blob, _old_model in candidates:
                if not item_budget():
                    break
                old_reason = _decode_vector(_old_blob, expected_dimension)
                if _old_blob is not None and old_reason is None and _old_model == embedding_model:
                    continue
                try:
                    encoded = embedding_fn([str(content)])
                    blob = _normalise_vector(encoded[0], expected_dimension)
                except (IndexError, TypeError, ValueError, RuntimeError, OSError) as exc:
                    _record(skipped, str(exc) or type(exc).__name__)
                    continue

                def replace_embedding(node_id=node_id, content=content, blob=blob) -> None:
                    current = conn.execute(
                        "SELECT content FROM thought_nodes WHERE id=?", (node_id,)
                    ).fetchone()
                    if current is None or current[0] != content:
                        raise RuntimeError("node_content_changed")
                    now = _now()
                    conn.execute(
                        "INSERT OR REPLACE INTO embeddings "
                        "(node_id, vector, model, updated_at) VALUES (?, ?, ?, ?)",
                        (node_id, blob, embedding_model, now),
                    )
                    if vec_ready:
                        conn.execute("DELETE FROM vec_embeddings WHERE node_id=?", (node_id,))
                        conn.execute(
                            "INSERT INTO vec_embeddings(node_id, embedding) VALUES (?, ?)",
                            (node_id, blob),
                        )

                if run_item(replace_embedding):
                    repairs["embeddings_repaired"] += 1

    # These are intentionally explicit and never part of the default action
    # set.  The source data cannot tell us which side of the contradiction was
    # originally intended.
    if permanence_policy != "report":
        conflict_rows = conn.execute(
            "SELECT id FROM thought_nodes "
            "WHERE COALESCE(permanent,0)!=0 AND COALESCE(decayed,0)!=0 "
            "ORDER BY id LIMIT ?",
            (min(batch_size, max_items),),
        ).fetchall()
        for (node_id,) in conflict_rows:
            if not item_budget():
                break

            def resolve_conflict(node_id=node_id) -> None:
                column = "decayed" if permanence_policy == "preserve_permanent" else "permanent"
                conn.execute(f"UPDATE thought_nodes SET {column}=0 WHERE id=?", (node_id,))

            if run_item(resolve_conflict):
                repairs["permanence_conflicts_resolved"] += 1
    if "promote_core_memories" in selected:
        core_rows = conn.execute(
            "SELECT id FROM thought_nodes "
            "WHERE node_type='core_memory' AND COALESCE(permanent,0)=0 "
            "ORDER BY id LIMIT ?",
            (min(batch_size, max_items),),
        ).fetchall()
        for (node_id,) in core_rows:
            if not item_budget():
                break

            def promote_core(node_id=node_id) -> None:
                conn.execute(
                    "UPDATE thought_nodes SET permanent=1 WHERE id=?", (node_id,)
                )

            if run_item(promote_core):
                repairs["core_memories_promoted"] += 1

    remaining_report = inspect_integrity(
        conn,
        expected_model=embedding_model,
        expected_dimension=expected_dimension,
    )
    remaining_counts = remaining_report.get("counts", {})
    report["remaining"] = {
        key: value
        for key, value in remaining_counts.items()
        if key != "embeddings" and value
    }
    actionable_keys = set()
    if "remove_orphan_embeddings" in selected:
        actionable_keys.add("orphan_embeddings")
    if "remove_orphan_edges" in selected:
        actionable_keys.add("orphan_edges")
    if "repair_vec" in selected:
        actionable_keys.update(
            {"missing_vec", "stale_vec", "invalid_vec", "vec_mismatched"}
        )
    if "repair_embeddings" in selected:
        actionable_keys.update({"missing_embeddings", "invalid_embeddings"})
    if "remove_self_edges" in selected:
        actionable_keys.add("self_edges")
    if permanence_policy != "report":
        actionable_keys.add("permanent_and_decayed")
    if "promote_core_memories" in selected:
        actionable_keys.add("core_memory_not_permanent")
    remaining_actionable = any(
        remaining_counts.get(key, 0) for key in actionable_keys
    )
    report["status"] = (
        "partial" if skipped or failures or remaining_actionable else "completed"
    )
    report["uncertainty"] = ["historical_consolidation", "commit_owned_by_caller"]
    return report


__all__ = ["inspect_integrity", "repair_integrity"]
