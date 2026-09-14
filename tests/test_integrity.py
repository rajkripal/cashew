"""Contract tests for the connection-owned integrity API."""

from __future__ import annotations

import sqlite3
import struct
from pathlib import Path

import numpy as np
import pytest

import core.integrity as integrity
from core.integrity import inspect_integrity, repair_integrity


def _blob(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def _create_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        PRAGMA user_version = 3;
        CREATE TABLE thought_nodes (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            node_type TEXT NOT NULL,
            decayed INTEGER DEFAULT 0,
            permanent INTEGER DEFAULT 0,
            access_count INTEGER DEFAULT 0,
            last_updated TEXT
        );
        CREATE TABLE embeddings (
            node_id TEXT PRIMARY KEY,
            vector BLOB NOT NULL,
            model TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE derivation_edges (
            parent_id TEXT,
            child_id TEXT,
            weight REAL,
            reasoning TEXT,
            timestamp TEXT,
            PRIMARY KEY (parent_id, child_id)
        );
        """
    )
    conn.commit()
    return conn


def _add_node(conn: sqlite3.Connection, node_id: str, content: str = "content") -> None:
    conn.execute(
        "INSERT INTO thought_nodes (id, content, node_type) VALUES (?, ?, 'fact')",
        (node_id, content),
    )


def test_inspection_is_query_only_and_keeps_connection_transaction_and_journal(
    tmp_path: Path,
) -> None:
    conn = _create_db(tmp_path / "inspect.db")
    assert conn.execute("PRAGMA journal_mode=WAL").fetchone()[0].lower() == "wal"
    _add_node(conn, "n1")
    conn.commit()
    conn.execute("INSERT INTO thought_nodes (id, content, node_type) VALUES ('pending', 'x', 'fact')")
    before_mode = conn.execute("PRAGMA journal_mode").fetchone()[0].lower()

    report = inspect_integrity(conn, expected_model="model-a", expected_dimension=4)

    assert report["status"] == "findings"
    assert report["transaction_owner"] == "caller"
    assert report["committed"] is False
    assert conn.in_transaction is True
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == before_mode
    assert conn.execute("SELECT 1 FROM thought_nodes WHERE id='pending'").fetchone() == (1,)
    conn.rollback()
    conn.close()


def test_orphan_repairs_preserve_outer_transaction_and_are_idempotent(
    tmp_path: Path,
) -> None:
    path = tmp_path / "orphan.db"
    conn = _create_db(path)
    _add_node(conn, "live")
    conn.execute(
        "INSERT INTO embeddings VALUES ('ghost', ?, 'model-a', 'now')",
        (_blob([1.0, 0.0, 0.0, 0.0]),),
    )
    conn.execute(
        "INSERT INTO derivation_edges VALUES ('ghost', 'live', 1.0, 'bad', 'now')"
    )
    conn.commit()
    conn.execute(
        "INSERT INTO thought_nodes (id, content, node_type) VALUES ('caller', 'keep', 'fact')"
    )

    first = repair_integrity(
        conn,
        embedding_model="model-a",
        expected_dimension=4,
        actions={"remove_orphan_embeddings", "remove_orphan_edges"},
    )
    assert first["status"] == "completed"
    assert first["mutated"] is True
    assert first["committed"] is False
    assert conn.execute("SELECT 1 FROM thought_nodes WHERE id='caller'").fetchone() == (1,)
    # The caller can still roll back all work, including the repair.
    conn.rollback()
    assert conn.execute("SELECT 1 FROM embeddings WHERE node_id='ghost'").fetchone() == (1,)
    conn.close()

    conn = sqlite3.connect(path)
    # Recreate and commit the repair, then prove a second pass is a no-op.
    conn.execute("BEGIN")
    second = repair_integrity(
        conn,
        embedding_model="model-a",
        expected_dimension=4,
        actions={"remove_orphan_embeddings", "remove_orphan_edges"},
    )
    conn.commit()
    assert second["repairs"]["orphan_embeddings_removed"] == 1
    assert second["repairs"]["orphan_edges_removed"] == 1
    conn.execute("BEGIN")
    again = repair_integrity(
        conn,
        embedding_model="model-a",
        expected_dimension=4,
        actions={"remove_orphan_embeddings", "remove_orphan_edges"},
    )
    assert again["mutated"] is False
    assert again["repairs"]["orphan_embeddings_removed"] == 0
    assert again["repairs"]["orphan_edges_removed"] == 0
    conn.rollback()
    conn.close()


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        (np.zeros(4, dtype=np.float32), "embedding_zero_norm"),
        (np.array([np.nan, 0, 0, 1], dtype=np.float32), "embedding_nonfinite"),
        (np.ones(3, dtype=np.float32), "embedding_dimension_mismatch"),
    ],
)
def test_embedding_output_validation_does_not_write_invalid_pairs(
    tmp_path: Path, value: np.ndarray, reason: str
) -> None:
    conn = _create_db(tmp_path / f"invalid-{reason}.db")
    _add_node(conn, "n1", "repair me")
    conn.commit()
    conn.execute("BEGIN")

    result = repair_integrity(
        conn,
        embedding_fn=lambda _texts: np.asarray([value]),
        embedding_model="model-a",
        expected_dimension=4,
        require_vec_parity=False,
        actions={"repair_embeddings"},
    )

    assert result["status"] == "partial"
    assert result["skipped"][reason] == 1
    assert conn.execute("SELECT COUNT(*) FROM embeddings").fetchone() == (0,)
    conn.rollback()
    conn.close()


def test_model_mismatch_and_missing_embedding_are_repaired_from_current_content(
    tmp_path: Path,
) -> None:
    conn = _create_db(tmp_path / "reembed.db")
    _add_node(conn, "missing", "new text")
    _add_node(conn, "stale", "old text")
    conn.execute(
        "INSERT INTO embeddings VALUES ('stale', ?, 'old-model', 'now')",
        (_blob([1.0, 0.0, 0.0, 0.0]),),
    )
    conn.commit()
    conn.execute("BEGIN")
    seen: list[str] = []

    def embed(texts):
        seen.extend(texts)
        return np.asarray([[0.0, 1.0, 0.0, 0.0] for _ in texts], dtype=np.float32)

    result = repair_integrity(
        conn,
        embedding_fn=embed,
        embedding_model="model-a",
        expected_dimension=4,
        require_vec_parity=False,
        actions={"repair_embeddings"},
    )
    conn.commit()

    assert result["status"] == "completed"
    assert result["repairs"]["embeddings_repaired"] == 2
    assert set(seen) == {"new text", "old text"}
    assert conn.execute("SELECT COUNT(*) FROM embeddings").fetchone() == (2,)
    assert conn.execute("SELECT DISTINCT model FROM embeddings").fetchone() == ("model-a",)
    conn.close()


def test_savepoint_rollback_keeps_other_repairs_and_reports_failure(tmp_path: Path) -> None:
    conn = _create_db(tmp_path / "savepoint.db")
    _add_node(conn, "bad", "bad")
    _add_node(conn, "good", "good")
    conn.execute(
        "CREATE TRIGGER reject_bad BEFORE INSERT ON embeddings "
        "WHEN NEW.node_id='bad' BEGIN SELECT RAISE(ABORT, 'reject bad'); END"
    )
    conn.commit()
    conn.execute("BEGIN")

    result = repair_integrity(
        conn,
        embedding_fn=lambda texts: np.asarray(
            [[1.0, 0.0, 0.0, 0.0] for _ in texts], dtype=np.float32
        ),
        embedding_model="model-a",
        expected_dimension=4,
        require_vec_parity=False,
        actions={"repair_embeddings"},
    )

    assert result["status"] == "partial"
    assert result["repairs"]["embeddings_repaired"] == 1
    assert result["failures"]
    assert conn.execute("SELECT node_id FROM embeddings").fetchall() == [("good",)]
    conn.rollback()
    conn.close()


def test_ambiguous_permanence_and_self_edges_are_report_only_by_default(
    tmp_path: Path,
) -> None:
    conn = _create_db(tmp_path / "ambiguous.db")
    _add_node(conn, "n1")
    conn.execute("UPDATE thought_nodes SET permanent=1, decayed=1 WHERE id='n1'")
    conn.execute(
        "INSERT INTO derivation_edges VALUES ('n1', 'n1', 1.0, 'self', 'now')"
    )
    conn.commit()
    conn.execute("BEGIN")

    report = repair_integrity(conn, actions=set(), expected_dimension=4)
    assert report["status"] == "completed"
    assert report["counts"]["permanent_and_decayed"] == 1
    assert report["counts"]["self_edges"] == 1
    assert conn.execute("SELECT decayed, permanent FROM thought_nodes").fetchone() == (1, 1)
    assert conn.execute("SELECT COUNT(*) FROM derivation_edges").fetchone() == (1,)

    fixed = repair_integrity(
        conn,
        actions={"remove_self_edges", "promote_core_memories"},
        permanence_policy="preserve_permanent",
        expected_dimension=4,
    )
    assert fixed["repairs"]["self_edges_removed"] == 1
    assert fixed["repairs"]["permanence_conflicts_resolved"] == 1
    conn.commit()
    assert conn.execute("SELECT decayed FROM thought_nodes").fetchone() == (0,)
    assert conn.execute("SELECT COUNT(*) FROM derivation_edges").fetchone() == (0,)
    conn.close()


def test_invalid_arguments_and_schema_states_are_explicit(tmp_path: Path) -> None:
    conn = _create_db(tmp_path / "states.db")
    assert repair_integrity(conn, batch_size=0)["status"] == "rejected"
    assert repair_integrity(conn, actions={"invented"})["status"] == "rejected"
    conn.close()

    incomplete = sqlite3.connect(tmp_path / "incomplete.db")
    incomplete.execute("CREATE TABLE thought_nodes (id TEXT PRIMARY KEY)")
    incomplete.commit()
    report = inspect_integrity(incomplete)
    assert report["status"] == "unavailable"
    assert repair_integrity(incomplete)["status"] == "unavailable"
    incomplete.close()

    deceptive = _create_db(tmp_path / "deceptive.db")
    deceptive.execute("CREATE TABLE vec_embeddings (node_id TEXT, embedding BLOB)")
    deceptive.commit()
    deceptive.execute("BEGIN")
    result = repair_integrity(deceptive, actions={"repair_vec"})
    assert result["status"] == "partial"
    assert "vec_schema_invalid" in result["reasons"]
    deceptive.close()


def test_real_vec_repairs_are_atomic_when_available(tmp_path: Path) -> None:
    sqlite_vec = pytest.importorskip("sqlite_vec")
    conn = _create_db(tmp_path / "real-vec.db")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute(
        "CREATE VIRTUAL TABLE vec_embeddings USING vec0("
        "node_id TEXT PRIMARY KEY, embedding float[4] distance_metric=cosine)"
    )
    _add_node(conn, "valid")
    _add_node(conn, "stale")
    blob = _blob([1.0, 0.0, 0.0, 0.0])
    conn.execute("INSERT INTO embeddings VALUES ('valid', ?, 'model-a', 'now')", (blob,))
    conn.execute("INSERT INTO embeddings VALUES ('stale', ?, 'model-a', 'now')", (blob,))
    conn.execute("INSERT INTO vec_embeddings VALUES ('stale', ?)", (blob,))
    conn.commit()
    # Establish the caller-owned outer transaction before invoking repair;
    # releasing a savepoint without an outer transaction necessarily commits
    # that savepoint in SQLite.
    conn.execute("BEGIN")

    result = repair_integrity(
        conn,
        expected_dimension=4,
        embedding_model="model-a",
        actions={"repair_vec"},
    )
    assert result["status"] == "completed"
    assert result["repairs"]["vec_rows_inserted"] == 1
    assert result["repairs"]["vec_rows_removed"] == 0
    assert conn.execute("SELECT node_id FROM vec_embeddings ORDER BY node_id").fetchall() == [
        ("stale",),
        ("valid",),
    ]
    # The API never commits; the caller can still roll back the complete
    # ordinary-table and sqlite-vec change as one transaction.
    conn.rollback()
    assert conn.execute("SELECT node_id FROM vec_embeddings ORDER BY node_id").fetchall() == [
        ("stale",),
    ]
    conn.close()


def test_null_orphans_are_deleted_by_rowid(tmp_path: Path) -> None:
    conn = _create_db(tmp_path / "null-orphans.db")
    _add_node(conn, "live")
    blob = _blob([1.0, 0.0, 0.0, 0.0])
    conn.execute(
        "INSERT INTO embeddings (node_id, vector, model, updated_at) "
        "VALUES (NULL, ?, 'model-a', 'now')",
        (blob,),
    )
    conn.execute(
        "INSERT INTO derivation_edges VALUES (NULL, 'live', 1.0, 'bad', 'now')"
    )
    conn.commit()
    conn.execute("BEGIN")

    result = repair_integrity(
        conn,
        require_vec_parity=False,
        actions={"remove_orphan_embeddings", "remove_orphan_edges"},
    )

    assert result["status"] == "completed"
    assert result["repairs"]["orphan_embeddings_removed"] == 1
    assert result["repairs"]["orphan_edges_removed"] == 1
    assert conn.execute("SELECT COUNT(*) FROM embeddings").fetchone() == (0,)
    assert conn.execute("SELECT COUNT(*) FROM derivation_edges").fetchone() == (0,)
    conn.rollback()
    conn.close()


def test_inventory_embeddings_do_not_make_healthy_status_findings(
    tmp_path: Path,
) -> None:
    conn = _create_db(tmp_path / "healthy.db")
    _add_node(conn, "live")
    conn.execute(
        "INSERT INTO embeddings VALUES ('live', ?, 'model-a', 'now')",
        (_blob([1.0, 0.0, 0.0, 0.0]),),
    )
    conn.commit()

    report = inspect_integrity(conn, expected_model="model-a", expected_dimension=4)

    assert report["counts"]["embeddings"] == 1
    assert report["status"] == "ok"
    conn.close()


def test_real_vec_invalid_and_mismatched_rows_are_replaced(tmp_path: Path) -> None:
    sqlite_vec = pytest.importorskip("sqlite_vec")
    conn = _create_db(tmp_path / "vec-invalid.db")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute(
        "CREATE VIRTUAL TABLE vec_embeddings USING vec0("
        "node_id TEXT PRIMARY KEY, embedding float[4] distance_metric=cosine)"
    )
    ordinary = _blob([1.0, 0.0, 0.0, 0.0])
    for node_id in ("zero", "nan", "mismatch"):
        _add_node(conn, node_id)
        conn.execute(
            "INSERT INTO embeddings VALUES (?, ?, 'model-a', 'now')",
            (node_id, ordinary),
        )
    conn.execute(
        "INSERT INTO vec_embeddings VALUES ('zero', ?)",
        (_blob([0.0, 0.0, 0.0, 0.0]),),
    )
    conn.execute(
        "INSERT INTO vec_embeddings VALUES ('nan', ?)",
        (struct.pack("<4f", float("nan"), 0.0, 0.0, 0.0),),
    )
    conn.execute(
        "INSERT INTO vec_embeddings VALUES ('mismatch', ?)",
        (_blob([0.0, 1.0, 0.0, 0.0]),),
    )
    conn.commit()
    before = inspect_integrity(conn, expected_dimension=4)
    assert before["counts"]["invalid_vec"] == 2
    assert before["counts"]["vec_mismatched"] == 1
    assert before["reasons_by_kind"]["vec_embedding_zero_norm"] == 1
    assert before["reasons_by_kind"]["vec_embedding_nonfinite"] == 1
    assert before["reasons_by_kind"]["vec_value_mismatch"] == 1
    wrong_schema_dimension = inspect_integrity(conn, expected_dimension=3)
    assert wrong_schema_dimension["counts"]["invalid_vec"] >= 1
    assert "vec_dimension_mismatch" in wrong_schema_dimension["reasons"]
    conn.execute("BEGIN")

    result = repair_integrity(
        conn,
        expected_dimension=4,
        embedding_model="model-a",
        actions={"repair_vec"},
        batch_size=10,
    )

    assert result["status"] == "completed"
    assert result["repairs"]["vec_rows_inserted"] == 3
    assert result["remaining"] == {}
    assert conn.execute(
        "SELECT node_id, embedding FROM vec_embeddings ORDER BY node_id"
    ).fetchall() == [(node_id, ordinary) for node_id in ("mismatch", "nan", "zero")]
    conn.rollback()
    conn.close()


def test_real_vec_repair_does_not_copy_old_model_vector(tmp_path: Path) -> None:
    sqlite_vec = pytest.importorskip("sqlite_vec")
    conn = _create_db(tmp_path / "vec-old-model.db")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute(
        "CREATE VIRTUAL TABLE vec_embeddings USING vec0("
        "node_id TEXT PRIMARY KEY, embedding float[4] distance_metric=cosine)"
    )
    _add_node(conn, "old")
    ordinary = _blob([1.0, 0.0, 0.0, 0.0])
    conn.execute(
        "INSERT INTO embeddings VALUES ('old', ?, 'old-model', 'now')", (ordinary,)
    )
    conn.execute(
        "INSERT INTO vec_embeddings VALUES ('old', ?)",
        (_blob([0.0, 0.0, 0.0, 0.0]),),
    )
    conn.commit()
    conn.execute("BEGIN")

    result = repair_integrity(
        conn,
        expected_dimension=4,
        embedding_model="new-model",
        actions={"repair_vec"},
    )

    assert result["status"] == "partial"
    assert result["skipped"]["embedding_model_mismatch"] == 1
    assert result["remaining"]["invalid_vec"] == 1
    assert conn.execute(
        "SELECT embedding FROM vec_embeddings WHERE node_id='old'"
    ).fetchone() == (_blob([0.0, 0.0, 0.0, 0.0]),)
    conn.rollback()
    conn.close()


def test_bounded_repairs_report_remaining_work(tmp_path: Path) -> None:
    conn = _create_db(tmp_path / "bounded.db")
    for node_id in ("a", "b", "c"):
        _add_node(conn, node_id)
        conn.execute(
            "INSERT INTO embeddings VALUES (?, ?, 'model-a', 'now')",
            (f"ghost-{node_id}", _blob([1.0, 0.0, 0.0, 0.0])),
        )
        conn.execute(
            "INSERT INTO derivation_edges VALUES (?, ?, 1.0, 'bad', 'now')",
            (f"ghost-{node_id}", node_id),
        )
    conn.commit()
    conn.execute("BEGIN")

    result = repair_integrity(
        conn,
        require_vec_parity=False,
        actions={"remove_orphan_embeddings", "remove_orphan_edges"},
        batch_size=10,
        max_items=1,
    )

    assert result["status"] == "partial"
    assert result["limits"]["items_considered"] == 2
    assert result["remaining"]["orphan_embeddings"] == 2
    assert result["remaining"]["orphan_edges"] == 3
    conn.rollback()
    conn.close()


def test_release_failure_is_structured_and_savepoint_is_rolled_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conn = _create_db(tmp_path / "release-failure.db")
    conn.execute(
        "INSERT INTO embeddings VALUES ('ghost', ?, 'model-a', 'now')",
        (_blob([1.0, 0.0, 0.0, 0.0]),),
    )
    conn.commit()
    conn.execute("BEGIN")

    def fail_release(_conn, _name):
        raise sqlite3.OperationalError("synthetic release failure")

    monkeypatch.setattr(integrity, "_release_savepoint", fail_release)
    result = repair_integrity(
        conn,
        require_vec_parity=False,
        actions={"remove_orphan_embeddings"},
    )

    assert result["status"] == "partial"
    assert result["failures"]["savepoint_release_failed"] == 1
    assert result["repairs"]["orphan_embeddings_removed"] == 0
    assert conn.execute("SELECT COUNT(*) FROM embeddings").fetchone() == (1,)
    conn.rollback()
    conn.close()


def test_rollback_failure_reports_mutation_uncertainty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conn = _create_db(tmp_path / "rollback-uncertain.db")
    conn.execute(
        "INSERT INTO embeddings VALUES ('ghost', ?, 'model-a', 'now')",
        (_blob([1.0, 0.0, 0.0, 0.0]),),
    )
    conn.commit()
    conn.execute("BEGIN")

    def fail_release(_conn, _name):
        raise sqlite3.OperationalError("synthetic release failure")

    def fail_rollback(_conn, _name):
        raise sqlite3.OperationalError("synthetic rollback failure")

    monkeypatch.setattr(integrity, "_release_savepoint", fail_release)
    monkeypatch.setattr(integrity, "_rollback_savepoint", fail_rollback)
    result = repair_integrity(
        conn,
        require_vec_parity=False,
        actions={"remove_orphan_embeddings"},
    )

    assert result["status"] == "partial"
    assert result["mutated"] is True
    assert result["mutation_uncertain"] is True
    assert result["failures"]["savepoint_release_failed"] == 1
    assert result["failures"]["savepoint_rollback_failed"] == 1
    conn.rollback()
    conn.close()


def test_repairs_require_a_caller_owned_outer_transaction(tmp_path: Path) -> None:
    conn = _create_db(tmp_path / "transaction-required.db")

    result = repair_integrity(conn, actions=set())

    assert result == {
        "status": "rejected",
        "reason": "outer_transaction_required",
        "transaction_owner": "caller",
        "committed": False,
    }
    conn.close()
