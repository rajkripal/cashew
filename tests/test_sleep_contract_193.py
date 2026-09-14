"""Production-boundary contracts for the upstream sleep adapter."""

from __future__ import annotations

import sqlite3
import inspect
from pathlib import Path

import numpy as np
import pytest

from core.decay_audit import ensure_decay_audit_schema
import core.sleep as sleep_module
from core.sleep import (
    _batch_cross_links, _embed_orphans, _empty_sleep_result,
    _ensure_sleep_state_schema,
    _select_cycle_node_ids,
    _vec_write_capability, run_sleep_cycle,
)


class _CountingConnection(sqlite3.Connection):
    commit_count = 0

    def commit(self):
        self.commit_count += 1
        return super().commit()


def _edge_db(tmp_path: Path, name: str = "edges.db", factory=None) -> sqlite3.Connection:
    kwargs = {"factory": factory} if factory is not None else {}
    conn = sqlite3.connect(str(tmp_path / name), **kwargs)
    conn.executescript("""
        CREATE TABLE derivation_edges(parent_id TEXT, child_id TEXT,
            weight REAL, reasoning TEXT, PRIMARY KEY(parent_id, child_id));
    """)
    return conn


def _cycle_db(tmp_path: Path, name: str = "cycle.db") -> Path:
    path = tmp_path / name
    conn = sqlite3.connect(str(path))
    conn.executescript("""
        CREATE TABLE thought_nodes(
            id TEXT PRIMARY KEY, content TEXT, decayed INTEGER DEFAULT 0,
            timestamp TEXT DEFAULT '', source_file TEXT, access_count INTEGER DEFAULT 0,
            permanent INTEGER DEFAULT 0, last_accessed TEXT, domain TEXT, node_type TEXT,
            confidence REAL, metadata TEXT, mood_state TEXT
        );
        CREATE TABLE embeddings(
            node_id TEXT PRIMARY KEY, vector BLOB, model TEXT, updated_at TEXT
        );
        CREATE TABLE derivation_edges(
            parent_id TEXT, child_id TEXT, weight REAL, reasoning TEXT,
            PRIMARY KEY(parent_id, child_id)
        );
    """)
    vector = np.ones(1024, dtype=np.float32).tobytes()
    for nid in ("a", "b"):
        conn.execute("INSERT INTO thought_nodes(id, content) VALUES (?, ?)", (nid, nid))
        conn.execute(
            "INSERT INTO embeddings(node_id, vector, model) VALUES (?, ?, ?)",
            (nid, vector, "thenlper/gte-large"),
        )
    conn.commit()
    conn.close()
    return path


def _add_four_fairness_nodes(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    conn.execute("DELETE FROM embeddings")
    conn.execute("DELETE FROM thought_nodes")
    for index, node_id in enumerate(("a", "b", "c", "d"), start=1):
        vector = np.zeros(1024, dtype=np.float32)
        vector[index - 1] = 1.0
        conn.execute(
            "INSERT INTO thought_nodes(id, content, timestamp) VALUES (?, ?, ?)",
            (node_id, node_id, f"2026-01-0{index}T00:00:00+00:00"),
        )
        conn.execute(
            "INSERT INTO embeddings(node_id, vector, model) VALUES (?, ?, ?)",
            (node_id, vector.tobytes(), "thenlper/gte-large"),
        )
    conn.commit()
    conn.close()


def _neutralize_post_pair_phases(monkeypatch) -> None:
    monkeypatch.setattr(sleep_module, "_compute_metrics", lambda _conn: {})
    monkeypatch.setattr(sleep_module, "_garbage_collect", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        sleep_module, "_evaluate_permanence", lambda _conn: {"nodes_promoted": 0}
    )
    monkeypatch.setattr(
        sleep_module,
        "_promote_core_memories",
        lambda _conn, _metrics: {"promoted": 0, "demoted": 0},
    )


def test_cross_link_cap_flushes_two_directed_rows(tmp_path):
    conn = _edge_db(tmp_path)
    stats = _batch_cross_links(
        conn, ["a", "b"], np.array([[0, 1]]), np.eye(2), max_edges=1
    )
    assert stats["created"] == 1
    assert stats["directed_rows"] == 2
    assert stats["dream_pairs"] == [("a", "b", 0.0)]
    assert conn.execute("SELECT count(*) FROM derivation_edges").fetchone()[0] == 2
    conn.close()


def test_capped_candidate_pages_rotate_across_four_nodes(tmp_path, monkeypatch):
    path = _cycle_db(tmp_path, "fair-pages.db")
    _add_four_fairness_nodes(path)
    selected = []

    def capture(ids, _matrix, **_kwargs):
        selected.append(list(ids))
        empty = np.empty((0, 2), dtype=int)
        return empty, empty, np.eye(len(ids))

    monkeypatch.setattr(sleep_module, "_find_pairs", capture)
    _neutralize_post_pair_phases(monkeypatch)
    before_conn = sqlite3.connect(str(path))
    before = before_conn.execute(
        "SELECT id, timestamp FROM thought_nodes ORDER BY id"
    ).fetchall()
    before_conn.close()

    for _ in range(2):
        result = run_sleep_cycle(db_path=str(path), limit=2, journal_policy="preserve")
        assert result["status"] == "completed"

    assert selected == [["a", "b"], ["c", "d"]]
    check = sqlite3.connect(str(path))
    assert check.execute(
        "SELECT epoch FROM _cashew_sleep_state WHERE name='candidate_cursor_v1'"
    ).fetchone()[0] == 2
    assert check.execute(
        "SELECT id, timestamp FROM thought_nodes ORDER BY id"
    ).fetchall() == before
    check.close()


def test_cursor_survives_interruption_then_wrap_repairs_half_pair(tmp_path, monkeypatch):
    path = _cycle_db(tmp_path, "fair-restart.db")
    _add_four_fairness_nodes(path)
    conn = sqlite3.connect(str(path))
    conn.execute("INSERT INTO derivation_edges VALUES ('a','b',.92,'existing half')")
    conn.commit()
    before = conn.execute(
        "SELECT id, timestamp FROM thought_nodes ORDER BY id"
    ).fetchall()
    conn.close()
    calls = []

    def interrupt_once(ids, _matrix, **_kwargs):
        calls.append(list(ids))
        if len(calls) == 1:
            raise RuntimeError("interrupted after durable page claim")
        sim = np.eye(2, dtype=np.float64)
        sim[0, 1] = sim[1, 0] = 0.92
        return np.asarray([[0, 1]]), np.empty((0, 2), dtype=int), sim

    monkeypatch.setattr(sleep_module, "_find_pairs", interrupt_once)
    _neutralize_post_pair_phases(monkeypatch)
    failed = run_sleep_cycle(db_path=str(path), limit=2, journal_policy="preserve")
    assert failed["status"] == "failed"
    assert failed["error"] == "sleep_cycle_failed"

    next_page = run_sleep_cycle(db_path=str(path), limit=2, journal_policy="preserve")
    wrapped = run_sleep_cycle(db_path=str(path), limit=2, journal_policy="preserve")
    assert calls == [["a", "b"], ["c", "d"], ["a", "b"]]
    assert next_page["cross_links_created"] == 1
    assert wrapped["cross_links_created"] == 0
    assert wrapped["cross_links_repaired"] == 1

    check = sqlite3.connect(str(path))
    assert check.execute(
        "SELECT count(*) FROM derivation_edges WHERE "
        "(parent_id='a' AND child_id='b') OR (parent_id='b' AND child_id='a')"
    ).fetchone()[0] == 2
    assert check.execute(
        "SELECT epoch FROM _cashew_sleep_state WHERE name='candidate_cursor_v1'"
    ).fetchone()[0] == 3
    assert check.execute(
        "SELECT id, timestamp FROM thought_nodes ORDER BY id"
    ).fetchall() == before
    check.close()


def test_cross_link_repairs_half_pair_and_counts_one_row(tmp_path):
    conn = _edge_db(tmp_path)
    conn.execute("INSERT INTO derivation_edges VALUES ('a','b',.9,'old')")
    conn.commit()
    stats = _batch_cross_links(
        conn, ["a", "b"], np.array([[0, 1]]), np.eye(2), max_edges=1
    )
    assert stats["created"] == 0 and stats["repaired"] == 1
    assert stats["directed_rows"] == 1
    assert stats["dream_pairs"] == [("a", "b", 0.0)]
    assert conn.execute("SELECT count(*) FROM derivation_edges").fetchone()[0] == 2
    conn.close()


def test_public_cycle_reports_exact_prefix_when_final_cross_commit_fails(
    tmp_path, monkeypatch,
):
    path = _cycle_db(tmp_path, "cross-prefix.db")
    conn = sqlite3.connect(str(path))
    conn.execute("DELETE FROM embeddings")
    conn.execute("DELETE FROM thought_nodes")
    vector = np.ones(1024, dtype=np.float32).tobytes()
    rows = [(f"n{i:04d}", f"node {i}") for i in range(1002)]
    conn.executemany("INSERT INTO thought_nodes(id, content) VALUES (?, ?)", rows)
    conn.executemany(
        "INSERT INTO embeddings(node_id, vector, model) VALUES (?, ?, ?)",
        [(node_id, vector, "thenlper/gte-large") for node_id, _ in rows],
    )
    conn.commit()
    conn.close()

    pairs = np.asarray([(i, i + 1) for i in range(0, 1002, 2)], dtype=int)
    monkeypatch.setattr(
        sleep_module,
        "_find_pairs",
        lambda ids, _matrix, **_kwargs: (
            pairs, np.empty((0, 2), dtype=int), np.ones((len(ids), len(ids))),
        ),
    )
    _neutralize_post_pair_phases(monkeypatch)
    real_connect = sqlite3.connect

    class FailFinalCommit(sqlite3.Connection):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.commit_calls = 0

        def commit(self):
            self.commit_calls += 1
            if self.commit_calls == 2:
                raise sqlite3.OperationalError("synthetic final commit failure")
            return super().commit()

    first = True

    def connect(database, *args, **kwargs):
        nonlocal first
        if first and str(database) == str(path):
            first = False
            kwargs["factory"] = FailFinalCommit
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(sleep_module.sqlite3, "connect", connect)
    dream_inputs = []

    def capture_dream(_conn, tuples, model_fn):
        dream_inputs.extend(tuples)
        return None

    monkeypatch.setattr(sleep_module, "_generate_dream", capture_dream)
    result = run_sleep_cycle(
        db_path=str(path), model_fn=lambda _prompt: "unused",
        journal_policy="preserve",
    )

    assert result["status"] == "partial"
    assert result["error"] == "cross_link_failed"
    assert result["cross_links_created"] == 500
    assert result["cross_link_directed_rows"] == 1000
    assert len(dream_inputs) == 500
    assert dream_inputs[-1][:2] == ("n0998", "n0999")
    assert set(result) == set(_empty_sleep_result("partial", "cross_link_failed"))
    check = real_connect(str(path))
    assert check.execute("SELECT count(*) FROM derivation_edges").fetchone()[0] == 1000
    check.close()


def test_public_cycle_reports_uncertain_when_commit_cannot_be_verified(
    tmp_path, monkeypatch,
):
    path = _cycle_db(tmp_path, "cross-uncertain.db")
    monkeypatch.setattr(
        sleep_module,
        "_find_pairs",
        lambda *_args, **_kwargs: (
            np.asarray([[0, 1]]), np.empty((0, 2), dtype=int), np.eye(2),
        ),
    )
    real_connect = sqlite3.connect

    class UnknownCommit(sqlite3.Connection):
        def commit(self):
            raise sqlite3.OperationalError("synthetic ambiguous commit")

        def execute(self, sql, *args, **kwargs):
            if sql.startswith("SELECT count(*) FROM derivation_edges"):
                raise sqlite3.OperationalError("synthetic verification failure")
            return super().execute(sql, *args, **kwargs)

    first = True

    def connect(database, *args, **kwargs):
        nonlocal first
        if first and str(database) == str(path):
            first = False
            kwargs["factory"] = UnknownCommit
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(sleep_module.sqlite3, "connect", connect)
    result = run_sleep_cycle(db_path=str(path), journal_policy="preserve")
    assert result["status"] == "uncertain"
    assert result["error"] == "cross_link_commit_uncertain"
    assert result["cross_links_created"] == 0
    assert result["cross_link_directed_rows"] == 0
    assert set(result) == set(
        _empty_sleep_result("uncertain", "cross_link_commit_uncertain")
    )


def test_public_cycle_counts_commit_that_succeeds_before_wrapper_error(
    tmp_path, monkeypatch,
):
    path = _cycle_db(tmp_path, "cross-post-commit.db")
    monkeypatch.setattr(
        sleep_module,
        "_find_pairs",
        lambda *_args, **_kwargs: (
            np.asarray([[0, 1]]), np.empty((0, 2), dtype=int), np.eye(2),
        ),
    )
    _neutralize_post_pair_phases(monkeypatch)
    real_connect = sqlite3.connect

    class PostCommitError(sqlite3.Connection):
        def commit(self):
            super().commit()
            raise sqlite3.OperationalError("synthetic post-commit wrapper failure")

    first = True

    def connect(database, *args, **kwargs):
        nonlocal first
        if first and str(database) == str(path):
            first = False
            kwargs["factory"] = PostCommitError
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(sleep_module.sqlite3, "connect", connect)
    result = run_sleep_cycle(db_path=str(path), journal_policy="preserve")
    assert result["status"] == "completed"
    assert result["error"] is None
    assert result["cross_links_created"] == 1
    assert result["cross_link_directed_rows"] == 2
    check = real_connect(str(path))
    assert check.execute("SELECT count(*) FROM derivation_edges").fetchone() == (2,)
    check.close()


def test_cross_link_trigger_suppression_is_not_claimed(tmp_path):
    conn = _edge_db(tmp_path)
    conn.execute(
        """CREATE TRIGGER suppress_reverse BEFORE INSERT ON derivation_edges
           WHEN NEW.parent_id='b' AND NEW.child_id='a'
           BEGIN SELECT RAISE(IGNORE); END"""
    )
    conn.commit()
    stats = _batch_cross_links(
        conn, ["a", "b"], np.array([[0, 1]]), np.eye(2), max_edges=1
    )
    assert stats["created"] == 0
    assert stats["failed"] == 1
    assert stats["dream_pairs"] == []
    assert conn.execute("SELECT count(*) FROM derivation_edges").fetchone()[0] == 0
    conn.close()


def test_existing_complete_pair_is_not_reused_for_dream(tmp_path):
    conn = _edge_db(tmp_path)
    conn.executemany(
        "INSERT INTO derivation_edges VALUES (?, ?, .9, 'old')",
        [("a", "b"), ("b", "a")],
    )
    conn.commit()
    stats = _batch_cross_links(
        conn, ["a", "b"], np.array([[0, 1]]), np.eye(2), max_edges=1
    )
    assert stats["skipped"] == 1
    assert stats["dream_pairs"] == []
    conn.close()


def test_dream_pairs_stop_at_successful_pair_cap(tmp_path):
    conn = _edge_db(tmp_path)
    stats = _batch_cross_links(
        conn,
        ["a", "b", "c", "d"],
        np.array([[0, 1], [2, 3]]),
        np.eye(4),
        max_edges=1,
    )
    assert stats["capped"] is True
    assert stats["dream_pairs"] == [("a", "b", 0.0)]
    assert conn.execute("SELECT count(*) FROM derivation_edges").fetchone()[0] == 2
    conn.close()


@pytest.mark.parametrize("pair_count", [500, 501])
def test_cross_link_cap_batches_at_five_hundred_pairs(tmp_path, pair_count):
    ids = [f"n{i}" for i in range(pair_count * 2)]
    pairs = np.asarray([[2 * i, 2 * i + 1] for i in range(pair_count)])
    sim = np.eye(len(ids), dtype=np.float32)
    conn = _edge_db(tmp_path, name=f"edges-{pair_count}.db", factory=_CountingConnection)
    baseline_commit_count = conn.commit_count
    stats = _batch_cross_links(conn, ids, pairs, sim, max_edges=pair_count)
    assert stats["created"] == pair_count
    assert stats["directed_rows"] == pair_count * 2
    assert len(stats["dream_pairs"]) == pair_count
    assert conn.execute("SELECT count(*) FROM derivation_edges").fetchone()[0] == pair_count * 2
    assert conn.commit_count - baseline_commit_count == (pair_count + 499) // 500
    conn.close()


def test_run_sleep_cycle_preserves_six_positional_arguments():
    signature = inspect.signature(run_sleep_cycle)
    signature.bind("/tmp/example.db", None, None, False, 1, True)
    assert signature.parameters["orphan_limit"].kind is inspect.Parameter.KEYWORD_ONLY
    assert (
        signature.parameters["orphan_batch_size"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"orphan_limit": -1}, "invalid_orphan_limit"),
        ({"orphan_limit": True}, "invalid_orphan_limit"),
        ({"orphan_batch_size": 0}, "invalid_orphan_batch_size"),
        ({"orphan_batch_size": 101}, "invalid_orphan_batch_size"),
    ],
)
def test_orphan_bounds_are_rejected_before_database_open(monkeypatch, kwargs, error):
    monkeypatch.setattr(
        sqlite3,
        "connect",
        lambda *_args, **_kwargs: pytest.fail("database must not open"),
    )
    result = run_sleep_cycle(db_path="/tmp/does-not-exist.db", **kwargs)
    assert result["status"] == "rejected"
    assert result["error"] == error


def test_partial_embedding_triad_is_rejected_before_database_open(monkeypatch):
    def fail_connect(*_args, **_kwargs):
        raise AssertionError("database must not open during argument rejection")

    monkeypatch.setattr(sqlite3, "connect", fail_connect)
    result = run_sleep_cycle(db_path="/tmp/does-not-exist.db", embedding_model="m")
    assert result["status"] == "rejected"
    assert result["error"] == "invalid_embedding_contract"


def test_embedding_profile_dimension_is_rejected_before_database_open(monkeypatch):
    def fail_connect(*_args, **_kwargs):
        raise AssertionError("database must not open for profile mismatch")

    monkeypatch.setattr(sqlite3, "connect", fail_connect)
    result = run_sleep_cycle(
        db_path="/tmp/does-not-exist.db",
        embedding_client=_Client(lambda n: np.ones((n, 4), dtype=np.float32)),
        embedding_model="all-MiniLM-L6-v2",
        expected_dimension=4,
    )
    assert result["status"] == "rejected"
    assert result["error"] == "embedding_dimension_mismatch"


def test_active_profile_failure_without_explicit_model_is_pre_db(monkeypatch):
    monkeypatch.setattr(sleep_module, "_get_active_profile", lambda *_args: (_ for _ in ()).throw(RuntimeError("profile")))
    monkeypatch.setattr(sqlite3, "connect", lambda *_args, **_kwargs: pytest.fail("database must not open"))
    result = run_sleep_cycle(db_path="/tmp/does-not-exist.db")
    assert result["status"] in {"rejected", "unavailable"}
    assert result["error"] == "uncalibrated_embedding_model"


class _Client:
    def __init__(self, value):
        self.value = value
        self.calls = []

    def encode(self, texts):
        self.calls.append(texts)
        return self.value(len(texts)) if callable(self.value) else self.value


def _orphan_db(tmp_path: Path, vec: bool = True, dimension: int = 4):
    conn = sqlite3.connect(str(tmp_path / "orphans.db"))
    conn.execute(
        "CREATE TABLE thought_nodes("
        "id TEXT PRIMARY KEY, content TEXT, decayed INTEGER DEFAULT 0, "
        "timestamp TEXT DEFAULT '', source_file TEXT, access_count INTEGER DEFAULT 0, "
        "permanent INTEGER DEFAULT 0, node_type TEXT DEFAULT 'observation', "
        "last_accessed TEXT, domain TEXT)"
    )
    conn.execute(
        "CREATE TABLE embeddings("
        "node_id TEXT PRIMARY KEY, vector BLOB, model TEXT, updated_at TEXT)"
    )
    conn.execute(
        "CREATE TABLE derivation_edges("
        "parent_id TEXT, child_id TEXT, weight REAL, reasoning TEXT, "
        "PRIMARY KEY(parent_id, child_id))"
    )
    if vec:
        pytest.importorskip("sqlite_vec")
        from core.embeddings import _load_vec

        _load_vec(conn)
        conn.execute(
            "CREATE VIRTUAL TABLE vec_embeddings USING vec0("
            f"node_id text primary key, embedding float[{dimension}] "
            "distance_metric=cosine)"
        )
    conn.execute(
        "INSERT INTO thought_nodes (id, content, decayed) VALUES ('n1','orphan',0)"
    )
    conn.commit()
    return conn


def test_orphan_invalid_batch_writes_nothing(tmp_path):
    conn = _orphan_db(tmp_path)
    client = _Client(lambda n: np.array([[np.nan, 1, 1, 1]], dtype=np.float32))
    stats = {}
    assert (
        _embed_orphans(
            conn,
            embedding_client=client,
            embedding_model="m",
            expected_dimension=4,
            stats=stats,
        )
        == 0
    )
    assert conn.execute("SELECT count(*) FROM embeddings").fetchone()[0] == 0
    assert stats["orphan_write_failed"] == 1
    conn.close()


def test_orphan_batches_101_rows_with_supervisor_compatible_ceiling(
    tmp_path, monkeypatch
):
    conn = _orphan_db(tmp_path, dimension=384)
    conn.execute("UPDATE thought_nodes SET id='n000' WHERE id='n1'")
    conn.executemany(
        "INSERT INTO thought_nodes(id, content, timestamp) VALUES (?, ?, ?)",
        [(f"n{i:03d}", f"orphan {i}", "2026-01-01T00:00:00+00:00")
         for i in range(1, 101)],
    )
    conn.commit()

    class BoundedClient:
        def __init__(self):
            self.calls = []
            self.offset = 0

        def encode(self, texts):
            assert len(texts) <= 100
            self.calls.append(list(texts))
            result = np.zeros((len(texts), 384), dtype=np.float32)
            for row in range(len(texts)):
                result[row, self.offset + row] = 1.0
            self.offset += len(texts)
            return result

    client = BoundedClient()
    conn.close()
    _neutralize_post_pair_phases(monkeypatch)
    result = run_sleep_cycle(
        db_path=str(tmp_path / "orphans.db"),
        limit=2,
        embedding_client=client,
        embedding_model="all-MiniLM-L6-v2",
        expected_dimension=384,
        journal_policy="preserve",
        orphan_limit=101,
        orphan_batch_size=100,
    )
    assert [len(call) for call in client.calls] == [100, 1]
    assert client.calls[0][:2] == ["orphan", "orphan 1"]
    assert client.calls[1] == ["orphan 100"]
    assert result["orphans_embedded"] == 101
    assert result["orphan_write_failed"] == 0
    assert set(result) == set(_empty_sleep_result("completed", None))
    check = sqlite3.connect(str(tmp_path / "orphans.db"))
    from core.embeddings import _load_vec
    _load_vec(check)
    assert check.execute("SELECT count(*) FROM embeddings").fetchone()[0] == 101
    assert check.execute("SELECT count(*) FROM vec_embeddings").fetchone()[0] == 101
    check.close()


def test_orphan_encode_runs_before_write_transaction(tmp_path):
    conn = _orphan_db(tmp_path)
    conn.execute("CREATE TABLE writer_probe(value TEXT)")
    conn.commit()

    class ConcurrentWriterClient:
        def encode(self, texts):
            assert conn.in_transaction is False
            writer = sqlite3.connect(str(tmp_path / "orphans.db"), timeout=0.1)
            writer.execute("INSERT INTO writer_probe VALUES ('during encode')")
            writer.commit()
            writer.close()
            return np.ones((len(texts), 4), dtype=np.float32)

    assert _embed_orphans(
        conn,
        embedding_client=ConcurrentWriterClient(),
        embedding_model="m",
        expected_dimension=4,
        batch_size=1,
    ) == 1
    assert conn.execute("SELECT value FROM writer_probe").fetchone()[0] == "during encode"
    conn.close()


def test_orphan_batch_failure_preserves_committed_prefix_for_restart(
    tmp_path, monkeypatch
):
    conn = _orphan_db(tmp_path, dimension=384)
    conn.execute("UPDATE thought_nodes SET id='n000' WHERE id='n1'")
    conn.executemany(
        "INSERT INTO thought_nodes(id, content, timestamp) VALUES (?, ?, ?)",
        [(f"n{i:03d}", f"orphan {i}", f"{i:03d}") for i in range(1, 101)],
    )
    conn.commit()

    class InterruptedClient:
        def __init__(self):
            self.calls = 0

        def encode(self, texts):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("worker exited")
            return np.ones((len(texts), 384), dtype=np.float32)

    conn.close()
    empty = np.empty((0, 2), dtype=int)
    monkeypatch.setattr(
        sleep_module,
        "_find_pairs",
        lambda ids, _matrix, **_kwargs: (empty, empty, np.eye(len(ids))),
    )
    _neutralize_post_pair_phases(monkeypatch)
    result = run_sleep_cycle(
        db_path=str(tmp_path / "orphans.db"),
        limit=2,
        embedding_client=InterruptedClient(),
        embedding_model="all-MiniLM-L6-v2",
        expected_dimension=384,
        journal_policy="preserve",
        orphan_limit=101,
        orphan_batch_size=100,
    )
    assert result["status"] == "partial"
    assert result["error"] == "orphan_write_failed"
    assert result["orphans_embedded"] == 100
    assert result["orphan_write_failed"] == 1
    assert set(result) == set(_empty_sleep_result("partial", "orphan_write_failed"))

    conn = sqlite3.connect(str(tmp_path / "orphans.db"))
    from core.embeddings import _load_vec
    _load_vec(conn)
    assert conn.execute("SELECT count(*) FROM embeddings").fetchone()[0] == 100
    assert conn.execute("SELECT count(*) FROM vec_embeddings").fetchone()[0] == 100

    restart = _Client(lambda n: np.ones((n, 384), dtype=np.float32))
    assert _embed_orphans(
        conn,
        embedding_client=restart,
        embedding_model="all-MiniLM-L6-v2",
        expected_dimension=384,
        limit=1,
        batch_size=1,
    ) == 1
    assert conn.execute("SELECT count(*) FROM embeddings").fetchone()[0] == 101
    assert conn.execute("SELECT count(*) FROM vec_embeddings").fetchone()[0] == 101
    conn.close()


def test_orphan_limit_caps_ordinary_present_vec_missing_rows(tmp_path):
    conn = _orphan_db(tmp_path)
    vec = np.ones(4, dtype=np.float32).tobytes()
    conn.executemany(
        "INSERT INTO thought_nodes(id, content, timestamp) VALUES (?, ?, ?)",
        [("n2", "two", "2"), ("n3", "three", "3")],
    )
    conn.executemany(
        "INSERT INTO embeddings(node_id, vector, model) VALUES (?, ?, 'm')",
        [("n1", vec), ("n2", vec), ("n3", vec)],
    )
    conn.commit()

    class MustNotEncode:
        def encode(self, _texts):
            raise AssertionError("existing vectors must be reused")

    stats = {}
    assert _embed_orphans(
        conn,
        embedding_client=MustNotEncode(),
        embedding_model="m",
        expected_dimension=4,
        limit=2,
        batch_size=1,
        stats=stats,
    ) == 2
    assert stats["orphan_examined"] == 2
    assert [row[0] for row in conn.execute(
        "SELECT node_id FROM vec_embeddings ORDER BY node_id"
    )] == ["n1", "n2"]
    conn.close()


def test_capped_orphan_failures_advance_durable_cursor(tmp_path):
    conn = _orphan_db(tmp_path, vec=False)
    conn.execute("UPDATE thought_nodes SET id='one', content='one', timestamp='1'")
    conn.executemany(
        "INSERT INTO thought_nodes(id, content, timestamp) VALUES (?, ?, ?)",
        [("two", "two", "2"), ("three", "three", "3")],
    )
    conn.commit()

    class RejectEveryPage:
        def __init__(self):
            self.calls = []

        def encode(self, texts):
            self.calls.append(list(texts))
            raise RuntimeError("synthetic worker rejection")

    client = RejectEveryPage()
    for _ in range(3):
        stats = {}
        assert _embed_orphans(
            conn,
            embedding_client=client,
            embedding_model="m",
            expected_dimension=4,
            limit=1,
            batch_size=1,
            stats=stats,
        ) == 0
        assert stats["orphan_examined"] == 1
        assert stats["orphan_write_failed"] == 1
    assert client.calls == [["one"], ["two"], ["three"]]
    check = sqlite3.connect(str(tmp_path / "orphans.db"))
    assert check.execute(
        "SELECT epoch FROM _cashew_sleep_state "
        "WHERE name='orphan_missing_cursor_v1'"
    ).fetchone() == (3,)
    check.close()
    conn.close()


def test_capped_orphan_phase_rotation_prevents_repair_starvation(tmp_path):
    conn = _orphan_db(tmp_path)
    vector = np.ones(4, dtype=np.float32).tobytes()
    conn.execute("UPDATE thought_nodes SET id='missing', content='missing'")
    conn.execute(
        "INSERT INTO thought_nodes(id, content, timestamp) VALUES ('repair','repair','2')"
    )
    conn.execute(
        "INSERT INTO embeddings VALUES ('repair', ?, 'm', datetime('now'))",
        (vector,),
    )
    conn.commit()

    class RejectOrdinary:
        def __init__(self):
            self.calls = []

        def encode(self, texts):
            self.calls.append(list(texts))
            raise RuntimeError("synthetic worker rejection")

    client = RejectOrdinary()
    first_stats = {}
    assert _embed_orphans(
        conn, embedding_client=client, embedding_model="m", expected_dimension=4,
        limit=1, batch_size=1, stats=first_stats,
    ) == 0
    assert first_stats["orphan_write_failed"] == 1
    second_stats = {}
    assert _embed_orphans(
        conn, embedding_client=client, embedding_model="m", expected_dimension=4,
        limit=1, batch_size=1, stats=second_stats,
    ) == 1
    assert client.calls == [["missing"]]
    assert conn.execute(
        "SELECT count(*) FROM vec_embeddings WHERE node_id='repair'"
    ).fetchone() == (1,)
    conn.close()


def test_orphan_vec_failure_rolls_back_ordinary_row(tmp_path):
    conn = _orphan_db(tmp_path)
    conn.close()

    class BrokenVecWrite(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            if sql.startswith("INSERT OR REPLACE INTO vec_embeddings"):
                raise sqlite3.OperationalError("synthetic vec write failure")
            return super().execute(sql, *args, **kwargs)

    conn = sqlite3.connect(
        str(tmp_path / "orphans.db"), factory=BrokenVecWrite,
    )
    client = _Client(lambda n: np.ones((n, 4), dtype=np.float32))
    stats = {}
    assert (
        _embed_orphans(
            conn,
            embedding_client=client,
            embedding_model="m",
            expected_dimension=4,
            stats=stats,
        )
        == 0
    )
    assert conn.execute("SELECT count(*) FROM embeddings").fetchone()[0] == 0
    assert stats["orphan_write_failed"] == 1
    conn.close()


def test_orphan_repair_works_without_two_anchors(tmp_path):
    conn = _orphan_db(tmp_path)
    vec = np.ones(4, dtype=np.float32).tobytes()
    conn.execute(
        "INSERT INTO embeddings VALUES ('n1', ?, 'm', datetime('now'))", (vec,)
    )
    conn.commit()
    # Repair is deliberately skipped without the explicit client/model/dim
    # triad; sleep must not borrow a configured model by accident.
    stats = {}
    assert _embed_orphans(conn, expected_dimension=4, stats=stats) == 0
    assert stats["capability_missing"] is True
    assert conn.execute("SELECT count(*) FROM vec_embeddings").fetchone()[0] == 0
    conn.close()


def test_orphan_without_vec_is_explicit_ordinary_only(tmp_path):
    conn = _orphan_db(tmp_path, vec=False)
    client = _Client(lambda n: np.ones((n, 4), dtype=np.float32))
    stats = {}
    assert (
        _embed_orphans(
            conn,
            embedding_client=client,
            embedding_model="m",
            expected_dimension=4,
            stats=stats,
        )
        == 0
    )
    assert stats["orphan_vec_unavailable"] == 1
    conn.close()


def test_malformed_existing_embedding_is_contained(tmp_path):
    conn = _orphan_db(tmp_path)
    conn.execute(
        "INSERT INTO embeddings VALUES ('n1', ?, 'm', datetime('now'))", (b"bad",)
    )
    conn.commit()
    stats = {}
    assert _embed_orphans(
        conn, embedding_client=_Client(lambda n: np.ones((n, 4), dtype=np.float32)),
        embedding_model="m", expected_dimension=4, stats=stats,
    ) == 0
    assert stats["orphan_write_failed"] == 1
    conn.close()


def test_real_sqlite_vec0_dual_write_when_extension_is_available(tmp_path):
    pytest.importorskip("sqlite_vec")

    conn = _orphan_db(tmp_path)
    assert _vec_write_capability(conn) is True
    conn.close()


def _real_vec_db(tmp_path: Path, name: str):
    pytest.importorskip("sqlite_vec")
    conn = _orphan_db(tmp_path)
    conn.close()
    return sqlite3.connect(str(tmp_path / "orphans.db"))


def test_vec_enable_failure_is_unavailable(tmp_path):
    conn = _real_vec_db(tmp_path, "enable-failure")

    class Disabled(sqlite3.Connection):
        def enable_load_extension(self, _enabled):
            raise sqlite3.OperationalError("extension disabled")

    conn.close()
    disabled = sqlite3.connect(str(tmp_path / "orphans.db"), factory=Disabled)
    assert _vec_write_capability(disabled) is False
    disabled.close()


def test_plain_table_named_vec_embeddings_is_not_vec_capability(tmp_path):
    conn = _orphan_db(tmp_path, vec=False)
    conn.execute(
        "CREATE TABLE vec_embeddings(node_id TEXT PRIMARY KEY, embedding BLOB)"
    )
    conn.commit()
    assert _vec_write_capability(conn) is False
    stats = {}
    assert _embed_orphans(
        conn,
        embedding_client=_Client(
            lambda n: np.ones((n, 4), dtype=np.float32)
        ),
        embedding_model="m",
        expected_dimension=4,
        stats=stats,
    ) == 0
    assert stats["orphan_vec_unavailable"] == 1
    assert conn.execute("SELECT count(*) FROM embeddings").fetchone() == (1,)
    assert conn.execute("SELECT count(*) FROM vec_embeddings").fetchone() == (0,)
    conn.close()


def test_vec_post_load_query_failure_is_hard(tmp_path):
    _real_vec_db(tmp_path, "query-failure").close()

    class BrokenQuery(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            if sql == "SELECT count(*) FROM vec_embeddings":
                raise sqlite3.OperationalError("corrupt vec schema")
            return super().execute(sql, *args, **kwargs)

    conn = sqlite3.connect(str(tmp_path / "orphans.db"), factory=BrokenQuery)
    with pytest.raises(sqlite3.OperationalError, match="corrupt vec schema"):
        _vec_write_capability(conn)
    conn.close()
    conn = sqlite3.connect(str(tmp_path / "orphans.db"))
    client = _Client(lambda n: np.ones((n, 4), dtype=np.float32))
    assert (
        _embed_orphans(
            conn,
            embedding_client=client,
            embedding_model="m",
            expected_dimension=4,
        )
        == 1
    )
    assert conn.execute("SELECT count(*) FROM vec_embeddings").fetchone()[0] == 1
    conn.close()


def test_public_cycle_reports_committed_prefix_when_later_phase_fails(tmp_path, monkeypatch):
    path = _cycle_db(tmp_path)
    monkeypatch.setattr(
        sleep_module, "_find_pairs",
        lambda *_args, **_kwargs: (
            np.asarray([[0, 1]]), np.empty((0, 2), dtype=int), np.eye(2)
        ),
    )
    monkeypatch.setattr(sleep_module, "_compute_metrics", lambda *_args: (_ for _ in ()).throw(RuntimeError("later")))
    result = run_sleep_cycle(db_path=str(path), journal_policy="preserve")
    assert result["status"] == "partial"
    assert result["cross_links_created"] == 1
    check = sqlite3.connect(str(path))
    assert check.execute("SELECT count(*) FROM derivation_edges").fetchone()[0] == 2
    check.close()


def test_early_orphan_commit_survives_candidate_discovery_failure(tmp_path, monkeypatch):
    conn = _orphan_db(tmp_path, dimension=384)
    vector = np.ones(384, dtype=np.float32).tobytes()
    conn.execute(
        "INSERT INTO embeddings VALUES ('anchor', ?, 'all-MiniLM-L6-v2', datetime('now'))",
        (vector,),
    )
    conn.execute("INSERT INTO thought_nodes(id, content) VALUES ('anchor', 'anchor')")
    conn.commit()
    conn.close()
    monkeypatch.setattr(
        sleep_module, "_find_pairs",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("discovery")),
    )
    result = run_sleep_cycle(
        db_path=str(tmp_path / "orphans.db"),
        embedding_client=_Client(lambda n: np.ones((n, 384), dtype=np.float32)),
        embedding_model="all-MiniLM-L6-v2",
        expected_dimension=384,
        journal_policy="preserve",
    )
    assert result["status"] == "partial"
    assert result["error"] == "sleep_cycle_failed"
    assert result["orphans_embedded"] == 2
    assert set(result) == set(_empty_sleep_result("partial", "sleep_cycle_failed"))
    check = sqlite3.connect(str(tmp_path / "orphans.db"))
    assert check.execute("SELECT count(*) FROM embeddings").fetchone()[0] == 2
    check.close()


def test_late_orphan_failure_retains_committed_dream_fields(tmp_path, monkeypatch):
    path = _cycle_db(tmp_path, "dream-orphan-failure.db")
    conn = sqlite3.connect(str(path))
    v1 = np.zeros(1024, dtype=np.float32)
    v1[0] = 1.0
    v2 = np.zeros(1024, dtype=np.float32)
    v2[0] = 0.92
    v2[1] = np.sqrt(1.0 - 0.92 ** 2)
    conn.execute("UPDATE thought_nodes SET source_file='one.md' WHERE id='a'")
    conn.execute("UPDATE thought_nodes SET source_file='two.md' WHERE id='b'")
    conn.execute("INSERT INTO thought_nodes(id, content) VALUES ('orphan', 'late orphan')")
    conn.execute("UPDATE embeddings SET vector=? WHERE node_id='a'", (v1.tobytes(),))
    conn.execute("UPDATE embeddings SET vector=? WHERE node_id='b'", (v2.tobytes(),))
    conn.commit()
    conn.close()
    monkeypatch.setattr(
        sleep_module, "_embed_orphans",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("orphan")),
    )
    result = run_sleep_cycle(
        db_path=str(path),
        model_fn=lambda _prompt: "A durable relationship connects these observations.",
        journal_policy="preserve",
    )
    assert result["status"] == "partial"
    assert result["dream_generation"] == "ran"
    assert result["dream_id"]


def _force_late_dream_failure(monkeypatch):
    monkeypatch.setattr(
        sleep_module, "_find_pairs",
        lambda *_args, **_kwargs: (
            np.asarray([[0, 1]]), np.empty((0, 2), dtype=int), np.eye(2)
        ),
    )
    monkeypatch.setattr(
        sleep_module, "_generate_dream",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("late")),
    )


def test_late_failure_preserves_permanence_promotion(tmp_path, monkeypatch):
    path = _cycle_db(tmp_path, "permanence-late.db")

    def promote(conn, **_kwargs):
        conn.execute("UPDATE thought_nodes SET permanent=1 WHERE id='a'")
        conn.commit()
        return {"nodes_promoted": 1}

    monkeypatch.setattr(sleep_module, "_evaluate_permanence", promote)
    monkeypatch.setattr(sleep_module, "_promote_core_memories", lambda *_args: {"promoted": 0, "demoted": 0})
    _force_late_dream_failure(monkeypatch)
    result = run_sleep_cycle(db_path=str(path), model_fn=lambda _prompt: "unused", journal_policy="preserve")
    assert result["status"] == "partial"
    assert result["nodes_made_permanent"] == 1


def test_late_failure_preserves_core_promotion(tmp_path, monkeypatch):
    path = _cycle_db(tmp_path, "core-promotion-late.db")

    def promote_core(conn, _metrics):
        conn.execute("UPDATE thought_nodes SET node_type='core_memory' WHERE id='a'")
        conn.commit()
        return {"promoted": 1, "demoted": 0}

    monkeypatch.setattr(sleep_module, "_evaluate_permanence", lambda *_args: {"nodes_promoted": 0})
    monkeypatch.setattr(sleep_module, "_promote_core_memories", promote_core)
    _force_late_dream_failure(monkeypatch)
    result = run_sleep_cycle(db_path=str(path), model_fn=lambda _prompt: "unused", journal_policy="preserve")
    assert result["status"] == "partial"
    assert result["core_promoted"] == 1


def test_late_failure_preserves_core_demotion(tmp_path, monkeypatch):
    path = _cycle_db(tmp_path, "core-demotion-late.db")
    conn = sqlite3.connect(str(path))
    conn.execute("UPDATE thought_nodes SET node_type='core_memory' WHERE id='a'")
    conn.commit()
    conn.close()

    def demote_core(conn, _metrics):
        conn.execute("UPDATE thought_nodes SET node_type='derived' WHERE id='a'")
        conn.commit()
        return {"promoted": 0, "demoted": 1}

    monkeypatch.setattr(sleep_module, "_evaluate_permanence", lambda *_args: {"nodes_promoted": 0})
    monkeypatch.setattr(sleep_module, "_promote_core_memories", demote_core)
    _force_late_dream_failure(monkeypatch)
    result = run_sleep_cycle(db_path=str(path), model_fn=lambda _prompt: "unused", journal_policy="preserve")
    assert result["status"] == "partial"
    assert result["core_demoted"] == 1


def test_public_cycle_exposes_cross_link_failure(tmp_path, monkeypatch):
    path = _cycle_db(tmp_path)
    monkeypatch.setattr(
        sleep_module, "_find_pairs",
        lambda *_args, **_kwargs: (
            np.asarray([[0, 1]]), np.empty((0, 2), dtype=int), np.eye(2)
        ),
    )
    monkeypatch.setattr(
        sleep_module, "_batch_cross_links",
        lambda *_args, **_kwargs: {"created": 0, "repaired": 0, "skipped": 0,
                                   "failed": 1, "directed_rows": 0},
    )
    result = run_sleep_cycle(db_path=str(path), journal_policy="preserve")
    assert result["status"] in {"failed", "partial"}
    assert result["error"] == "cross_link_failed"


def test_max_edges_zero_never_dreams_from_uncommitted_candidates(tmp_path, monkeypatch):
    path = _cycle_db(tmp_path, "zero-dream.db")
    conn = sqlite3.connect(str(path))
    first = np.zeros(1024, dtype=np.float32)
    first[0] = 1.0
    second = np.zeros(1024, dtype=np.float32)
    second[0] = 0.92
    second[1] = np.sqrt(1.0 - 0.92 ** 2)
    conn.execute("UPDATE thought_nodes SET source_file='one.md' WHERE id='a'")
    conn.execute("UPDATE thought_nodes SET source_file='two.md' WHERE id='b'")
    conn.execute("UPDATE embeddings SET vector=? WHERE node_id='a'", (first.tobytes(),))
    conn.execute("UPDATE embeddings SET vector=? WHERE node_id='b'", (second.tobytes(),))
    conn.commit()
    conn.close()
    _neutralize_post_pair_phases(monkeypatch)
    prompts = []
    result = run_sleep_cycle(
        db_path=str(path), model_fn=prompts.append, max_edges=0,
        journal_policy="preserve",
    )
    assert result["cross_link_candidates"] == 1
    assert result["cross_link_capped"] is True
    assert result["cross_links_created"] == 0
    assert result["dream_generation"] == "skipped"
    assert result["status"] == "completed"
    assert prompts == []
    check = sqlite3.connect(str(path))
    assert check.execute("SELECT count(*) FROM derivation_edges").fetchone()[0] == 0
    assert check.execute("SELECT count(*) FROM thought_nodes WHERE node_type='dream'").fetchone()[0] == 0
    check.close()


def test_sleep_state_partial_schema_migrates_and_keeps_cursor(tmp_path, monkeypatch):
    path = _cycle_db(tmp_path, "partial-state.db")
    _add_four_fairness_nodes(path)
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE _cashew_sleep_state("
        "name TEXT, cursor_timestamp TEXT, cursor_node_id TEXT)"
    )
    conn.execute(
        "INSERT INTO _cashew_sleep_state VALUES (?, ?, ?)",
        ("candidate_cursor_v1", "2026-01-02T00:00:00+00:00", "b"),
    )
    conn.commit()
    conn.close()
    selected = []

    def capture(ids, _matrix, **_kwargs):
        selected.append(list(ids))
        empty = np.empty((0, 2), dtype=int)
        return empty, empty, np.eye(len(ids))

    monkeypatch.setattr(sleep_module, "_find_pairs", capture)
    _neutralize_post_pair_phases(monkeypatch)
    result = run_sleep_cycle(db_path=str(path), limit=2, journal_policy="preserve")
    assert result["status"] == "completed"
    assert selected == [["c", "d"]]
    check = sqlite3.connect(str(path))
    columns = [row[1] for row in check.execute("PRAGMA table_info(_cashew_sleep_state)")]
    assert columns == ["name", "cursor_timestamp", "cursor_node_id", "epoch"]
    assert check.execute(
        "SELECT epoch FROM _cashew_sleep_state WHERE name='candidate_cursor_v1'"
    ).fetchone() == (1,)
    check.close()


def test_sleep_state_malformed_schema_rebuilds_idempotently(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "state.db"))
    conn.execute("CREATE TABLE _cashew_sleep_state(unrelated TEXT)")
    conn.execute("BEGIN IMMEDIATE")
    _ensure_sleep_state_schema(conn)
    _ensure_sleep_state_schema(conn)
    conn.commit()
    shape = [
        (row[1], (row[2] or "").upper(), row[3], row[5])
        for row in conn.execute("PRAGMA table_info(_cashew_sleep_state)")
    ]
    assert shape == [
        ("name", "TEXT", 0, 1),
        ("cursor_timestamp", "TEXT", 1, 0),
        ("cursor_node_id", "TEXT", 1, 0),
        ("epoch", "INTEGER", 1, 0),
    ]
    conn.close()


@pytest.mark.parametrize(
    "rows",
    [
        [
            ("candidate_cursor_v1", "2026-01-01", "a", 1),
            ("candidate_cursor_v1", "2026-01-02", "b", 2),
        ],
        [("candidate_cursor_v1", b"bad", "a", 1)],
        [("candidate_cursor_v1", "2026-01-01", "a", -1)],
    ],
)
def test_sleep_state_ambiguous_or_invalid_cursor_resets_to_origin(tmp_path, rows):
    conn = sqlite3.connect(str(tmp_path / "state-reset.db"))
    conn.execute(
        "CREATE TABLE _cashew_sleep_state("
        "name TEXT, cursor_timestamp TEXT, cursor_node_id TEXT, epoch)"
    )
    conn.executemany("INSERT INTO _cashew_sleep_state VALUES (?, ?, ?, ?)", rows)
    conn.commit()
    conn.execute("BEGIN IMMEDIATE")
    _ensure_sleep_state_schema(conn)
    conn.commit()
    assert conn.execute(
        "SELECT * FROM _cashew_sleep_state WHERE name='candidate_cursor_v1'"
    ).fetchall() == []
    conn.close()


def test_sleep_state_exact_schema_drops_type_invalid_cursor(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "state-invalid.db"))
    conn.execute(
        "CREATE TABLE _cashew_sleep_state("
        "name TEXT PRIMARY KEY, cursor_timestamp TEXT NOT NULL, "
        "cursor_node_id TEXT NOT NULL, epoch INTEGER NOT NULL)"
    )
    conn.execute(
        "INSERT INTO _cashew_sleep_state VALUES (?, ?, ?, ?)",
        ("candidate_cursor_v1", sqlite3.Binary(b"bad"), "a", 1),
    )
    conn.commit()
    conn.execute("BEGIN IMMEDIATE")
    _ensure_sleep_state_schema(conn)
    conn.commit()
    assert conn.execute("SELECT * FROM _cashew_sleep_state").fetchall() == []
    conn.close()


def test_sleep_cursor_migration_is_atomic_under_writer_contention(tmp_path):
    path = _cycle_db(tmp_path, "cursor-contention.db")
    _add_four_fairness_nodes(path)
    owner = sqlite3.connect(str(path))
    contender = sqlite3.connect(str(path), timeout=0.0)
    owner.execute("BEGIN IMMEDIATE")
    with pytest.raises(sqlite3.OperationalError, match="locked"):
        _select_cycle_node_ids(contender, 2)
    assert owner.execute(
        "SELECT name FROM sqlite_master WHERE name='_cashew_sleep_state'"
    ).fetchone() is None
    owner.rollback()
    assert _select_cycle_node_ids(contender, 2) == ["a", "b"]
    assert contender.execute(
        "SELECT name, epoch FROM _cashew_sleep_state"
    ).fetchall() == [("candidate_cursor_v1", 1)]
    contender.close()
    owner.close()


def test_synchronous_dream_reports_ran(tmp_path, monkeypatch):
    path = _cycle_db(tmp_path, "dream.db")
    conn = sqlite3.connect(str(path))
    v1 = np.zeros(1024, dtype=np.float32)
    v1[0] = 1.0
    v2 = np.zeros(1024, dtype=np.float32)
    v2[0] = 0.92
    v2[1] = np.sqrt(1.0 - 0.92 ** 2)
    conn.execute("UPDATE thought_nodes SET source_file='one.md' WHERE id='a'")
    conn.execute("UPDATE thought_nodes SET source_file='two.md' WHERE id='b'")
    conn.execute("UPDATE embeddings SET vector=? WHERE node_id='a'", (v1.tobytes(),))
    conn.execute("UPDATE embeddings SET vector=? WHERE node_id='b'", (v2.tobytes(),))
    conn.commit()
    conn.close()
    result = run_sleep_cycle(
        db_path=str(path),
        model_fn=lambda _prompt: "A durable relationship connects these two observations.",
        journal_policy="preserve",
    )
    assert result["cross_links_created"] == 1
    assert result["dream_generation"] == "ran"
    assert result["dream_id"]
    check = sqlite3.connect(str(path))
    assert check.execute(
        "SELECT count(*) FROM thought_nodes WHERE id=?", (result["dream_id"],)
    ).fetchone()[0] == 1
    assert check.execute(
        "SELECT count(*) FROM derivation_edges WHERE child_id=?", (result["dream_id"],)
    ).fetchone()[0] == 2
    check.close()


def test_repaired_half_pair_can_drive_dream(tmp_path, monkeypatch):
    path = _cycle_db(tmp_path, "repair-dream.db")
    conn = sqlite3.connect(str(path))
    first = np.zeros(1024, dtype=np.float32)
    first[0] = 1.0
    second = np.zeros(1024, dtype=np.float32)
    second[0] = 0.92
    second[1] = np.sqrt(1.0 - 0.92 ** 2)
    conn.execute("UPDATE thought_nodes SET source_file='one.md' WHERE id='a'")
    conn.execute("UPDATE thought_nodes SET source_file='two.md' WHERE id='b'")
    conn.execute("UPDATE embeddings SET vector=? WHERE node_id='a'", (first.tobytes(),))
    conn.execute("UPDATE embeddings SET vector=? WHERE node_id='b'", (second.tobytes(),))
    conn.execute(
        "INSERT INTO derivation_edges VALUES "
        "('a', 'b', .92, 'existing half')"
    )
    conn.commit()
    conn.close()
    result = run_sleep_cycle(
        db_path=str(path),
        model_fn=lambda _prompt: "A durable relationship connects these observations.",
        journal_policy="preserve",
    )
    assert result["cross_links_created"] == 0
    assert result["cross_links_repaired"] == 1
    assert result["dream_generation"] == "ran"
    assert result["dream_id"]


def test_public_cycle_repairs_orphan_before_anchor_requirement(tmp_path):
    conn = _orphan_db(tmp_path, dimension=384)
    dimension = 384
    anchor = np.ones(dimension, dtype=np.float32).tobytes()
    conn.execute(
        "INSERT INTO embeddings VALUES "
        "('anchor', ?, 'all-MiniLM-L6-v2', datetime('now'))",
        (anchor,),
    )
    conn.execute(
        "INSERT INTO thought_nodes (id, content, decayed) VALUES ('anchor','anchor',0)"
    )
    conn.commit()
    conn.close()
    client = _Client(lambda n: np.ones((n, dimension), dtype=np.float32))
    result = run_sleep_cycle(
        db_path=str(tmp_path / "orphans.db"),
        embedding_client=client,
        embedding_model="all-MiniLM-L6-v2",
        expected_dimension=dimension,
        journal_policy="preserve",
    )
    assert client.calls == [["orphan"]]
    assert result["orphans_embedded"] == 2
    assert result["nodes_selected"] == 2
    assert result["dedup_nodes_merged"] == 1
    check = sqlite3.connect(str(tmp_path / "orphans.db"))
    from core.embeddings import _load_vec
    _load_vec(check)
    assert check.execute("SELECT count(*) FROM vec_embeddings").fetchone()[0] == 1
    check.close()


def test_public_cycle_reports_ordinary_only_orphan_write(tmp_path):
    conn = _orphan_db(tmp_path, vec=False)
    conn.close()
    client = _Client(lambda n: np.ones((n, 384), dtype=np.float32))
    result = run_sleep_cycle(
        db_path=str(tmp_path / "orphans.db"),
        embedding_client=client,
        embedding_model="all-MiniLM-L6-v2",
        expected_dimension=384,
        journal_policy="preserve",
    )
    assert result["status"] == "partial"
    assert result["error"] == "vec_capability_unavailable"
    assert result["orphan_vec_unavailable"] == 1
    check = sqlite3.connect(str(tmp_path / "orphans.db"))
    assert check.execute("SELECT count(*) FROM embeddings").fetchone()[0] == 1
    check.close()


def test_decay_audit_schema_is_idempotent_and_preserves_graph(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "audit.db"))
    conn.execute(
        "CREATE TABLE thought_nodes("
        "id TEXT PRIMARY KEY, content TEXT, decayed INTEGER DEFAULT 0, "
        "timestamp TEXT DEFAULT '')"
    )
    conn.execute("INSERT INTO thought_nodes (id, content) VALUES ('n','keep')")
    ensure_decay_audit_schema(conn)
    ensure_decay_audit_schema(conn)
    assert conn.execute("SELECT content FROM thought_nodes").fetchone()[0] == "keep"
    assert conn.execute(
        "SELECT name FROM sqlite_master WHERE name='decay_audit'"
    ).fetchone()
    conn.close()


def test_decay_audit_partial_schema_migrates_and_keeps_history(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "partial-audit.db"))
    conn.execute("CREATE TABLE decay_audit(node_id TEXT, decay_reason TEXT)")
    conn.execute("INSERT INTO decay_audit VALUES ('old', 'dedup_loser')")
    conn.commit()
    ensure_decay_audit_schema(conn)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(decay_audit)")}
    assert {"id", "content_summary", "decay_timestamp", "metadata"} <= columns
    assert conn.execute("SELECT node_id FROM decay_audit").fetchone()[0] == "old"
    conn.close()


def test_preserve_journal_policy_does_not_change_mode(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "empty.db"))
    conn.execute(
        "CREATE TABLE thought_nodes("
        "id TEXT PRIMARY KEY, content TEXT, decayed INTEGER DEFAULT 0, "
        "timestamp TEXT DEFAULT '')"
    )
    conn.commit()
    conn.close()
    result = run_sleep_cycle(
        db_path=str(tmp_path / "empty.db"), journal_policy="preserve"
    )
    assert result["status"] == "unavailable"
    check = sqlite3.connect(str(tmp_path / "empty.db"))
    assert check.execute("PRAGMA journal_mode").fetchone()[0].lower() == "delete"
    check.close()


def test_result_contract_no_llm_has_skipped_dream(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "noemb.db"))
    conn.execute(
        "CREATE TABLE thought_nodes("
        "id TEXT PRIMARY KEY, content TEXT, decayed INTEGER DEFAULT 0, "
        "timestamp TEXT DEFAULT '')"
    )
    conn.execute(
        "CREATE TABLE embeddings("
        "node_id TEXT PRIMARY KEY, vector BLOB, model TEXT, updated_at TEXT)"
    )
    conn.commit()
    conn.close()
    result = run_sleep_cycle(
        db_path=str(tmp_path / "noemb.db"), journal_policy="preserve"
    )
    assert result["status"] == "unavailable"
    assert result["error"] == "too_few_embeddings"


@pytest.mark.parametrize('mode', ['default', 'injected', 'disabled', 'failure'])
def test_public_orphan_encoder_policy(tmp_path, monkeypatch, mode):
    import core.config as config_module
    from core.embedding_service import LocalBackend

    conn = _orphan_db(tmp_path, dimension=384)
    conn.close()
    monkeypatch.setattr(config_module, 'get_embedding_model', lambda: 'all-MiniLM-L6-v2')
    calls = []

    def encode(self, texts):
        calls.append((self.model_name, texts))
        if mode == 'failure':
            raise RuntimeError('encoder unavailable')
        return np.ones((len(texts), 384), dtype=np.float32)

    monkeypatch.setattr(LocalBackend, 'encode', encode)
    kwargs = {}
    client = _Client(lambda n: np.ones((n, 384), dtype=np.float32))
    if mode == 'injected':
        kwargs = dict(embedding_client=client, embedding_model='all-MiniLM-L6-v2',
                      expected_dimension=384)
    elif mode == 'disabled':
        kwargs = dict(auto_embed=False)
    result = run_sleep_cycle(str(tmp_path / 'orphans.db'), journal_policy='preserve', **kwargs)
    check = sqlite3.connect(str(tmp_path / 'orphans.db'))
    rows = check.execute('SELECT model FROM embeddings').fetchall()
    check.close()
    if mode in {'default', 'injected'}:
        assert rows == [('all-MiniLM-L6-v2',)]
        assert result['orphans_embedded'] == 1
    else:
        assert rows == []
    assert bool(calls) == (mode in {'default', 'failure'})
    if mode == 'failure':
        assert result['error'] == 'orphan_write_failed'


@pytest.mark.parametrize('repair', [False, True])
@pytest.mark.parametrize('value,valid', [(1e-9, True), (0.0, False), (float('nan'), False)])
def test_orphan_tiny_vectors_and_exact_zero(tmp_path, repair, value, valid):
    conn = _orphan_db(tmp_path)
    vector = np.full(4, value, dtype=np.float32)
    if repair:
        conn.execute("INSERT INTO embeddings VALUES ('n1', ?, 'm', datetime('now'))",
                     (vector.tobytes(),))
        conn.commit()
    stats = {}
    count = _embed_orphans(conn, embedding_client=_Client(lambda n: np.tile(vector, (n, 1))),
                           embedding_model='m', expected_dimension=4, stats=stats)
    assert count == int(valid)
    assert stats['orphan_write_failed'] == int(not valid)
    conn.close()


def test_verified_cross_link_commit_has_no_failed_count(tmp_path):
    class PostCommitError(sqlite3.Connection):
        fail = False

        def commit(self):
            super().commit()
            if self.fail:
                raise sqlite3.OperationalError('raised after commit')
    conn = _edge_db(tmp_path, factory=PostCommitError)
    conn.fail = True
    stats = _batch_cross_links(conn, ['a', 'b'], np.array([[0, 1]]), np.eye(2))
    assert stats['created'] == 1
    assert stats['failed'] == 0
    assert len(stats['dream_pairs']) == 1
    conn.close()


def test_default_encoder_is_lazy_without_orphans(tmp_path, monkeypatch):
    from core.embedding_service import LocalBackend
    monkeypatch.setattr(LocalBackend, '_ensure_model', lambda self: pytest.fail('model loaded'))
    _neutralize_post_pair_phases(monkeypatch)
    result = run_sleep_cycle(str(_cycle_db(tmp_path)), journal_policy='preserve')
    assert result['status'] == 'completed'


def test_invalid_auto_embed_rejected_before_open(monkeypatch):
    monkeypatch.setattr(sqlite3, 'connect', lambda *a, **kw: pytest.fail('database opened'))
    assert run_sleep_cycle('unused', auto_embed='false')['error'] == 'invalid_auto_embed'


def test_cli_default_cycle_embeds_orphans(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace
    import core.config as config_module
    from core.embedding_service import LocalBackend
    from scripts import cashew_context

    conn = _orphan_db(tmp_path, dimension=384)
    conn.execute("INSERT INTO thought_nodes(id, content) VALUES ('n2', 'second')")
    conn.commit()
    conn.close()
    monkeypatch.setattr(config_module, 'get_embedding_model', lambda: 'all-MiniLM-L6-v2')
    monkeypatch.setattr(cashew_context, '_build_model_fn', lambda: None)
    monkeypatch.setattr(LocalBackend, 'encode', lambda self, texts: np.eye(len(texts), 384, dtype=np.float32))
    _neutralize_post_pair_phases(monkeypatch)
    cashew_context.cmd_complete_sleep(SimpleNamespace(db=str(tmp_path / 'orphans.db'), debug=True))
    output = capsys.readouterr().out
    assert '"status": "completed"' in output
    assert '"orphans_embedded": 2' in output


def test_injected_encoder_failure_never_falls_back(tmp_path, monkeypatch):
    from core.embedding_service import LocalBackend
    conn = _orphan_db(tmp_path, dimension=384)
    conn.close()
    monkeypatch.setattr(LocalBackend, '__init__', lambda *a, **kw: pytest.fail('fallback constructed'))
    def fail(_n):
        raise RuntimeError('worker unavailable')
    result = run_sleep_cycle(str(tmp_path / 'orphans.db'), embedding_client=_Client(fail),
                            embedding_model='all-MiniLM-L6-v2', expected_dimension=384)
    assert result['error'] == 'orphan_write_failed'


def test_repair_preserves_model_identity_per_row(tmp_path):
    conn = _orphan_db(tmp_path)
    vector = np.ones(4, dtype=np.float32).tobytes()
    conn.execute("INSERT INTO embeddings VALUES ('n1', ?, 'old-model', datetime('now'))", (vector,))
    conn.execute("INSERT INTO thought_nodes(id, content) VALUES ('n2', 'matching')")
    conn.execute("INSERT INTO embeddings VALUES ('n2', ?, 'new-model', datetime('now'))", (vector,))
    conn.commit()
    stats = {}
    count = _embed_orphans(conn, embedding_client=_Client(None), embedding_model='new-model',
                           expected_dimension=4, stats=stats)
    assert count == 1
    assert stats['orphan_write_failed'] == 1
    assert conn.execute('SELECT node_id FROM vec_embeddings').fetchall() == [('n2',)]
    assert conn.execute("SELECT model, vector FROM embeddings WHERE node_id='n1'").fetchone() == ('old-model', vector)
    conn.close()
