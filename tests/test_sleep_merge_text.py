"""Dedup keeps the newest value of an updated fact.

Without a model the newest node wins the keeper tiebreak. With one, the
keeper's content becomes a merged statement dated to the cluster's newest.
"""

import json
import sqlite3
from unittest.mock import patch

import numpy as np
import pytest

from core.decay_audit import ensure_decay_audit_schema
from core.sleep import MAX_LLM_MERGES_PER_CYCLE, _merge_cluster, _run_dedup, run_sleep_cycle

OLD = ("old", "User has added 17 new postcards to their collection", "2023-08-11T15:58:00")
NEW = ("new", "User has added 25 new postcards to their collection", "2023-11-30T03:48:00")
MERGED = "User has added 25 new postcards to their collection as of 2023-11-30 (17 as of 2023-08-11)"


def _db(tmp_path):
    path = tmp_path / "g.db"
    conn = sqlite3.connect(str(path))
    conn.executescript("""
        CREATE TABLE thought_nodes(id TEXT PRIMARY KEY, content TEXT NOT NULL,
            node_type TEXT NOT NULL DEFAULT 'fact', timestamp TEXT, referent_time TEXT,
            mood_state TEXT, metadata TEXT, source_file TEXT, domain TEXT DEFAULT 'user',
            decayed INTEGER DEFAULT 0, permanent INTEGER DEFAULT 0,
            access_count INTEGER DEFAULT 0, last_accessed TEXT, tags TEXT);
        CREATE TABLE derivation_edges(parent_id TEXT NOT NULL, child_id TEXT NOT NULL,
            weight REAL NOT NULL DEFAULT 1.0, reasoning TEXT,
            PRIMARY KEY (parent_id, child_id));
        CREATE TABLE embeddings(node_id TEXT PRIMARY KEY, vector BLOB NOT NULL,
            model TEXT NOT NULL, updated_at TEXT NOT NULL);
    """)
    ensure_decay_audit_schema(conn)
    for nid, content, ts in (OLD, NEW):
        conn.execute(
            "INSERT INTO thought_nodes(id, content, timestamp, referent_time) VALUES (?, ?, ?, ?)",
            (nid, content, ts, ts),
        )
    conn.executemany(
        "INSERT INTO thought_nodes(id, content, timestamp) VALUES (?, ?, '2023-01-01')",
        [("a", "User started collecting postcards again"), ("b", "User likes antique shops")],
    )
    conn.executemany(
        "INSERT INTO derivation_edges(parent_id, child_id) VALUES (?, ?)",
        [("a", "old"), ("new", "b")],
    )
    conn.commit()
    return path, conn


def _node(conn, nid):
    return conn.execute(
        "SELECT content, timestamp, referent_time, decayed FROM thought_nodes WHERE id=?", (nid,)
    ).fetchone()


def _edges(conn):
    return set(conn.execute("SELECT parent_id, child_id FROM derivation_edges").fetchall())


def test_keeper_tiebreak_prefers_newest(tmp_path):
    _, conn = _db(tmp_path)

    assert _merge_cluster(conn, ["old", "new"]) == "new"
    assert _node(conn, "new") == (NEW[1], NEW[2], NEW[2], 0)
    assert _node(conn, "old")[3] == 1


def test_model_fn_writes_merged_text_into_keeper(tmp_path):
    _, conn = _db(tmp_path)
    # Access makes the older node the keeper, so the merge must carry the new value over.
    conn.execute("UPDATE thought_nodes SET access_count=3 WHERE id='old'")
    prompts = []

    def model_fn(prompt):
        prompts.append(prompt)
        return f'"{MERGED}"\n'

    stats = {}
    assert _merge_cluster(conn, ["old", "new"], model_fn=model_fn, stats=stats) == "old"

    assert stats == {"llm_merge_calls": 1, "llm_merges": 1}
    assert _node(conn, "old") == (MERGED, NEW[2], NEW[2], 0)
    assert _node(conn, "new")[3] == 1
    assert _edges(conn) == {("a", "old"), ("old", "b")}
    # Notes go oldest first with their dates so the model can tell which is current.
    notes = prompts[0][prompts[0].index("NOTE 1"):]
    assert notes.index("2023-08-11") < notes.index(OLD[1]) < notes.index("2023-11-30") < notes.index(NEW[1])
    meta = conn.execute("SELECT metadata FROM decay_audit WHERE node_id='new'").fetchone()[0]
    assert json.loads(meta) == {"merged_text": True}


@pytest.mark.parametrize("model_fn", [
    lambda p: (_ for _ in ()).throw(TimeoutError("model timed out")),
    lambda p: "",
    lambda p: None,
    lambda p: "## Merged\n- " + "x" * 400,
])
def test_failed_model_fn_falls_back_to_newest_keeper(tmp_path, model_fn):
    _, conn = _db(tmp_path)

    stats = {}
    assert _merge_cluster(conn, ["old", "new"], model_fn=model_fn, stats=stats) == "new"

    assert stats == {"llm_merge_calls": 1}
    assert _node(conn, "new") == (NEW[1], NEW[2], NEW[2], 0)
    assert _node(conn, "old")[3] == 1
    assert _edges(conn) == {("a", "new"), ("new", "b")}
    assert conn.execute("SELECT metadata FROM decay_audit WHERE node_id='old'").fetchone()[0] is None


def test_pinned_keeper_is_not_rewritten(tmp_path):
    _, conn = _db(tmp_path)
    conn.execute("UPDATE thought_nodes SET permanent=2 WHERE id='old'")
    calls = []

    assert _merge_cluster(conn, ["old", "new"], model_fn=lambda p: calls.append(p) or MERGED) == "old"

    assert calls == []
    assert _node(conn, "old") == (OLD[1], OLD[2], OLD[2], 0)
    assert _node(conn, "new")[3] == 1


def test_llm_merges_are_capped_per_cycle(tmp_path):
    _, conn = _db(tmp_path)
    n = MAX_LLM_MERGES_PER_CYCLE + 2
    ids = []
    for i in range(n):
        ids += [f"x{i}", f"y{i}"]
        conn.executemany(
            "INSERT INTO thought_nodes(id, content, timestamp) VALUES (?, ?, ?)",
            [(f"x{i}", f"Duplicate fact number {i}", "2024-01-01"),
             (f"y{i}", f"Duplicate fact number {i}.", "2024-02-01")],
        )
    pairs = np.array([[2 * i, 2 * i + 1] for i in range(n)])
    calls = []

    stats = _run_dedup(conn, ids, pairs, model_fn=lambda p: calls.append(p) or "Merged duplicate fact text")

    assert len(calls) == MAX_LLM_MERGES_PER_CYCLE
    assert stats["llm_merges"] == MAX_LLM_MERGES_PER_CYCLE
    assert stats["components"] == n


def test_sleep_cycle_merges_and_reembeds_keeper(tmp_path):
    path, conn = _db(tmp_path)
    vec = np.full(384, 384 ** -0.5, dtype=np.float32)
    for nid in ("old", "new"):
        conn.execute(
            "INSERT INTO embeddings VALUES (?, ?, 'all-MiniLM-L6-v2', '2023-01-01')",
            (nid, vec.tobytes()),
        )
    conn.commit()
    conn.close()
    fresh = np.full(384, 0.5, dtype=np.float32)

    with (
        patch("core.sleep._resolve_expected_dim", return_value=384),
        patch("core.config.get_embedding_model", return_value="all-MiniLM-L6-v2"),
        patch("core.embedding_service.LocalBackend.encode",
              side_effect=lambda texts: np.array([fresh for _ in texts])),
        patch("core.sleep.config") as cfg,
    ):
        cfg.gc_mode = "off"
        result = run_sleep_cycle(db_path=str(path), model_fn=lambda p: MERGED)

    assert result["dedup_llm_merges"] == 1
    conn = sqlite3.connect(str(path))
    assert _node(conn, "new")[0] == MERGED
    blob = conn.execute("SELECT vector FROM embeddings WHERE node_id='new'").fetchone()[0]
    assert np.array_equal(np.frombuffer(blob, dtype=np.float32), fresh)
