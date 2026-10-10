"""Sleep closes commitment nodes cited by ID in a Completed: node."""

import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.db import ensure_schema
from core.decay_audit import ensure_decay_audit_schema
from core.sleep import _close_completed_todos, close_completed_todos

TODO_A = "aaaaaaaaaaa1"
TODO_B = "bbbbbbbbbbb2"
FACT = "ccccccccccc3"
OLD_TODO = "ddddddddddd4"
PINNED = "eeeeeeeeeee5"
DONE_1 = "f00000000001"
DONE_2 = "f00000000002"


@pytest.fixture
def db(tmp_path):
    path = str(tmp_path / "graph.db")
    ensure_schema(path)
    conn = sqlite3.connect(path)
    ensure_decay_audit_schema(conn)
    conn.commit()
    conn.close()
    return path


def _add(path, nid, content, node_type="commitment", decayed=0, permanent=0):
    conn = sqlite3.connect(path)
    conn.execute(
        "INSERT INTO thought_nodes (id, content, node_type, timestamp, decayed, permanent) "
        "VALUES (?, ?, ?, '2026-01-01T00:00:00+00:00', ?, ?)",
        (nid, content, node_type, decayed, permanent),
    )
    conn.commit()
    conn.close()


def _decayed(path):
    conn = sqlite3.connect(path)
    rows = conn.execute("SELECT id FROM thought_nodes WHERE decayed = 1").fetchall()
    conn.close()
    return {r[0] for r in rows}


def _audit(path):
    conn = sqlite3.connect(path)
    rows = conn.execute(
        "SELECT node_id, decay_reason FROM decay_audit ORDER BY id"
    ).fetchall()
    conn.close()
    return rows


def test_decays_cited_commitment(db):
    _add(db, TODO_A, "TODO: renew passport")
    _add(db, DONE_1, f"Completed: renewed passport ({TODO_A})", "fact")
    assert close_completed_todos(db) == [(DONE_1, TODO_A)]
    assert _decayed(db) == {TODO_A}
    assert _audit(db) == [(TODO_A, "completed_todo")]


def test_ignores_cited_non_commitment(db):
    _add(db, FACT, "Passport office is on 5th street", "fact")
    _add(db, DONE_1, f"Completed: see {FACT}", "fact")
    assert close_completed_todos(db) == []
    assert _decayed(db) == set()


def test_ignores_already_decayed(db):
    _add(db, OLD_TODO, "TODO: old thing", decayed=1)
    _add(db, DONE_1, f"Completed: {OLD_TODO}", "fact")
    assert close_completed_todos(db) == []
    assert _audit(db) == []


def test_never_decays_permanent(db):
    _add(db, PINNED, "TODO: pinned commitment", permanent=1)
    _add(db, DONE_1, f"Completed: {PINNED}", "fact")
    assert close_completed_todos(db) == []
    assert _decayed(db) == set()


def test_never_decays_completed_node(db):
    _add(db, DONE_1, "Completed: first pass", "commitment")
    _add(db, DONE_2, f"Completed: follow-up to {DONE_1}", "commitment")
    assert close_completed_todos(db) == []
    assert _decayed(db) == set()


def test_multiple_ids_in_one_note(db):
    _add(db, TODO_A, "TODO: a")
    _add(db, TODO_B, "TODO: b")
    _add(db, FACT, "a fact", "fact")
    _add(db, DONE_1, f"Completed: {TODO_A}, {TODO_B} and {FACT}", "fact")
    assert close_completed_todos(db) == [(DONE_1, TODO_A), (DONE_1, TODO_B)]
    assert _decayed(db) == {TODO_A, TODO_B}


def test_unknown_ids_and_non_ids_ignored(db):
    _add(db, TODO_A, "TODO: a")
    # 999999999999 is not a node; 13- and 11-char hex runs are not IDs.
    _add(db, DONE_1, "Completed: 999999999999 aaaaaaaaaaa1b aaaaaaaaaaa", "fact")
    assert close_completed_todos(db) == []
    assert _decayed(db) == set()


def test_requires_completed_prefix(db):
    _add(db, TODO_A, "TODO: a")
    _add(db, DONE_1, f"Not completed yet: {TODO_A}", "fact")
    assert close_completed_todos(db) == []


def test_decayed_completion_note_is_ignored(db):
    _add(db, TODO_A, "TODO: a")
    _add(db, DONE_1, f"Completed: {TODO_A}", "fact", decayed=1)
    assert close_completed_todos(db) == []


def test_idempotent(db):
    _add(db, TODO_A, "TODO: a")
    _add(db, DONE_1, f"Completed: {TODO_A}", "fact")
    _add(db, DONE_2, f"Completed: also {TODO_A}", "fact")
    first = close_completed_todos(db)
    assert len(first) == 1 and first[0][1] == TODO_A
    assert close_completed_todos(db) == []
    assert len(_audit(db)) == 1


def test_dry_run_writes_nothing(db):
    _add(db, TODO_A, "TODO: a")
    _add(db, DONE_1, f"Completed: {TODO_A}", "fact")
    conn = sqlite3.connect(db)
    before = conn.total_changes
    pairs = _close_completed_todos(conn, dry_run=True)
    assert conn.total_changes == before
    conn.close()
    assert pairs == [(DONE_1, TODO_A)]
    assert _decayed(db) == set()
    assert _audit(db) == []


def test_id_in_later_sentence_not_closed(db):
    _add(db, TODO_A, "TODO: a")
    _add(db, TODO_B, "TODO: b")
    _add(db, DONE_1, f"Completed: closed {TODO_A}. Still tracked: {TODO_B}.", "fact")
    assert close_completed_todos(db) == [(DONE_1, TODO_A)]


def test_id_after_dash_separator_not_closed(db):
    _add(db, TODO_A, "TODO: a")
    _add(db, TODO_B, "TODO: b")
    _add(db, OLD_TODO, "TODO: d")
    _add(db, DONE_1, f"Completed: closed {TODO_A} (desc) \u2014 see {TODO_B}", "fact")
    _add(db, DONE_2, f"Completed: closed something - see {OLD_TODO}", "fact")
    assert close_completed_todos(db) == [(DONE_1, TODO_A)]


@pytest.mark.parametrize("hedge", [
    "attempted commitment", "Partially closed", "not closed:", "still open:",
    "still pending:", "blocked on", "NOT DONE:",
])
def test_hedged_leading_clause_closes_nothing(db, hedge):
    _add(db, TODO_A, "TODO: a")
    _add(db, DONE_1, f"Completed: {hedge} {TODO_A} (desc)", "fact")
    assert close_completed_todos(db) == []


def test_id_list_in_leading_parenthetical_closed(db):
    _add(db, TODO_A, "TODO: a")
    _add(db, TODO_B, "TODO: b")
    _add(db, OLD_TODO, "TODO: d")
    _add(db, DONE_1,
         f"Completed: closed commitments {TODO_A}, {TODO_B}, {OLD_TODO} "
         "(dim-mismatch bug). Result: fixed in #130.", "fact")
    assert close_completed_todos(db) == [
        (DONE_1, TODO_A), (DONE_1, TODO_B), (DONE_1, OLD_TODO)
    ]
