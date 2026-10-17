"""Demotion releases permanence; CASHEW_NO_TOUCH skips access tracking."""

import sqlite3

from core.session import _update_access_tracking
from core.sleep import _promote_core_memories


def _db(tmp_path):
    path = tmp_path / "g.db"
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE thought_nodes(id TEXT PRIMARY KEY, content TEXT, "
        "node_type TEXT, permanent INTEGER DEFAULT 0, decayed INTEGER DEFAULT 0, "
        "access_count INTEGER DEFAULT 0, last_accessed TEXT)"
    )
    return path, conn


def test_demoted_core_memory_loses_permanence(tmp_path):
    _, conn = _db(tmp_path)
    conn.executemany(
        "INSERT INTO thought_nodes(id, content, node_type, permanent) VALUES (?, ?, ?, ?)",
        [("old", "old", "core_memory", 1)]
        + [(f"n{i}", f"n{i}", "fact", 0) for i in range(4)],
    )
    metrics = {"old": {"fitness": 0.0}}
    metrics.update({f"n{i}": {"fitness": 10.0 - i} for i in range(4)})

    result = _promote_core_memories(conn, metrics)

    assert result["demoted"] == 1
    node_type, permanent = conn.execute(
        "SELECT node_type, permanent FROM thought_nodes WHERE id='old'"
    ).fetchone()
    assert (node_type, permanent) == ("derived", 0)
    assert conn.execute(
        "SELECT node_type, permanent FROM thought_nodes WHERE id='n0'"
    ).fetchone() == ("core_memory", 1)


def test_no_touch_env_skips_access_tracking(tmp_path, monkeypatch):
    path, conn = _db(tmp_path)
    conn.execute("INSERT INTO thought_nodes(id, content) VALUES ('a', 'a')")
    conn.commit()
    conn.close()

    monkeypatch.setenv("CASHEW_NO_TOUCH", "1")
    _update_access_tracking(str(path), ["a"])
    check = sqlite3.connect(str(path))
    assert check.execute("SELECT access_count FROM thought_nodes").fetchone()[0] == 0
    check.close()

    monkeypatch.delenv("CASHEW_NO_TOUCH")
    _update_access_tracking(str(path), ["a"])
    check = sqlite3.connect(str(path))
    assert check.execute("SELECT access_count FROM thought_nodes").fetchone()[0] == 1
    check.close()
