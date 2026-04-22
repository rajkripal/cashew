"""Tests for the public schema API (issues #4, #5, #6).

Covers:
- ensure_schema bootstraps a fresh database from scratch
- ensure_schema is idempotent
- ensure_schema upgrades a legacy database missing newer columns
- PRAGMA user_version carries the applied schema version
- core.db exposes get_schema_version / schema_version / ensure_schema
"""
from __future__ import annotations

import os
import sqlite3
import tempfile

import pytest

from core.session import SCHEMA_VERSION, _ensure_schema, get_schema_version
from core import db as cdb


OWNED_TABLES = {
    "thought_nodes",
    "derivation_edges",
    "embeddings",
    "hotspots",
    "metrics",
}

OWNED_NODE_COLUMNS = {
    "id", "content", "node_type", "domain", "timestamp",
    "access_count", "last_accessed", "confidence", "source_file",
    "decayed", "metadata", "last_updated", "mood_state", "permanent",
    "tags", "referent_time",
}


def _tables(db_path: str) -> set[str]:
    conn = sqlite3.connect(db_path)
    try:
        return {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    finally:
        conn.close()


def _columns(db_path: str, table: str) -> set[str]:
    conn = sqlite3.connect(db_path)
    try:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    finally:
        conn.close()


def test_ensure_schema_creates_fresh_database(tmp_path):
    db = str(tmp_path / "fresh.db")
    _ensure_schema(db)
    assert OWNED_TABLES.issubset(_tables(db))
    assert OWNED_NODE_COLUMNS.issubset(_columns(db, "thought_nodes"))


def test_ensure_schema_is_idempotent(tmp_path):
    db = str(tmp_path / "idem.db")
    _ensure_schema(db)
    _ensure_schema(db)
    _ensure_schema(db)
    assert get_schema_version(db) == SCHEMA_VERSION


def test_ensure_schema_migrates_legacy_database(tmp_path):
    db = str(tmp_path / "legacy.db")
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE thought_nodes ("
        " id TEXT PRIMARY KEY, content TEXT, node_type TEXT, timestamp TEXT)"
    )
    conn.commit()
    conn.close()

    _ensure_schema(db)

    assert OWNED_NODE_COLUMNS.issubset(_columns(db, "thought_nodes"))
    assert OWNED_TABLES.issubset(_tables(db))
    assert get_schema_version(db) == SCHEMA_VERSION


def test_fresh_database_accepts_writes(tmp_path):
    db = str(tmp_path / "write.db")
    _ensure_schema(db)
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            "INSERT INTO thought_nodes(id, content, node_type) VALUES (?,?,?)",
            ("n1", "hello", "fact"),
        )
        conn.execute(
            "INSERT INTO derivation_edges(parent_id, child_id, weight) VALUES (?,?,?)",
            ("n1", "n1", 1.0),
        )
        conn.commit()
        assert conn.execute("SELECT COUNT(*) FROM thought_nodes").fetchone()[0] == 1
    finally:
        conn.close()


def test_version_zero_for_unmanaged_database(tmp_path):
    db = str(tmp_path / "unmanaged.db")
    sqlite3.connect(db).close()
    assert get_schema_version(db) == 0


def test_core_db_exposes_public_api(tmp_path, monkeypatch):
    db = str(tmp_path / "api.db")
    # Force resolve_db_path to use our temp file regardless of config.
    monkeypatch.setenv("CASHEW_DB_PATH", db)

    cdb.ensure_schema()
    assert cdb.get_schema_version() == cdb.schema_version() == SCHEMA_VERSION
