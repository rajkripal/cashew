"""Tests for dynamic-dim sqlite-vec table creation.

Bug history: `core/embeddings.py` hardcoded `embedding float[384]` in the
sqlite-vec virtual table even after PR #40 made the rest of the embedding
stack honor CASHEW_EMBEDDING_MODEL. Result: a fresh DB created under
gte-large still got a 384-dim vec table, every dual-write rejected.
These tests pin the contract that the vec table dim follows the configured
model, and that legacy DBs at the wrong dim degrade to brute-force search
without crashing.
"""

from __future__ import annotations

import importlib
import os
import sqlite3

import pytest


def _reload_embedding_modules():
    """Reload modules so they pick up a freshly-set CASHEW_EMBEDDING_MODEL."""
    import core.config
    importlib.reload(core.config)
    import core.embedding_service
    importlib.reload(core.embedding_service)
    import core.embeddings
    importlib.reload(core.embeddings)
    return core.embeddings


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("CASHEW_EMBEDDING_MODEL", raising=False)
    yield


def _vec_table_sql(db_path: str) -> str:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='vec_embeddings'"
        ).fetchone()
        return row[0] if row else ""
    finally:
        conn.close()


class TestVecTableCreatedAtModelDim:
    def test_fresh_db_under_gte_large_gets_1024_dim_vec_table(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CASHEW_EMBEDDING_MODEL", "thenlper/gte-large")
        emb = _reload_embedding_modules()
        db = str(tmp_path / "gte.db")
        from core.session import _ensure_schema as ensure_session_schema
        ensure_session_schema(db)
        emb.ensure_schema(db)
        sql = _vec_table_sql(db)
        assert "float[1024]" in sql, f"expected float[1024] in {sql!r}"

    def test_fresh_db_under_minilm_gets_384_dim_vec_table(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CASHEW_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        emb = _reload_embedding_modules()
        db = str(tmp_path / "minilm.db")
        from core.session import _ensure_schema as ensure_session_schema
        ensure_session_schema(db)
        emb.ensure_schema(db)
        sql = _vec_table_sql(db)
        assert "float[384]" in sql, f"expected float[384] in {sql!r}"


class TestDualWriteAtNewDim:
    def test_dual_write_succeeds_at_1024_dim(self, tmp_path, monkeypatch):
        """A fresh gte-large DB should accept 1024-dim vectors into
        vec_embeddings without sqlite-vec rejecting them."""
        monkeypatch.setenv("CASHEW_EMBEDDING_MODEL", "thenlper/gte-large")
        emb = _reload_embedding_modules()
        db = str(tmp_path / "gte_dual.db")

        # Bootstrap the canonical schema so embed_nodes has thought_nodes.
        from core.session import _ensure_schema as ensure_session_schema
        ensure_session_schema(db)
        emb.ensure_schema(db)

        # Insert a node and synthesize a 1024-dim vector directly to avoid
        # loading the actual SentenceTransformer in CI.
        import numpy as np
        conn = sqlite3.connect(db)
        conn.execute(
            "INSERT INTO thought_nodes(id, content, node_type, timestamp) "
            "VALUES (?, ?, ?, ?)",
            ("n1", "hello world", "fact", "2026-05-08T00:00:00"),
        )
        conn.commit()
        conn.close()

        # Open a vec-loaded conn and dual-write a fake 1024-dim vector.
        conn = sqlite3.connect(db)
        conn.enable_load_extension(True)
        import sqlite_vec
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)

        vec = np.zeros(1024, dtype=np.float32)
        vec[0] = 1.0
        # Should NOT raise.
        conn.execute(
            "INSERT INTO vec_embeddings(node_id, embedding) VALUES (?, ?)",
            ("n1", vec.tobytes()),
        )
        conn.commit()

        row = conn.execute(
            "SELECT node_id FROM vec_embeddings WHERE node_id = ?", ("n1",)
        ).fetchone()
        assert row == ("n1",)
        conn.close()


class TestLegacyDimMismatchFallsBackGracefully:
    def test_search_on_384_dim_table_under_1024_model_does_not_crash(self, tmp_path, monkeypatch):
        """Pre-existing 384-dim vec table opened under a 1024-dim model must
        not crash — the code skips vec and falls back to brute force."""
        # Build a legacy DB at 384 dim.
        monkeypatch.setenv("CASHEW_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        emb = _reload_embedding_modules()
        db = str(tmp_path / "legacy.db")

        from core.session import _ensure_schema as ensure_session_schema
        ensure_session_schema(db)
        emb.ensure_schema(db)
        assert "float[384]" in _vec_table_sql(db)

        # Drop in a stored 384-dim embedding so brute-force has something.
        import numpy as np
        v384 = np.random.RandomState(0).randn(384).astype(np.float32)
        conn = sqlite3.connect(db)
        conn.execute(
            "INSERT INTO thought_nodes(id, content, node_type, timestamp, decayed) "
            "VALUES (?, ?, ?, ?, 0)",
            ("n1", "hello", "fact", "2026-05-08"),
        )
        conn.execute(
            "INSERT INTO embeddings(node_id, vector, model, updated_at) "
            "VALUES (?, ?, ?, ?)",
            ("n1", v384.tobytes(), "all-MiniLM-L6-v2", "2026-05-08"),
        )
        conn.commit()
        conn.close()

        # Switch model to 1024-dim and re-import.
        monkeypatch.setenv("CASHEW_EMBEDDING_MODEL", "thenlper/gte-large")
        emb = _reload_embedding_modules()

        # Fake out embed_text so we don't load SentenceTransformer.
        v1024 = np.random.RandomState(1).randn(1024).astype(np.float32).tolist()
        emb.embed_text = lambda text: v1024  # type: ignore

        # ensure_schema must not crash on dim mismatch.
        emb.ensure_schema(db)
        # Vec table dim is unchanged (we don't drop+recreate).
        assert "float[384]" in _vec_table_sql(db)

        # Search must not crash; falls back to brute force on the stored
        # 384-dim vector. Cosine of mismatched-shape vectors would error in
        # numpy, but since we skip the brute-force pair when shapes differ,
        # we just expect no crash and a (possibly empty) list.
        results = emb.search(db, "any query", top_k=5)
        assert isinstance(results, list)

    def test_backfill_refuses_dim_mismatch(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CASHEW_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        emb = _reload_embedding_modules()
        db = str(tmp_path / "legacy_bf.db")

        from core.session import _ensure_schema as ensure_session_schema
        ensure_session_schema(db)
        emb.ensure_schema(db)

        monkeypatch.setenv("CASHEW_EMBEDDING_MODEL", "thenlper/gte-large")
        emb = _reload_embedding_modules()

        result = emb.backfill_vec_index(db)
        assert "error" in result
        assert "dim mismatch" in result["error"].lower()
