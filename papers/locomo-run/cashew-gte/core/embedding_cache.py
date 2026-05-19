"""
Content-hash embedding cache.

Embeddings are a pure function of (model, text). That makes caching by
sha256(text) under a model-version namespace correct without expiry logic —
a model swap invalidates via the namespace, nothing else can go stale.

Stored in SQLite as raw float32 bytes to match the rest of the embeddings
pipeline. A separate DB from the main graph so cache corruption never
threatens the graph, and so CI can point at /tmp without spinning up the
full schema.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np

_SCHEMA = """
CREATE TABLE IF NOT EXISTS embedding_cache (
    model   TEXT NOT NULL,
    hash    TEXT NOT NULL,
    vector  BLOB NOT NULL,
    PRIMARY KEY (model, hash)
) WITHOUT ROWID;
"""


def default_cache_path() -> str:
    base = os.environ.get("CASHEW_EMBD_CACHE")
    if base:
        return base
    home = Path.home() / ".cashew"
    home.mkdir(parents=True, exist_ok=True)
    return str(home / "embedding_cache.db")


def _key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """Thin SQLite-backed key-value store for embedding vectors.

    Usage:
        cache = EmbeddingCache()
        hit = cache.get("all-MiniLM-L6-v2", "hello world")  # None or np.ndarray
        cache.put("all-MiniLM-L6-v2", "hello world", vector)

    Thread-safe across processes via SQLite's locking. Not optimized for
    concurrent writers; the daemon is the only expected writer.
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path or default_cache_path()
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def get(self, model: str, text: str) -> Optional[np.ndarray]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT vector FROM embedding_cache WHERE model=? AND hash=?",
                (model, _key(text)),
            ).fetchone()
        if row is None:
            return None
        return np.frombuffer(row[0], dtype=np.float32)

    def get_many(self, model: str, texts: List[str]) -> List[Optional[np.ndarray]]:
        """Parallel lookup for a batch. Returns one entry per text, None on miss."""
        if not texts:
            return []
        hashes = [_key(t) for t in texts]
        placeholders = ",".join("?" * len(hashes))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT hash, vector FROM embedding_cache "
                f"WHERE model=? AND hash IN ({placeholders})",
                (model, *hashes),
            ).fetchall()
        by_hash = {h: np.frombuffer(v, dtype=np.float32) for h, v in rows}
        return [by_hash.get(h) for h in hashes]

    def put(self, model: str, text: str, vector: np.ndarray) -> None:
        vec = np.asarray(vector, dtype=np.float32)
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO embedding_cache(model, hash, vector) VALUES (?, ?, ?)",
                (model, _key(text), vec.tobytes()),
            )

    def put_many(self, model: str, pairs: Iterable[Tuple[str, np.ndarray]]) -> int:
        rows = [
            (model, _key(t), np.asarray(v, dtype=np.float32).tobytes())
            for t, v in pairs
        ]
        if not rows:
            return 0
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO embedding_cache(model, hash, vector) VALUES (?, ?, ?)",
                rows,
            )
        return len(rows)

    def size(self, model: Optional[str] = None) -> int:
        with self._connect() as conn:
            if model is None:
                row = conn.execute("SELECT COUNT(*) FROM embedding_cache").fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) FROM embedding_cache WHERE model=?", (model,)
                ).fetchone()
        return int(row[0])

    def invalidate_model(self, model: str) -> int:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM embedding_cache WHERE model=?", (model,))
            return cur.rowcount
