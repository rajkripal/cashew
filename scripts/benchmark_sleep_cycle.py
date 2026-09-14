#!/usr/bin/env python3
"""Opt-in deterministic sleep-cycle measurement on an isolated fixture."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.model_profiles import get_active_profile
from core.sleep import run_sleep_cycle


def _build_fixture(path: Path, nodes: int, dimension: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE thought_nodes(
            id TEXT PRIMARY KEY, content TEXT, decayed INTEGER DEFAULT 0,
            timestamp TEXT DEFAULT '', source_file TEXT,
            access_count INTEGER DEFAULT 0, permanent INTEGER DEFAULT 0,
            last_accessed TEXT, domain TEXT, node_type TEXT,
            confidence REAL, metadata TEXT, mood_state TEXT
        );
        CREATE TABLE embeddings(
            node_id TEXT PRIMARY KEY, vector BLOB, model TEXT, updated_at TEXT
        );
        CREATE TABLE derivation_edges(
            parent_id TEXT, child_id TEXT, weight REAL, reasoning TEXT,
            PRIMARY KEY(parent_id, child_id)
        );
        """
    )
    shared = np.zeros(dimension, dtype=np.float32)
    shared[0] = 1.0
    for index in range(nodes):
        node_id = f"n{index:08d}"
        residual = rng.normal(size=dimension).astype(np.float32)
        residual[0] = 0.0
        residual /= np.linalg.norm(residual)
        # Pair similarities cluster around 0.92: inside gte-large's
        # cross-link band while remaining below its 0.94 dedup threshold.
        vector = np.asarray(
            np.sqrt(0.92) * shared + np.sqrt(0.08) * residual,
            dtype=np.float32,
        )
        conn.execute(
            "INSERT INTO thought_nodes(id, content, timestamp, source_file) "
            "VALUES (?, ?, ?, ?)",
            (node_id, f"synthetic node {index}", f"2026-01-{index % 28 + 1:02d}",
             f"source-{index % 7}.md"),
        )
        conn.execute(
            "INSERT INTO embeddings(node_id, vector, model) VALUES (?, ?, ?)",
            (node_id, vector.tobytes(), get_active_profile().name),
        )
    conn.commit()
    conn.close()


def _hold_writer(path: Path, milliseconds: int, acquired: threading.Event) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("BEGIN IMMEDIATE")
        acquired.set()
        time.sleep(milliseconds / 1000)
        conn.rollback()
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=500)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--max-edges", type=int, default=100)
    parser.add_argument("--seed", type=int, default=193)
    parser.add_argument(
        "--hold-writer-ms", type=int, default=0,
        help="hold a competing write transaction for this many milliseconds",
    )
    args = parser.parse_args()
    if (
        args.nodes < 2 or args.limit < 2 or args.max_edges < 0
        or args.hold_writer_ms < 0
    ):
        parser.error("nodes/limit must be at least 2; limits must be non-negative")

    profile = get_active_profile()
    with tempfile.TemporaryDirectory(prefix="cashew-sleep-benchmark-") as tmp:
        path = Path(tmp) / "fixture.db"
        _build_fixture(path, args.nodes, profile.dim, args.seed)
        writer = None
        if args.hold_writer_ms:
            acquired = threading.Event()
            writer = threading.Thread(
                target=_hold_writer,
                args=(path, args.hold_writer_ms, acquired),
                daemon=True,
            )
            writer.start()
            if not acquired.wait(timeout=2):
                raise RuntimeError("competing writer did not acquire its transaction")
        started = time.perf_counter()
        result = run_sleep_cycle(
            str(path), limit=args.limit, max_edges=args.max_edges,
            journal_policy="preserve",
        )
        if writer is not None:
            writer.join(timeout=max(2.0, args.hold_writer_ms / 1000 + 1.0))
            if writer.is_alive():
                raise RuntimeError("competing writer did not exit")
        report = {
            "fixture_nodes": args.nodes,
            "limit": args.limit,
            "max_edges": args.max_edges,
            "hold_writer_ms": args.hold_writer_ms,
            "model": profile.name,
            "dimension": profile.dim,
            "wall_seconds": round(time.perf_counter() - started, 6),
            "result": result,
        }
    print(json.dumps(report, sort_keys=True))
    return 0 if result["status"] in {"completed", "partial"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
