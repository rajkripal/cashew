#!/usr/bin/env python3
"""Tests for the referent_time (event clock) column.

Covers:
- normalization accepts tz-aware ISO8601, rejects naive input
- session extractor passes through per-message timestamp
- similarity retrieval uses COALESCE(referent_time, timestamp)
- decay ignores referent_time (stays on ingestion clock)
- backfill handles missing/malformed source times without crashing
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from core.session import (  # noqa: E402
    _create_node,
    _ensure_schema,
    _normalize_referent_time,
)


# ---------------------------------------------------------------------------
# normalization
# ---------------------------------------------------------------------------

class TestNormalization:
    def test_accepts_utc_z(self):
        assert _normalize_referent_time("2019-03-14T12:00:00Z") == (
            "2019-03-14T12:00:00+00:00"
        )

    def test_accepts_offset(self):
        out = _normalize_referent_time("2019-03-14T12:00:00-05:00")
        assert out == "2019-03-14T17:00:00+00:00"

    def test_rejects_naive(self):
        with pytest.raises(ValueError):
            _normalize_referent_time("2019-03-14T12:00:00")

    def test_rejects_garbage(self):
        with pytest.raises(ValueError):
            _normalize_referent_time("not a date")

    def test_none_and_empty_pass_through(self):
        assert _normalize_referent_time(None) is None
        assert _normalize_referent_time("   ") is None


# ---------------------------------------------------------------------------
# CLI flag end-to-end: node gets referent_time stored
# ---------------------------------------------------------------------------

class TestCreateNodeStoresReferentTime:
    def test_stored(self, temp_db):
        _ensure_schema(temp_db)
        nid = _create_node(
            temp_db,
            "Event X happened in 2019",
            "observation",
            "sess",
            domain="default",
            referent_time="2019-03-14T12:00:00+00:00",
        )
        conn = sqlite3.connect(temp_db)
        row = conn.execute(
            "SELECT timestamp, referent_time FROM thought_nodes WHERE id = ?",
            (nid,),
        ).fetchone()
        conn.close()
        assert row is not None
        ts, ref = row
        # Ingestion clock is "now", event clock is 2019.
        assert ref == "2019-03-14T12:00:00+00:00"
        assert ts.startswith(datetime.now(timezone.utc).strftime("%Y-"))
        assert ts != ref


# ---------------------------------------------------------------------------
# session extractor per-message timestamp pass-through
# ---------------------------------------------------------------------------

class TestSessionExtractorPassThrough:
    def test_llm_path_tags_referent_time(self):
        from extractors.sessions import SessionExtractor

        extractor = SessionExtractor()
        messages = [
            {"role": "user", "content": "I moved to Seattle in 2019",
             "timestamp": "2019-06-15T10:00:00Z"},
            {"role": "assistant", "content": "Got it, Seattle 2019 noted",
             "timestamp": "2019-06-15T10:00:05Z"},
        ]

        # Stub model_fn that returns one extracted statement.
        def fake_model(prompt):
            return "User relocated to Seattle in mid-2019."

        result = extractor._extract_with_llm(messages, fake_model, "session_abc")
        assert result, "expected at least one extracted node"
        # Every extracted node must carry referent_time from message batch.
        for node in result:
            assert node.get("referent_time") == "2019-06-15T10:00:05Z"

    def test_simple_path_tags_referent_time(self):
        from extractors.sessions import SessionExtractor

        extractor = SessionExtractor()
        long_content = "x" * 200
        messages = [
            {"role": "user", "content": long_content,
             "timestamp": "2018-01-01T00:00:00Z"},
        ]
        result = extractor._extract_simple(messages, "sess")
        assert result
        assert result[0]["referent_time"] == "2018-01-01T00:00:00Z"


# ---------------------------------------------------------------------------
# similarity retrieval uses COALESCE(referent_time, timestamp)
# ---------------------------------------------------------------------------

class TestSimilarityUsesCoalesce:
    def test_recency_clock_loader_prefers_referent_time(self, temp_db):
        _ensure_schema(temp_db)
        now_iso = datetime.now(timezone.utc).isoformat()
        old_iso = "2019-01-01T00:00:00+00:00"
        conn = sqlite3.connect(temp_db)
        conn.execute(
            "INSERT INTO thought_nodes (id, content, node_type, timestamp, "
            "source_file, referent_time) VALUES (?, ?, ?, ?, ?, ?)",
            ("nA", "old event imported today", "observation", now_iso, "t", old_iso),
        )
        conn.execute(
            "INSERT INTO thought_nodes (id, content, node_type, timestamp, "
            "source_file, referent_time) VALUES (?, ?, ?, ?, ?, ?)",
            ("nB", "fresh event", "observation", now_iso, "t", None),
        )
        conn.commit()
        conn.close()

        from core.retrieval import _load_recency_clock, _recency_weight

        clocks = _load_recency_clock(temp_db, ["nA", "nB"])
        # nA uses referent_time (2019), nB falls back to timestamp (now)
        assert clocks["nA"] == old_iso
        assert clocks["nB"] == now_iso

        wA = _recency_weight(clocks["nA"])
        wB = _recency_weight(clocks["nB"])
        # Old event gets lower weight than fresh one.
        assert wA < wB
        # But still within [0.5, 1.0] — no node buried entirely.
        assert 0.5 <= wA <= 1.0
        assert 0.5 <= wB <= 1.0


# ---------------------------------------------------------------------------
# decay ignores referent_time
# ---------------------------------------------------------------------------

class TestDecayIgnoresReferentTime:
    def test_fresh_ingest_old_referent_not_decayed(self, temp_db):
        """A 2019 WhatsApp note imported today must NOT be decayed just
        because its event time is old. Decay reads ingestion clock only."""
        _ensure_schema(temp_db)
        now = datetime.now(timezone.utc)
        fresh_ts = now.isoformat()
        ancient_ref = "2019-01-01T00:00:00+00:00"
        conn = sqlite3.connect(temp_db)
        conn.execute(
            "INSERT INTO thought_nodes "
            "(id, content, node_type, timestamp, source_file, "
            "access_count, referent_time, decayed) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
            ("imp1", "old WA note imported fresh", "observation", fresh_ts,
             "whatsapp:2019", 0, ancient_ref),
        )
        conn.commit()
        conn.close()

        from core.decay import auto_decay

        result = auto_decay(
            temp_db, min_age_days=30,
            enable_cascading=False,
        )
        assert result["pruned"] == 0, (
            "node with fresh ingestion timestamp must not be decayed, "
            "regardless of old referent_time"
        )

    def test_old_ingest_decays_even_with_future_referent(self, temp_db):
        """Inverse: if ingestion timestamp is old and confidence/access
        qualify, decay proceeds. referent_time is not consulted."""
        _ensure_schema(temp_db)
        old_ts = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        future_ref = "2099-01-01T00:00:00+00:00"
        conn = sqlite3.connect(temp_db)
        conn.execute(
            "INSERT INTO thought_nodes "
            "(id, content, node_type, timestamp, source_file, "
            "access_count, referent_time, decayed) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
            ("old1", "stale ingest", "observation", old_ts, "x", 0, future_ref),
        )
        conn.commit()
        conn.close()

        from core.decay import auto_decay

        result = auto_decay(
            temp_db, min_age_days=14,
            enable_cascading=False,
        )
        assert result["pruned"] == 1


# ---------------------------------------------------------------------------
# backfill handles missing/malformed source times without crashing
# ---------------------------------------------------------------------------

class TestBackfill:
    def _make_db_with_rows(self, temp_db, rows):
        _ensure_schema(temp_db)
        conn = sqlite3.connect(temp_db)
        now_iso = datetime.now(timezone.utc).isoformat()
        for (nid, metadata, source_file) in rows:
            conn.execute(
                "INSERT INTO thought_nodes (id, content, node_type, "
                "timestamp, source_file, metadata, referent_time) "
                "VALUES (?, ?, ?, ?, ?, ?, NULL)",
                (nid, f"content {nid}", "observation", now_iso,
                 source_file, metadata),
            )
        conn.commit()
        conn.close()

    def test_recovers_from_metadata_and_filename_and_skips_garbage(self, temp_db):
        self._make_db_with_rows(temp_db, [
            ("a", json.dumps({"event_time": "2020-05-05T00:00:00Z"}), "random.txt"),
            ("b", json.dumps({}), "whatsapp-2019-06-15-chat.txt"),
            ("c", "{not json", "nope.txt"),
            ("d", None, None),
            ("e", json.dumps({"event_time": "garbage"}), "also.txt"),
        ])

        script = _REPO / "scripts" / "backfill_referent_time.py"
        # Dry run first: must not modify rows.
        r = subprocess.run(
            [sys.executable, str(script), "--db", temp_db],
            capture_output=True, text=True, check=True,
        )
        assert "Dry run" in r.stdout
        conn = sqlite3.connect(temp_db)
        null_count = conn.execute(
            "SELECT COUNT(*) FROM thought_nodes WHERE referent_time IS NULL"
        ).fetchone()[0]
        conn.close()
        assert null_count == 5

        # Apply.
        r = subprocess.run(
            [sys.executable, str(script), "--db", temp_db, "--apply"],
            capture_output=True, text=True, check=True,
        )
        assert "Wrote referent_time to" in r.stdout

        conn = sqlite3.connect(temp_db)
        got = dict(conn.execute(
            "SELECT id, referent_time FROM thought_nodes"
        ).fetchall())
        conn.close()

        # a: from metadata event_time
        assert got["a"] == "2020-05-05T00:00:00+00:00"
        # b: from source_file date (UTC midnight)
        assert got["b"] == "2019-06-15T00:00:00+00:00"
        # c,d,e: unrecoverable → NULL (NOT defaulted to timestamp)
        assert got["c"] is None
        assert got["d"] is None
        assert got["e"] is None
