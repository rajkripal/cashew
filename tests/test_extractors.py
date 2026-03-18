#!/usr/bin/env python3
"""
Tests for cashew.core.extractors module
"""

import pytest
import sqlite3
import json
import tempfile
import os
from pathlib import Path

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.extractors import BaseExtractor, ExtractorRegistry


class DummyExtractor(BaseExtractor):
    """Test extractor that returns canned nodes."""
    name = "dummy"

    def __init__(self):
        self._state = {}

    def extract(self, source_path, model_fn, db_path):
        return [
            {"content": "Dummy fact number one for testing purposes", "type": "fact", "confidence": 0.9},
            {"content": "Dummy observation about the world around us", "type": "observation", "confidence": 0.6},
        ]

    def get_state(self):
        return self._state

    def set_state(self, state):
        self._state = state


class FailingExtractor(BaseExtractor):
    """Extractor that always raises."""
    name = "failing"

    def extract(self, source_path, model_fn, db_path):
        raise RuntimeError("extractor exploded")


class StatefulExtractor(BaseExtractor):
    """Extractor with meaningful state."""
    name = "stateful"

    def __init__(self):
        self._cursor = 0

    def extract(self, source_path, model_fn, db_path):
        self._cursor += 1
        return [{"content": f"Stateful extraction run number {self._cursor} completed", "type": "observation", "confidence": 0.7}]

    def get_state(self):
        return {"cursor": self._cursor}

    def set_state(self, state):
        self._cursor = state.get("cursor", 0)


@pytest.fixture
def gc_db():
    """Temp database with full schema for extractor commit tests."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE thought_nodes (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            node_type TEXT NOT NULL,
            domain TEXT,
            timestamp TEXT,
            access_count INTEGER DEFAULT 0,
            last_accessed TEXT,
            confidence REAL,
            source_file TEXT,
            decayed INTEGER DEFAULT 0,
            metadata TEXT DEFAULT '{}',
            mood_state TEXT,
            permanent INTEGER DEFAULT 0
        )
    """)
    c.execute("""
        CREATE TABLE derivation_edges (
            parent_id TEXT,
            child_id TEXT,
            weight REAL,
            reasoning TEXT,
            PRIMARY KEY (parent_id, child_id)
        )
    """)
    c.execute("""
        CREATE TABLE embeddings (
            node_id TEXT PRIMARY KEY,
            vector BLOB NOT NULL,
            model TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

    yield path
    os.unlink(path)


@pytest.fixture
def data_dir():
    """Temporary data directory for state persistence."""
    d = tempfile.mkdtemp()
    yield d
    import shutil
    shutil.rmtree(d, ignore_errors=True)


class TestExtractorRegistry:

    def test_register_and_get(self, data_dir):
        reg = ExtractorRegistry(data_dir=data_dir)
        ext = DummyExtractor()
        reg.register(ext)

        assert reg.get("dummy") is ext

    def test_register_duplicate_raises(self, data_dir):
        reg = ExtractorRegistry(data_dir=data_dir)
        reg.register(DummyExtractor())

        with pytest.raises(ValueError, match="already registered"):
            reg.register(DummyExtractor())

    def test_register_no_name_raises(self, data_dir):
        reg = ExtractorRegistry(data_dir=data_dir)

        class NoName(BaseExtractor):
            name = ""
            def extract(self, *a):
                return []

        with pytest.raises(ValueError, match="non-empty name"):
            reg.register(NoName())

    def test_get_missing_raises(self, data_dir):
        reg = ExtractorRegistry(data_dir=data_dir)
        with pytest.raises(KeyError, match="not registered"):
            reg.get("nope")

    def test_list_extractors(self, data_dir):
        reg = ExtractorRegistry(data_dir=data_dir)
        reg.register(DummyExtractor())
        reg.register(StatefulExtractor())

        names = reg.list_extractors()
        assert "dummy" in names
        assert "stateful" in names
        assert len(names) == 2

    def test_unregister(self, data_dir):
        reg = ExtractorRegistry(data_dir=data_dir)
        reg.register(DummyExtractor())
        reg.unregister("dummy")
        assert "dummy" not in reg.list_extractors()

    def test_unregister_missing_raises(self, data_dir):
        reg = ExtractorRegistry(data_dir=data_dir)
        with pytest.raises(KeyError):
            reg.unregister("nope")


class TestExtractorRun:

    def test_run_creates_nodes(self, data_dir, gc_db):
        reg = ExtractorRegistry(data_dir=data_dir)
        reg.register(DummyExtractor())

        result = reg.run("dummy", "/dev/null", model_fn=None, db_path=gc_db)

        assert result["nodes_created"] == 2
        assert result["errors"] == []

        # Verify nodes in DB
        conn = sqlite3.connect(gc_db)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM thought_nodes")
        assert c.fetchone()[0] == 2
        conn.close()

    def test_run_failing_extractor(self, data_dir, gc_db):
        reg = ExtractorRegistry(data_dir=data_dir)
        reg.register(FailingExtractor())

        result = reg.run("failing", "/dev/null", model_fn=None, db_path=gc_db)

        assert result["nodes_created"] == 0
        assert len(result["errors"]) == 1
        assert "exploded" in result["errors"][0]

    def test_run_all(self, data_dir, gc_db):
        reg = ExtractorRegistry(data_dir=data_dir)
        reg.register(DummyExtractor())
        reg.register(FailingExtractor())

        results = reg.run_all(model_fn=None, db_path=gc_db)

        assert "dummy" in results
        assert "failing" in results
        assert results["dummy"]["nodes_created"] == 2
        assert results["dummy"]["errors"] == []
        assert results["failing"]["nodes_created"] == 0
        assert len(results["failing"]["errors"]) == 1

    def test_run_skips_duplicate_nodes(self, data_dir, gc_db):
        """Running the same extractor twice should not create duplicate nodes."""
        reg = ExtractorRegistry(data_dir=data_dir)
        reg.register(DummyExtractor())

        reg.run("dummy", "/dev/null", model_fn=None, db_path=gc_db)
        result2 = reg.run("dummy", "/dev/null", model_fn=None, db_path=gc_db)

        # Second run should create 0 new nodes (content hashes match)
        assert result2["nodes_created"] == 0

        conn = sqlite3.connect(gc_db)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM thought_nodes")
        assert c.fetchone()[0] == 2
        conn.close()


class TestExtractorStatePersistence:

    def test_state_saved_after_run(self, data_dir, gc_db):
        reg = ExtractorRegistry(data_dir=data_dir)
        ext = StatefulExtractor()
        reg.register(ext)

        reg.run("stateful", "/dev/null", model_fn=None, db_path=gc_db)

        state_file = Path(data_dir) / "extractor_state" / "stateful.json"
        assert state_file.exists()

        with open(state_file) as f:
            saved = json.load(f)
        assert saved["cursor"] == 1

    def test_state_restored_on_register(self, data_dir, gc_db):
        # First: run and save state
        reg1 = ExtractorRegistry(data_dir=data_dir)
        ext1 = StatefulExtractor()
        reg1.register(ext1)
        reg1.run("stateful", "/dev/null", model_fn=None, db_path=gc_db)
        assert ext1._cursor == 1

        # Second: new registry, new instance — state should be restored
        reg2 = ExtractorRegistry(data_dir=data_dir)
        ext2 = StatefulExtractor()
        reg2.register(ext2)

        assert ext2._cursor == 1  # restored from JSON

    def test_state_not_saved_when_empty(self, data_dir, gc_db):
        """Extractors with empty state should not create state files."""
        reg = ExtractorRegistry(data_dir=data_dir)

        class NoState(BaseExtractor):
            name = "nostate"
            def extract(self, *a):
                return []
            def get_state(self):
                return {}

        reg.register(NoState())
        reg.run("nostate", "/dev/null", model_fn=None, db_path=gc_db)

        state_file = Path(data_dir) / "extractor_state" / "nostate.json"
        assert not state_file.exists()
