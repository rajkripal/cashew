#!/usr/bin/env python3
"""Tests for ClaudeArchiveExtractor."""

import copy
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.extractors import ExtractorRegistry
from extractors.claude_archive import ClaudeArchiveExtractor


def _make_conversations_json(path, conversations):
    """Write a synthetic conversations.json to path."""
    with open(path, 'w') as f:
        json.dump(conversations, f, indent=2)


def _make_db(path):
    """Create a minimal cashew DB at path and return the path."""
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS thought_nodes (
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
            metadata TEXT,
            last_updated TEXT,
            mood_state TEXT,
            tags TEXT DEFAULT ""
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS derivation_edges (
            parent_id TEXT,
            child_id TEXT,
            weight REAL,
            reasoning TEXT,
            confidence REAL,
            timestamp TEXT,
            PRIMARY KEY (parent_id, child_id)
        )
    """)
    conn.commit()
    conn.close()
    return path


def _count_nodes(db_path):
    """Return count of rows in thought_nodes."""
    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM thought_nodes").fetchone()[0]
    conn.close()
    return count


# ── Fixtures ──────────────────────────────────────────────────────────

SIMPLE_CONV = {
    "uuid": "conv-001",
    "name": "Refactoring Discussion",
    "created_at": "2025-06-01T10:00:00Z",
    "updated_at": "2025-06-01T11:00:00Z",
    "chat_messages": [
        {
            "uuid": "msg-001",
            "sender": "human",
            "text": "I need to refactor the database layer to use async connections.",
            "content": [{"type": "text", "text": "I need to refactor the database layer to use async connections."}],
            "created_at": "2025-06-01T10:00:00Z",
            "parent_message_uuid": "00000000-0000-4000-8000-000000000000",
        },
        {
            "uuid": "msg-002",
            "sender": "assistant",
            "text": "Let's use async SQLAlchemy with connection pooling.",
            "content": [{"type": "text", "text": "Let's use async SQLAlchemy with connection pooling."}],
            "created_at": "2025-06-01T10:05:00Z",
            "parent_message_uuid": "msg-001",
        },
    ],
}

CONV_WITH_TOOL_CALLS = {
    "uuid": "conv-002",
    "name": "Weather Check and Analysis",
    "created_at": "2025-06-02T14:00:00Z",
    "updated_at": "2025-06-02T14:30:00Z",
    "chat_messages": [
        {
            "uuid": "msg-101",
            "sender": "human",
            "text": "Can you check the current weather in Raleigh and tell me if it's good for an afternoon bike ride on the greenway trails?",
            "content": [{"type": "text", "text": "Can you check the current weather in Raleigh and tell me if it's good for an afternoon bike ride on the greenway trails?"}],
            "created_at": "2025-06-02T14:00:00Z",
            "parent_message_uuid": "00000000-0000-4000-8000-000000000000",
        },
        {
            "uuid": "msg-102",
            "sender": "assistant",
            "text": "I checked the weather. Currently 84°F and partly cloudy in Raleigh. Good conditions for a bike ride — low chance of rain, moderate temperature. I'd recommend the Neuse River Trail since it has good tree coverage.",
            "content": [
                {"type": "text", "text": "Let me look up the current weather data for Raleigh, NC."},
                {"type": "tool_use", "text": "{}", "name": "get_weather"},
                {"type": "tool_result", "text": "{\"temp\":84,\"conditions\":\"partly cloudy\"}"},
                {"type": "text", "text": "Currently 84°F and partly cloudy in Raleigh. Good conditions for a bike ride — low chance of rain, moderate temperature. I'd recommend the Neuse River Trail since it has good tree coverage."},
            ],
            "created_at": "2025-06-02T14:02:00Z",
            "parent_message_uuid": "msg-101",
        },
    ],
}

SHORT_CONV = {
    "uuid": "conv-003",
    "name": "Greeting",
    "created_at": "2025-06-03T09:00:00Z",
    "updated_at": "2025-06-03T09:01:00Z",
    "chat_messages": [
        {
            "uuid": "msg-201",
            "sender": "human",
            "text": "hi",
            "content": [{"type": "text", "text": "hi"}],
            "created_at": "2025-06-03T09:00:00Z",
            "parent_message_uuid": "00000000-0000-4000-8000-000000000000",
        },
    ],
}

ALL_CONVERSATIONS = [SIMPLE_CONV, CONV_WITH_TOOL_CALLS, SHORT_CONV]


class TestClaudeArchiveExtractor(unittest.TestCase):
    """Test ClaudeArchiveExtractor."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.fixture_dir = Path(self.temp_dir.name)
        self.db_path = str(self.fixture_dir / "test.db")
        _make_db(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _run_extractor(self, source_path=None, model_fn=None):
        """Helper: run the claude_archive extractor and return result."""
        if source_path is None:
            source_path = str(self.fixture_dir)
        ext = ClaudeArchiveExtractor()
        return ext.extract(source_path, model_fn=model_fn, db_path=self.db_path)

    # ── Basic extraction ──────────────────────────────────────────────

    def test_extracts_substantive_conversations(self):
        """Should extract nodes from substantive conversations."""
        _make_conversations_json(self.fixture_dir / "conversations.json", [SIMPLE_CONV])
        nodes = self._run_extractor()
        # Two substantive messages -> 2 nodes (50+ chars each)
        self.assertEqual(len(nodes), 2)

    def test_strips_tool_calls(self):
        """Should filter out tool_use and tool_result content blocks."""
        _make_conversations_json(self.fixture_dir / "conversations.json", [CONV_WITH_TOOL_CALLS])
        nodes = self._run_extractor()
        # Human message + assistant message (only text blocks kept)
        self.assertEqual(len(nodes), 2)
        # Neither node should contain tool output
        for node in nodes:
            self.assertNotIn("tool_use", node["content"].lower())
            self.assertNotIn("tool_result", node["content"].lower())

    def test_skips_short_conversations(self):
        """Should skip messages under 50 characters."""
        _make_conversations_json(self.fixture_dir / "conversations.json", [SHORT_CONV])
        nodes = self._run_extractor()
        self.assertEqual(len(nodes), 0)

    def test_skips_conversations_entirely_of_tool_artifacts(self):
        """Should skip conversations where all substantive content is tool artifacts."""
        artifact_conv = {
            "uuid": "conv-artifact",
            "name": "Tool Only",
            "created_at": "2025-06-04T12:00:00Z",
            "updated_at": "2025-06-04T12:00:00Z",
            "chat_messages": [
                {
                    "uuid": "msg-art-1",
                    "sender": "assistant",
                    "text": "This block is not supported on your current device yet.",
                    "content": [{"type": "text", "text": "This block is not supported on your current device yet."}],
                    "created_at": "2025-06-04T12:00:00Z",
                    "parent_message_uuid": "00000000-0000-4000-8000-000000000000",
                },
            ],
        }
        _make_conversations_json(self.fixture_dir / "conversations.json", [artifact_conv])
        nodes = self._run_extractor()
        self.assertEqual(len(nodes), 0)

    # ── Domain and source_file ────────────────────────────────────────

    def test_domain_is_claude_conversations(self):
        """All extracted nodes should have domain='claude_conversations'."""
        _make_conversations_json(self.fixture_dir / "conversations.json", [SIMPLE_CONV])
        nodes = self._run_extractor()
        for node in nodes:
            self.assertEqual(node.get("domain"), "claude_conversations")

    def test_source_file_contains_extractor_prefix(self):
        """Source file should follow extractor:claude_archive:<uuid> pattern."""
        _make_conversations_json(self.fixture_dir / "conversations.json", [SIMPLE_CONV])
        nodes = self._run_extractor()
        for node in nodes:
            self.assertIn("extractor:claude_archive:conv-001", node.get("source_file", ""))

    # ── referent_time ─────────────────────────────────────────────────

    def test_referent_time_from_message_timestamps(self):
        """Nodes should carry the created_at timestamp of their message."""
        _make_conversations_json(self.fixture_dir / "conversations.json", [SIMPLE_CONV])
        nodes = self._run_extractor()
        # First message is human at 10:00:00Z
        self.assertEqual(nodes[0].get("referent_time"), "2025-06-01T10:00:00Z")
        # Second message is assistant at 10:05:00Z
        self.assertEqual(nodes[1].get("referent_time"), "2025-06-01T10:05:00Z")

    # ── Incremental processing ────────────────────────────────────────

    def test_incremental_skips_unchanged_conversations(self):
        """Second run should produce no nodes for unchanged conversations."""
        _make_conversations_json(self.fixture_dir / "conversations.json", [SIMPLE_CONV])
        ext = ClaudeArchiveExtractor()
        # First pass
        nodes1 = ext.extract(str(self.fixture_dir), model_fn=None, db_path=self.db_path)
        self.assertEqual(len(nodes1), 2)
        # Second pass - same conversations, same updated_at
        nodes2 = ext.extract(str(self.fixture_dir), model_fn=None, db_path=self.db_path)
        self.assertEqual(len(nodes2), 0)

    def test_incremental_processes_updated_conversations(self):
        """When a conversation's updated_at changes, it should be re-extracted."""
        conv = copy.deepcopy(SIMPLE_CONV)
        _make_conversations_json(self.fixture_dir / "conversations.json", [conv])
        ext = ClaudeArchiveExtractor()
        # First pass
        nodes1 = ext.extract(str(self.fixture_dir), model_fn=None, db_path=self.db_path)
        self.assertEqual(len(nodes1), 2)
        # Update the conversation with new messages and a later updated_at
        conv["updated_at"] = "2025-06-01T12:00:00Z"
        conv["chat_messages"].append({
            "uuid": "msg-003",
            "sender": "human",
            "text": "What about migration strategy? We should do it incrementally.",
            "content": [{"type": "text", "text": "What about migration strategy? We should do it incrementally."}],
            "created_at": "2025-06-01T11:00:00Z",
            "parent_message_uuid": "msg-002",
        })
        _make_conversations_json(self.fixture_dir / "conversations.json", [conv])
        # Second pass - should process new messages
        nodes2 = ext.extract(str(self.fixture_dir), model_fn=None, db_path=self.db_path)
        # Re-processing returns all messages in the conversation (dedup happens at registry level)
        self.assertEqual(len(nodes2), 3)
        self.assertTrue(any("migration strategy" in n["content"] for n in nodes2))

    # ── Accepts both directory and file path ──────────────────────────

    def test_accepts_directory_path(self):
        """Should find conversations.json in a directory."""
        _make_conversations_json(self.fixture_dir / "conversations.json", [SIMPLE_CONV])
        nodes = self._run_extractor(source_path=str(self.fixture_dir))
        self.assertEqual(len(nodes), 2)

    def test_accepts_direct_file_path(self):
        """Should accept a direct path to conversations.json."""
        _make_conversations_json(self.fixture_dir / "conversations.json", [SIMPLE_CONV])
        nodes = self._run_extractor(source_path=str(self.fixture_dir / "conversations.json"))
        self.assertEqual(len(nodes), 2)

    # ── Via the registry (integration) ────────────────────────────────

    def test_works_through_registry(self):
        """Should work when registered and run through ExtractorRegistry."""
        _make_conversations_json(self.fixture_dir / "conversations.json", [SIMPLE_CONV])
        registry = ExtractorRegistry(data_dir=str(self.fixture_dir))
        registry.register(ClaudeArchiveExtractor())
        result = registry.run("claude_archive", str(self.fixture_dir),
                              model_fn=None, db_path=self.db_path)
        self.assertEqual(result["nodes_created"], 2)
        self.assertEqual(result["errors"], [])

    def test_listed_in_registry(self):
        """Extractor should be discoverable via list_extractors()."""
        registry = ExtractorRegistry(data_dir=str(self.fixture_dir))
        registry.register(ClaudeArchiveExtractor())
        self.assertIn("claude_archive", registry.list_extractors())

    # ── State persistence ─────────────────────────────────────────────

    def test_state_persistence(self):
        """get_state/set_state round-trips processed UUIDs."""
        ext = ClaudeArchiveExtractor()
        ext._processed = {"conv-001": "2025-06-01T11:00:00Z"}
        state = ext.get_state()
        self.assertIn("processed", state)
        self.assertEqual(state["processed"]["conv-001"], "2025-06-01T11:00:00Z")
        ext2 = ClaudeArchiveExtractor()
        ext2.set_state(state)
        self.assertEqual(ext2._processed, ext._processed)

    # ── Error handling ────────────────────────────────────────────────

    def test_nonexistent_path_returns_empty(self):
        """Non-existent paths should produce an empty result, not crash."""
        nodes = self._run_extractor(source_path="/nonexistent/path")
        self.assertEqual(nodes, [])

    def test_missing_conversations_json_returns_empty(self):
        """Empty directory with no conversations.json should produce empty result."""
        nodes = self._run_extractor(source_path=str(self.fixture_dir))
        self.assertEqual(nodes, [])

    def test_invalid_json_returns_empty(self):
        """Malformed conversations.json should produce an empty result, not crash."""
        with open(self.fixture_dir / "conversations.json", 'w') as f:
            f.write("this is not json")
        nodes = self._run_extractor()
        self.assertEqual(nodes, [])


if __name__ == "__main__":
    unittest.main()
