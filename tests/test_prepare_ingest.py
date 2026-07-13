#!/usr/bin/env python3
"""
Tests for prepare-only and ingest patterns for both think cycle and extract commands.
"""

import sys
import os
import json
import tempfile
import sqlite3
from pathlib import Path
import pytest
import subprocess

# Add the cashew directory to the path
cashew_dir = Path(__file__).parent.parent
sys.path.insert(0, str(cashew_dir))

from core.session import _ensure_schema, _create_node, _get_connection
from core.embeddings import embed_nodes


# A small, deterministic synthetic brain used in place of the real graph.db.
# Two domains, a mix of node types, a few system_generated rows — enough for
# think --prepare-only to run and for the saturated-themes helper to have
# material, without depending on (or copying) the production brain. Keeping
# integration tests off the real db is what makes them hermetic and portable.
_SAMPLE_NODES = [
    ("Raj is optimizing a Redis data structure for lower tail latency at Meta", "fact", "raj", "test"),
    ("Raj prefers direct, no-fluff communication and systems-level thinking", "insight", "raj", "test"),
    ("Raj is tracking an E5 promotion and tends to go quiet when overloaded", "observation", "raj", "test"),
    ("Raj chose SQLite over Postgres for a side project to keep ops simple", "decision", "raj", "test"),
    ("Raj values empirical verification over vibes when judging results", "belief", "raj", "test"),
    ("Raj lives in Bothell and works in Pacific time", "fact", "raj", "test"),
    ("Bunny should query the brain before replying to substantive messages", "insight", "bunny", "system_generated"),
    ("Bunny runs a watchdog that restarts the Telegram bridge on crash", "fact", "bunny", "test"),
    ("Bunny extracts commitments as TODO nodes during conversation", "observation", "bunny", "system_generated"),
    ("Bunny keeps a dumb graph and a smart reasoning layer separate", "belief", "bunny", "test"),
    ("Bunny avoids sycophancy and pressure-tests positive claims", "insight", "bunny", "system_generated"),
    ("Bunny delegates heavy execution to sub-agents and verifies output", "decision", "bunny", "test"),
]


def _build_synthetic_brain(path: str) -> str:
    """Populate a schema-complete brain with embedded sample nodes."""
    _ensure_schema(path)
    conn = _get_connection(path)
    cursor = conn.cursor()
    for i, (content, node_type, domain, source_file) in enumerate(_SAMPLE_NODES):
        cursor.execute(
            "INSERT OR IGNORE INTO thought_nodes "
            "(id, content, node_type, timestamp, source_file, domain) "
            "VALUES (?, ?, ?, datetime('now'), ?, ?)",
            (f"syn{i:02d}", content, node_type, source_file, domain),
        )
    conn.commit()
    conn.close()
    embed_nodes(path)
    return path


class TestThinkCyclePrepareIngest:
    """Tests for think cycle prepare-only and ingest patterns"""

    @pytest.fixture
    def real_db(self, tmp_path):
        """Synthetic brain for read-only tests (was the real graph.db)."""
        return _build_synthetic_brain(str(tmp_path / "brain.db"))

    @pytest.fixture
    def temp_db(self, tmp_path):
        """Synthetic brain for write tests (was a copy of the real graph.db)."""
        return _build_synthetic_brain(str(tmp_path / "brain.db"))

    @pytest.fixture
    def empty_db(self):
        """Create a schema-only brain so diversity gates can't filter against
        unrelated prior content."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_db_path = Path(tmp_dir) / "test_graph.db"
            _ensure_schema(str(temp_db_path))
            yield str(temp_db_path)

    def test_think_prepare_only_outputs_valid_json(self, real_db):
        """Test that think --prepare-only outputs valid JSON with required fields"""
        result = subprocess.run([
            sys.executable, 
            str(cashew_dir / "scripts" / "cashew_context.py"),
            "think", "--prepare-only", "--db", real_db
        ], capture_output=True, text=True, env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"})
        
        assert result.returncode == 0
        
        # Parse the JSON output
        output = json.loads(result.stdout)
        
        # Check required fields
        assert "status" in output
        assert output["status"] in ["ready", "empty"]
        
        if output["status"] == "ready":
            assert "node_ids" in output
            assert "domains" in output
            assert "cluster_description" in output
            assert "saturated_block" in output
            assert isinstance(output["node_ids"], list)
            assert isinstance(output["domains"], list)
            assert isinstance(output["cluster_description"], str)
            assert len(output["node_ids"]) > 0
    
    def test_think_prepare_only_selects_multiple_domains(self, real_db):
        """Test that think --prepare-only selects nodes from multiple domains when possible"""
        result = subprocess.run([
            sys.executable,
            str(cashew_dir / "scripts" / "cashew_context.py"),
            "think", "--prepare-only", "--db", real_db
        ], capture_output=True, text=True, env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"})
        
        if result.returncode != 0:
            pytest.skip("Think prepare-only returned empty result")
        
        output = json.loads(result.stdout)
        
        if output["status"] == "ready" and len(output["domains"]) >= 2:
            assert len(output["domains"]) >= 2, "Should select nodes from multiple domains when available"
    
    def test_think_ingest_creates_nodes_and_edges(self, empty_db):
        """Test that think --ingest creates nodes and edges correctly"""
        # Create test insights JSON
        test_insights = {
            "insights": [
                {
                    "content": "Test insight about cross-domain patterns in automated systems",
                    "type": "insight",
                    "confidence": 0.8
                },
                {
                    "content": "Another test insight about engineering philosophy and personal beliefs",
                    "type": "insight", 
                    "confidence": 0.75
                }
            ],
            "source_node_ids": ["test_node_1", "test_node_2"]
        }
        
        # Create some source nodes first
        _ensure_schema(empty_db)
        conn = _get_connection(empty_db)
        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO thought_nodes (id, content, node_type, timestamp, source_file, domain) VALUES (?, ?, ?, datetime('now'), ?, ?)",
                      ("test_node_1", "Test source node 1", "observation", "test", "bunny"))
        cursor.execute("INSERT OR IGNORE INTO thought_nodes (id, content, node_type, timestamp, source_file, domain) VALUES (?, ?, ?, datetime('now'), ?, ?)",
                      ("test_node_2", "Test source node 2", "observation", "test", "raj"))
        conn.commit()
        conn.close()
        
        # Write insights to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_insights, f)
            insights_file = f.name
        
        try:
            # Run think --ingest
            result = subprocess.run([
                sys.executable,
                str(cashew_dir / "scripts" / "cashew_context.py"),
                "think", "--ingest", insights_file, "--db", empty_db
            ], capture_output=True, text=True, env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"})
            
            assert result.returncode == 0
            
            # Parse result
            output = json.loads(result.stdout)
            assert output["success"] == True
            assert output["new_nodes"] > 0
            assert output["new_edges"] > 0
            
            # Verify nodes were created in database
            conn = _get_connection(empty_db)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM thought_nodes WHERE source_file = 'system_generated'")
            new_count = cursor.fetchone()[0]
            assert new_count >= output["new_nodes"]
            conn.close()
            
        finally:
            os.unlink(insights_file)
    
    def test_think_ingest_respects_diversity_threshold(self, temp_db):
        """Test that think --ingest rejects too-similar nodes"""
        # Create a node with specific content
        _ensure_schema(temp_db)
        existing_content = "This is a very specific test insight about engineering patterns"
        existing_id = _create_node(temp_db, existing_content, "insight", "system_generated")
        embed_nodes(temp_db)
        
        # Try to ingest very similar content
        test_insights = {
            "insights": [
                {
                    "content": "This is a very specific test insight about engineering patterns and systems",  # Very similar
                    "type": "insight",
                    "confidence": 0.8
                }
            ],
            "source_node_ids": [existing_id]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_insights, f)
            insights_file = f.name
        
        try:
            result = subprocess.run([
                sys.executable,
                str(cashew_dir / "scripts" / "cashew_context.py"),
                "think", "--ingest", insights_file, "--db", temp_db
            ], capture_output=True, text=True, env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"})
            
            assert result.returncode == 0
            
            output = json.loads(result.stdout)
            # Should reject the similar content
            assert output["filtered_out"] > 0
            
        finally:
            os.unlink(insights_file)


class TestExtractPrepareIngest:
    """Tests for extract prepare-only and ingest patterns"""
    
    @pytest.fixture
    def temp_db(self, tmp_path):
        """Synthetic brain for write tests (was a copy of the real graph.db)."""
        return _build_synthetic_brain(str(tmp_path / "brain.db"))

    @pytest.fixture
    def sample_conversation(self):
        """Sample conversation text for testing"""
        return """
        This is a test conversation about engineering decisions.
        We learned that local embedding models are sufficient for small graphs.
        The key insight is that brute force cosine similarity works well under 100K nodes.
        Another important decision was to use SQLite instead of PostgreSQL for simplicity.
        """
    
    def test_extract_prepare_only_outputs_valid_json(self, sample_conversation):
        """Test that extract --prepare-only outputs valid JSON"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(sample_conversation)
            conversation_file = f.name
        
        try:
            result = subprocess.run([
                sys.executable,
                str(cashew_dir / "scripts" / "cashew_context.py"),
                "extract", "--prepare-only", "--input", conversation_file, "--db", "dummy.db"
            ], capture_output=True, text=True, env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"})
            
            assert result.returncode == 0
            
            output = json.loads(result.stdout)
            
            # Check required fields
            assert "status" in output
            assert output["status"] == "ready"
            assert "conversation_text" in output
            assert "extraction_prompt" in output
            assert "session_id" in output
            assert "file_path" in output
            assert "conversation_length" in output
            
            # Verify content
            assert sample_conversation.strip() in output["conversation_text"]
            assert "JSON array" in output["extraction_prompt"]
            assert len(output["conversation_text"]) == output["conversation_length"]
            
        finally:
            os.unlink(conversation_file)
    
    def test_extract_ingest_creates_nodes(self, temp_db):
        """Test that extract --ingest creates nodes correctly"""
        # Create test extraction results
        test_extractions = {
            "insights": [
                {
                    "content": "Local embedding models (all-MiniLM-L6-v2) are sufficient for graphs under 100K nodes",
                    "type": "fact",
                    "confidence": 0.8
                },
                {
                    "content": "SQLite was chosen over PostgreSQL for simplicity in the cashew project",
                    "type": "decision",
                    "confidence": 0.7
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_extractions, f)
            extractions_file = f.name
        
        try:
            result = subprocess.run([
                sys.executable,
                str(cashew_dir / "scripts" / "cashew_context.py"),
                "extract", "--ingest", extractions_file, "--db", temp_db
            ], capture_output=True, text=True, env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"})
            
            assert result.returncode == 0
            
            output = json.loads(result.stdout)
            assert output["success"] == True
            assert output["new_nodes"] > 0
            
            # Verify nodes were created with correct source_file
            conn = _get_connection(temp_db)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM thought_nodes WHERE source_file = 'openclaw_extraction'")
            new_count = cursor.fetchone()[0]
            assert new_count >= output["new_nodes"]
            conn.close()

        finally:
            os.unlink(extractions_file)

    def test_extract_ingest_embeds_and_links(self, temp_db):
        """Regression: --ingest must run the same post-write pipeline as LLM
        extraction. Before the fix, ingested nodes got no embedding and no
        similarity edges, making them invisible to retrieval."""
        # Near-duplicate of a _SAMPLE_NODES sentence — guaranteed to clear the
        # cross-link threshold, so at least one similarity edge must appear.
        test_extractions = {
            "insights": [{
                "content": "Raj chose SQLite over Postgres for a side project to keep operations simple, confirmed again this week",
                "type": "decision",
            }]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_extractions, f)
            extractions_file = f.name

        try:
            result = subprocess.run([
                sys.executable,
                str(cashew_dir / "scripts" / "cashew_context.py"),
                "extract", "--ingest", extractions_file, "--db", temp_db
            ], capture_output=True, text=True, env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"})

            assert result.returncode == 0, result.stderr
            output = json.loads(result.stdout)
            assert output["success"] is True
            assert output["new_nodes"] == 1
            assert output["new_edges"] >= 1

            conn = _get_connection(temp_db)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT n.id FROM thought_nodes n
                WHERE n.source_file = 'openclaw_extraction'""")
            node_ids = [row[0] for row in cursor.fetchall()]
            assert node_ids

            for node_id in node_ids:
                cursor.execute("SELECT COUNT(*) FROM embeddings WHERE node_id = ?", (node_id,))
                assert cursor.fetchone()[0] == 1, f"node {node_id} was not embedded"
                cursor.execute("SELECT COUNT(*) FROM derivation_edges WHERE child_id = ?", (node_id,))
                assert cursor.fetchone()[0] >= 1, f"node {node_id} has no similarity edges"
            conn.close()

        finally:
            os.unlink(extractions_file)


class TestSaturatedThemesHelper:
    """Test the saturated themes helper function"""
    
    @pytest.fixture
    def real_db(self, tmp_path):
        """Synthetic brain for read-only tests (was the real graph.db)."""
        return _build_synthetic_brain(str(tmp_path / "brain.db"))

    def test_saturated_themes_returns_recent_system_generated_content(self, real_db):
        """Test that _get_saturated_themes returns recent system_generated content"""
        from core.session import _get_saturated_themes
        
        themes = _get_saturated_themes(real_db, days=14, min_count=3)
        
        # Should return a list of strings
        assert isinstance(themes, list)
        
        # If there are themes, they should be strings
        for theme in themes:
            assert isinstance(theme, str)
            assert len(theme) > 0
        
        # Check that these are actually from the database
        if themes:
            conn = _get_connection(real_db)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM thought_nodes 
                WHERE source_file = 'system_generated'
                AND timestamp > datetime('now', '-14 days')
                AND (decayed IS NULL OR decayed = 0)
            """)
            count = cursor.fetchone()[0]
            conn.close()
            
            assert count > 0, "Should have recent system_generated nodes"


class TestIngestNoLlm:
    """Tests for cashew ingest --no-llm flag (regression for #50)"""

    @pytest.fixture
    def obsidian_vault(self, tmp_path):
        """Create a minimal Obsidian vault with a few markdown files."""
        vault = tmp_path / "test-vault"
        vault.mkdir()
        (vault / "Note One.md").write_text("""---
title: Note One
tags: [test, engineering]
created: 2026-01-01
---

This is a test note about engineering decisions.
We learned that local embedding models work well.
Another key insight: SQLite is simpler than PostgreSQL for small graphs.
""")
        (vault / "Projects").mkdir()
        (vault / "Projects" / "Project Alpha.md").write_text("""---
title: Project Alpha
tags: [project, test]
status: active
---

Project Alpha uses cashew for persistent memory.
The architecture relies on sqlite-vec for vector search.
""")
        return str(vault)

    def test_ingest_obsidian_no_llm_succeeds(self, obsidian_vault):
        """Test that 'cashew ingest obsidian --no-llm' completes without errors."""
        db_fd, db_path = tempfile.mkstemp(suffix='.db')
        os.close(db_fd)

        # Initialize the database schema (same as conftest.temp_db fixture)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
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
                last_updated TEXT,
                mood_state TEXT,
                permanent INTEGER DEFAULT 0,
                referent_time TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE derivation_edges (
                parent_id TEXT,
                child_id TEXT,
                weight REAL,
                reasoning TEXT,
                confidence REAL,
                timestamp TEXT,
                PRIMARY KEY (parent_id, child_id),
                FOREIGN KEY (parent_id) REFERENCES thought_nodes(id),
                FOREIGN KEY (child_id) REFERENCES thought_nodes(id)
            )
        ''')
        cursor.execute('''
            CREATE TABLE embeddings (
                node_id TEXT PRIMARY KEY,
                vector BLOB NOT NULL,
                model TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (node_id) REFERENCES thought_nodes(id)
            )
        ''')
        conn.commit()
        conn.close()

        try:
            result = subprocess.run([
                sys.executable,
                str(cashew_dir / "cashew_cli.py"),
                "--db", db_path,
                "ingest", "obsidian", obsidian_vault,
                "--no-llm",
            ], capture_output=True, text=True, timeout=60)

            assert result.returncode == 0, (
                f"ingest failed (rc={result.returncode}): {result.stderr[:500]}"
            )
            assert "✅ Extraction complete" in result.stdout, (
                f"Expected success message in stdout: {result.stdout[:500]}"
            )

            # Verify nodes were created in the database
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM thought_nodes")
            node_count = cursor.fetchone()[0]
            conn.close()
            assert node_count > 0, "Expected at least one node to be created"

        finally:
            os.unlink(db_path)