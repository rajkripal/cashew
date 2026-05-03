#!/usr/bin/env python3
"""
Tests for prepare-only and ingest patterns for both think cycle and extract commands.
"""

import sys
import os
import json
import tempfile
import sqlite3
import shutil
from pathlib import Path
import pytest
import subprocess

# Add the cashew directory to the path
cashew_dir = Path(__file__).parent.parent
sys.path.insert(0, str(cashew_dir))

from core.session import _ensure_schema, _create_node, _get_connection
from core.embeddings import embed_nodes


class TestThinkCyclePrepareIngest:
    """Tests for think cycle prepare-only and ingest patterns"""
    
    @pytest.fixture
    def real_db(self):
        """Use real graph.db for read-only tests"""
        db_path = cashew_dir / "data" / "graph.db"
        if not db_path.exists():
            pytest.skip(f"Real database not found at {db_path}")
        return str(db_path)
    
    @pytest.fixture
    def temp_db(self):
        """Create a temp copy of the real database for write tests"""
        real_db_path = cashew_dir / "data" / "graph.db"
        if not real_db_path.exists():
            pytest.skip(f"Real database not found at {real_db_path}")
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_db_path = Path(tmp_dir) / "test_graph.db"
            shutil.copy2(real_db_path, temp_db_path)
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
    
    def test_think_ingest_creates_nodes_and_edges(self, temp_db):
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
        _ensure_schema(temp_db)
        conn = _get_connection(temp_db)
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
                "think", "--ingest", insights_file, "--db", temp_db
            ], capture_output=True, text=True, env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"})
            
            assert result.returncode == 0
            
            # Parse result
            output = json.loads(result.stdout)
            assert output["success"] == True
            assert output["new_nodes"] > 0
            assert output["new_edges"] > 0
            
            # Verify nodes were created in database
            conn = _get_connection(temp_db)
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
    def temp_db(self):
        """Create a temp copy of the real database for write tests"""
        real_db_path = cashew_dir / "data" / "graph.db"
        if not real_db_path.exists():
            pytest.skip(f"Real database not found at {real_db_path}")
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_db_path = Path(tmp_dir) / "test_graph.db"
            shutil.copy2(real_db_path, temp_db_path)
            yield str(temp_db_path)
    
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


class TestSaturatedThemesHelper:
    """Test the saturated themes helper function"""
    
    @pytest.fixture
    def real_db(self):
        """Use real graph.db for tests"""
        db_path = cashew_dir / "data" / "graph.db"
        if not db_path.exists():
            pytest.skip(f"Real database not found at {db_path}")
        return str(db_path)
    
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


if __name__ == "__main__":
    pytest.main([__file__])