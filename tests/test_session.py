#!/usr/bin/env python3
"""
Tests for Cashew Session Integration Layer
"""

import pytest
import sqlite3
import tempfile
import os
import json
from datetime import datetime, timezone
from unittest.mock import Mock, patch

# Add the parent directory to the path to import our modules
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from core.session import (
    start_session, end_session, think_cycle,
    SessionContext, ExtractionResult, ThinkResult,
    _create_node, _find_similar_nodes, _estimate_tokens
)
from core.config import CashewConfig

class TestSessionIntegration:
    """Test suite for session integration functions"""
    
    @pytest.fixture
    def test_db_path(self):
        """Create a temporary database for testing"""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        
        # Initialize test database with schema
        conn = sqlite3.connect(path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE thought_nodes (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                node_type TEXT NOT NULL,
                timestamp TEXT,
                confidence REAL DEFAULT 0.5,
                source_file TEXT,
                metadata TEXT DEFAULT '{}',
                decayed INTEGER DEFAULT 0,
                last_accessed TEXT,
                last_updated TEXT,
                access_count INTEGER DEFAULT 0,
                domain TEXT DEFAULT 'raj',
                permanent INTEGER DEFAULT 0,
                referent_time TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE derivation_edges (
                parent_id TEXT NOT NULL,
                child_id TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                reasoning TEXT,
                UNIQUE(parent_id, child_id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE embeddings (
                node_id TEXT PRIMARY KEY,
                vector BLOB NOT NULL,
                model TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (node_id) REFERENCES thought_nodes(id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE hotspots (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                status TEXT,
                domain TEXT,
                file_pointers TEXT,
                cluster_node_ids TEXT,
                tags TEXT,
                created TEXT,
                last_updated TEXT
            )
        """)
        
        # Add some test data
        test_nodes = [
            ("node1", "This is about machine learning and AI", "fact", "work"),
            ("node2", "Python programming is useful for data science", "observation", "work"), 
            ("node3", "Exercise improves mental health", "belief", "health"),
            ("node4", "Meeting scheduled for tomorrow at 2pm", "fact", "work"),
            ("node5", "Need to review quarterly goals", "decision", "work")
        ]
        
        now = datetime.now(timezone.utc).isoformat()
        for node_id, content, node_type, domain in test_nodes:
            metadata = json.dumps({"domain": domain})
            cursor.execute("""
                INSERT INTO thought_nodes 
                (id, content, node_type, timestamp, confidence, metadata, source_file, access_count)
                VALUES (?, ?, ?, ?, 0.8, ?, 'test', 0)
            """, (node_id, content, node_type, now, metadata))
        
        # Add some edges
        test_edges = [
            ("node1", "node2", 0.7, "related_to - Both about technical work"),
            ("node4", "node5", 0.8, "leads_to - Meeting may discuss goals")
        ]
        
        for parent, child, weight, reasoning in test_edges:
            cursor.execute("""
                INSERT INTO derivation_edges (parent_id, child_id, weight, reasoning)
                VALUES (?, ?, ?, ?)
            """, (parent, child, weight, reasoning))
        
        conn.commit()
        conn.close()
        
        yield path
        
        # Cleanup
        try:
            os.unlink(path)
        except OSError:
            pass
    
    @pytest.fixture
    def mock_model_fn(self):
        """Mock model function for testing"""
        def mock_fn(prompt: str) -> str:
            if "conversation" in prompt.lower():
                return json.dumps([
                    {"content": "User prefers Python for data analysis", "type": "belief", "confidence": 0.8},
                    {"content": "Meeting went well", "type": "observation", "confidence": 0.7},
                    {"content": "Will start new project next week", "type": "decision", "confidence": 0.9}
                ])
            else:
                return json.dumps([
                    {"content": "Work and health domains are interconnected", "type": "insight", "confidence": 0.8}
                ])
        return mock_fn
    
    def test_estimate_tokens(self):
        """Test token estimation function"""
        # Test empty string
        assert _estimate_tokens("") == 0
        
        # Test short string
        assert _estimate_tokens("hello") == 1  # 5 chars / 3 = 1.67 -> 1
        
        # Test longer string
        text = "This is a longer sentence with multiple words."
        expected = len(text) // 3
        assert _estimate_tokens(text) == expected
    
    def test_start_session_no_hints(self, test_db_path):
        """Test starting session without hints returns overview context"""
        result = start_session(test_db_path, "test_session_1")
        
        assert isinstance(result, SessionContext)
        assert "GRAPH OVERVIEW" in result.context_str
        assert result.nodes_used == []  # No hint-driven nodes
        assert result.token_estimate > 0  # Overview always has tokens
    
    @patch('core.session.retrieve_recursive_bfs')
    @patch('core.embeddings.embed_nodes')
    def test_start_session_with_results(self, mock_embed, mock_retrieve, test_db_path):
        """Test starting session with retrieval results"""
        from core.retrieval import RetrievalResult
        
        # Mock retrieval results
        mock_results = [
            RetrievalResult(
                node_id="node1",
                content="This is about machine learning and AI",
                node_type="fact",
                domain="work",
                score=0.8,
                path=["node1"]
            ),
            RetrievalResult(
                node_id="node2", 
                content="Python programming is useful for data science",
                node_type="observation",
                domain="work",
                score=0.7,
                path=["node1", "node2"]
            )
        ]
        
        mock_retrieve.return_value = mock_results
        
        result = start_session(test_db_path, "test_session_2", ["machine learning", "python"])
        
        assert isinstance(result, SessionContext)
        assert len(result.nodes_used) == 2
        assert "node1" in result.nodes_used
        assert "node2" in result.nodes_used
        assert result.token_estimate > 0
        assert "RELEVANT CONTEXT" in result.context_str
        assert "machine learning" in result.context_str
        
        # Verify access tracking was updated
        conn = sqlite3.connect(test_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT access_count FROM thought_nodes WHERE id = 'node1'")
        access_count = cursor.fetchone()[0]
        assert access_count == 1
        conn.close()
    
    def test_start_session_token_budget(self, test_db_path):
        """Test that session respects token budget"""
        with patch('core.session.retrieve') as mock_retrieve:
            from core.retrieval import RetrievalResult
            
            # Create a very long content that exceeds budget
            long_content = "x" * 10000  # Very long content
            
            mock_results = [
                RetrievalResult(
                    node_id="long_node",
                    content=long_content,
                    node_type="fact",
                    domain="test", 
                    score=0.9,
                    path=["long_node"]
                )
            ]
            
            mock_retrieve.return_value = mock_results
            
            # Mock a small token budget
            with patch('core.session.get_token_budget', return_value=100):
                result = start_session(test_db_path, "test_session_budget")
                
                # Should respect budget and truncate results
                assert result.token_estimate <= 200  # Graph overview adds baseline tokens
    
    def test_create_node(self, test_db_path):
        """Test node creation"""
        content = "This is a new test node"
        node_type = "test"
        session_id = "test_create"
        
        node_id = _create_node(test_db_path, content, node_type, session_id)
        
        # Verify node was created
        conn = sqlite3.connect(test_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT content, node_type, source_file FROM thought_nodes WHERE id = ?", (node_id,))
        row = cursor.fetchone()
        conn.close()
        
        assert row is not None
        assert row[0] == content
        assert row[1] == node_type
        assert row[2] == session_id
        
        # Test creating duplicate (should return existing ID)
        duplicate_id = _create_node(test_db_path, content, node_type, session_id)
        assert duplicate_id == node_id
    
    @patch('core.embeddings.search')
    def test_find_similar_nodes(self, mock_search, test_db_path):
        """Test finding similar nodes"""
        # Mock embedding search results
        mock_search.return_value = [
            ("similar_node_1", 0.8),
            ("similar_node_2", 0.6),
            ("node1", 0.99),  # Should be filtered out (self)
            ("low_similarity", 0.1)      # Should be filtered out
        ]
        
        similar = _find_similar_nodes(test_db_path, "node1", threshold=0.5)
        
        # Should exclude self and low similarity
        assert len(similar) == 2
        assert ("similar_node_1", 0.8) in similar
        assert ("similar_node_2", 0.6) in similar
        assert all(node_id != "node1" for node_id, _ in similar)
        assert all(score >= 0.5 for _, score in similar)
    
    @patch('core.embeddings.embed_nodes')
    @patch('core.session._find_similar_nodes')
    def test_end_session_with_model(self, mock_similar, mock_embed, test_db_path, mock_model_fn):
        """Test ending session with model function"""
        mock_similar.return_value = [("node1", 0.8), ("node2", 0.7)]
        
        conversation = """
        User: I really like using Python for data analysis. 
        Assistant: That's great! Python has excellent libraries.
        User: Yes, and our meeting went well today. We decided to start a new project next week.
        """
        
        result = end_session(test_db_path, "test_end", conversation, mock_model_fn)
        
        assert isinstance(result, ExtractionResult)
        assert len(result.new_nodes) >= 1  # Should extract at least some nodes
        assert len(result.new_edges) >= 1  # Should create some edges
        
        # Verify nodes were actually created
        conn = sqlite3.connect(test_db_path)
        cursor = conn.cursor()
        
        for node_id in result.new_nodes:
            cursor.execute("SELECT content, node_type FROM thought_nodes WHERE id = ?", (node_id,))
            row = cursor.fetchone()
            assert row is not None
            
        conn.close()
    
    @patch('core.embeddings.embed_nodes')
    def test_end_session_heuristics(self, mock_embed, test_db_path):
        """Test ending session with heuristic extraction (no model)"""
        conversation = """
        I decided to learn more about machine learning. 
        I think Python is the best language for this.
        I will start with basic tutorials.
        """
        
        result = end_session(test_db_path, "test_heuristic", conversation, model_fn=None)
        
        assert isinstance(result, ExtractionResult)
        # Heuristics should extract at least the decision
        assert len(result.new_nodes) >= 1
    
    def test_end_session_minimal_content(self, test_db_path):
        """Test ending session with minimal content"""
        result = end_session(test_db_path, "test_minimal", "ok", model_fn=None)
        
        assert isinstance(result, ExtractionResult)
        assert len(result.new_nodes) == 0
        assert len(result.new_edges) == 0
    
    @patch('core.embeddings.embed_nodes')
    @patch('core.session._create_edge')
    @patch('core.session._create_node')
    def test_think_cycle(self, mock_create_node, mock_create_edge, mock_embed, test_db_path, mock_model_fn):
        """Test think cycle functionality"""
        # Mock node creation to return predictable IDs
        mock_create_node.return_value = "new_insight_node"
        
        result = think_cycle(test_db_path, mock_model_fn)
        
        assert isinstance(result, ThinkResult)
        assert "cluster" in result.cluster_topic.lower()
        
        # Should call model function and create nodes
        mock_create_node.assert_called()
        mock_create_edge.assert_called()
    
    def test_think_cycle_with_domain_focus(self, test_db_path, mock_model_fn):
        """Test think cycle with domain focus"""
        with patch('core.session._create_node') as mock_create_node:
            mock_create_node.return_value = "domain_insight"
            
            result = think_cycle(test_db_path, mock_model_fn, focus_domain="work")
            
            assert isinstance(result, ThinkResult)
            # With few test nodes, clustering may not find domain-specific clusters
            assert result.cluster_topic is not None
    
    def test_think_cycle_no_nodes(self, test_db_path, mock_model_fn):
        """Test think cycle when no nodes available"""
        # Create empty database
        conn = sqlite3.connect(test_db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM thought_nodes")
        conn.commit()
        conn.close()
        
        result = think_cycle(test_db_path, mock_model_fn)
        
        assert isinstance(result, ThinkResult)
        assert result.cluster_topic == "No cluster found"
        assert len(result.new_nodes) == 0
    
    def test_config_integration(self):
        """Test that session functions use configuration properly"""
        # Test with custom config
        os.environ['CASHEW_TOKEN_BUDGET'] = '1500'
        os.environ['CASHEW_TOP_K'] = '15'
        os.environ['CASHEW_WALK_DEPTH'] = '3'
        
        from core.config import reload_config, get_token_budget, get_top_k, get_walk_depth
        reload_config()
        
        assert get_token_budget() == 1500
        assert get_top_k() == 15
        assert get_walk_depth() == 3
        
        # Cleanup
        del os.environ['CASHEW_TOKEN_BUDGET']
        del os.environ['CASHEW_TOP_K'] 
        del os.environ['CASHEW_WALK_DEPTH']
        reload_config()

class TestCashewConfig:
    """Test suite for configuration management"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = CashewConfig()
        
        assert config.token_budget == 2000
        assert config.top_k == 10
        assert config.walk_depth == 2
        assert config.embedding_model == 'all-MiniLM-L6-v2'
    
    def test_env_override(self):
        """Test environment variable overrides"""
        os.environ['CASHEW_TOKEN_BUDGET'] = '3000'
        os.environ['CASHEW_TOP_K'] = '20'
        os.environ['CASHEW_EMBEDDING_MODEL'] = 'different-model'
        
        config = CashewConfig()
        
        assert config.token_budget == 3000
        assert config.top_k == 20
        assert config.embedding_model == 'different-model'
        
        # Cleanup
        del os.environ['CASHEW_TOKEN_BUDGET']
        del os.environ['CASHEW_TOP_K']
        del os.environ['CASHEW_EMBEDDING_MODEL']
    
    def test_config_validation(self):
        """Test configuration validation"""
        os.environ['CASHEW_TOKEN_BUDGET'] = '-100'
        
        with pytest.raises(ValueError):
            CashewConfig()
        
        del os.environ['CASHEW_TOKEN_BUDGET']
    
    def test_scoring_weights(self):
        """Test scoring weight calculation"""
        config = CashewConfig()
        weights = config.get_scoring_weights()
        
        assert 'embedding' in weights
        assert 'access' in weights  
        assert 'temporal' in weights
        
        # Weights should sum to 1.0
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.001
    
    def test_scoring_weights_invalid(self):
        """Test scoring weights validation — weights > 1.0 should raise ValueError"""
        os.environ['CASHEW_ACCESS_WEIGHT'] = '0.8'
        os.environ['CASHEW_TEMPORAL_WEIGHT'] = '0.5'  # Total > 1.0
        
        try:
            with pytest.raises(ValueError):
                CashewConfig()  # Validation fires in __init__
        finally:
            # Cleanup
            del os.environ['CASHEW_ACCESS_WEIGHT']
            del os.environ['CASHEW_TEMPORAL_WEIGHT']

    def test_extractor_types_match_config(self):
        """All extractor _VALID_TYPES must be present in default config node types."""
        from extractors.sessions import SessionExtractor
        from extractors.markdown_dir import MarkdownDirExtractor
        from extractors.obsidian import ObsidianExtractor

        config = CashewConfig()
        configured = config.node_type_names

        for extractor_cls in (SessionExtractor, MarkdownDirExtractor, ObsidianExtractor):
            if hasattr(extractor_cls, '_VALID_TYPES'):
                for t in extractor_cls._VALID_TYPES:
                    assert t in configured, (
                        f"{extractor_cls.__name__}._VALID_TYPES contains '{t}' "
                        f"which is not in CashewConfig default node types"
                    )


class TestSessionDataClasses:
    """Test the dataclass structures"""
    
    def test_session_context(self):
        """Test SessionContext dataclass"""
        context = SessionContext(
            context_str="test context",
            nodes_used=["node1", "node2"],
            token_estimate=100
        )
        
        result = context.to_dict()
        assert result['context_str'] == "test context"
        assert result['nodes_used'] == ["node1", "node2"]
        assert result['token_estimate'] == 100
    
    def test_extraction_result(self):
        """Test ExtractionResult dataclass"""
        result = ExtractionResult(
            new_nodes=["new1", "new2"],
            new_edges=[("parent", "child", "reasoning")],
            updated_nodes=["updated1"]
        )
        
        dict_result = result.to_dict()
        assert dict_result['new_nodes'] == ["new1", "new2"]
        assert len(dict_result['new_edges']) == 1
    
    def test_think_result(self):
        """Test ThinkResult dataclass"""
        result = ThinkResult(
            new_nodes=["insight1"],
            new_edges=[("source", "insight", "derived")],
            cluster_topic="test cluster"
        )
        
        dict_result = result.to_dict()
        assert dict_result['cluster_topic'] == "test cluster"
        assert len(dict_result['new_nodes']) == 1

if __name__ == "__main__":
    pytest.main([__file__, "-v"])