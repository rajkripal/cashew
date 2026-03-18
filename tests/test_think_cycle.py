"""Tests for core/think_cycle.py — random sampling + LLM via model_fn"""

import sqlite3
import json
import pytest
import random
from unittest.mock import MagicMock
from core.think_cycle import ThinkCycle, NewThought


@pytest.fixture
def temp_db(tmp_path):
    """Create a minimal test DB with thought_nodes and derivation_edges."""
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE thought_nodes (
            id TEXT PRIMARY KEY,
            content TEXT,
            node_type TEXT DEFAULT 'concept',
            timestamp TEXT,
            confidence REAL DEFAULT 0.8,
            source_file TEXT DEFAULT 'test',
            last_updated TEXT,
            domain TEXT DEFAULT 'bunny',
            metadata TEXT DEFAULT '{}'
        )
    """)
    cursor.execute("""
        CREATE TABLE derivation_edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            parent_id TEXT,
            child_id TEXT,
            weight REAL DEFAULT 0.8,
            reasoning TEXT DEFAULT ''
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS node_embeddings (
            node_id TEXT PRIMARY KEY,
            embedding BLOB
        )
    """)
    conn.commit()
    conn.close()
    return db_path


def _populate_db(db_path, n_nodes=20, n_edges=30):
    """Populate DB with nodes and edges for testing."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    for i in range(n_nodes):
        node_type = 'question' if i % 5 == 0 else 'concept'
        cursor.execute("""
            INSERT INTO thought_nodes (id, content, node_type, timestamp, confidence, domain)
            VALUES (?, ?, ?, datetime('now'), 0.8, 'bunny')
        """, (f"node_{i}", f"Test node content {i} about topic {i % 3}", node_type))
    
    # Create edges — only first half of nodes get outgoing edges (rest are leaves)
    hub_count = n_nodes // 2
    for i in range(n_edges):
        parent = f"node_{i % hub_count}"
        child = f"node_{(i + 3) % n_nodes}"
        reasoning = "contradicts" if i % 7 == 0 else "relates to"
        cursor.execute("""
            INSERT INTO derivation_edges (parent_id, child_id, weight, reasoning)
            VALUES (?, ?, 0.8, ?)
        """, (parent, child, reasoning))
    
    conn.commit()
    conn.close()


class TestRandomSampling:
    """Test that analyze_graph_structure uses random sampling."""
    
    def test_no_crash_on_empty_db(self, temp_db):
        tc = ThinkCycle(db_path=temp_db)
        analysis = tc.analyze_graph_structure()
        assert analysis['hub_nodes'] == []
        assert analysis['leaf_nodes'] == []
        assert analysis['question_nodes'] == []
        assert analysis['contradiction_nodes'] == []
    
    def test_returns_nodes_from_populated_db(self, temp_db):
        _populate_db(temp_db)
        tc = ThinkCycle(db_path=temp_db)
        analysis = tc.analyze_graph_structure()
        # Should have some results in each category
        assert len(analysis['hub_nodes']) > 0
        assert len(analysis['leaf_nodes']) > 0
    
    def test_random_sampling_produces_different_results(self, temp_db):
        """Run analyze twice — with enough nodes, results should differ."""
        _populate_db(temp_db, n_nodes=50, n_edges=80)
        tc = ThinkCycle(db_path=temp_db)
        
        # Run many times and check we get at least 2 unique samplings
        all_leaf_sets = []
        for seed in range(20):
            random.seed(seed)
            analysis = tc.analyze_graph_structure()
            ids = frozenset(n[0] for n in analysis['leaf_nodes'])
            all_leaf_sets.append(ids)
        
        unique = set(all_leaf_sets)
        assert len(unique) > 1, "Random sampling should produce different results across seeds"
    
    def test_respects_sample_sizes(self, temp_db):
        _populate_db(temp_db, n_nodes=50, n_edges=100)
        tc = ThinkCycle(db_path=temp_db)
        analysis = tc.analyze_graph_structure(k_hubs=3, k_leaves=4, k_questions=2)
        assert len(analysis['hub_nodes']) <= 3
        assert len(analysis['leaf_nodes']) <= 4
        assert len(analysis['question_nodes']) <= 2
    
    def test_hub_query_filters_low_degree(self, temp_db):
        """Hubs should only include nodes with edge_count > 2."""
        _populate_db(temp_db, n_nodes=10, n_edges=5)
        tc = ThinkCycle(db_path=temp_db)
        analysis = tc.analyze_graph_structure()
        for hub in analysis['hub_nodes']:
            assert hub[3] > 2, f"Hub node {hub[0]} has edge_count {hub[3]} <= 2"


class TestModelFn:
    """Test model_fn wiring and graceful fallback."""
    
    def test_no_model_fn_returns_empty(self, temp_db):
        _populate_db(temp_db)
        tc = ThinkCycle(db_path=temp_db)
        insights = tc.generate_insights_from_analysis(
            tc.analyze_graph_structure(), model_fn=None
        )
        assert insights == []
    
    def test_model_fn_called_for_each_category(self, temp_db):
        _populate_db(temp_db, n_nodes=30, n_edges=60)
        tc = ThinkCycle(db_path=temp_db)
        analysis = tc.analyze_graph_structure()
        
        mock_fn = MagicMock(return_value="INSIGHT: Test insight\nCONFIDENCE: 0.7\nREASONING: Test\n---")
        tc.generate_insights_from_analysis(analysis, model_fn=mock_fn)
        
        # model_fn should be called at least once (for whichever categories have nodes)
        assert mock_fn.call_count > 0
    
    def test_model_fn_exception_handled_gracefully(self, temp_db):
        _populate_db(temp_db, n_nodes=30, n_edges=60)
        tc = ThinkCycle(db_path=temp_db)
        analysis = tc.analyze_graph_structure()
        
        mock_fn = MagicMock(side_effect=Exception("LLM exploded"))
        insights = tc.generate_insights_from_analysis(analysis, model_fn=mock_fn)
        # Should not raise, just return empty or partial
        assert isinstance(insights, list)


class TestParseLLMInsights:
    """Test _parse_llm_insights parsing."""
    
    def test_single_insight(self, temp_db):
        tc = ThinkCycle(db_path=temp_db)
        response = "INSIGHT: Something interesting\nCONFIDENCE: 0.75\nREASONING: Because reasons\n---"
        results = tc._parse_llm_insights(response, ["node_1", "node_2"])
        assert len(results) == 1
        assert "Something interesting" in results[0].content
        assert results[0].confidence == 0.75
        assert results[0].reasoning == "Because reasons"
    
    def test_multiple_insights(self, temp_db):
        tc = ThinkCycle(db_path=temp_db)
        response = """INSIGHT: First insight
CONFIDENCE: 0.6
REASONING: Reason one
---
INSIGHT: Second insight
CONFIDENCE: 0.8
REASONING: Reason two
---"""
        results = tc._parse_llm_insights(response, ["node_1"])
        assert len(results) == 2
    
    def test_confidence_clamping(self, temp_db):
        tc = ThinkCycle(db_path=temp_db)
        response = "INSIGHT: Too confident\nCONFIDENCE: 1.5\nREASONING: Hubris\n---"
        results = tc._parse_llm_insights(response, ["node_1"])
        assert results[0].confidence == 0.95  # clamped
        
        response2 = "INSIGHT: Too shy\nCONFIDENCE: 0.01\nREASONING: Timid\n---"
        results2 = tc._parse_llm_insights(response2, ["node_1"])
        assert results2[0].confidence == 0.1  # clamped
    
    def test_prefix_injection(self, temp_db):
        tc = ThinkCycle(db_path=temp_db)
        response = "INSIGHT: No prefix here\nCONFIDENCE: 0.6\nREASONING: Test\n---"
        results = tc._parse_llm_insights(response, ["node_1"], prefix="[cross-domain insight]")
        assert results[0].content.startswith("[cross-domain insight]")
    
    def test_existing_prefix_not_doubled(self, temp_db):
        tc = ThinkCycle(db_path=temp_db)
        response = "INSIGHT: [think cycle] Already prefixed\nCONFIDENCE: 0.6\nREASONING: Test\n---"
        results = tc._parse_llm_insights(response, ["node_1"], prefix="[think cycle]")
        assert not results[0].content.startswith("[think cycle] [think cycle]")
    
    def test_bad_confidence_uses_default(self, temp_db):
        tc = ThinkCycle(db_path=temp_db)
        response = "INSIGHT: Bad number\nCONFIDENCE: not_a_number\nREASONING: Test\n---"
        results = tc._parse_llm_insights(response, ["node_1"])
        assert results[0].confidence == 0.6  # default
    
    def test_empty_response(self, temp_db):
        tc = ThinkCycle(db_path=temp_db)
        results = tc._parse_llm_insights("", ["node_1"])
        assert results == []
    
    def test_no_connections_response(self, temp_db):
        tc = ThinkCycle(db_path=temp_db)
        results = tc._parse_llm_insights("NO_CONNECTIONS", ["node_1"])
        assert results == []  # no INSIGHT: line found
    
    def test_parent_ids_capped_at_3(self, temp_db):
        tc = ThinkCycle(db_path=temp_db)
        response = "INSIGHT: Test\nCONFIDENCE: 0.6\nREASONING: Test\n---"
        results = tc._parse_llm_insights(response, ["n1", "n2", "n3", "n4", "n5"])
        assert len(results[0].parent_ids) <= 3


class TestRunThinkCycle:
    """Test the full run_think_cycle path."""
    
    def test_run_without_model_fn(self, temp_db):
        tc = ThinkCycle(db_path=temp_db)
        # Need the stats functions to work — add the source_file column check
        result = tc.run_think_cycle(model_fn=None)
        assert result['new_thoughts_generated'] == 0
    
    def test_run_with_model_fn(self, temp_db):
        _populate_db(temp_db, n_nodes=30, n_edges=60)
        tc = ThinkCycle(db_path=temp_db)
        
        mock_fn = MagicMock(return_value="INSIGHT: A genuine new insight about patterns\nCONFIDENCE: 0.7\nREASONING: Cross-cluster analysis\n---")
        result = tc.run_think_cycle(model_fn=mock_fn)
        
        assert 'new_thoughts_generated' in result
        assert 'initial_count' in result
        assert 'final_count' in result
