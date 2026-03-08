#!/usr/bin/env python3
"""
Tests for cashew.core.traversal module
"""

import pytest
import sqlite3
import json
import tempfile
import os
from typing import List, Dict

import sys
sys.path.insert(0, '/Users/bunny/.openclaw/workspace/cashew')

from core.traversal import TraversalEngine, ThoughtNode, DerivationEdge, AuditReport

class TestTraversalEngine:
    """Test the traversal engine functionality"""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary test database"""
        fd, db_path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        
        # Create schema and test data
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE thought_nodes (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                node_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                confidence REAL NOT NULL,
                mood_state TEXT,
                metadata TEXT,
                source_file TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE derivation_edges (
                parent_id TEXT NOT NULL,
                child_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                weight REAL NOT NULL,
                reasoning TEXT,
                FOREIGN KEY (parent_id) REFERENCES thought_nodes(id),
                FOREIGN KEY (child_id) REFERENCES thought_nodes(id),
                PRIMARY KEY (parent_id, child_id, relation)
            )
        """)
        
        # Insert test data
        nodes = [
            ("seed1", "God exists", "seed", "2023-01-01T00:00:00", 1.0, "certain", "{}", "test"),
            ("belief1", "Prayer works", "belief", "2023-01-02T00:00:00", 0.8, "hopeful", "{}", "test"),
            ("derived1", "Therefore I should pray", "derived", "2023-01-03T00:00:00", 0.7, "confident", "{}", "test"),
            ("question1", "Why doesn't prayer always work?", "question", "2023-01-04T00:00:00", 0.6, "curious", "{}", "test"),
            ("derived2", "Prayer outcomes seem random", "derived", "2023-01-05T00:00:00", 0.5, "doubtful", "{}", "test"),
            ("orphan1", "Isolated thought", "derived", "2023-01-06T00:00:00", 0.4, "neutral", "{}", "test"),
            ("weak1", "Weakly derived thought", "derived", "2023-01-07T00:00:00", 0.3, "uncertain", "{}", "test")
        ]
        
        cursor.executemany("""
            INSERT INTO thought_nodes 
            (id, content, node_type, timestamp, confidence, mood_state, metadata, source_file)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, nodes)
        
        # Create edges to form derivation chains
        edges = [
            ("seed1", "belief1", "supports", 0.9, "Core belief supports prayer"),
            ("belief1", "derived1", "derived_from", 0.8, "Logical conclusion from belief"),
            ("derived1", "question1", "questions", 0.7, "Observation leads to question"),
            ("question1", "derived2", "derived_from", 0.6, "Question leads to new understanding"),
            ("belief1", "derived2", "contradicts", 0.5, "New understanding contradicts old belief"),
            # Add a weak chain for testing
            ("seed1", "weak1", "supports", 0.1, "Very weak connection")
        ]
        
        cursor.executemany("""
            INSERT INTO derivation_edges 
            (parent_id, child_id, relation, weight, reasoning)
            VALUES (?, ?, ?, ?, ?)
        """, edges)
        
        conn.commit()
        conn.close()
        
        yield db_path
        
        # Cleanup
        os.unlink(db_path)
    
    def test_why_returns_derivation_chain(self, temp_db):
        """Test that why() returns correct derivation chain"""
        engine = TraversalEngine(temp_db)
        
        # Test derivation chain for derived2
        chain = engine.why("derived2")
        
        assert len(chain) == 1
        assert chain[0]["node"]["id"] == "derived2"
        assert chain[0]["node"]["content"] == "Prayer outcomes seem random"
        assert "derived_from" in chain[0]
        assert len(chain[0]["derived_from"]) == 2  # Two parents: question1 and belief1 (contradiction)
        
        # Check that it traces back to seed
        found_seed = False
        def check_for_seed(step):
            nonlocal found_seed
            if step.get("is_seed", False):
                found_seed = True
                return
            for derivation in step.get("derived_from", []):
                for parent_step in derivation.get("parent_chain", []):
                    check_for_seed(parent_step)
        
        check_for_seed(chain[0])
        assert found_seed, "Should trace back to seed node"
    
    def test_why_handles_orphan_nodes(self, temp_db):
        """Test that why() handles orphan nodes correctly"""
        engine = TraversalEngine(temp_db)
        
        chain = engine.why("orphan1")
        
        assert len(chain) == 1
        assert chain[0]["node"]["id"] == "orphan1"
        assert chain[0]["is_seed"] == False  # Not a seed, just has no parents
        assert "derived_from" not in chain[0] or len(chain[0].get("derived_from", [])) == 0
    
    def test_how_finds_shortest_path(self, temp_db):
        """Test that how() finds shortest path between nodes"""
        engine = TraversalEngine(temp_db)
        
        # Test path from seed1 to derived2
        path = engine.how("seed1", "derived2")
        
        assert path is not None
        assert len(path) >= 2
        assert path[0]["node"]["id"] == "seed1"
        assert path[-1]["node"]["id"] == "derived2"
        
        # Check distances are increasing
        for i in range(1, len(path)):
            assert path[i]["distance"] == path[i-1]["distance"] + 1
    
    def test_how_returns_none_for_disconnected_nodes(self, temp_db):
        """Test that how() returns None for disconnected nodes"""
        engine = TraversalEngine(temp_db)
        
        # orphan1 is not connected to the main graph
        path = engine.how("seed1", "orphan1")
        assert path is None
        
        path = engine.how("orphan1", "seed1")
        assert path is None
    
    def test_audit_detects_contradictions(self, temp_db):
        """Test that audit() detects contradictions"""
        engine = TraversalEngine(temp_db)
        
        report = engine.audit()
        
        # Should detect the contradiction between belief1 and derived2
        assert len(report.contradictions) == 1
        contradiction = report.contradictions[0]
        assert "Prayer works" in contradiction[0] or "Prayer works" in contradiction[1]
        assert "Prayer outcomes seem random" in contradiction[0] or "Prayer outcomes seem random" in contradiction[1]
    
    def test_audit_detects_orphan_nodes(self, temp_db):
        """Test that audit() detects orphan nodes"""
        engine = TraversalEngine(temp_db)
        
        report = engine.audit()
        
        # Should detect orphan1 as an orphan (no parents, not a seed)
        assert len(report.orphan_nodes) == 1
        assert "orphan1" in report.orphan_nodes
    
    def test_audit_detects_weak_chains(self, temp_db):
        """Test that audit() detects weak derivation chains"""
        engine = TraversalEngine(temp_db)
        
        report = engine.audit()
        
        # Should identify nodes with weak derivation chains
        assert len(report.weak_chains) > 0
        
        # Weak chains should be sorted by weight (weakest first)
        if len(report.weak_chains) > 1:
            assert report.weak_chains[0][1] <= report.weak_chains[1][1]
    
    def test_audit_handles_cycles(self, temp_db):
        """Test that audit() detects cycles"""
        # Add a cycle to the test database
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        
        # Create a cycle: derived2 -> belief1 (which already goes belief1 -> derived1 -> question1 -> derived2)
        cursor.execute("""
            INSERT INTO derivation_edges 
            (parent_id, child_id, relation, weight, reasoning)
            VALUES ('derived2', 'belief1', 'questions', 0.3, 'Circular reasoning test')
        """)
        conn.commit()
        conn.close()
        
        engine = TraversalEngine(temp_db)
        report = engine.audit()
        
        # Should detect the cycle
        assert len(report.cycles) > 0
        
        # Check that the cycle contains the expected nodes
        cycle_found = False
        for cycle in report.cycles:
            if "belief1" in cycle and "derived2" in cycle:
                cycle_found = True
                break
        
        assert cycle_found, "Should detect cycle involving belief1 and derived2"
    
    def test_load_node_returns_correct_data(self, temp_db):
        """Test that _load_node returns correct node data"""
        engine = TraversalEngine(temp_db)
        
        node = engine._load_node("seed1")
        
        assert node is not None
        assert node.id == "seed1"
        assert node.content == "God exists"
        assert node.node_type == "seed"
        assert node.confidence == 1.0
        assert node.mood_state == "certain"
    
    def test_load_node_returns_none_for_missing_node(self, temp_db):
        """Test that _load_node returns None for missing nodes"""
        engine = TraversalEngine(temp_db)
        
        node = engine._load_node("nonexistent")
        assert node is None
    
    def test_get_parents_returns_correct_edges(self, temp_db):
        """Test that _get_parents returns correct parent relationships"""
        engine = TraversalEngine(temp_db)
        
        parents = engine._get_parents("derived2")
        
        assert len(parents) == 2  # Has two parents: question1 and belief1
        
        parent_ids = [parent.id for parent, edge in parents]
        assert "question1" in parent_ids
        assert "belief1" in parent_ids
        
        # Check edge information
        for parent, edge in parents:
            assert edge.child_id == "derived2"
            assert edge.parent_id == parent.id
            if parent.id == "question1":
                assert edge.relation == "derived_from"
            elif parent.id == "belief1":
                assert edge.relation == "contradicts"
    
    def test_get_children_returns_correct_ids(self, temp_db):
        """Test that _get_children returns correct child node IDs"""
        engine = TraversalEngine(temp_db)
        
        children = engine._get_children("belief1")
        
        assert len(children) == 2  # Has two children: derived1 and derived2
        assert "derived1" in children
        assert "derived2" in children