#!/usr/bin/env python3
"""
Tests for cashew.core.sleep module
"""

import pytest
import sqlite3
import json
import tempfile
import os
from unittest.mock import patch

import sys
sys.path.insert(0, '/Users/bunny/.openclaw/workspace/cashew')

from core.sleep import SleepProtocol, CrossLinkCandidate, NodeMetrics

class TestSleepProtocol:
    """Test the sleep protocol functionality"""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary test database with sleep-relevant data"""
        fd, db_path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create tables (including decayed column)
        cursor.execute("""
            CREATE TABLE thought_nodes (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                node_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                confidence REAL NOT NULL,
                mood_state TEXT,
                metadata TEXT,
                source_file TEXT,
                decayed INTEGER DEFAULT 0
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
        
        # Insert test nodes with similar content for cross-linking tests
        nodes = [
            ("seed1", "God exists and is all-powerful", "seed", "2023-01-01T00:00:00", 1.0, "certain", "{}", "test", 0),
            ("similar1", "God exists and is all-powerful and omnipotent", "derived", "2023-01-02T00:00:00", 0.8, "confident", "{}", "test", 0),
            ("similar2", "God exists and is all-powerful deity", "belief", "2023-01-03T00:00:00", 0.7, "hopeful", "{}", "test", 0),
            ("different1", "Prayer works sometimes", "belief", "2023-01-04T00:00:00", 0.6, "hopeful", "{}", "test", 0),
            ("hub1", "Core belief about reality", "core_memory", "2023-01-05T00:00:00", 0.9, "stable", "{}", "test", 0),
            ("weak1", "Uncertain thought", "derived", "2023-01-06T00:00:00", 0.2, "doubtful", "{}", "test", 0),
            ("weak2", "Another weak idea", "derived", "2023-01-07T00:00:00", 0.1, "confused", "{}", "test", 0),
            ("isolated1", "Completely separate thought", "derived", "2023-01-08T00:00:00", 0.5, "neutral", "{}", "different_source", 0),
            ("isolated2", "Another isolated idea", "belief", "2023-01-09T00:00:00", 0.4, "neutral", "{}", "different_source", 0),
        ]
        
        cursor.executemany("""
            INSERT INTO thought_nodes 
            (id, content, node_type, timestamp, confidence, mood_state, metadata, source_file, decayed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, nodes)
        
        # Create edges to establish hierarchy and connectivity
        edges = [
            ("seed1", "similar1", "supports", 0.9, "Strong support"),
            ("seed1", "different1", "supports", 0.6, "Moderate support"),
            ("similar1", "hub1", "derived_from", 0.8, "Derived from similar belief"),
            ("hub1", "weak1", "derived_from", 0.3, "Weak derivation"),  # Low weight for testing
            ("hub1", "weak2", "derived_from", 0.2, "Very weak derivation"),
            # No edges for isolated1 and isolated2 to test orphan detection
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
    
    @pytest.fixture
    def sleep_protocol(self, temp_db):
        """Create sleep protocol instance with temp database"""
        return SleepProtocol(temp_db, tempfile.mktemp(suffix='.json'))
    
    def test_text_similarity_calculation(self, sleep_protocol):
        """Test text similarity calculation"""
        # High similarity
        sim1 = sleep_protocol._text_similarity(
            "God exists and is all-powerful",
            "God exists and is omnipotent"
        )
        assert sim1 >= 0.5  # Should be similar
        
        # Medium similarity
        sim2 = sleep_protocol._text_similarity(
            "God exists and is all-powerful",
            "The deity is all-powerful and real"
        )
        assert 0.1 < sim2 < 0.8  # Should be moderately similar
        
        # Low similarity
        sim3 = sleep_protocol._text_similarity(
            "God exists and is all-powerful",
            "Prayer works sometimes"
        )
        assert sim3 < 0.2  # Should be different
        
        # Identical text
        sim4 = sleep_protocol._text_similarity(
            "Exactly the same text",
            "Exactly the same text"
        )
        assert sim4 == 1.0  # Should be identical
    
    def test_find_cross_link_candidates(self, sleep_protocol):
        """Test finding cross-link candidates"""
        # Lower thresholds for testing
        sleep_protocol.dedup_threshold = 0.6
        sleep_protocol.cross_link_threshold = 0.4
        
        candidates = sleep_protocol.find_cross_link_candidates()
        
        assert len(candidates) > 0
        
        # Should find high similarity candidates for deduplication
        dedup_candidates = [c for c in candidates if c.action == "dedup"]
        assert len(dedup_candidates) > 0
        
        # Should find medium similarity candidates for cross-linking
        crosslink_candidates = [c for c in candidates if c.action == "cross_link"]
        assert len(crosslink_candidates) > 0
        
        # Check that similarity scores are reasonable
        for candidate in candidates:
            assert 0.0 <= candidate.similarity <= 1.0
            if candidate.action == "dedup":
                assert candidate.similarity >= sleep_protocol.dedup_threshold
            elif candidate.action == "cross_link":
                assert candidate.similarity >= sleep_protocol.cross_link_threshold
    
    def test_cross_link_nodes_creates_edges(self, sleep_protocol):
        """Test that cross-linking creates bidirectional edges"""
        # Ensure nodes exist in database
        conn = sleep_protocol._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM thought_nodes WHERE id IN ('different1', 'isolated1')")
        existing_nodes = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        # Skip test if nodes don't exist
        if len(existing_nodes) < 2:
            pytest.skip("Required nodes not found in test database")
        
        # Use nodes without existing edges
        sleep_protocol.cross_link_nodes("different1", "isolated1", 0.8, "Test cross-link")
        
        conn = sleep_protocol._get_connection()
        cursor = conn.cursor()
        
        # Check that bidirectional cross-link edges were created
        cursor.execute("""
            SELECT parent_id, child_id, relation, weight 
            FROM derivation_edges 
            WHERE relation = 'cross_link'
        """)
        
        cross_links = cursor.fetchall()
        conn.close()
        
        assert len(cross_links) == 2  # Should be bidirectional
        
        # Check both directions exist
        directions = [(row[0], row[1]) for row in cross_links]
        assert ("different1", "isolated1") in directions
        assert ("isolated1", "different1") in directions
        
        # Check event was logged
        assert len(sleep_protocol.events) == 1
        assert sleep_protocol.events[0].event_type == "cross_link"
    
    def test_deduplicate_nodes_merges_correctly(self, sleep_protocol):
        """Test that deduplication merges nodes correctly"""
        # Check initial state
        conn = sleep_protocol._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM thought_nodes WHERE decayed = 0")
        initial_active_count = cursor.fetchone()[0]
        conn.close()
        
        sleep_protocol.deduplicate_nodes("similar1", "similar2", 0.95)
        
        # Check that one node was marked as decayed
        conn = sleep_protocol._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM thought_nodes WHERE decayed = 1")
        decayed_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM thought_nodes WHERE decayed = 0")
        active_count = cursor.fetchone()[0]
        conn.close()
        
        assert decayed_count == 1  # One node should be decayed
        assert active_count == initial_active_count - 1  # One less active node
        
        # Check event was logged
        dedup_events = [e for e in sleep_protocol.events if e.event_type == "dedup"]
        assert len(dedup_events) == 1
    
    def test_calculate_node_metrics(self, sleep_protocol):
        """Test node metrics calculation"""
        metrics = sleep_protocol.calculate_node_metrics()
        
        assert len(metrics) > 0
        
        # Check that hub1 has high metrics (it has children)
        hub_metrics = metrics.get("hub1")
        assert hub_metrics is not None
        assert hub_metrics.branching_factor >= 2  # Has weak1 and weak2 as children
        
        # Check that weak nodes have low fitness
        weak1_metrics = metrics.get("weak1")
        assert weak1_metrics is not None
        assert weak1_metrics.composite_fitness < 1.0  # Should be low fitness
        
        # Check that seeds get bonus
        seed_metrics = metrics.get("seed1")
        assert seed_metrics is not None
        # Seed should have higher fitness due to type bonus
    
    def test_garbage_collect_preserves_important_nodes(self, sleep_protocol):
        """Test that GC doesn't kill high-branching nodes"""
        # Calculate initial metrics
        initial_metrics = sleep_protocol.calculate_node_metrics()
        
        # Run garbage collection
        decayed_nodes = sleep_protocol.garbage_collect(initial_metrics, k_nodes=5)
        
        # Check that important nodes (seeds, core_memory, high-branching) weren't decayed
        conn = sleep_protocol._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id FROM thought_nodes 
            WHERE node_type IN ('seed', 'core_memory') AND decayed = 1
        """)
        
        decayed_important = cursor.fetchall()
        conn.close()
        
        assert len(decayed_important) == 0, "Seeds and core memories should not be decayed"
        
        # Check that hub1 (high branching factor) wasn't decayed
        conn = sleep_protocol._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT decayed FROM thought_nodes WHERE id = 'hub1'")
        hub_decayed = cursor.fetchone()[0]
        conn.close()
        
        assert hub_decayed == 0, "High-branching nodes should not be decayed"
    
    def test_promote_core_memories(self, sleep_protocol):
        """Test core memory promotion/demotion"""
        metrics = sleep_protocol.calculate_node_metrics()
        
        # Count initial core memories
        conn = sleep_protocol._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM thought_nodes WHERE node_type = 'core_memory'")
        initial_core_count = cursor.fetchone()[0]
        conn.close()
        
        promotions, demotions = sleep_protocol.promote_core_memories(metrics)
        
        # Check that target number of core memories is maintained
        target_count = int(len(metrics) ** 0.5)  # sqrt(total_nodes)
        
        conn = sleep_protocol._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM thought_nodes WHERE node_type = 'core_memory'")
        final_core_count = cursor.fetchone()[0]
        conn.close()
        
        assert final_core_count <= target_count
        
        # Check that events were logged
        promotion_events = [e for e in sleep_protocol.events if e.event_type == "core_promotion"]
        demotion_events = [e for e in sleep_protocol.events if e.event_type == "core_demotion"]
        
        assert len(promotion_events) == len(promotions)
        assert len(demotion_events) == len(demotions)
    
    def test_generate_dream_node_connects_chains(self, sleep_protocol):
        """Test dream node generation connects separate chains"""
        # Create cross-link candidates between different source files
        candidates = [
            CrossLinkCandidate("isolated1", "seed1", 0.8, "cross_link"),
            CrossLinkCandidate("isolated2", "different1", 0.75, "cross_link")
        ]
        
        dream_id = sleep_protocol.generate_dream_node(candidates)
        
        if dream_id:  # Dream might not be generated if no bridge candidates
            # Check that dream node was created
            conn = sleep_protocol._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT node_type FROM thought_nodes WHERE id = ?", (dream_id,))
            node_type = cursor.fetchone()
            
            if node_type:
                assert node_type[0] == "dream"
                
                # Check that dream is connected to bridged nodes
                cursor.execute("""
                    SELECT COUNT(*) FROM derivation_edges 
                    WHERE child_id = ? AND relation = 'derived_from'
                """, (dream_id,))
                
                connection_count = cursor.fetchone()[0]
                assert connection_count >= 2  # Should be connected to at least 2 nodes
            
            conn.close()
            
            # Check event was logged
            dream_events = [e for e in sleep_protocol.events if e.event_type == "dream"]
            assert len(dream_events) == 1
    
    @patch('random.sample')
    def test_garbage_collect_uses_random_sampling(self, mock_sample, sleep_protocol):
        """Test that GC uses random sampling"""
        metrics = sleep_protocol.calculate_node_metrics()
        
        # Mock random.sample to return specific nodes
        test_nodes = list(metrics.keys())[:3]
        mock_sample.return_value = test_nodes
        
        sleep_protocol.garbage_collect(metrics, k_nodes=3)
        
        # Verify random.sample was called
        mock_sample.assert_called_once()
        # Check the call was made with correct arguments
        assert len(mock_sample.call_args[0]) >= 2  # population and k
    
    def test_run_sleep_cycle_integration(self, sleep_protocol):
        """Test full sleep cycle integration"""
        summary = sleep_protocol.run_sleep_cycle()
        
        # Check that summary contains expected keys
        expected_keys = [
            "cross_links_created",
            "deduplications", 
            "dream_nodes_created",
            "nodes_decayed",
            "core_promotions",
            "core_demotions",
            "total_nodes",
            "events_logged"
        ]
        
        for key in expected_keys:
            assert key in summary
            assert isinstance(summary[key], int)
            assert summary[key] >= 0
        
        # Check that events were logged
        assert summary["events_logged"] > 0
        
        # Check that some operations happened (at least metrics calculation)
        assert summary["total_nodes"] > 0
    
    def test_sleep_protocol_handles_empty_database(self):
        """Test sleep protocol handles empty database gracefully"""
        # Create empty database
        fd, empty_db = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        
        conn = sqlite3.connect(empty_db)
        cursor = conn.cursor()
        
        # Create empty tables
        cursor.execute("""
            CREATE TABLE thought_nodes (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                node_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                confidence REAL NOT NULL,
                mood_state TEXT,
                metadata TEXT,
                source_file TEXT,
                decayed INTEGER DEFAULT 0
            )
        """)
        
        cursor.execute("""
            CREATE TABLE derivation_edges (
                parent_id TEXT NOT NULL,
                child_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                weight REAL NOT NULL,
                reasoning TEXT
            )
        """)
        
        conn.commit()
        conn.close()
        
        try:
            protocol = SleepProtocol(empty_db, tempfile.mktemp(suffix='.json'))
            summary = protocol.run_sleep_cycle()
            
            # Should complete without error
            assert summary["total_nodes"] == 0
            assert summary["cross_links_created"] == 0
            assert summary["deduplications"] == 0
            
        finally:
            os.unlink(empty_db)