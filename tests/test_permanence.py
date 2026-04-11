#!/usr/bin/env python3
"""
Tests for cashew.core.permanence module
"""

import pytest
import sqlite3
import tempfile
import os
from datetime import datetime, timezone

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.permanence import (
    promote_permanent_nodes,
    get_permanence_stats,
    calculate_recommended_threshold,
    validate_permanence_integrity
)


class TestPermanence:
    """Test the permanence module functionality"""
    
    @pytest.fixture
    def test_db(self):
        """Create a test database with sample nodes"""
        fd, db_path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create table with all necessary columns
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
                decayed INTEGER DEFAULT 0,
                permanent INTEGER DEFAULT 0,
                access_count INTEGER DEFAULT 0,
                last_updated TEXT,
                last_accessed TEXT
            )
        """)
        
        # Insert test data with various access patterns
        test_nodes = [
            # (id, content, node_type, timestamp, confidence, mood, metadata, source, decayed, permanent, access_count)
            ("frequent_1", "Very frequent thought", "derived", "2023-01-01T00:00:00", 0.9, "confident", "{}", "test", 0, 0, 50),
            ("frequent_2", "Another frequent thought", "belief", "2023-01-02T00:00:00", 0.8, "stable", "{}", "test", 0, 0, 30),
            ("moderate_1", "Moderately accessed", "derived", "2023-01-03T00:00:00", 0.7, "neutral", "{}", "test", 0, 0, 12),
            ("moderate_2", "Another moderate", "belief", "2023-01-04T00:00:00", 0.6, "uncertain", "{}", "test", 0, 0, 8),
            ("low_access", "Low access node", "derived", "2023-01-05T00:00:00", 0.5, "doubtful", "{}", "test", 0, 0, 3),
            ("zero_access", "Never accessed", "derived", "2023-01-06T00:00:00", 0.4, "forgotten", "{}", "test", 0, 0, 0),
            ("already_permanent", "Pre-existing permanent", "derived", "2023-01-07T00:00:00", 0.8, "stable", "{}", "test", 0, 1, 20),
            ("permanent_low", "Permanent with low access", "derived", "2023-01-08T00:00:00", 0.6, "stable", "{}", "test", 0, 1, 5),
            ("decayed_high", "Decayed but high access", "derived", "2023-01-09T00:00:00", 0.3, "forgotten", "{}", "test", 1, 0, 25),
        ]
        
        cursor.executemany("""
            INSERT INTO thought_nodes 
            (id, content, node_type, timestamp, confidence, mood_state, metadata, source_file, decayed, permanent, access_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, test_nodes)
        
        conn.commit()
        conn.close()
        
        yield db_path
        
        # Cleanup
        try:
            os.unlink(db_path)
        except OSError:
            pass

    def test_promote_permanent_nodes_basic(self, test_db):
        """Test basic permanence promotion functionality"""
        result = promote_permanent_nodes(test_db, access_threshold=10)
        
        # Should evaluate 3 nodes (frequent_1, frequent_2, moderate_1) and promote them
        assert result["nodes_evaluated"] == 3
        assert result["nodes_promoted"] == 3
        assert result["access_threshold"] == 10
        
        # Verify the database was updated correctly
        conn = sqlite3.connect(test_db)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, permanent FROM thought_nodes WHERE access_count >= 10 AND decayed = 0")
        updated_nodes = cursor.fetchall()
        
        # Should have 4 permanent nodes now (3 newly promoted + 1 already permanent)
        cursor.execute("SELECT COUNT(*) FROM thought_nodes WHERE permanent = 1")
        total_permanent = cursor.fetchone()[0]
        assert total_permanent == 5  # 3 new + already_permanent + permanent_low
        
        conn.close()

    def test_promote_permanent_nodes_skips_decayed(self, test_db):
        """Test that decayed nodes are not promoted even with high access"""
        result = promote_permanent_nodes(test_db, access_threshold=20)
        
        # Should evaluate 2 nodes (frequent_1=50, frequent_2=30) but skip decayed_high=25
        assert result["nodes_evaluated"] == 2
        assert result["nodes_promoted"] == 2
        
        # Verify decayed node was not promoted
        conn = sqlite3.connect(test_db)
        cursor = conn.cursor()
        
        cursor.execute("SELECT permanent FROM thought_nodes WHERE id = 'decayed_high'")
        decayed_permanent = cursor.fetchone()[0]
        assert decayed_permanent == 0, "Decayed nodes should not be promoted"
        
        conn.close()

    def test_promote_permanent_nodes_idempotent(self, test_db):
        """Test that running promotion multiple times doesn't double-promote"""
        # Run promotion twice with same threshold
        result1 = promote_permanent_nodes(test_db, access_threshold=10)
        result2 = promote_permanent_nodes(test_db, access_threshold=10)
        
        # First run should promote nodes, second run should find 0 to promote
        assert result1["nodes_promoted"] > 0
        assert result2["nodes_promoted"] == 0
        assert result2["nodes_evaluated"] == 0  # No candidates found

    def test_get_permanence_stats(self, test_db):
        """Test permanence statistics collection"""
        # Get initial stats
        stats = get_permanence_stats(test_db)
        
        assert stats["permanent_count"] == 2  # already_permanent + permanent_low
        assert stats["non_permanent_count"] == 7  # The rest
        assert stats["total_nodes"] == 9
        
        # Check access count statistics
        assert stats["permanent_stats"]["max_access"] == 20  # already_permanent
        assert stats["permanent_stats"]["min_access"] == 5   # permanent_low
        assert stats["non_permanent_stats"]["max_access"] == 50  # frequent_1
        assert stats["highest_non_permanent_access"] == 50

    def test_calculate_recommended_threshold(self, test_db):
        """Test threshold calculation based on existing permanent nodes"""
        threshold = calculate_recommended_threshold(test_db)
        
        # With 2 permanent nodes having access counts [5, 20], 75th percentile should be around 20
        # But we ensure minimum of 5, so result should be reasonable
        assert threshold >= 5
        assert threshold <= 20

    def test_calculate_recommended_threshold_empty(self):
        """Test threshold calculation with no permanent nodes"""
        # Create empty database
        fd, db_path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE thought_nodes (
                id TEXT PRIMARY KEY,
                content TEXT,
                node_type TEXT,
                timestamp TEXT,
                confidence REAL,
                permanent INTEGER DEFAULT 0,
                access_count INTEGER DEFAULT 0
            )
        """)
        
        # Insert nodes with no permanent ones
        cursor.execute("INSERT INTO thought_nodes (id, permanent, access_count) VALUES ('test1', 0, 5)")
        conn.commit()
        conn.close()
        
        try:
            threshold = calculate_recommended_threshold(db_path)
            # Should return conservative default when no permanent nodes exist
            assert threshold == 10
        finally:
            os.unlink(db_path)

    def test_validate_permanence_integrity(self, test_db):
        """Test permanence integrity validation"""
        # Get initial validation
        integrity = validate_permanence_integrity(test_db)
        
        assert integrity["permanent_but_decayed"] == 0  # No permanent nodes should be decayed
        assert integrity["core_not_permanent"] == 0     # No core memories in test data
        assert integrity["high_access_not_permanent"] > 0  # Should find some high-access non-permanent nodes
        assert integrity["integrity_ok"] == True

    def test_validate_permanence_integrity_with_violations(self, test_db):
        """Test validation when there are integrity violations"""
        # Create a violation: make a permanent node decayed
        conn = sqlite3.connect(test_db)
        cursor = conn.cursor()
        
        cursor.execute("UPDATE thought_nodes SET decayed = 1 WHERE id = 'already_permanent'")
        
        # Add a core memory that isn't permanent
        cursor.execute("""
            INSERT INTO thought_nodes 
            (id, content, node_type, timestamp, confidence, permanent, access_count)
            VALUES ('core_test', 'Core memory', 'core_memory', '2023-01-10T00:00:00', 0.9, 0, 15)
        """)
        
        conn.commit()
        conn.close()
        
        # Validate integrity
        integrity = validate_permanence_integrity(test_db)
        
        assert integrity["permanent_but_decayed"] == 1  # One violation
        assert integrity["core_not_permanent"] == 1     # One core memory not permanent
        assert integrity["integrity_ok"] == False       # Should fail integrity check

    def test_permanence_threshold_edge_cases(self, test_db):
        """Test edge cases for threshold handling"""
        # Test with threshold of 0 (should promote everything non-decayed)
        result = promote_permanent_nodes(test_db, access_threshold=0)
        
        # Should promote all non-decayed, non-permanent nodes
        expected_promotions = 6  # All except decayed_high and the 2 already permanent
        assert result["nodes_promoted"] == expected_promotions
        
        # Test with very high threshold (should promote nothing)
        result2 = promote_permanent_nodes(test_db, access_threshold=1000)
        assert result2["nodes_promoted"] == 0
        assert result2["nodes_evaluated"] == 0

    def test_timestamp_updates(self, test_db):
        """Test that last_updated timestamps are set correctly"""
        before_time = datetime.now(timezone.utc)
        
        promote_permanent_nodes(test_db, access_threshold=10)
        
        after_time = datetime.now(timezone.utc)
        
        # Check that promoted nodes have updated timestamps
        conn = sqlite3.connect(test_db)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT last_updated FROM thought_nodes 
            WHERE id IN ('frequent_1', 'frequent_2', 'moderate_1')
        """)
        timestamps = cursor.fetchall()
        
        for (timestamp_str,) in timestamps:
            assert timestamp_str is not None
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            assert before_time <= timestamp <= after_time
        
        conn.close()

    def test_permanence_consistency_with_decay_module(self, test_db):
        """Test that permanence works correctly with decay protection"""
        # This test ensures our permanence module is consistent with 
        # how the decay module checks permanent status
        
        # Promote some nodes to permanent status
        promote_permanent_nodes(test_db, access_threshold=10)
        
        # Simulate what decay.py does: check permanent column
        conn = sqlite3.connect(test_db)
        cursor = conn.cursor()
        
        # This mimics the check in decay.py lines 65-76 (auto_decay function)
        cursor.execute("""
            SELECT COUNT(*) FROM thought_nodes
            WHERE (permanent IS NULL OR permanent = 0)
            AND access_count = 0
            AND confidence < 0.5
        """)
        decay_candidates = cursor.fetchone()[0]
        
        # Should find some candidates but permanent nodes should be excluded
        cursor.execute("""
            SELECT COUNT(*) FROM thought_nodes
            WHERE permanent > 0
            AND access_count = 0
            AND confidence < 0.5
        """)
        permanent_that_would_be_candidates = cursor.fetchone()[0]
        
        # This ensures permanent nodes are properly excluded from decay
        assert permanent_that_would_be_candidates == 0, "Permanent nodes should never be decay candidates"
        
        conn.close()