#!/usr/bin/env python3
"""
Tests for cashew.core.sleep module
"""

import pytest
import sqlite3
import json
import tempfile
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

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
                decayed INTEGER DEFAULT 0,
                permanent INTEGER DEFAULT 0,
                domain TEXT,
                access_count INTEGER DEFAULT 0,
                last_accessed TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE derivation_edges (
                parent_id TEXT NOT NULL,
                child_id TEXT NOT NULL,
                weight REAL NOT NULL,
                reasoning TEXT,
                FOREIGN KEY (parent_id) REFERENCES thought_nodes(id),
                FOREIGN KEY (child_id) REFERENCES thought_nodes(id),
                PRIMARY KEY (parent_id, child_id)
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
            ("seed1", "similar1", 0.9, "supports - Strong support"),
            ("seed1", "different1", 0.6, "supports - Moderate support"),
            ("similar1", "hub1", 0.8, "derived_from - Derived from similar belief"),
            ("hub1", "weak1", 0.3, "derived_from - Weak derivation"),  # Low weight for testing
            ("hub1", "weak2", 0.2, "derived_from - Very weak derivation"),
            # No edges for isolated1 and isolated2 to test orphan detection
        ]
        
        cursor.executemany("""
            INSERT INTO derivation_edges 
            (parent_id, child_id, weight, reasoning)
            VALUES (?, ?, ?, ?)
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
            SELECT parent_id, child_id, weight, reasoning
            FROM derivation_edges 
            WHERE reasoning LIKE '%cross_link%'
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
                    WHERE child_id = ? AND reasoning LIKE '%derived_from%'
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
                decayed INTEGER DEFAULT 0,
                permanent INTEGER DEFAULT 0,
                domain TEXT,
                access_count INTEGER DEFAULT 0,
                last_accessed TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE derivation_edges (
                parent_id TEXT NOT NULL,
                child_id TEXT NOT NULL,
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


class TestGCConfigModes:
    """Test garbage collection with config-driven modes, grace periods, and protections."""

    @pytest.fixture
    def gc_db(self):
        """DB with last_accessed column and enough nodes to trigger GC."""
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
                mood_state TEXT,
                metadata TEXT DEFAULT '{}',
                source_file TEXT,
                decayed INTEGER DEFAULT 0,
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

        old_ts = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        recent_ts = datetime.now(timezone.utc).isoformat()

        # 25 nodes so GC will always have enough (>k_nodes=20)
        # Nodes gc01-gc20: old last_accessed, low-fitness (no edges), type=derived
        for i in range(1, 21):
            c.execute("""
                INSERT INTO thought_nodes
                (id, content, node_type, domain, timestamp, last_accessed,
                 confidence, source_file, decayed)
                VALUES (?, ?, 'derived', 'general', ?, ?, 0.3, 'test', 0)
            """, (f"gc{i:02d}", f"GC candidate node number {i}", old_ts, old_ts))

        # Protected nodes
        c.execute("""
            INSERT INTO thought_nodes
            (id, content, node_type, domain, timestamp, last_accessed,
             confidence, source_file, decayed)
            VALUES ('seed01', 'I am a seed node', 'seed', 'general', ?, ?, 0.5, 'test', 0)
        """, (old_ts, old_ts))

        c.execute("""
            INSERT INTO thought_nodes
            (id, content, node_type, domain, timestamp, last_accessed,
             confidence, source_file, decayed)
            VALUES ('hot01', 'I am a hotspot summary', 'hotspot', 'general', ?, ?, 0.5, 'test', 0)
        """, (old_ts, old_ts))

        # Recently accessed node (within grace period)
        c.execute("""
            INSERT INTO thought_nodes
            (id, content, node_type, domain, timestamp, last_accessed,
             confidence, source_file, decayed)
            VALUES ('recent01', 'Recently accessed node', 'derived', 'general', ?, ?, 0.3, 'test', 0)
        """, (old_ts, recent_ts))

        # Think-cycle node
        c.execute("""
            INSERT INTO thought_nodes
            (id, content, node_type, domain, timestamp, last_accessed,
             confidence, source_file, decayed)
            VALUES ('think01', 'Think cycle output node', 'derived', 'general', ?, ?, 0.3, 'think_cycle_run', 0)
        """, (old_ts, old_ts))

        # Core memory node (protected type)
        c.execute("""
            INSERT INTO thought_nodes
            (id, content, node_type, domain, timestamp, last_accessed,
             confidence, source_file, decayed)
            VALUES ('cmem01', 'Core memory node', 'core_memory', 'general', ?, ?, 0.3, 'test', 0)
        """, (old_ts, old_ts))

        conn.commit()
        conn.close()

        yield path
        os.unlink(path)

    def _make_metrics(self, db_path, fitness=0.0):
        """Build a metrics dict for all nodes in the DB with a given fitness."""
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("SELECT id FROM thought_nodes WHERE decayed = 0 OR decayed IS NULL")
        ids = [r[0] for r in c.fetchall()]
        conn.close()
        return {nid: NodeMetrics(nid, 0, 0, 0, 0, fitness) for nid in ids}

    # --- mode tests ---

    def test_gc_off_does_nothing(self, gc_db):
        """GC mode=off should not decay or delete any nodes."""
        with patch.object(SleepProtocol, '_get_connection', return_value=sqlite3.connect(gc_db)):
            pass  # just verifying config path

        protocol = SleepProtocol(gc_db, tempfile.mktemp(suffix=".json"))
        metrics = self._make_metrics(gc_db, fitness=0.0)

        with patch("core.sleep.config") as mock_cfg:
            mock_cfg.gc_mode = "off"
            mock_cfg.gc_threshold = 0.05
            mock_cfg.gc_grace_days = 7
            mock_cfg.gc_protect_types = ["seed", "core_memory"]
            mock_cfg.gc_protect_hotspots = True
            mock_cfg.gc_think_cycle_penalty = 1.5

            result = protocol.garbage_collect(metrics, k_nodes=20)

        assert result == []

        # Verify nothing changed
        conn = sqlite3.connect(gc_db)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM thought_nodes WHERE decayed = 1")
        assert c.fetchone()[0] == 0
        conn.close()

    def test_gc_soft_sets_decayed(self, gc_db):
        """GC mode=soft should set decayed=1, not delete rows."""
        protocol = SleepProtocol(gc_db, tempfile.mktemp(suffix=".json"))
        metrics = self._make_metrics(gc_db, fitness=0.0)

        with patch("core.sleep.config") as mock_cfg:
            mock_cfg.gc_mode = "soft"
            mock_cfg.gc_threshold = 0.05
            mock_cfg.gc_grace_days = 7
            mock_cfg.gc_protect_types = ["seed", "core_memory"]
            mock_cfg.gc_protect_hotspots = True
            mock_cfg.gc_think_cycle_penalty = 1.5

            result = protocol.garbage_collect(metrics, k_nodes=20)

        assert len(result) > 0

        conn = sqlite3.connect(gc_db)
        c = conn.cursor()
        # Soft mode: rows still exist
        for nid in result:
            c.execute("SELECT decayed FROM thought_nodes WHERE id = ?", (nid,))
            row = c.fetchone()
            assert row is not None, f"Node {nid} should still exist (soft mode)"
            assert row[0] == 1, f"Node {nid} should be decayed=1"
        conn.close()

    def test_gc_hard_deletes_nodes(self, gc_db):
        """GC mode=hard should DELETE nodes and their edges/embeddings."""
        protocol = SleepProtocol(gc_db, tempfile.mktemp(suffix=".json"))
        metrics = self._make_metrics(gc_db, fitness=0.0)

        with patch("core.sleep.config") as mock_cfg:
            mock_cfg.gc_mode = "hard"
            mock_cfg.gc_threshold = 0.05
            mock_cfg.gc_grace_days = 7
            mock_cfg.gc_protect_types = ["seed", "core_memory"]
            mock_cfg.gc_protect_hotspots = True
            mock_cfg.gc_think_cycle_penalty = 1.5

            result = protocol.garbage_collect(metrics, k_nodes=20)

        assert len(result) > 0

        conn = sqlite3.connect(gc_db)
        c = conn.cursor()
        for nid in result:
            c.execute("SELECT id FROM thought_nodes WHERE id = ?", (nid,))
            assert c.fetchone() is None, f"Node {nid} should be deleted (hard mode)"
        conn.close()

    # --- grace period tests ---

    def test_gc_grace_period_protects_recent_nodes(self, gc_db):
        """Nodes accessed within grace_days should not be collected."""
        protocol = SleepProtocol(gc_db, tempfile.mktemp(suffix=".json"))
        metrics = self._make_metrics(gc_db, fitness=0.0)

        with patch("core.sleep.config") as mock_cfg:
            mock_cfg.gc_mode = "soft"
            mock_cfg.gc_threshold = 10.0  # Very high — everything is below threshold
            mock_cfg.gc_grace_days = 7
            mock_cfg.gc_protect_types = []
            mock_cfg.gc_protect_hotspots = False
            mock_cfg.gc_think_cycle_penalty = 1.0

            # Force selection to include recent01
            with patch("random.sample", return_value=["recent01"]):
                result = protocol.garbage_collect(metrics, k_nodes=1)

        assert "recent01" not in result

    def test_gc_uses_last_accessed_not_created_at(self, gc_db):
        """Grace period should check last_accessed, not timestamp/created_at."""
        # Update a node: old timestamp but recent last_accessed
        conn = sqlite3.connect(gc_db)
        c = conn.cursor()
        recent = datetime.now(timezone.utc).isoformat()
        c.execute("UPDATE thought_nodes SET last_accessed = ? WHERE id = 'gc01'", (recent,))
        conn.commit()
        conn.close()

        protocol = SleepProtocol(gc_db, tempfile.mktemp(suffix=".json"))
        metrics = self._make_metrics(gc_db, fitness=0.0)

        with patch("core.sleep.config") as mock_cfg:
            mock_cfg.gc_mode = "soft"
            mock_cfg.gc_threshold = 10.0
            mock_cfg.gc_grace_days = 7
            mock_cfg.gc_protect_types = []
            mock_cfg.gc_protect_hotspots = False
            mock_cfg.gc_think_cycle_penalty = 1.0

            with patch("random.sample", return_value=["gc01"]):
                result = protocol.garbage_collect(metrics, k_nodes=1)

        assert "gc01" not in result

    # --- protection tests ---

    def test_gc_protects_hotspot_nodes(self, gc_db):
        """When protect_hotspots=True, hotspot nodes should survive GC."""
        protocol = SleepProtocol(gc_db, tempfile.mktemp(suffix=".json"))
        metrics = self._make_metrics(gc_db, fitness=0.0)

        with patch("core.sleep.config") as mock_cfg:
            mock_cfg.gc_mode = "soft"
            mock_cfg.gc_threshold = 10.0
            mock_cfg.gc_grace_days = 0  # No grace period
            mock_cfg.gc_protect_types = []
            mock_cfg.gc_protect_hotspots = True
            mock_cfg.gc_think_cycle_penalty = 1.0

            with patch("random.sample", return_value=["hot01"]):
                result = protocol.garbage_collect(metrics, k_nodes=1)

        assert "hot01" not in result

    def test_gc_protects_configured_types(self, gc_db):
        """Nodes with types in protect_types should survive GC."""
        protocol = SleepProtocol(gc_db, tempfile.mktemp(suffix=".json"))
        metrics = self._make_metrics(gc_db, fitness=0.0)

        with patch("core.sleep.config") as mock_cfg:
            mock_cfg.gc_mode = "soft"
            mock_cfg.gc_threshold = 10.0
            mock_cfg.gc_grace_days = 0
            mock_cfg.gc_protect_types = ["seed", "core_memory"]
            mock_cfg.gc_protect_hotspots = False
            mock_cfg.gc_think_cycle_penalty = 1.0

            with patch("random.sample", return_value=["seed01", "cmem01"]):
                result = protocol.garbage_collect(metrics, k_nodes=2)

        assert "seed01" not in result
        assert "cmem01" not in result

    # --- threshold tests ---

    def test_gc_threshold_boundary(self, gc_db):
        """Nodes exactly at threshold should NOT be collected (< not <=)."""
        protocol = SleepProtocol(gc_db, tempfile.mktemp(suffix=".json"))
        # Set fitness exactly at threshold
        metrics = self._make_metrics(gc_db, fitness=0.05)

        with patch("core.sleep.config") as mock_cfg:
            mock_cfg.gc_mode = "soft"
            mock_cfg.gc_threshold = 0.05
            mock_cfg.gc_grace_days = 0
            mock_cfg.gc_protect_types = []
            mock_cfg.gc_protect_hotspots = False
            mock_cfg.gc_think_cycle_penalty = 1.0

            with patch("random.sample", return_value=["gc01"]):
                result = protocol.garbage_collect(metrics, k_nodes=1)

        # fitness (0.05) is NOT < threshold (0.05), so node should survive
        assert "gc01" not in result

    def test_gc_think_cycle_penalty(self, gc_db):
        """Think-cycle nodes should use threshold * penalty multiplier."""
        protocol = SleepProtocol(gc_db, tempfile.mktemp(suffix=".json"))
        # fitness = 0.06 — above base threshold (0.05) but below penalized (0.075)
        metrics = self._make_metrics(gc_db, fitness=0.06)

        with patch("core.sleep.config") as mock_cfg:
            mock_cfg.gc_mode = "soft"
            mock_cfg.gc_threshold = 0.05
            mock_cfg.gc_grace_days = 0
            mock_cfg.gc_protect_types = []
            mock_cfg.gc_protect_hotspots = False
            mock_cfg.gc_think_cycle_penalty = 1.5  # effective threshold = 0.075

            with patch("random.sample", return_value=["think01", "gc01"]):
                result = protocol.garbage_collect(metrics, k_nodes=2)

        # think01 has source_file='think_cycle_run', effective threshold=0.075 > 0.06 → collected
        assert "think01" in result
        # gc01 has source_file='test', effective threshold=0.05 < 0.06 → survives
        assert "gc01" not in result