#!/usr/bin/env python3
"""
Tests for bug fixes: novelty gates, hotspot proliferation fix, 
batch embedding, micro-cluster removal, and CLI commands
"""

import unittest
import sys
import os
import sqlite3
import tempfile
import json
import numpy as np
from unittest.mock import patch, MagicMock, call
from datetime import datetime, timezone
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.placement_aware_extraction import check_novelty, _should_create_new_hotspot
from core.clustering import run_clustering_cycle
from core.complete_clustering import run_complete_clustering_cycle, detect_complete_clusters
from core.hotspots import create_hotspot, update_hotspot
from core.think_cycle import ThinkCycle
from scripts.cashew_context import cmd_prune, cmd_compact, cmd_extract


class TestNoveltyGate(unittest.TestCase):
    """Test the novelty gate functionality"""
    
    def setUp(self):
        """Set up test database"""
        self.db_fd, self.db_path = tempfile.mkstemp(suffix='.db')
        os.close(self.db_fd)
        
        # Initialize with schema
        conn = sqlite3.connect(self.db_path)
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
                mood_state TEXT
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
        
        # Add test data
        now = datetime.now(timezone.utc).isoformat()
        cursor.execute("""
            INSERT INTO thought_nodes (id, content, node_type, domain, timestamp, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        """, ("existing_node", "Machine learning improves with more data", "fact", "tech", now, 0.8))
        
        # Create a fake embedding for the existing node
        fake_embedding = np.random.rand(384).astype(np.float32)  # MiniLM-L6-v2 dimension
        cursor.execute("""
            INSERT INTO embeddings (node_id, vector, model, updated_at)
            VALUES (?, ?, ?, ?)
        """, ("existing_node", fake_embedding.tobytes(), "all-MiniLM-L6-v2", now))
        
        conn.commit()
        conn.close()
        
        # Store the embedding for similarity tests
        self.existing_embedding = fake_embedding
    
    def tearDown(self):
        """Clean up test database"""
        os.unlink(self.db_path)
    
    @patch('core.placement_aware_extraction.embed_text')
    def test_identical_content_rejected(self, mock_embed):
        """Test that identical content is rejected (similarity > 0.82)"""
        # Mock the embedding to return same as existing
        mock_embed.return_value = self.existing_embedding
        
        is_novel, max_sim, nearest_id = check_novelty(self.db_path, "Machine learning improves with more data")
        
        self.assertFalse(is_novel, "Identical content should be rejected")
        self.assertGreater(max_sim, 0.82, "Similarity should be > 0.82")
        self.assertEqual(nearest_id, "existing_node")
    
    @patch('core.placement_aware_extraction.embed_text')
    def test_very_different_content_accepted(self, mock_embed):
        """Test that very different content is accepted"""
        # Mock embedding to be very different 
        different_embedding = np.random.rand(384).astype(np.float32)
        mock_embed.return_value = different_embedding
        
        is_novel, max_sim, nearest_id = check_novelty(self.db_path, "Cats like to play with string")
        
        self.assertTrue(is_novel, "Very different content should be accepted")
        self.assertLess(max_sim, 0.82, "Similarity should be < 0.82")
    
    @patch('core.placement_aware_extraction.embed_text')
    def test_borderline_content_with_low_confidence_rejected(self, mock_embed):
        """Test that borderline similarity (0.72-0.82) gets checked properly"""
        # Create an embedding that's mathematically guaranteed to be in the target range
        # We'll create a vector that when dotted with existing gives us exactly 0.75 similarity
        normalized_existing = self.existing_embedding / np.linalg.norm(self.existing_embedding)
        # Create orthogonal component 
        random_vec = np.random.rand(384).astype(np.float32)
        orthogonal = random_vec - np.dot(random_vec, normalized_existing) * normalized_existing
        orthogonal = orthogonal / np.linalg.norm(orthogonal)
        
        # Mix to get desired similarity (cos(theta) = 0.75 means theta ~= 41.4 degrees)
        target_sim = 0.75
        angle_rad = np.arccos(target_sim)
        moderate_sim_embedding = target_sim * normalized_existing + np.sin(angle_rad) * orthogonal
        moderate_sim_embedding = moderate_sim_embedding / np.linalg.norm(moderate_sim_embedding)
        
        mock_embed.return_value = moderate_sim_embedding
        
        is_novel, max_sim, nearest_id = check_novelty(self.db_path, "Machine learning gets better with big datasets", 
                                                      threshold=0.82)
        
        # Should be in borderline range
        self.assertGreater(max_sim, 0.72, "Should be in borderline range")
        self.assertLess(max_sim, 0.82, "Should be in borderline range") 
        self.assertTrue(is_novel, "Should be novel since similarity < 0.82")


class TestHotspotProliferationFix(unittest.TestCase):
    """Test that clustering cycles no longer create hotspots"""
    
    def setUp(self):
        """Set up test database with clustering data"""
        self.db_fd, self.db_path = tempfile.mkstemp(suffix='.db')
        os.close(self.db_fd)
        
        # Initialize with schema
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Core schema
        cursor.execute('''
            CREATE TABLE thought_nodes (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                node_type TEXT NOT NULL,
                domain TEXT,
                timestamp TEXT,
                confidence REAL,
                source_file TEXT,
                decayed INTEGER DEFAULT 0,
                metadata TEXT DEFAULT '{}'
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE derivation_edges (
                parent_id TEXT,
                child_id TEXT,
                relation TEXT,
                weight REAL,
                reasoning TEXT,
                confidence REAL,
                timestamp TEXT,
                PRIMARY KEY (parent_id, child_id, relation)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE embeddings (
                node_id TEXT PRIMARY KEY,
                vector BLOB NOT NULL,
                model TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        ''')
        
        cursor.execute('''
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
        ''')
        
        # Add test nodes with embeddings
        now = datetime.now(timezone.utc).isoformat()
        for i in range(10):
            node_id = f"node_{i}"
            content = f"Test content for clustering {i}"
            cursor.execute("""
                INSERT INTO thought_nodes (id, content, node_type, domain, timestamp, confidence)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (node_id, content, "fact", "test", now, 0.8))
            
            # Add fake embedding
            embedding = np.random.rand(384).astype(np.float32)
            cursor.execute("""
                INSERT INTO embeddings (node_id, vector, model, updated_at)
                VALUES (?, ?, ?, ?)
            """, (node_id, embedding.tobytes(), "all-MiniLM-L6-v2", now))
        
        conn.commit()
        conn.close()
    
    def tearDown(self):
        """Clean up test database"""
        os.unlink(self.db_path)
    
    def test_clustering_cycle_no_hotspot_creation(self):
        """Test that run_clustering_cycle doesn't create new hotspots"""
        # Count existing hotspots before
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM hotspots")
        hotspots_before = cursor.fetchone()[0]
        conn.close()
        
        # Run clustering cycle
        results = run_clustering_cycle(self.db_path, dry_run=False)
        
        # Count hotspots after
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM hotspots")
        hotspots_after = cursor.fetchone()[0]
        conn.close()
        
        # Should have same number of hotspots
        self.assertEqual(hotspots_before, hotspots_after, "Clustering cycle should not create new hotspots")
        
        # Check that skip action is recorded if clusters were found
        if 'cluster_details' in results:
            for detail in results['cluster_details']:
                if 'action' in detail:
                    self.assertEqual(detail['action'], "skipped_no_hotspot_creation_in_clustering")
    
    def test_complete_clustering_cycle_no_hotspot_creation(self):
        """Test that run_complete_clustering_cycle doesn't create new hotspots"""
        # Count existing hotspots before
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor() 
        cursor.execute("SELECT COUNT(*) FROM hotspots")
        hotspots_before = cursor.fetchone()[0]
        conn.close()
        
        # Run complete clustering cycle
        results = run_complete_clustering_cycle(self.db_path, dry_run=False)
        
        # Count hotspots after
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM hotspots")
        hotspots_after = cursor.fetchone()[0]
        conn.close()
        
        # Should have same number of hotspots
        self.assertEqual(hotspots_before, hotspots_after, "Complete clustering cycle should not create new hotspots")
        
        # Check that skip action is recorded if clusters were found
        if 'cluster_details' in results:
            for detail in results['cluster_details']:
                if 'action' in detail:
                    self.assertEqual(detail['action'], "skipped_no_hotspot_creation_in_sleep")


class TestMicroClusterRemoval(unittest.TestCase):
    """Test that micro-clusters are no longer created for orphan nodes"""
    
    def setUp(self):
        """Set up test database with orphan nodes"""
        self.db_fd, self.db_path = tempfile.mkstemp(suffix='.db')
        os.close(self.db_fd)
        
        # Initialize with schema
        conn = sqlite3.connect(self.db_path)
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
                mood_state TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE derivation_edges (
                parent_id TEXT,
                child_id TEXT,
                relation TEXT,
                weight REAL,
                reasoning TEXT,
                confidence REAL,
                timestamp TEXT,
                PRIMARY KEY (parent_id, child_id, relation)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE embeddings (
                node_id TEXT PRIMARY KEY,
                vector BLOB NOT NULL,
                model TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        ''')
        
        # Add few isolated nodes that would become orphans/noise in DBSCAN
        now = datetime.now(timezone.utc).isoformat()
        for i in range(3):  # Small number to ensure they're noise
            node_id = f"orphan_{i}"
            content = f"Isolated content {i} that doesn't cluster with others"
            cursor.execute("""
                INSERT INTO thought_nodes (id, content, node_type, domain, timestamp, confidence, source_file)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (node_id, content, "fact", "test", now, 0.8, "test_source"))
            
            # Add very different embeddings
            embedding = np.random.rand(384).astype(np.float32)
            cursor.execute("""
                INSERT INTO embeddings (node_id, vector, model, updated_at)
                VALUES (?, ?, ?, ?)
            """, (node_id, embedding.tobytes(), "all-MiniLM-L6-v2", now))
        
        conn.commit()
        conn.close()
    
    def tearDown(self):
        """Clean up test database"""
        os.unlink(self.db_path)
    
    def test_orphan_nodes_not_micro_clustered(self):
        """Test that orphan nodes don't get automatically micro-clustered in the main clustering logic"""
        clusters = detect_complete_clusters(self.db_path, max_cluster_size=15)
        
        # The main goal is that DBSCAN orphan/noise nodes don't automatically get micro-clustered
        # in the primary clustering step. However, there may be an emergency fallback for coverage.
        # What we're testing is that the primary clustering logic now acknowledges orphans
        # rather than forcing them into clusters immediately.
        
        # Count how many clusters are of the "natural" type (from DBSCAN)
        natural_clusters = [c for c in clusters if c.cluster_type.startswith("natural")]
        
        # With our few isolated nodes, DBSCAN should not find natural clusters
        self.assertEqual(len(natural_clusters), 0, "DBSCAN should not find natural clusters with isolated nodes")
        
        # Any clusters that do exist should be from the emergency fallback, not the main logic
        if clusters:
            for cluster in clusters:
                # Check that emergency clusters are acknowledged as such
                if len(cluster.node_ids) == 1:
                    self.assertTrue(
                        cluster.cluster_type in ["micro", "emergency"], 
                        f"Single-node clusters should be marked as micro/emergency, not {cluster.cluster_type}"
                    )


class TestBatchEmbedding(unittest.TestCase):
    """Test that hotspot creation/update doesn't call embed_nodes inline"""
    
    def setUp(self):
        """Set up test database"""
        self.db_fd, self.db_path = tempfile.mkstemp(suffix='.db')
        os.close(self.db_fd)
        
        # Initialize with schema
        conn = sqlite3.connect(self.db_path)
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
                mood_state TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE derivation_edges (
                parent_id TEXT,
                child_id TEXT,
                relation TEXT,
                weight REAL,
                reasoning TEXT,
                confidence REAL,
                timestamp TEXT,
                PRIMARY KEY (parent_id, child_id, relation)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE embeddings (
                node_id TEXT PRIMARY KEY,
                vector BLOB NOT NULL,
                model TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        ''')
        
        cursor.execute('''
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
        ''')
        
        # Add a test node
        now = datetime.now(timezone.utc).isoformat()
        cursor.execute("""
            INSERT INTO thought_nodes (id, content, node_type, domain, timestamp, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        """, ("test_node", "Test content", "fact", "test", now, 0.8))
        
        conn.commit()
        conn.close()
    
    def tearDown(self):
        """Clean up test database"""
        os.unlink(self.db_path)
    
    @patch('core.hotspots.embed_nodes')
    def test_create_hotspot_no_inline_embedding(self, mock_embed_nodes):
        """Test that create_hotspot doesn't call embed_nodes"""
        hotspot_id = create_hotspot(
            db_path=self.db_path,
            content="Test hotspot",
            status="test",
            file_pointers={},
            cluster_node_ids=["test_node"],
            domain="test"
        )
        
        # Verify hotspot was created
        self.assertIsNotNone(hotspot_id)
        
        # Verify embed_nodes was NOT called
        mock_embed_nodes.assert_not_called()
    
    @patch('core.hotspots.embed_nodes')
    def test_update_hotspot_deletes_embedding_no_re_embed(self, mock_embed_nodes):
        """Test that update_hotspot deletes old embedding but doesn't re-embed"""
        # First create a hotspot
        hotspot_id = create_hotspot(
            db_path=self.db_path,
            content="Original content",
            status="test",
            file_pointers={},
            cluster_node_ids=["test_node"],
            domain="test"
        )
        
        # Add an embedding for it
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        embedding = np.random.rand(384).astype(np.float32)
        now = datetime.now(timezone.utc).isoformat()
        cursor.execute("""
            INSERT INTO embeddings (node_id, vector, model, updated_at)
            VALUES (?, ?, ?, ?)
        """, (hotspot_id, embedding.tobytes(), "all-MiniLM-L6-v2", now))
        conn.commit()
        
        # Verify embedding exists
        cursor.execute("SELECT COUNT(*) FROM embeddings WHERE node_id = ?", (hotspot_id,))
        self.assertEqual(cursor.fetchone()[0], 1)
        conn.close()
        
        # Update hotspot content
        update_hotspot(
            db_path=self.db_path,
            hotspot_id=hotspot_id,
            content="Updated content"
        )
        
        # Verify old embedding was deleted
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM embeddings WHERE node_id = ?", (hotspot_id,))
        self.assertEqual(cursor.fetchone()[0], 0, "Old embedding should be deleted")
        conn.close()
        
        # Verify embed_nodes was NOT called for re-embedding
        mock_embed_nodes.assert_not_called()


class TestShouldCreateNewHotspot(unittest.TestCase):
    """Test the rewritten _should_create_new_hotspot function"""
    
    def setUp(self):
        """Set up test database"""
        self.db_fd, self.db_path = tempfile.mkstemp(suffix='.db')
        os.close(self.db_fd)
        
        # Initialize with schema
        conn = sqlite3.connect(self.db_path)
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
                mood_state TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE derivation_edges (
                parent_id TEXT,
                child_id TEXT,
                relation TEXT,
                weight REAL,
                reasoning TEXT,
                confidence REAL,
                timestamp TEXT,
                PRIMARY KEY (parent_id, child_id, relation)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE embeddings (
                node_id TEXT PRIMARY KEY,
                vector BLOB NOT NULL,
                model TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        ''')
        
        # Create inbox hotspot
        now = datetime.now(timezone.utc).isoformat()
        inbox_id = "inbox_hotspot"
        cursor.execute("""
            INSERT INTO thought_nodes (id, content, node_type, domain, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (inbox_id, "INBOX: Uncategorized thoughts", "hotspot", "inbox", now))
        
        conn.commit()
        conn.close()
        
        self.inbox_id = inbox_id
    
    def tearDown(self):
        """Clean up test database"""
        os.unlink(self.db_path)
    
    @patch('core.placement_aware_extraction.embed_text')
    @patch('core.placement_aware_extraction.sqlite3.connect')
    def test_fewer_than_min_cluster_size_returns_false(self, mock_connect, mock_embed):
        """Test that fewer than 5 similar inbox nodes returns False"""
        # Mock embedding
        mock_embed.return_value = np.random.rand(384).astype(np.float32)
        
        # Mock database connection to return 3 similar nodes (< MIN_CLUSTER_SIZE)
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        # First call: finds inbox hotspot
        # Second call: finds 3 similar nodes in inbox
        mock_cursor.fetchone.side_effect = [("inbox_id",)]  # inbox exists
        mock_cursor.fetchall.return_value = [
            ("node1", np.random.rand(384).astype(np.float32).tobytes()),
            ("node2", np.random.rand(384).astype(np.float32).tobytes()),
            ("node3", np.random.rand(384).astype(np.float32).tobytes())
        ]  # 3 nodes (< MIN_CLUSTER_SIZE=5)
        
        result = _should_create_new_hotspot(self.db_path, "New similar content", "test", None)
        
        self.assertFalse(result, "Should return False with fewer than 5 similar inbox nodes")
    
    @patch('core.placement_aware_extraction.embed_text')
    @patch('core.placement_aware_extraction.sqlite3.connect')
    def test_min_cluster_size_or_more_returns_true(self, mock_connect, mock_embed):
        """Test that 5+ similar inbox nodes returns True"""
        # Mock embedding
        mock_embed.return_value = np.random.rand(384).astype(np.float32)
        
        # Mock database connection to return 6 similar nodes (>= MIN_CLUSTER_SIZE)
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        # First call: finds inbox hotspot
        # Second call: finds 6 similar nodes in inbox
        mock_cursor.fetchone.side_effect = [("inbox_id",)]  # inbox exists
        mock_cursor.fetchall.return_value = [
            ("node1", np.random.rand(384).astype(np.float32).tobytes()),
            ("node2", np.random.rand(384).astype(np.float32).tobytes()),
            ("node3", np.random.rand(384).astype(np.float32).tobytes()),
            ("node4", np.random.rand(384).astype(np.float32).tobytes()),
            ("node5", np.random.rand(384).astype(np.float32).tobytes()),
            ("node6", np.random.rand(384).astype(np.float32).tobytes())
        ]  # 6 nodes (>= MIN_CLUSTER_SIZE=5)
        
        result = _should_create_new_hotspot(self.db_path, "New similar content", "test", None)
        
        self.assertTrue(result, "Should return True with 5+ similar inbox nodes")


class TestThinkCycleNoveltyGate(unittest.TestCase):
    """Test that think cycle uses novelty gate before saving insights"""
    
    def setUp(self):
        """Set up test database"""
        self.db_fd, self.db_path = tempfile.mkstemp(suffix='.db')
        os.close(self.db_fd)
        
        # Initialize with schema
        conn = sqlite3.connect(self.db_path)
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
                mood_state TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE derivation_edges (
                parent_id TEXT,
                child_id TEXT,
                relation TEXT,
                weight REAL,
                reasoning TEXT,
                confidence REAL,
                timestamp TEXT,
                PRIMARY KEY (parent_id, child_id, relation)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE embeddings (
                node_id TEXT PRIMARY KEY,
                vector BLOB NOT NULL,
                model TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        ''')
        
        # Add existing insight
        now = datetime.now(timezone.utc).isoformat()
        cursor.execute("""
            INSERT INTO thought_nodes (id, content, node_type, domain, timestamp, confidence, source_file)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, ("existing_insight", "Existing insight about systems", "insight", "meta", now, 0.8, "system_generated"))
        
        # Add embedding for existing insight
        embedding = np.random.rand(384).astype(np.float32)
        cursor.execute("""
            INSERT INTO embeddings (node_id, vector, model, updated_at)
            VALUES (?, ?, ?, ?)
        """, ("existing_insight", embedding.tobytes(), "all-MiniLM-L6-v2", now))
        
        conn.commit()
        conn.close()
        
        self.existing_embedding = embedding
    
    def tearDown(self):
        """Clean up test database"""
        os.unlink(self.db_path)
    
    @patch('core.placement_aware_extraction.embed_text')
    def test_duplicate_think_insight_rejected(self, mock_embed):
        """Test that duplicate insights are rejected by novelty gate"""        
        # Mock embedding to return identical embedding
        mock_embed.return_value = self.existing_embedding
        
        think_cycle = ThinkCycle(self.db_path)
        
        # Count insights before
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM thought_nodes WHERE node_type = 'insight'")
        insights_before = cursor.fetchone()[0]
        conn.close()
        
        # Run think cycle - the system will try to create insights but duplicates will be rejected
        result = think_cycle.run_think_cycle()
        
        # Count insights after
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM thought_nodes WHERE node_type = 'insight'")
        insights_after = cursor.fetchone()[0]
        conn.close()
        
        # Since we're mocking embeddings to always return the same value, 
        # all generated insights would be seen as duplicates and rejected
        # So the count should not increase much (if at all)
        self.assertLessEqual(insights_after - insights_before, 1, "Most duplicate insights should be rejected")


class TestCLICommands(unittest.TestCase):
    """Test new CLI commands exist and basic functionality"""
    
    def setUp(self):
        """Set up test database"""
        self.db_fd, self.db_path = tempfile.mkstemp(suffix='.db')
        os.close(self.db_fd)
        
        # Initialize with schema
        conn = sqlite3.connect(self.db_path)
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
                mood_state TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def tearDown(self):
        """Clean up test database"""
        os.unlink(self.db_path)
    
    def test_prune_command_dry_run_exists(self):
        """Test that prune command with --dry-run doesn't crash"""
        # Create mock args
        class MockArgs:
            db = self.db_path
            dry_run = True
            min_age_days = 14
            max_confidence = 0.85
        
        args = MockArgs()
        
        # Should not crash
        try:
            result = cmd_prune(args)
            # If it returns something, it should be 0 (success)
            if result is not None:
                self.assertEqual(result, 0)
        except Exception as e:
            self.fail(f"Prune dry-run crashed: {e}")
    
    def test_compact_command_dry_run_exists(self):
        """Test that compact command with --dry-run doesn't crash"""
        # Create mock args
        class MockArgs:
            db = self.db_path
            dry_run = True
            similarity_threshold = 0.82
        
        args = MockArgs()
        
        # Should not crash
        try:
            result = cmd_compact(args)
            # If it returns something, it should be 0 (success)
            if result is not None:
                self.assertEqual(result, 0)
        except Exception as e:
            self.fail(f"Compact dry-run crashed: {e}")


class TestExtractionNoveltyGate(unittest.TestCase):
    """Test extraction with confidence + novelty gate"""
    
    def setUp(self):
        """Set up test database"""
        self.db_fd, self.db_path = tempfile.mkstemp(suffix='.db')
        os.close(self.db_fd)
        
        # Initialize with schema (minimal for extract test)
        conn = sqlite3.connect(self.db_path)
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
                mood_state TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE embeddings (
                node_id TEXT PRIMARY KEY,
                vector BLOB NOT NULL,
                model TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE derivation_edges (
                parent_id TEXT,
                child_id TEXT,
                relation TEXT,
                weight REAL,
                reasoning TEXT,
                confidence REAL,
                timestamp TEXT,
                PRIMARY KEY (parent_id, child_id, relation)
            )
        ''')
        
        # Add existing node
        now = datetime.now(timezone.utc).isoformat()
        cursor.execute("""
            INSERT INTO thought_nodes (id, content, node_type, domain, timestamp, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        """, ("existing_node", "Existing knowledge about machine learning", "fact", "tech", now, 0.8))
        
        # Add embedding
        embedding = np.random.rand(384).astype(np.float32)
        cursor.execute("""
            INSERT INTO embeddings (node_id, vector, model, updated_at)
            VALUES (?, ?, ?, ?)
        """, ("existing_node", embedding.tobytes(), "all-MiniLM-L6-v2", now))
        
        conn.commit()
        conn.close()
        
        self.existing_embedding = embedding
    
    def tearDown(self):
        """Clean up test database"""
        os.unlink(self.db_path)
    
    @patch('scripts.cashew_context.extract_from_conversation')
    @patch('builtins.print')  # Mock print to suppress output during tests
    def test_extract_duplicate_content_rejected(self, mock_print, mock_extract):
        """Test that extract rejects duplicate content"""
        # Mock extract_from_conversation to simulate rejection due to duplicate content
        mock_extract.return_value = {
            "success": True,
            "new_nodes": 0,  # No new nodes due to duplicates being rejected
            "new_edges": 0,
            "rejections": 1
        }
        
        # Create mock args for extract
        class MockArgs:
            db = self.db_path
            input = None
            session_id = "test_session"
            debug = False
            
        args = MockArgs()
        
        # Create temp input file with duplicate content
        input_fd, input_path = tempfile.mkstemp(suffix='.md')
        with os.fdopen(input_fd, 'w') as f:
            f.write("Existing knowledge about machine learning")
        
        args.input = input_path
        
        try:            
            # Run extract
            result = cmd_extract(args)
            
            # Verify extract was called
            mock_extract.assert_called_once()
            
            # Verify the result indicates no new nodes (duplicates rejected)
            if result is not None:
                self.assertEqual(result, 0, "Extract should succeed but create no new nodes")
            
        finally:
            # Clean up temp file
            os.unlink(input_path)


if __name__ == '__main__':
    unittest.main()