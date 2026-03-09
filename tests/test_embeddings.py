#!/usr/bin/env python3
"""
Tests for cashew.core.embeddings module
"""

import pytest
import sqlite3
import tempfile
import os
import numpy as np
from typing import List

import sys
sys.path.insert(0, '/Users/bunny/.openclaw/workspace/cashew')

from core.embeddings import embed_text, embed_nodes, search, get_embedding_stats

class TestEmbeddings:
    """Test the embeddings functionality"""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary test database with sample data"""
        fd, db_path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        
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
        
        # Insert test nodes with diverse content
        test_nodes = [
            ("node1", "The weather is sunny and warm today", "observation", "2023-01-01T10:00:00", 0.9, "happy", "{}", "test"),
            ("node2", "I love sunny weather and outdoor activities", "belief", "2023-01-01T11:00:00", 0.8, "joyful", "{}", "test"),
            ("node3", "Python is a great programming language", "fact", "2023-01-01T12:00:00", 1.0, "confident", "{}", "test"),
            ("node4", "Programming in Python makes me productive", "belief", "2023-01-01T13:00:00", 0.7, "satisfied", "{}", "test"),
            ("node5", "Machine learning models need training data", "fact", "2023-01-01T14:00:00", 0.9, "focused", "{}", "test"),
            ("node6", "Today's rain makes me feel gloomy", "observation", "2023-01-01T15:00:00", 0.6, "sad", "{}", "test"),
            ("node7", "", "empty", "2023-01-01T16:00:00", 0.0, "neutral", "{}", "test"),  # Empty content
            ("node8", "Decayed node should be ignored", "observation", "2023-01-01T17:00:00", 0.5, "neutral", "{}", "test")  # Will be marked as decayed
        ]
        
        cursor.executemany("""
            INSERT INTO thought_nodes 
            (id, content, node_type, timestamp, confidence, mood_state, metadata, source_file, decayed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
        """, test_nodes)
        
        # Mark one node as decayed
        cursor.execute("UPDATE thought_nodes SET decayed = 1 WHERE id = 'node8'")
        
        conn.commit()
        conn.close()
        
        yield db_path
        
        # Cleanup
        os.unlink(db_path)
    
    def test_embed_text_returns_correct_dimension(self):
        """Test that embed_text returns 384-dimensional vectors"""
        text = "This is a test sentence"
        embedding = embed_text(text)
        
        assert isinstance(embedding, list)
        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)
    
    def test_embed_text_handles_empty_string(self):
        """Test that embed_text handles empty strings gracefully"""
        embedding = embed_text("")
        assert len(embedding) == 384
        assert all(x == 0.0 for x in embedding)
        
        embedding = embed_text("   ")  # Whitespace only
        assert len(embedding) == 384
        assert all(x == 0.0 for x in embedding)
    
    def test_embed_text_similar_content_similar_embeddings(self):
        """Test that similar texts produce similar embeddings"""
        text1 = "I love sunny weather"
        text2 = "I enjoy sunny days"
        text3 = "Python programming language"
        
        emb1 = np.array(embed_text(text1))
        emb2 = np.array(embed_text(text2))
        emb3 = np.array(embed_text(text3))
        
        # Calculate cosine similarities
        similarity_12 = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        similarity_13 = np.dot(emb1, emb3) / (np.linalg.norm(emb1) * np.linalg.norm(emb3))
        
        # Similar weather texts should be more similar than weather vs programming
        assert similarity_12 > similarity_13
        assert similarity_12 > 0.5  # Should be reasonably similar
    
    def test_embed_nodes_creates_embeddings_table(self, temp_db):
        """Test that embed_nodes creates the embeddings table"""
        stats = embed_nodes(temp_db)
        
        # Check that table was created
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='embeddings'")
        assert cursor.fetchone() is not None
        conn.close()
    
    def test_embed_nodes_embeds_all_non_decayed_nodes(self, temp_db):
        """Test that embed_nodes processes all non-decayed nodes"""
        stats = embed_nodes(temp_db)
        
        # Should embed 7 nodes (8 total - 1 decayed)
        assert stats["total_nodes"] == 7
        assert stats["embedded"] == 7
        assert stats["skipped"] == 0
        
        # Check that embeddings were actually stored
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM embeddings")
        embedding_count = cursor.fetchone()[0]
        assert embedding_count == 7
        
        # Check that decayed node was not embedded
        cursor.execute("SELECT node_id FROM embeddings WHERE node_id = 'node8'")
        assert cursor.fetchone() is None
        
        conn.close()
    
    def test_embed_nodes_skips_already_embedded(self, temp_db):
        """Test that embed_nodes doesn't re-embed already embedded nodes"""
        # First embedding
        stats1 = embed_nodes(temp_db)
        assert stats1["embedded"] == 7
        
        # Second embedding should skip all
        stats2 = embed_nodes(temp_db)
        assert stats2["total_nodes"] == 0  # No nodes need embedding
        assert stats2["embedded"] == 0
        assert stats2["skipped"] == 0
    
    def test_search_returns_ranked_results(self, temp_db):
        """Test that search returns results ranked by similarity"""
        # First embed the nodes
        embed_nodes(temp_db)
        
        # Search for weather-related content
        results = search(temp_db, "sunny weather", top_k=5)
        
        assert len(results) > 0
        assert len(results) <= 5
        
        # Results should be tuples of (node_id, score)
        for node_id, score in results:
            assert isinstance(node_id, str)
            assert isinstance(score, float)
            assert -1.0 <= score <= 1.0  # Cosine similarity range
        
        # Results should be sorted by score (descending)
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)
        
        # Check that weather-related nodes are at the top
        top_node_ids = [node_id for node_id, _ in results[:2]]
        
        # Load node contents to verify
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute(f"SELECT id, content FROM thought_nodes WHERE id IN ({','.join('?' * len(top_node_ids))})", top_node_ids)
        top_contents = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()
        
        # At least one of the top results should be weather-related
        weather_found = any("sunny" in content or "weather" in content for content in top_contents.values())
        assert weather_found
    
    def test_search_handles_empty_query(self, temp_db):
        """Test that search handles empty queries gracefully"""
        embed_nodes(temp_db)
        
        assert search(temp_db, "", top_k=5) == []
        assert search(temp_db, "   ", top_k=5) == []
    
    def test_search_respects_top_k_limit(self, temp_db):
        """Test that search respects the top_k parameter"""
        embed_nodes(temp_db)
        
        # Test different top_k values
        results_3 = search(temp_db, "programming", top_k=3)
        results_10 = search(temp_db, "programming", top_k=10)
        
        assert len(results_3) <= 3
        assert len(results_10) <= 7  # Max available nodes (excluding empty and decayed)
        
        # First 3 results should be the same
        for i in range(min(3, len(results_3))):
            assert results_3[i][0] == results_10[i][0]
    
    def test_get_embedding_stats_returns_correct_info(self, temp_db):
        """Test that get_embedding_stats returns accurate statistics"""
        # Before embedding
        stats = get_embedding_stats(temp_db)
        assert stats["total_nodes"] == 7  # Non-decayed nodes
        assert stats["embedded_nodes"] == 0
        assert stats["missing_embeddings"] == 7
        assert stats["coverage_percentage"] == 0.0
        
        # After embedding
        embed_nodes(temp_db)
        stats = get_embedding_stats(temp_db)
        assert stats["total_nodes"] == 7
        assert stats["embedded_nodes"] == 7
        assert stats["missing_embeddings"] == 0
        assert stats["coverage_percentage"] == 100.0
        assert "all-MiniLM-L6-v2" in stats["models_used"]
        assert stats["last_updated"] is not None
    
    def test_embedding_storage_format(self, temp_db):
        """Test that embeddings are stored and retrieved correctly"""
        embed_nodes(temp_db)
        
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        
        # Get a stored embedding
        cursor.execute("SELECT vector FROM embeddings WHERE node_id = 'node1'")
        vector_bytes = cursor.fetchone()[0]
        
        # Convert back to numpy array
        stored_embedding = np.frombuffer(vector_bytes, dtype=np.float32)
        
        # Should be 384 dimensions
        assert len(stored_embedding) == 384
        
        # Should match what embed_text produces for the same content
        cursor.execute("SELECT content FROM thought_nodes WHERE id = 'node1'")
        content = cursor.fetchone()[0]
        fresh_embedding = np.array(embed_text(content))
        
        # Should be very close (allowing for minor float precision differences)
        # Embeddings can have small differences due to model randomness and storage precision
        np.testing.assert_allclose(stored_embedding, fresh_embedding, rtol=1e-3, atol=1e-6)
        
        conn.close()
    
    def test_batch_processing(self, temp_db):
        """Test that batch processing works with different batch sizes"""
        # Test with batch size 1
        stats1 = embed_nodes(temp_db, batch_size=1)
        assert stats1["embedded"] == 7
        
        # Clear embeddings
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM embeddings")
        conn.commit()
        conn.close()
        
        # Test with batch size 10 (larger than number of nodes)
        stats2 = embed_nodes(temp_db, batch_size=10)
        assert stats2["embedded"] == 7
        
        # Results should be identical
        results1 = search(temp_db, "sunny weather", top_k=5)
        
        # Clear and re-embed with different batch size
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM embeddings")
        conn.commit()
        conn.close()
        
        embed_nodes(temp_db, batch_size=3)
        results2 = search(temp_db, "sunny weather", top_k=5)
        
        # Should get same results regardless of batch size
        assert len(results1) == len(results2)
        for i in range(len(results1)):
            assert results1[i][0] == results2[i][0]  # Same node IDs
            assert abs(results1[i][1] - results2[i][1]) < 1e-5  # Same scores (within precision)