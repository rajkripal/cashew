#!/usr/bin/env python3
"""
Tests for context retrieval module
"""

import unittest
import sys
import os
import sqlite3
import tempfile
import json
from datetime import datetime, timezone

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.context import ContextRetriever

class TestContextRetrieval(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a temporary database for testing
        self.db_fd, self.db_path = tempfile.mkstemp(suffix='.db')
        os.close(self.db_fd)
        
        # Initialize with proper schema and test data
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create the schema
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
                mood_state TEXT,
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
        
        # Add test data for retrieval testing
        now = datetime.now(timezone.utc).isoformat()
        test_nodes = [
            ("god_seed", "God exists and created the universe", "seed", "philosophy", 0.9),
            ("christian_belief", "Christianity provides salvation through faith", "belief", "philosophy", 0.8),
            ("system_thinking", "Systems thinking reveals interconnected patterns", "insight", "meta", 0.7),
            ("religion_general", "Religion offers meaning and community", "observation", "philosophy", 0.6),
            ("programming_fact", "Python is used for data analysis", "fact", "tech", 0.8)
        ]
        
        for node_id, content, node_type, domain, confidence in test_nodes:
            cursor.execute("""
                INSERT INTO thought_nodes 
                (id, content, node_type, domain, timestamp, confidence, source_file)
                VALUES (?, ?, ?, ?, ?, ?, 'test')
            """, (node_id, content, node_type, domain, now, confidence))
        
        conn.commit()
        conn.close()
        
        self.retriever = ContextRetriever(self.db_path)
    
    def tearDown(self):
        """Clean up test database"""
        os.unlink(self.db_path)
    
    def test_keyword_extraction(self):
        """Test keyword extraction from queries"""
        keywords = self.retriever._extract_keywords("God religion Christianity belief")
        
        self.assertIn("god", keywords)
        self.assertIn("religion", keywords)
        self.assertIn("christianity", keywords)
        self.assertIn("belief", keywords)
        
        # Should filter stopwords
        self.assertNotIn("the", keywords)
        self.assertNotIn("and", keywords)
    
    def test_relevance_calculation(self):
        """Test relevance score calculation"""
        content = "God exists and provides salvation through faith"
        keywords = ["god", "faith", "salvation"]
        
        score = self.retriever._calculate_relevance_score(content, keywords)
        
        # Should be high relevance (all keywords match)
        self.assertGreater(score, 0.8)
    
    def test_god_query_returns_seed(self):
        """Test that querying 'God' returns the God seed node"""
        results = self.retriever.retrieve("God", max_nodes=5)
        
        # Should return at least one result
        self.assertGreater(len(results), 0)
        
        # Check if any result mentions God
        god_mentioned = any("god" in node.content.lower() or "God" in node.content 
                           for node in results)
        self.assertTrue(god_mentioned, "Query for 'God' should return God-related nodes")
    
    def test_religion_query(self):
        """Test querying for religion-related content"""
        results = self.retriever.retrieve("religion Christianity", max_nodes=3)
        
        # Should return relevant results
        self.assertGreater(len(results), 0)
        
        # Results should have relevance scores
        for result in results:
            self.assertGreater(result.relevance_score, 0)
    
    def test_format_context(self):
        """Test context formatting for LLM injection"""
        results = self.retriever.retrieve("systems thinking", max_nodes=2)
        
        if results:
            context = self.retriever.format_context(results)
            
            # Should contain the header
            self.assertIn("Existing reasoning", context)
    
    def test_search_by_content(self):
        """Test searching for specific content fragments"""
        # Look for nodes containing "system"
        results = self.retriever.search_by_content("system", max_nodes=3)
        
        if results:
            # All results should contain the search term
            for result in results:
                self.assertIn("system", result.content.lower())


if __name__ == "__main__":
    unittest.main()