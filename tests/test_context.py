#!/usr/bin/env python3
"""
Tests for context retrieval module
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.context import ContextRetriever

class TestContextRetrieval(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.retriever = ContextRetriever()
    
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
            self.assertIn("Raj's existing reasoning", context)
            
            # Should contain confidence scores
            self.assertIn("Confidence:", context)
    
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