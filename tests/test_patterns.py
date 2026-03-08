#!/usr/bin/env python3
"""
Tests for pattern extraction module
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.patterns import PatternExtractor

class TestPatternExtraction(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.extractor = PatternExtractor()
    
    def test_edge_type_analysis(self):
        """Test analysis of edge relation types"""
        edge_types = self.extractor.analyze_edge_types()
        
        # Should return a dictionary of percentages
        self.assertIsInstance(edge_types, dict)
        
        # Percentages should sum to ~100 (allowing for rounding)
        total_pct = sum(edge_types.values())
        self.assertAlmostEqual(total_pct, 100, delta=5)
        
        # Should have common relation types
        expected_relations = ['derived_from', 'supports', 'contradicts', 'questions']
        for relation in expected_relations:
            if relation in edge_types:
                self.assertGreaterEqual(edge_types[relation], 0)
    
    def test_branching_factor_calculation(self):
        """Test branching factor analysis"""
        branching = self.extractor.calculate_branching_factor()
        
        # Should return branching metrics
        self.assertIn("avg_branching", branching)
        self.assertIn("max_branching", branching)
        
        # Values should be non-negative
        self.assertGreaterEqual(branching["avg_branching"], 0)
        self.assertGreaterEqual(branching["max_branching"], 0)
    
    def test_question_pattern_analysis(self):
        """Test question pattern extraction"""
        questions = self.extractor.analyze_question_patterns()
        
        # Should return question metrics
        self.assertIn("total_questions", questions)
        self.assertIn("question_type_distribution", questions)
        self.assertIn("gap_type_distribution", questions)
        
        # Total questions should be non-negative
        self.assertGreaterEqual(questions["total_questions"], 0)
        
        # If there are questions, should have type distribution
        if questions["total_questions"] > 0:
            type_dist = questions["question_type_distribution"]
            self.assertIsInstance(type_dist, dict)
    
    def test_contradiction_tolerance(self):
        """Test contradiction tolerance analysis"""
        contradictions = self.extractor.calculate_contradiction_tolerance()
        
        # Should return contradiction metrics
        self.assertIn("contradiction_ratio", contradictions)
        self.assertIn("total_contradictions", contradictions)
        self.assertIn("total_edges", contradictions)
        
        # Ratio should be between 0-100
        ratio = contradictions["contradiction_ratio"]
        self.assertGreaterEqual(ratio, 0)
        self.assertLessEqual(ratio, 100)
    
    def test_confidence_patterns(self):
        """Test confidence pattern analysis"""
        confidence = self.extractor.analyze_confidence_patterns()
        
        # Should return confidence metrics
        self.assertIn("by_type", confidence)
        self.assertIn("overall_avg", confidence)
        
        # Overall average should be between 0-1
        overall = confidence["overall_avg"]
        self.assertGreaterEqual(overall, 0)
        self.assertLessEqual(overall, 1)
    
    def test_full_pattern_extraction(self):
        """Test complete pattern extraction"""
        patterns = self.extractor.extract_patterns()
        
        # Should contain all major pattern categories
        expected_keys = [
            "edge_types", "chain_depths", "branching", 
            "questions", "contradictions", "confidence", 
            "temporal", "composite"
        ]
        
        for key in expected_keys:
            self.assertIn(key, patterns, f"Missing pattern category: {key}")
    
    def test_pattern_description(self):
        """Test human-readable pattern description"""
        description = self.extractor.describe_patterns()
        
        # Should return a non-empty string
        self.assertIsInstance(description, str)
        self.assertGreater(len(description), 0)
        
        # Should contain key headers
        self.assertIn("REASONING PATTERNS", description)
        self.assertIn("Primary Style", description)
    
    def test_composite_metrics(self):
        """Test composite metric calculation"""
        patterns = self.extractor.extract_patterns()
        composite = patterns.get("composite", {})
        
        # Should have composite reasoning style
        self.assertIn("primary_reasoning_style", composite)
        self.assertIn("exploration_style", composite)
        self.assertIn("inquiry_style", composite)
        self.assertIn("confidence_level", composite)
        
        # Values should be from expected sets
        valid_styles = ["derivational", "supportive", "critical", "exploratory"]
        self.assertIn(composite.get("primary_reasoning_style"), valid_styles)
        
        valid_exploration = ["depth-focused", "breadth-focused", "balanced"]
        self.assertIn(composite.get("exploration_style"), valid_exploration)


if __name__ == "__main__":
    unittest.main()