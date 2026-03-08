#!/usr/bin/env python3
"""
Cashew Pattern Extraction
Extract Raj's reasoning PATTERNS from the graph — not what he thinks, but HOW he thinks.
Analyzes graph structure for recurring patterns in reasoning style.
"""

import sqlite3
import json
import math
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import argparse
import sys
from datetime import datetime
from dataclasses import dataclass

DB_PATH = "/Users/bunny/.openclaw/workspace/cashew/data/graph.db"

@dataclass
class ReasoningPattern:
    pattern_type: str
    description: str
    metric_value: float
    evidence: List[str]
    confidence: float

class PatternExtractor:
    """Extract reasoning patterns from the thought graph"""
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def analyze_edge_types(self) -> Dict[str, float]:
        """Analyze distribution of edge relation types"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT relation, COUNT(*) as count
            FROM derivation_edges
            GROUP BY relation
            ORDER BY count DESC
        """)
        
        edge_counts = {}
        total_edges = 0
        
        for relation, count in cursor.fetchall():
            edge_counts[relation] = count
            total_edges += count
        
        conn.close()
        
        # Convert to percentages
        edge_percentages = {}
        for relation, count in edge_counts.items():
            edge_percentages[relation] = (count / total_edges * 100) if total_edges > 0 else 0
        
        return edge_percentages
    
    def calculate_chain_depths(self) -> Dict[str, float]:
        """Calculate average chain depth from seeds to leaves"""
        try:
            from .traversal import TraversalEngine
            engine = TraversalEngine(self.db_path)
        except ImportError:
            return {"avg_depth": 0, "max_depth": 0, "depths": []}
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get all leaf nodes (no children, not questions)
        cursor.execute("""
            SELECT tn.id 
            FROM thought_nodes tn
            LEFT JOIN derivation_edges de ON tn.id = de.parent_id
            WHERE de.parent_id IS NULL 
            AND tn.node_type != 'question'
            AND (tn.decayed = 0 OR tn.decayed IS NULL)
        """)
        
        leaf_nodes = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        depths = []
        
        for leaf_id in leaf_nodes:
            chain = engine.why(leaf_id, max_depth=20)
            depth = self._calculate_chain_depth(chain)
            if depth > 0:
                depths.append(depth)
        
        if not depths:
            return {"avg_depth": 0, "max_depth": 0, "depths": []}
        
        return {
            "avg_depth": sum(depths) / len(depths),
            "max_depth": max(depths),
            "min_depth": min(depths),
            "depths": depths
        }
    
    def _calculate_chain_depth(self, chain: List[Dict]) -> int:
        """Calculate depth of a single derivation chain"""
        if not chain or any("error" in step or "cycle_detected" in step for step in chain):
            return 0
        
        def get_max_depth(step: dict, current_depth: int = 0) -> int:
            max_d = current_depth
            if "derived_from" in step:
                for derivation in step["derived_from"]:
                    if "parent_chain" in derivation and derivation["parent_chain"]:
                        for parent_step in derivation["parent_chain"]:
                            depth = get_max_depth(parent_step, current_depth + 1)
                            max_d = max(max_d, depth)
            return max_d
        
        return get_max_depth(chain[0]) if chain else 0
    
    def calculate_branching_factor(self) -> Dict[str, float]:
        """Calculate how broadly vs deeply Raj explores ideas"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get outgoing edge counts for each node
        cursor.execute("""
            SELECT tn.id, tn.node_type, COUNT(de.child_id) as out_degree
            FROM thought_nodes tn
            LEFT JOIN derivation_edges de ON tn.id = de.parent_id
            WHERE (tn.decayed = 0 OR tn.decayed IS NULL)
            AND tn.node_type != 'question'
            GROUP BY tn.id, tn.node_type
        """)
        
        out_degrees = []
        type_degrees = defaultdict(list)
        
        for node_id, node_type, out_degree in cursor.fetchall():
            out_degrees.append(out_degree)
            type_degrees[node_type].append(out_degree)
        
        conn.close()
        
        if not out_degrees:
            return {"avg_branching": 0, "max_branching": 0, "type_branching": {}}
        
        # Calculate type-specific branching
        type_avg = {}
        for node_type, degrees in type_degrees.items():
            type_avg[node_type] = sum(degrees) / len(degrees)
        
        return {
            "avg_branching": sum(out_degrees) / len(out_degrees),
            "max_branching": max(out_degrees),
            "type_branching": type_avg,
            "branching_distribution": out_degrees
        }
    
    def analyze_question_patterns(self) -> Dict:
        """Categorize questions by type and analyze patterns"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT content, metadata FROM thought_nodes 
            WHERE node_type = 'question'
        """)
        
        questions = []
        gap_types = Counter()
        question_types = Counter()
        
        for content, metadata_str in cursor.fetchall():
            questions.append(content)
            
            # Parse metadata for gap type
            if metadata_str:
                try:
                    metadata = json.loads(metadata_str)
                    gap_type = metadata.get('gap_type', 'unknown')
                    gap_types[gap_type] += 1
                except json.JSONDecodeError:
                    gap_types['unknown'] += 1
            else:
                gap_types['unknown'] += 1
            
            # Categorize by question word
            content_lower = content.lower()
            if content_lower.startswith('why'):
                question_types['why'] += 1
            elif content_lower.startswith('what'):
                question_types['what'] += 1
            elif content_lower.startswith('how'):
                question_types['how'] += 1
            elif content_lower.startswith('when'):
                question_types['when'] += 1
            elif content_lower.startswith('where'):
                question_types['where'] += 1
            else:
                question_types['other'] += 1
        
        total_questions = len(questions)
        
        # Convert to percentages
        gap_percentages = {gap: (count / total_questions * 100) if total_questions > 0 else 0
                          for gap, count in gap_types.items()}
        
        type_percentages = {qtype: (count / total_questions * 100) if total_questions > 0 else 0
                           for qtype, count in question_types.items()}
        
        conn.close()
        
        return {
            "total_questions": total_questions,
            "gap_type_distribution": gap_percentages,
            "question_type_distribution": type_percentages,
            "sample_questions": questions[:5]  # First 5 for examples
        }
    
    def calculate_contradiction_tolerance(self) -> Dict[str, float]:
        """Measure how comfortable Raj is with contradictory beliefs"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Count contradiction edges
        cursor.execute("""
            SELECT COUNT(*) FROM derivation_edges WHERE relation = 'contradicts'
        """)
        contradiction_count = cursor.fetchone()[0]
        
        # Count total edges
        cursor.execute("""
            SELECT COUNT(*) FROM derivation_edges
        """)
        total_edges = cursor.fetchone()[0]
        
        # Get contradiction examples
        cursor.execute("""
            SELECT de.reasoning, tn1.content, tn2.content
            FROM derivation_edges de
            JOIN thought_nodes tn1 ON de.parent_id = tn1.id
            JOIN thought_nodes tn2 ON de.child_id = tn2.id
            WHERE de.relation = 'contradicts'
            LIMIT 3
        """)
        
        contradictions = []
        for reasoning, content1, content2 in cursor.fetchall():
            contradictions.append({
                "reasoning": reasoning,
                "node1": content1[:60] + "...",
                "node2": content2[:60] + "..."
            })
        
        conn.close()
        
        contradiction_ratio = (contradiction_count / total_edges * 100) if total_edges > 0 else 0
        
        return {
            "contradiction_ratio": contradiction_ratio,
            "total_contradictions": contradiction_count,
            "total_edges": total_edges,
            "examples": contradictions
        }
    
    def analyze_confidence_patterns(self) -> Dict:
        """Analyze confidence levels across different node types and topics"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Confidence by node type
        cursor.execute("""
            SELECT node_type, AVG(confidence), MIN(confidence), MAX(confidence), COUNT(*)
            FROM thought_nodes 
            WHERE (decayed = 0 OR decayed IS NULL)
            GROUP BY node_type
        """)
        
        confidence_by_type = {}
        for node_type, avg_conf, min_conf, max_conf, count in cursor.fetchall():
            confidence_by_type[node_type] = {
                "avg": avg_conf,
                "min": min_conf,
                "max": max_conf,
                "count": count
            }
        
        # Overall confidence distribution
        cursor.execute("""
            SELECT confidence FROM thought_nodes 
            WHERE (decayed = 0 OR decayed IS NULL)
            ORDER BY confidence
        """)
        
        confidences = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        
        return {
            "by_type": confidence_by_type,
            "overall_avg": sum(confidences) / len(confidences) if confidences else 0,
            "overall_median": confidences[len(confidences)//2] if confidences else 0,
            "distribution": confidences
        }
    
    def analyze_temporal_patterns(self) -> Dict:
        """Analyze how reasoning patterns change over time"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get nodes by timestamp
        cursor.execute("""
            SELECT timestamp, node_type, confidence, content
            FROM thought_nodes 
            WHERE (decayed = 0 OR decayed IS NULL)
            AND timestamp IS NOT NULL
            ORDER BY timestamp
        """)
        
        temporal_data = []
        confidence_over_time = []
        types_over_time = defaultdict(list)
        
        for timestamp, node_type, confidence, content in cursor.fetchall():
            temporal_data.append({
                "timestamp": timestamp,
                "type": node_type,
                "confidence": confidence,
                "content_length": len(content)
            })
            
            confidence_over_time.append(confidence)
            types_over_time[node_type].append(timestamp)
        
        conn.close()
        
        # Analyze confidence trajectory
        if len(confidence_over_time) >= 5:
            early_avg = sum(confidence_over_time[:5]) / 5
            late_avg = sum(confidence_over_time[-5:]) / 5
            confidence_trend = "increasing" if late_avg > early_avg else "decreasing"
        else:
            confidence_trend = "insufficient_data"
            early_avg = late_avg = 0
        
        return {
            "total_timepoints": len(temporal_data),
            "confidence_trend": confidence_trend,
            "early_confidence": early_avg,
            "late_confidence": late_avg,
            "types_by_period": dict(types_over_time)
        }
    
    def extract_patterns(self) -> Dict:
        """Extract all reasoning patterns"""
        print("🧠 Extracting reasoning patterns...")
        
        patterns = {}
        
        # Edge type patterns
        patterns["edge_types"] = self.analyze_edge_types()
        
        # Depth patterns
        patterns["chain_depths"] = self.calculate_chain_depths()
        
        # Branching patterns
        patterns["branching"] = self.calculate_branching_factor()
        
        # Question patterns
        patterns["questions"] = self.analyze_question_patterns()
        
        # Contradiction tolerance
        patterns["contradictions"] = self.calculate_contradiction_tolerance()
        
        # Confidence patterns
        patterns["confidence"] = self.analyze_confidence_patterns()
        
        # Temporal patterns
        patterns["temporal"] = self.analyze_temporal_patterns()
        
        # Calculate composite metrics
        patterns["composite"] = self._calculate_composite_metrics(patterns)
        
        return patterns
    
    def _calculate_composite_metrics(self, patterns: Dict) -> Dict:
        """Calculate high-level composite reasoning metrics"""
        composite = {}
        
        # Primary reasoning style
        edge_types = patterns.get("edge_types", {})
        derived_pct = edge_types.get("derived_from", 0)
        supports_pct = edge_types.get("supports", 0)
        contradicts_pct = edge_types.get("contradicts", 0)
        
        if derived_pct > 50:
            primary_style = "derivational"
        elif supports_pct > 30:
            primary_style = "supportive"
        elif contradicts_pct > 10:
            primary_style = "critical"
        else:
            primary_style = "exploratory"
        
        composite["primary_reasoning_style"] = primary_style
        
        # Exploration vs Exploitation
        branching = patterns.get("branching", {}).get("avg_branching", 0)
        depth = patterns.get("chain_depths", {}).get("avg_depth", 0)
        
        if depth > branching * 2:
            exploration_style = "depth-focused"
        elif branching > depth * 2:
            exploration_style = "breadth-focused"
        else:
            exploration_style = "balanced"
        
        composite["exploration_style"] = exploration_style
        
        # Question-driven vs assertion-driven
        questions = patterns.get("questions", {})
        total_questions = questions.get("total_questions", 0)
        
        # Estimate total assertions (non-question nodes)
        total_assertions = 40  # Rough estimate from data we've seen
        question_ratio = total_questions / (total_questions + total_assertions) if total_assertions > 0 else 0
        
        if question_ratio > 0.3:
            inquiry_style = "question-driven"
        elif question_ratio < 0.1:
            inquiry_style = "assertion-driven"
        else:
            inquiry_style = "mixed"
        
        composite["inquiry_style"] = inquiry_style
        
        # Confidence evolution
        confidence = patterns.get("confidence", {})
        overall_avg = confidence.get("overall_avg", 0.5)
        
        if overall_avg > 0.8:
            confidence_level = "high"
        elif overall_avg < 0.5:
            confidence_level = "low"
        else:
            confidence_level = "moderate"
        
        composite["confidence_level"] = confidence_level
        
        return composite
    
    def describe_patterns(self) -> str:
        """Generate human-readable description of reasoning patterns"""
        patterns = self.extract_patterns()
        
        description = []
        description.append("🧠 RAJ'S REASONING PATTERNS")
        description.append("=" * 50)
        
        # Primary findings
        composite = patterns.get("composite", {})
        description.append(f"\n📊 Primary Style: {composite.get('primary_reasoning_style', 'unknown').title()}")
        description.append(f"🔍 Exploration: {composite.get('exploration_style', 'unknown').title()}")
        description.append(f"❓ Inquiry: {composite.get('inquiry_style', 'unknown').title()}")
        description.append(f"🎯 Confidence: {composite.get('confidence_level', 'unknown').title()}")
        
        # Edge type distribution
        edge_types = patterns.get("edge_types", {})
        if edge_types:
            description.append(f"\n🔗 Reasoning Relations:")
            for relation, pct in sorted(edge_types.items(), key=lambda x: x[1], reverse=True):
                description.append(f"  {relation}: {pct:.1f}%")
        
        # Depth and branching
        depths = patterns.get("chain_depths", {})
        branching = patterns.get("branching", {})
        if depths and branching:
            avg_depth = depths.get("avg_depth", 0)
            avg_branch = branching.get("avg_branching", 0)
            description.append(f"\n⛓️  Chain Analysis:")
            description.append(f"  Average depth: {avg_depth:.1f} layers")
            description.append(f"  Average branching: {avg_branch:.1f} children per node")
            
            if avg_depth > 3:
                description.append(f"  → Goes deep into ideas (depth > 3)")
            if avg_branch > 2:
                description.append(f"  → Explores multiple angles (branching > 2)")
        
        # Question patterns
        questions = patterns.get("questions", {})
        if questions:
            question_types = questions.get("question_type_distribution", {})
            description.append(f"\n❓ Question Patterns:")
            for qtype, pct in sorted(question_types.items(), key=lambda x: x[1], reverse=True):
                if pct > 5:  # Only show significant types
                    description.append(f"  {qtype}: {pct:.1f}%")
            
            # Dominant question type
            dominant = max(question_types.items(), key=lambda x: x[1]) if question_types else None
            if dominant and dominant[1] > 40:
                description.append(f"  → Primarily asks '{dominant[0]}' questions")
        
        # Contradiction tolerance
        contradictions = patterns.get("contradictions", {})
        if contradictions:
            ratio = contradictions.get("contradiction_ratio", 0)
            description.append(f"\n💭 Contradiction Tolerance: {ratio:.1f}%")
            
            if ratio > 15:
                description.append(f"  → High tolerance for contradictory ideas")
            elif ratio < 5:
                description.append(f"  → Prefers consistent belief systems")
            else:
                description.append(f"  → Moderate comfort with contradictions")
        
        # Confidence patterns
        confidence = patterns.get("confidence", {})
        if confidence:
            by_type = confidence.get("by_type", {})
            description.append(f"\n🎯 Confidence by Type:")
            for node_type, conf_data in sorted(by_type.items()):
                avg_conf = conf_data.get("avg", 0)
                description.append(f"  {node_type}: {avg_conf:.2f}")
        
        # Temporal trends
        temporal = patterns.get("temporal", {})
        if temporal:
            trend = temporal.get("confidence_trend", "unknown")
            if trend != "insufficient_data":
                early = temporal.get("early_confidence", 0)
                late = temporal.get("late_confidence", 0)
                description.append(f"\n⏰ Evolution:")
                description.append(f"  Confidence trend: {trend}")
                description.append(f"  Early avg: {early:.2f} → Late avg: {late:.2f}")
        
        return "\n".join(description)


def main():
    """CLI interface for pattern extraction"""
    parser = argparse.ArgumentParser(description="Cashew Pattern Extraction")
    parser.add_argument("command", choices=["analyze", "describe", "export"], help="Command to run")
    parser.add_argument("--output", help="Output file for export", default="patterns.json")
    
    args = parser.parse_args()
    
    extractor = PatternExtractor()
    
    if args.command == "analyze":
        patterns = extractor.extract_patterns()
        
        print(f"\n🧠 Reasoning Pattern Analysis:")
        print("=" * 50)
        
        # Show key metrics
        composite = patterns.get("composite", {})
        for key, value in composite.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        
        print(f"\n📊 Detailed Metrics:")
        
        # Edge types
        edge_types = patterns.get("edge_types", {})
        if edge_types:
            print(f"Edge Distribution:")
            for relation, pct in edge_types.items():
                print(f"  {relation}: {pct:.1f}%")
        
        # Depth and branching
        depths = patterns.get("chain_depths", {})
        branching = patterns.get("branching", {})
        print(f"Structural Metrics:")
        if depths:
            print(f"  Avg chain depth: {depths.get('avg_depth', 0):.1f}")
        if branching:
            print(f"  Avg branching: {branching.get('avg_branching', 0):.1f}")
        
        # Questions
        questions = patterns.get("questions", {})
        if questions:
            print(f"  Total questions: {questions.get('total_questions', 0)}")
        
        # Contradictions
        contradictions = patterns.get("contradictions", {})
        if contradictions:
            print(f"  Contradiction ratio: {contradictions.get('contradiction_ratio', 0):.1f}%")
    
    elif args.command == "describe":
        description = extractor.describe_patterns()
        print(description)
    
    elif args.command == "export":
        patterns = extractor.extract_patterns()
        
        with open(args.output, 'w') as f:
            json.dump(patterns, f, indent=2)
        
        print(f"✅ Patterns exported to: {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())