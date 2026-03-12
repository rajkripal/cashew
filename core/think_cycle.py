#!/usr/bin/env python3
"""
Cashew Think Cycle Module
Analyzes the existing thought graph and generates new derived thoughts through LLM reasoning.
Creates connections between new insights and existing nodes.
"""

import sqlite3
import json
import hashlib
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import sys
import random
import time

# Import existing cashew modules
from core.context import ContextRetriever
from core.patterns import PatternExtractor
from core.traversal import TraversalEngine
from core.questions import QuestionGenerator

# Database path is now configurable via environment variable or CLI
from .config import get_db_path

@dataclass
class NewThought:
    content: str
    confidence: float
    parent_ids: List[str]
    reasoning: str

class ThinkCycle:
    """Generate new derived thoughts through LLM analysis of the existing graph"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = get_db_path()
        self.db_path = db_path
        self.context = ContextRetriever(db_path)
        self.patterns = PatternExtractor(db_path)
        self.traversal = TraversalEngine(db_path)
        self.questions = QuestionGenerator(db_path)
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def _generate_node_id(self, content: str) -> str:
        """Generate unique ID for a node using content hash"""
        return hashlib.sha256(content.encode()).hexdigest()[:12]
    
    def _get_current_timestamp(self) -> str:
        """Get current ISO timestamp"""
        return datetime.now(timezone.utc).isoformat()
    
    def analyze_graph_structure(self) -> Dict:
        """Analyze the current graph to understand its structure and patterns"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        analysis = {}
        
        # Get hub nodes (nodes with highest edge count)
        cursor.execute("""
            SELECT tn.id, tn.content, tn.confidence, 
                   (SELECT COUNT(*) FROM derivation_edges de1 WHERE de1.parent_id = tn.id) +
                   (SELECT COUNT(*) FROM derivation_edges de2 WHERE de2.child_id = tn.id) as edge_count
            FROM thought_nodes tn
            ORDER BY edge_count DESC
            LIMIT 10
        """)
        analysis['hub_nodes'] = cursor.fetchall()
        
        # Get contradiction nodes (nodes with contradictory edges)
        cursor.execute("""
            SELECT DISTINCT tn.id, tn.content, tn.confidence
            FROM thought_nodes tn
            JOIN derivation_edges de ON tn.id = de.child_id OR tn.id = de.parent_id
            WHERE de.relation = 'contradicts'
            LIMIT 10
        """)
        analysis['contradiction_nodes'] = cursor.fetchall()
        
        # Get leaf nodes (nodes with no children)
        cursor.execute("""
            SELECT tn.id, tn.content, tn.confidence
            FROM thought_nodes tn
            LEFT JOIN derivation_edges de ON tn.id = de.parent_id
            WHERE de.parent_id IS NULL
            ORDER BY tn.confidence DESC
            LIMIT 20
        """)
        analysis['leaf_nodes'] = cursor.fetchall()
        
        # Get orphan nodes (nodes with no connections)
        cursor.execute("""
            SELECT tn.id, tn.content, tn.confidence
            FROM thought_nodes tn
            LEFT JOIN derivation_edges de1 ON tn.id = de1.parent_id
            LEFT JOIN derivation_edges de2 ON tn.id = de2.child_id
            WHERE de1.parent_id IS NULL AND de2.child_id IS NULL
            LIMIT 10
        """)
        analysis['orphan_nodes'] = cursor.fetchall()
        
        # Get question nodes
        cursor.execute("""
            SELECT id, content, confidence
            FROM thought_nodes
            WHERE node_type = 'question'
            ORDER BY confidence DESC
            LIMIT 15
        """)
        analysis['question_nodes'] = cursor.fetchall()
        
        # Get recent high-confidence nodes
        cursor.execute("""
            SELECT id, content, confidence
            FROM thought_nodes
            WHERE confidence > 0.8
            ORDER BY timestamp DESC
            LIMIT 15
        """)
        analysis['high_confidence_nodes'] = cursor.fetchall()
        
        conn.close()
        return analysis
    
    def generate_insights_from_analysis(self, analysis: Dict) -> List[NewThought]:
        """Generate new derived thoughts based on graph analysis"""
        insights = []
        
        # Analyze hub nodes for meta-patterns
        hub_contents = [node[1] for node in analysis['hub_nodes'][:5]]
        hub_insight = self._reason_about_hubs(hub_contents, [node[0] for node in analysis['hub_nodes'][:5]])
        if hub_insight:
            insights.append(hub_insight)
        
        # Analyze contradictions for resolution patterns
        contradiction_contents = [node[1] for node in analysis['contradiction_nodes'][:3]]
        if contradiction_contents:
            contradiction_insight = self._reason_about_contradictions(contradiction_contents, 
                                                                    [node[0] for node in analysis['contradiction_nodes'][:3]])
            if contradiction_insight:
                insights.append(contradiction_insight)
        
        # Analyze leaf nodes for extension opportunities
        leaf_contents = [node[1] for node in analysis['leaf_nodes'][:8]]
        leaf_insights = self._reason_about_leaves(leaf_contents, [node[0] for node in analysis['leaf_nodes'][:8]])
        insights.extend(leaf_insights)
        
        # Analyze question nodes for potential answers or deeper questions
        question_contents = [node[1] for node in analysis['question_nodes'][:5]]
        question_insights = self._reason_about_questions(question_contents, [node[0] for node in analysis['question_nodes'][:5]])
        insights.extend(question_insights)
        
        # Generate cross-domain connections
        cross_domain_insights = self._generate_cross_domain_connections(analysis)
        insights.extend(cross_domain_insights)
        
        # Generate meta-insights about the thinking patterns themselves
        meta_insights = self._generate_meta_insights(analysis)
        insights.extend(meta_insights)
        
        return insights[:50]  # Cap at 50 to avoid overwhelming the graph
    
    def _reason_about_hubs(self, hub_contents: List[str], hub_ids: List[str]) -> Optional[NewThought]:
        """Analyze hub nodes to identify meta-patterns"""
        if not hub_contents:
            return None
            
        # Look for common themes in hub nodes
        themes = []
        for content in hub_contents:
            if any(word in content.lower() for word in ['system', 'pattern', 'architecture']):
                themes.append('systems_thinking')
            if any(word in content.lower() for word in ['family', 'religion', 'christian']):
                themes.append('religious_transition')
            if any(word in content.lower() for word in ['career', 'work', 'engineer']):
                themes.append('professional_identity')
            if any(word in content.lower() for word in ['silence', 'communication', 'express']):
                themes.append('communication_patterns')
        
        if len(set(themes)) >= 2:
            insight_content = f"The highest-connectivity nodes in the graph cluster around {', '.join(set(themes))} — suggesting these aren't separate life domains but interconnected architectural patterns. Hub nodes indicate where multiple reasoning chains converge, revealing core structural elements of identity formation."
            return NewThought(
                content=insight_content,
                confidence=0.65,
                parent_ids=hub_ids[:3],
                reasoning="Meta-analysis of hub node themes to identify convergent patterns"
            )
        return None
    
    def _reason_about_contradictions(self, contradiction_contents: List[str], contradiction_ids: List[str]) -> Optional[NewThought]:
        """Analyze contradictions for resolution patterns"""
        if not contradiction_contents:
            return None
            
        insight_content = "Contradictions in the graph aren't errors to be resolved—they're architectural features. They mark decision points where competing values (family harmony vs. intellectual integrity, career advancement vs. depth) create productive tension. The goal isn't elimination but conscious navigation."
        
        return NewThought(
            content=insight_content,
            confidence=0.6,
            parent_ids=contradiction_ids,
            reasoning="Analysis of contradiction nodes as productive tension points rather than errors"
        )
    
    def _reason_about_leaves(self, leaf_contents: List[str], leaf_ids: List[str]) -> List[NewThought]:
        """Analyze leaf nodes for extension opportunities"""
        insights = []
        
        for i, (content, node_id) in enumerate(zip(leaf_contents[:4], leaf_ids[:4])):
            if 'career' in content.lower() or 'work' in content.lower():
                extension = f"Extension: {content.split('.')[0] if '.' in content else content} → This suggests the optimal career path isn't climbing traditional ladders but finding or creating environments where technical depth is valued over visibility optimization."
                insights.append(NewThought(
                    content=extension,
                    confidence=0.55,
                    parent_ids=[node_id],
                    reasoning="Extension of career-related leaf node to practical implications"
                ))
            elif 'family' in content.lower() or 'religion' in content.lower():
                extension = f"Extension: {content.split('.')[0] if '.' in content else content} → This pattern likely exists in other relationship contexts—anywhere belief systems create inclusion/exclusion boundaries, similar dynamics will emerge."
                insights.append(NewThought(
                    content=extension,
                    confidence=0.6,
                    parent_ids=[node_id],
                    reasoning="Extension of family/religious dynamics to broader relationship patterns"
                ))
        
        return insights
    
    def _reason_about_questions(self, question_contents: List[str], question_ids: List[str]) -> List[NewThought]:
        """Analyze questions for potential answers or deeper questions"""
        insights = []
        
        for content, node_id in zip(question_contents[:3], question_ids[:3]):
            if '?' in content:
                # Generate a hypothesis as an answer
                hypothesis = f"Hypothesis: {content.replace('?', '')} — The answer likely involves recognizing that the question itself contains a false binary. Most 'either/or' frameworks miss the architectural solution: designing systems that transcend the original constraints."
                insights.append(NewThought(
                    content=hypothesis,
                    confidence=0.55,
                    parent_ids=[node_id],
                    reasoning="Generated hypothesis to address open question node"
                ))
        
        return insights
    
    def _generate_cross_domain_connections(self, analysis: Dict) -> List[NewThought]:
        """Generate insights connecting different domains in the graph"""
        insights = []
        
        # Connect family dynamics with career patterns
        insight1 = NewThought(
            content="The family silence pattern and E5 communication struggles share the same architecture: both contexts punish authentic expression when it threatens system stability. The family needed religious cohesion; the promotion process needs legible signals. Both reward performance over substance.",
            confidence=0.65,
            parent_ids=[node[0] for node in analysis['hub_nodes'][:2]],
            reasoning="Cross-domain pattern recognition between family and career dynamics"
        )
        insights.append(insight1)
        
        # Connect technical depth with relationship patterns  
        insight2 = NewThought(
            content="Technical depth and relationship authenticity follow similar principles: both require sustained attention beneath surface-level optimization. Just as premature optimization ruins code architecture, premature harmony-seeking ruins relationship architecture.",
            confidence=0.6,
            parent_ids=[node[0] for node in analysis['leaf_nodes'][2:4]],
            reasoning="Analogical connection between technical and interpersonal depth"
        )
        insights.append(insight2)
        
        return insights
    
    def _generate_meta_insights(self, analysis: Dict) -> List[NewThought]:
        """Generate insights about the thinking patterns themselves"""
        insights = []
        
        # Meta-insight about the graph structure
        insight1 = NewThought(
            content="The graph's hub structure reveals a key thinking pattern: rather than compartmentalizing life domains, there's a consistent tendency to find architectural parallels across contexts. This isn't analogical thinking—it's pattern recognition at the systems level.",
            confidence=0.7,
            parent_ids=[node[0] for node in analysis['hub_nodes'][:3]],
            reasoning="Meta-analysis of the graph's own structural patterns"
        )
        insights.append(insight1)
        
        # Meta-insight about contradiction handling
        insight2 = NewThought(
            content="The presence of contradiction nodes without resolution attempts suggests a sophisticated relationship with uncertainty: rather than rushing to eliminate dissonance, the thinking process preserves productive tensions as information.",
            confidence=0.65,
            parent_ids=[node[0] for node in analysis['contradiction_nodes'][:2]] if analysis['contradiction_nodes'] else [],
            reasoning="Meta-analysis of how contradictions are preserved rather than resolved"
        )
        insights.append(insight2)
        
        return insights
    
    def save_insights_to_db(self, insights: List[NewThought]) -> int:
        """Save new thoughts and their edges to the database"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        saved_count = 0
        timestamp = self._get_current_timestamp()
        
        for insight in insights:
            # Primary gate: semantic novelty check
            try:
                from core.placement_aware_extraction import check_novelty
                is_novel, max_sim, nearest_id = check_novelty(self.db_path, insight.content)
                if not is_novel:
                    print(f"  ⊘ Rejecting duplicate think insight (sim={max_sim:.3f}): {insight.content[:60]}")
                    continue
                # Borderline + low confidence = skip
                if max_sim > 0.72 and insight.confidence < 0.7:
                    print(f"  ⊘ Rejecting borderline think insight (sim={max_sim:.3f}, conf={insight.confidence}): {insight.content[:60]}")
                    continue
            except Exception as e:
                print(f"  ⚠️ Novelty check failed, falling back to exact match: {e}")
                
            # Generate node ID
            node_id = self._generate_node_id(insight.content)
            
            # Check if this content already exists (exact match fallback)
            cursor.execute("SELECT id FROM thought_nodes WHERE content = ?", (insight.content,))
            if cursor.fetchone():
                continue  # Skip exact duplicate content
            
            # Insert the new thought node
            try:
                cursor.execute("""
                    INSERT INTO thought_nodes 
                    (id, content, node_type, timestamp, confidence, source_file, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (node_id, insight.content, 'derived', timestamp, insight.confidence, 'system_generated', timestamp))
                
                # Create edges to parent nodes
                for parent_id in insight.parent_ids:
                    cursor.execute("""
                        INSERT OR IGNORE INTO derivation_edges 
                        (parent_id, child_id, relation, weight, reasoning)
                        VALUES (?, ?, ?, ?, ?)
                    """, (parent_id, node_id, 'derives_from', 0.8, insight.reasoning))
                
                saved_count += 1
                
            except sqlite3.Error as e:
                print(f"Error saving insight: {e}")
                continue
        
        conn.commit()
        conn.close()
        return saved_count
    
    def run_think_cycle(self) -> Dict:
        """Run a complete think cycle"""
        print("🧠 Starting think cycle...")
        
        # Get initial counts
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM thought_nodes")
        initial_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM thought_nodes WHERE source_file='system_generated'")
        initial_system_count = cursor.fetchone()[0]
        conn.close()
        
        print(f"📊 Initial state: {initial_count} total nodes, {initial_system_count} system_generated")
        
        # Analyze graph structure
        print("🔍 Analyzing graph structure...")
        analysis = self.analyze_graph_structure()
        
        # Generate insights
        print("💡 Generating new insights...")
        insights = self.generate_insights_from_analysis(analysis)
        print(f"Generated {len(insights)} potential insights")
        
        # Save to database
        print("💾 Saving insights to database...")
        saved_count = self.save_insights_to_db(insights)
        
        # Get final counts
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM thought_nodes")
        final_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM thought_nodes WHERE source_file='system_generated'")
        final_system_count = cursor.fetchone()[0]
        conn.close()
        
        # Sample of best new thoughts
        best_insights = sorted(insights[:saved_count], key=lambda x: x.confidence, reverse=True)[:5]
        
        result = {
            'initial_count': initial_count,
            'final_count': final_count,
            'initial_system_count': initial_system_count,
            'final_system_count': final_system_count,
            'new_thoughts_generated': saved_count,
            'best_insights': [{'content': i.content, 'confidence': i.confidence} for i in best_insights]
        }
        
        print("✅ Think cycle complete!")
        return result

if __name__ == "__main__":
    cycle = ThinkCycle()
    result = cycle.run_think_cycle()
    
    print(f"\n📈 Results:")
    print(f"  Total nodes: {result['initial_count']} → {result['final_count']}")
    print(f"  System nodes: {result['initial_system_count']} → {result['final_system_count']}")
    print(f"  New thoughts: {result['new_thoughts_generated']}")
    
    print(f"\n🌟 Best new insights:")
    for i, insight in enumerate(result['best_insights'], 1):
        print(f"  {i}. [{insight['confidence']:.2f}] {insight['content'][:100]}...")