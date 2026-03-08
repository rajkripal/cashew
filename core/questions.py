#!/usr/bin/env python3
"""
Cashew Question Generation
Self-generated questions about graph gaps, tensions, and weak edges
"""

import sqlite3
import json
import hashlib
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass
import argparse
import sys
from datetime import datetime

DB_PATH = "/Users/bunny/.openclaw/workspace/cashew/data/graph.db"

@dataclass
class QuestionCandidate:
    content: str
    reasoning: str
    source_nodes: List[str]
    gap_type: str  # "leaf", "contradiction", "weak_edge", "orphan", "cycle"
    confidence: float

class QuestionGenerator:
    """Generate questions about graph gaps and tensions"""
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def _load_node(self, node_id: str) -> Optional[Dict]:
        """Load a single node"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, content, node_type, confidence, mood_state, metadata, source_file
            FROM thought_nodes WHERE id = ?
        """, (node_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "id": row[0],
                "content": row[1],
                "node_type": row[2],
                "confidence": row[3],
                "mood_state": row[4],
                "metadata": json.loads(row[5]) if row[5] else {},
                "source_file": row[6]
            }
        return None
    
    def find_leaf_nodes(self) -> List[str]:
        """Find leaf nodes (no children) that could spawn questions"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
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
        return leaf_nodes
    
    def find_contradictions(self) -> List[Tuple[str, str, str]]:
        """Find contradiction edges"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT de.parent_id, de.child_id, de.reasoning
            FROM derivation_edges de
            WHERE de.relation = 'contradicts'
        """)
        
        contradictions = [(row[0], row[1], row[2]) for row in cursor.fetchall()]
        conn.close()
        return contradictions
    
    def find_weak_edges(self, threshold: float = 0.3) -> List[Tuple[str, str, float, str]]:
        """Find edges with low weight"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT de.parent_id, de.child_id, de.weight, de.reasoning
            FROM derivation_edges de
            WHERE de.weight < ? AND de.relation != 'cross_link'
            ORDER BY de.weight ASC
        """, (threshold,))
        
        weak_edges = [(row[0], row[1], row[2], row[3]) for row in cursor.fetchall()]
        conn.close()
        return weak_edges
    
    def find_orphan_nodes(self) -> List[str]:
        """Find nodes with no parents that aren't seeds"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT tn.id 
            FROM thought_nodes tn
            LEFT JOIN derivation_edges de ON tn.id = de.child_id
            WHERE de.child_id IS NULL 
            AND tn.node_type != 'seed'
            AND tn.node_type != 'question'
            AND (tn.decayed = 0 OR tn.decayed IS NULL)
        """)
        
        orphans = [row[0] for row in cursor.fetchall()]
        conn.close()
        return orphans
    
    def detect_potential_cycles(self) -> List[List[str]]:
        """Detect potential circular reasoning patterns"""
        # Use the audit functionality from traversal
        try:
            from .traversal import TraversalEngine
            engine = TraversalEngine(self.db_path)
            audit_report = engine.audit()
            return audit_report.cycles
        except ImportError:
            return []
    
    def generate_leaf_questions(self, leaf_nodes: List[str]) -> List[QuestionCandidate]:
        """Generate questions about leaf nodes - focus on implications"""
        questions = []
        
        for node_id in leaf_nodes[:10]:  # Limit to avoid spam
            node = self._load_node(node_id)
            if not node:
                continue
            
            content = node["content"]
            
            # Generate implication questions for leaf nodes
            if node["node_type"] == "belief":
                question_content = f"If '{content[:50]}...' is true, what follows? What would this mean for other beliefs?"
                question_type = "implication"
                reasoning = "Belief node with no derived thoughts - needs exploration of consequences"
            elif node["node_type"] == "derived":
                question_content = f"If '{content[:50]}...' is correct, what practical implications does this have?"
                question_type = "implication"
                reasoning = "Derived thought with no further implications explored"
            elif node["node_type"] == "core_memory":
                question_content = f"How does the memory '{content[:50]}...' shape current thinking patterns?"
                question_type = "implication"
                reasoning = "Core memory that may influence other thoughts"
            else:
                question_content = f"What consequences follow from '{content[:50]}...'?"
                question_type = "implication"
                reasoning = "Isolated thought that needs development"
            
            questions.append(QuestionCandidate(
                content=question_content,
                reasoning=reasoning,
                source_nodes=[node_id],
                gap_type="leaf",
                confidence=0.7
            ))
        
        return questions
    
    def generate_contradiction_questions(self, contradictions: List[Tuple[str, str, str]]) -> List[QuestionCandidate]:
        """Generate questions about contradictions - probe the tension"""
        questions = []
        
        for parent_id, child_id, reasoning in contradictions:
            parent = self._load_node(parent_id)
            child = self._load_node(child_id)
            
            if not parent or not child:
                continue
            
            # Generate tension-probing questions
            parent_short = parent['content'][:40] + "..." if len(parent['content']) > 40 else parent['content']
            child_short = child['content'][:40] + "..." if len(child['content']) > 40 else child['content']
            
            question_content = f"You believe '{parent_short}' but also '{child_short}' — what resolves this tension?"
            
            questions.append(QuestionCandidate(
                content=question_content,
                reasoning=f"Tension between contradictory beliefs: {reasoning}",
                source_nodes=[parent_id, child_id],
                gap_type="tension",
                confidence=0.9
            ))
        
        return questions
    
    def generate_weak_edge_questions(self, weak_edges: List[Tuple[str, str, float, str]]) -> List[QuestionCandidate]:
        """Generate questions about weak derivation links"""
        questions = []
        
        for parent_id, child_id, weight, reasoning in weak_edges[:5]:  # Top 5 weakest
            parent = self._load_node(parent_id)
            child = self._load_node(child_id)
            
            if not parent or not child:
                continue
            
            question_content = f"Why does '{parent['content'][:40]}...' lead to '{child['content'][:40]}...'?"
            
            questions.append(QuestionCandidate(
                content=question_content,
                reasoning=f"Weak derivation link (weight: {weight:.2f}): {reasoning}",
                source_nodes=[parent_id, child_id],
                gap_type="weak_edge",
                confidence=0.7
            ))
        
        return questions
    
    def generate_orphan_questions(self, orphan_nodes: List[str]) -> List[QuestionCandidate]:
        """Generate questions about orphaned thoughts - probe connections"""
        questions = []
        
        for node_id in orphan_nodes[:5]:  # Limit to avoid spam
            node = self._load_node(node_id)
            if not node:
                continue
            
            # Find nearest non-orphan node for connection probing
            nearest_node = self._find_nearest_connected_node(node_id)
            
            if nearest_node:
                question_content = f"How does '{node['content'][:40]}...' relate to '{nearest_node['content'][:40]}...'?"
                reasoning = f"Orphaned thought needs connection to existing reasoning network"
            else:
                question_content = f"What foundational beliefs or experiences led to '{node['content'][:50]}...'?"
                reasoning = "Isolated thought with no clear derivation or connections"
            
            questions.append(QuestionCandidate(
                content=question_content,
                reasoning=reasoning,
                source_nodes=[node_id],
                gap_type="connection",
                confidence=0.6
            ))
        
        return questions
    
    def _find_nearest_connected_node(self, orphan_id: str) -> Optional[Dict]:
        """Find the nearest non-orphan node for connection questions"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get connected nodes (nodes that have at least one edge)
        cursor.execute("""
            SELECT DISTINCT tn.id, tn.content, tn.node_type, tn.confidence
            FROM thought_nodes tn
            JOIN derivation_edges de ON (tn.id = de.parent_id OR tn.id = de.child_id)
            WHERE tn.node_type != 'question'
            AND (tn.decayed = 0 OR tn.decayed IS NULL)
            ORDER BY tn.confidence DESC
            LIMIT 5
        """)
        
        connected_nodes = []
        for row in cursor.fetchall():
            connected_nodes.append({
                "id": row[0],
                "content": row[1],
                "node_type": row[2],
                "confidence": row[3]
            })
        
        conn.close()
        
        # Return highest confidence connected node
        return connected_nodes[0] if connected_nodes else None
    
    def _determine_question_type(self, gap_type: str, question_content: str) -> str:
        """Determine question type based on gap type and content"""
        if gap_type == "tension" or "contradiction" in gap_type:
            return "tension"
        elif gap_type == "leaf" or "implication" in question_content.lower():
            return "implication"
        elif gap_type == "connection" or gap_type == "orphan":
            return "connection"
        elif "depth" in gap_type or "why" in question_content.lower():
            return "depth"
        else:
            return "exploration"
    
    def generate_cycle_questions(self, cycles: List[List[str]]) -> List[QuestionCandidate]:
        """Generate questions about circular reasoning"""
        questions = []
        
        for cycle in cycles[:3]:  # Limit cycles to examine
            if len(cycle) < 2:
                continue
            
            # Get the nodes in the cycle
            cycle_nodes = []
            for node_id in cycle:
                node = self._load_node(node_id)
                if node:
                    cycle_nodes.append(node)
            
            if len(cycle_nodes) < 2:
                continue
            
            # Create more specific circular reasoning question
            first_node = cycle_nodes[0]['content'][:30] + "..."
            last_node = cycle_nodes[-1]['content'][:30] + "..."
            
            question_content = f"Does '{first_node}' depend on '{last_node}' for its validity, creating circular logic?"
            
            questions.append(QuestionCandidate(
                content=question_content,
                reasoning="Potential circular reasoning detected in derivation chain",
                source_nodes=cycle,
                gap_type="cycle",
                confidence=0.9
            ))
        
        return questions
    
    def create_question_node(self, question: QuestionCandidate) -> str:
        """Create a question node in the database"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Generate question ID
        question_id = hashlib.sha256(question.content.encode()).hexdigest()[:12]
        
        # Create metadata with usefulness tracking and question type
        metadata = {
            "useful": None,  # To be set later based on whether it spawns derivations
            "gap_type": question.gap_type,
            "question_type": self._determine_question_type(question.gap_type, question.content),
            "source_nodes": question.source_nodes,
            "generated_reasoning": question.reasoning
        }
        
        # Insert question node
        cursor.execute("""
            INSERT OR REPLACE INTO thought_nodes
            (id, content, node_type, timestamp, confidence, mood_state, metadata, source_file)
            VALUES (?, ?, 'question', ?, ?, 'curious', ?, 'question_generator')
        """, (
            question_id,
            question.content,
            datetime.now().isoformat(),
            question.confidence,
            json.dumps(metadata)
        ))
        
        # Link question to source nodes
        for source_id in question.source_nodes:
            cursor.execute("""
                INSERT OR IGNORE INTO derivation_edges
                (parent_id, child_id, relation, weight, reasoning)
                VALUES (?, ?, 'questions', 0.8, ?)
            """, (source_id, question_id, question.reasoning))
        
        conn.commit()
        conn.close()
        
        return question_id
    
    def generate_all_questions(self) -> Dict[str, List[str]]:
        """Generate all types of questions"""
        print("🤔 Analyzing graph for question opportunities...")
        
        # Find different types of gaps
        leaf_nodes = self.find_leaf_nodes()
        contradictions = self.find_contradictions()
        weak_edges = self.find_weak_edges()
        orphan_nodes = self.find_orphan_nodes()
        cycles = self.detect_potential_cycles()
        
        print(f"Found: {len(leaf_nodes)} leaf nodes, {len(contradictions)} contradictions, "
              f"{len(weak_edges)} weak edges, {len(orphan_nodes)} orphans, {len(cycles)} cycles")
        
        # Generate questions
        all_questions = []
        
        all_questions.extend(self.generate_leaf_questions(leaf_nodes))
        all_questions.extend(self.generate_contradiction_questions(contradictions))
        all_questions.extend(self.generate_weak_edge_questions(weak_edges))
        all_questions.extend(self.generate_orphan_questions(orphan_nodes))
        all_questions.extend(self.generate_cycle_questions(cycles))
        
        # Create question nodes
        created_questions = defaultdict(list)
        
        for question in all_questions:
            question_id = self.create_question_node(question)
            created_questions[question.gap_type].append(question_id)
        
        return dict(created_questions)
    
    def get_question_stats(self) -> Dict:
        """Get statistics about questions and their usefulness"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Total questions
        cursor.execute("SELECT COUNT(*) FROM thought_nodes WHERE node_type = 'question'")
        total_questions = cursor.fetchone()[0]
        
        # Questions with children (spawned derivations)
        cursor.execute("""
            SELECT COUNT(DISTINCT tn.id)
            FROM thought_nodes tn
            JOIN derivation_edges de ON tn.id = de.parent_id
            WHERE tn.node_type = 'question'
        """)
        useful_questions = cursor.fetchone()[0]
        
        # Questions without children (dead ends)
        dead_end_questions = total_questions - useful_questions
        dead_end_ratio = dead_end_questions / total_questions if total_questions > 0 else 0
        
        # Question types distribution
        cursor.execute("""
            SELECT JSON_EXTRACT(metadata, '$.gap_type') as gap_type, COUNT(*)
            FROM thought_nodes 
            WHERE node_type = 'question' AND metadata IS NOT NULL
            GROUP BY gap_type
        """)
        
        gap_types = {}
        for row in cursor.fetchall():
            gap_type = row[0] or "unknown"
            count = row[1]
            gap_types[gap_type] = count
        
        # Average confidence of questions
        cursor.execute("""
            SELECT AVG(confidence) FROM thought_nodes WHERE node_type = 'question'
        """)
        avg_confidence = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            "total_questions": total_questions,
            "useful_questions": useful_questions,
            "dead_end_questions": dead_end_questions,
            "dead_end_ratio": dead_end_ratio,
            "gap_types": gap_types,
            "avg_confidence": avg_confidence
        }
    
    def update_question_usefulness(self):
        """Update usefulness metadata for questions based on whether they spawned derivations"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get all questions
        cursor.execute("""
            SELECT tn.id, tn.metadata
            FROM thought_nodes tn
            WHERE tn.node_type = 'question'
        """)
        
        questions = cursor.fetchall()
        updated_count = 0
        
        for question_id, metadata_str in questions:
            metadata = json.loads(metadata_str) if metadata_str else {}
            
            # Check if question has spawned any derivations
            cursor.execute("""
                SELECT COUNT(*) FROM derivation_edges WHERE parent_id = ?
            """, (question_id,))
            
            has_children = cursor.fetchone()[0] > 0
            
            # Update usefulness if not already set
            if metadata.get("useful") is None:
                metadata["useful"] = has_children
                
                cursor.execute("""
                    UPDATE thought_nodes SET metadata = ? WHERE id = ?
                """, (json.dumps(metadata), question_id))
                
                updated_count += 1
        
        conn.commit()
        conn.close()
        
        return updated_count


def main():
    """CLI interface for question generation"""
    parser = argparse.ArgumentParser(description="Cashew Question Generator")
    parser.add_argument("command", choices=["generate", "stats", "update"], help="Command to run")
    
    args = parser.parse_args()
    
    generator = QuestionGenerator()
    
    if args.command == "generate":
        created_questions = generator.generate_all_questions()
        
        print(f"\n❓ Generated Questions by Type:")
        total_created = 0
        for gap_type, question_ids in created_questions.items():
            print(f"  {gap_type}: {len(question_ids)} questions")
            total_created += len(question_ids)
        
        print(f"\n✅ Total questions created: {total_created}")
        
        # Show some example questions
        if total_created > 0:
            print(f"\n📝 Example Questions:")
            conn = sqlite3.connect(generator.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT content FROM thought_nodes 
                WHERE node_type = 'question' 
                ORDER BY timestamp DESC 
                LIMIT 5
            """)
            
            for i, (content,) in enumerate(cursor.fetchall(), 1):
                print(f"  {i}. {content}")
            
            conn.close()
    
    elif args.command == "stats":
        stats = generator.get_question_stats()
        
        print(f"\n📊 Question Statistics:")
        print(f"  Total questions: {stats['total_questions']}")
        print(f"  Useful questions (spawned derivations): {stats['useful_questions']}")
        print(f"  Dead-end questions: {stats['dead_end_questions']}")
        print(f"  Dead-end ratio: {stats['dead_end_ratio']:.2%}")
        print(f"  Average confidence: {stats['avg_confidence']:.2f}")
        
        if stats['gap_types']:
            print(f"\n📈 Questions by Gap Type:")
            for gap_type, count in stats['gap_types'].items():
                print(f"  {gap_type}: {count}")
    
    elif args.command == "update":
        updated = generator.update_question_usefulness()
        print(f"✅ Updated usefulness metadata for {updated} questions")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())