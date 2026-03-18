#!/usr/bin/env python3
"""
Cashew Edge Densification
Automatically create edges between related nodes to reduce graph sparsity.
Addresses the problem of too many orphan nodes by finding semantic connections.
"""

import sqlite3
import json
import re
from typing import List, Dict, Tuple, Set
from collections import defaultdict, Counter
import argparse
import sys
from datetime import datetime

# Database path is now configurable via environment variable or CLI
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core.config import get_db_path

DB_PATH = get_db_path()

class EdgeDensifier:
    """Densify graph by creating connections between semantically related nodes"""
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'were', 'will', 'with', 'i', 'you', 'they', 'we',
            'this', 'but', 'not', 'or', 'can', 'had', 'would', 'could',
            'what', 'when', 'where', 'why', 'how', 'all', 'also', 'just',
            'so', 'do', 'does', 'did', 'have', 'been', 'being', 'if', 'up',
            'then', 'than', 'more', 'very', 'much', 'now', 'here', 'there'
        }
        # Minimum overlapping keywords to consider connection
        self.keyword_threshold = 3
        # Minimum concept flow score for derived_from edge  
        self.concept_flow_threshold = 0.7
        
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract meaningful keywords from text"""
        # Normalize and tokenize
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter stopwords and short words
        keywords = {word for word in words 
                   if word not in self.stopwords and len(word) > 2}
        
        return keywords
    
    def _extract_concepts(self, text: str) -> Set[str]:
        """Extract key concepts (noun phrases, important terms)"""
        concepts = set()
        
        # Religious concepts
        religious_terms = {
            'god', 'jesus', 'christ', 'christianity', 'christian', 'bible', 
            'prayer', 'faith', 'belief', 'salvation', 'judgment', 'heaven',
            'hell', 'sin', 'grace', 'holy', 'spirit', 'resurrection',
            'atheism', 'atheist', 'religion', 'theology', 'doctrine'
        }
        
        # Philosophical concepts
        philosophical_terms = {
            'truth', 'evidence', 'reasoning', 'logic', 'morality', 'ethics',
            'existence', 'reality', 'consciousness', 'meaning', 'purpose',
            'system', 'pattern', 'depth', 'complexity'
        }
        
        # Personal/emotional concepts
        personal_terms = {
            'love', 'compassion', 'guilt', 'shame', 'fear', 'hope',
            'family', 'mother', 'father', 'child', 'passion', 'interest',
            'authenticity', 'identity', 'recognition', 'belonging'
        }
        
        text_lower = text.lower()
        all_concept_terms = religious_terms | philosophical_terms | personal_terms
        
        for term in all_concept_terms:
            if term in text_lower:
                concepts.add(term)
        
        # Also extract capitalized terms (likely important concepts)
        capitalized = re.findall(r'\b[A-Z][a-z]+\b', text)
        concepts.update(word.lower() for word in capitalized)
        
        return concepts
    
    def _calculate_keyword_overlap(self, keywords1: Set[str], keywords2: Set[str]) -> Tuple[int, float]:
        """Calculate keyword overlap between two sets"""
        intersection = keywords1.intersection(keywords2)
        union = keywords1.union(keywords2)
        
        overlap_count = len(intersection)
        jaccard_similarity = len(intersection) / len(union) if union else 0.0
        
        return overlap_count, jaccard_similarity
    
    def _calculate_concept_flow(self, node1_content: str, node2_content: str) -> float:
        """
        Calculate if node1 concepts logically flow into node2
        Higher score means node2 likely derives from node1
        """
        concepts1 = self._extract_concepts(node1_content)
        concepts2 = self._extract_concepts(node2_content)
        
        if not concepts1:
            return 0.0
        
        # Check how many node1 concepts appear in node2
        shared_concepts = concepts1.intersection(concepts2)
        concept_flow_ratio = len(shared_concepts) / len(concepts1)
        
        # Bonus for definitional/foundational terms in node1
        foundational_bonus = 0.0
        foundational_terms = {'god', 'bible', 'christianity', 'system', 'truth', 'love'}
        if concepts1.intersection(foundational_terms):
            foundational_bonus = 0.2
        
        # Check for logical flow keywords
        flow_keywords = {
            'because', 'since', 'therefore', 'thus', 'so', 'leads', 'results',
            'implies', 'means', 'shows', 'proves', 'demonstrates'
        }
        
        text2_lower = node2_content.lower()
        flow_bonus = 0.1 if any(keyword in text2_lower for keyword in flow_keywords) else 0.0
        
        return concept_flow_ratio + foundational_bonus + flow_bonus
    
    def get_current_orphan_count(self) -> Tuple[int, int, List[str]]:
        """Get current orphan node count and total nodes"""
        from core.stats import get_active_node_count

        conn = self._get_connection()
        cursor = conn.cursor()

        # Total non-decayed nodes
        total_nodes = get_active_node_count(cursor)
        
        # Orphan nodes (no parents, not seeds, not questions)
        cursor.execute("""
            SELECT tn.id
            FROM thought_nodes tn
            LEFT JOIN derivation_edges de ON tn.id = de.child_id
            WHERE de.child_id IS NULL 
            AND tn.node_type != 'seed'
            AND tn.node_type != 'question'
            AND (tn.decayed = 0 OR tn.decayed IS NULL)
        """)
        
        orphan_ids = [row[0] for row in cursor.fetchall()]
        orphan_count = len(orphan_ids)
        
        conn.close()
        return total_nodes, orphan_count, orphan_ids
    
    def get_all_nodes(self) -> List[Dict]:
        """Get all non-decayed nodes"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, content, node_type, confidence
            FROM thought_nodes 
            WHERE (decayed = 0 OR decayed IS NULL)
            AND node_type != 'question'
            ORDER BY node_type, confidence DESC
        """)
        
        nodes = []
        for row in cursor.fetchall():
            nodes.append({
                'id': row[0],
                'content': row[1],
                'node_type': row[2],
                'confidence': row[3],
                'keywords': self._extract_keywords(row[1]),
                'concepts': self._extract_concepts(row[1])
            })
        
        conn.close()
        return nodes
    
    def edge_exists(self, node1_id: str, node2_id: str) -> bool:
        """Check if edge exists between two nodes (either direction)"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COUNT(*) FROM derivation_edges 
            WHERE (parent_id = ? AND child_id = ?) OR (parent_id = ? AND child_id = ?)
        """, (node1_id, node2_id, node2_id, node1_id))
        
        exists = cursor.fetchone()[0] > 0
        conn.close()
        return exists
    
    def create_edge(self, parent_id: str, child_id: str, relation: str, weight: float, reasoning: str):
        """Create new derivation edge"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO derivation_edges (parent_id, child_id, relation, weight, reasoning)
                VALUES (?, ?, ?, ?, ?)
            """, (parent_id, child_id, relation, weight, reasoning))
            
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            # Edge already exists
            return False
        finally:
            conn.close()
    
    def find_keyword_connections(self, nodes: List[Dict]) -> List[Dict]:
        """Find nodes that should be connected based on keyword overlap"""
        connections = []
        
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if i >= j:  # Avoid duplicate comparisons
                    continue
                
                if self.edge_exists(node1['id'], node2['id']):
                    continue
                
                # Calculate keyword overlap
                overlap_count, jaccard = self._calculate_keyword_overlap(
                    node1['keywords'], node2['keywords']
                )
                
                if overlap_count >= self.keyword_threshold:
                    connections.append({
                        'node1': node1,
                        'node2': node2,
                        'overlap_count': overlap_count,
                        'jaccard_similarity': jaccard,
                        'relation': 'supports',
                        'weight': min(0.8, jaccard + 0.1),  # Cap at 0.8
                        'reasoning': f"Semantic similarity: {overlap_count} shared concepts"
                    })
        
        # Sort by overlap strength
        connections.sort(key=lambda c: c['overlap_count'], reverse=True)
        return connections
    
    def find_concept_flows(self, nodes: List[Dict]) -> List[Dict]:
        """Find derived_from relationships based on concept flow"""
        flows = []
        
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if i == j:
                    continue
                
                if self.edge_exists(node1['id'], node2['id']):
                    continue
                
                # Check if node2 derives from node1 (concept flow)
                flow_score = self._calculate_concept_flow(node1['content'], node2['content'])
                
                if flow_score >= self.concept_flow_threshold:
                    flows.append({
                        'parent_node': node1,
                        'child_node': node2,
                        'flow_score': flow_score,
                        'relation': 'derived_from',
                        'weight': min(0.9, flow_score),
                        'reasoning': f"Concept flow detected (score: {flow_score:.2f})"
                    })
        
        # Sort by flow strength
        flows.sort(key=lambda f: f['flow_score'], reverse=True)
        return flows
    
    def connect_seeds_to_references(self, nodes: List[Dict]) -> List[Dict]:
        """Connect seed nodes to any nodes that reference their concepts"""
        connections = []
        
        # Get seed nodes
        seed_nodes = [n for n in nodes if n['node_type'] == 'seed']
        other_nodes = [n for n in nodes if n['node_type'] != 'seed']
        
        for seed in seed_nodes:
            seed_concepts = seed['concepts']
            
            for other in other_nodes:
                if self.edge_exists(seed['id'], other['id']):
                    continue
                
                # Check if other node references seed concepts
                other_concepts = other['concepts']
                shared = seed_concepts.intersection(other_concepts)
                
                if shared and len(shared) >= 1:  # Any shared concept
                    connections.append({
                        'parent_node': seed,
                        'child_node': other,
                        'shared_concepts': shared,
                        'relation': 'supports',
                        'weight': min(0.8, len(shared) * 0.2 + 0.3),
                        'reasoning': f"References seed concepts: {', '.join(shared)}"
                    })
        
        return connections
    
    def densify_graph(self) -> Dict:
        """Main densification process"""
        print("🌐 Starting graph densification...")
        
        # Get current state
        total_nodes, initial_orphans, orphan_ids = self.get_current_orphan_count()
        print(f"Initial state: {total_nodes} total nodes, {initial_orphans} orphans")
        
        # Get all nodes
        nodes = self.get_all_nodes()
        
        edges_created = 0
        
        # 1. Find keyword-based connections
        print("🔍 Finding keyword-based connections...")
        keyword_connections = self.find_keyword_connections(nodes)
        
        for conn in keyword_connections[:20]:  # Limit to top 20 to avoid spam
            success = self.create_edge(
                conn['node1']['id'], 
                conn['node2']['id'],
                conn['relation'],
                conn['weight'],
                conn['reasoning']
            )
            if success:
                edges_created += 1
                print(f"  ✓ Connected: {conn['node1']['content'][:40]}... ↔ {conn['node2']['content'][:40]}...")
        
        # 2. Find concept flow relationships
        print("🔀 Finding concept flow relationships...")
        concept_flows = self.find_concept_flows(nodes)
        
        for flow in concept_flows[:15]:  # Limit to top 15
            success = self.create_edge(
                flow['parent_node']['id'],
                flow['child_node']['id'], 
                flow['relation'],
                flow['weight'],
                flow['reasoning']
            )
            if success:
                edges_created += 1
                print(f"  ✓ Flow: {flow['parent_node']['content'][:40]}... → {flow['child_node']['content'][:40]}...")
        
        # 3. Connect seeds to references
        print("🌱 Connecting seeds to references...")
        seed_connections = self.connect_seeds_to_references(nodes)
        
        for conn in seed_connections:
            success = self.create_edge(
                conn['parent_node']['id'],
                conn['child_node']['id'],
                conn['relation'], 
                conn['weight'],
                conn['reasoning']
            )
            if success:
                edges_created += 1
                print(f"  ✓ Seed link: {conn['parent_node']['content'][:30]}... → {conn['child_node']['content'][:40]}...")
        
        # Get final state
        _, final_orphans, _ = self.get_current_orphan_count()
        orphans_resolved = initial_orphans - final_orphans
        
        result = {
            'edges_created': edges_created,
            'initial_orphans': initial_orphans,
            'final_orphans': final_orphans,
            'orphans_resolved': orphans_resolved,
            'total_nodes': total_nodes,
            'keyword_connections': len(keyword_connections),
            'concept_flows': len(concept_flows), 
            'seed_connections': len(seed_connections)
        }
        
        print(f"\n✅ Densification complete:")
        print(f"  Edges created: {edges_created}")
        print(f"  Orphans: {initial_orphans} → {final_orphans} (resolved: {orphans_resolved})")
        print(f"  Orphan ratio: {final_orphans/total_nodes:.1%}")
        
        return result


def main():
    """CLI interface for edge densification"""
    parser = argparse.ArgumentParser(description="Cashew Edge Densification")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--keyword-threshold", type=int, default=3, help="Minimum keyword overlap for connection")
    parser.add_argument("--concept-threshold", type=float, default=0.7, help="Minimum concept flow score")
    
    args = parser.parse_args()
    
    densifier = EdgeDensifier()
    densifier.keyword_threshold = args.keyword_threshold
    densifier.concept_flow_threshold = args.concept_threshold
    
    if args.dry_run:
        print("🔍 DRY RUN: Analyzing potential connections...")
        
        # Show current state
        total_nodes, orphan_count, orphan_ids = densifier.get_current_orphan_count()
        print(f"Current state: {total_nodes} nodes, {orphan_count} orphans ({orphan_count/total_nodes:.1%})")
        
        # Analyze potential connections
        nodes = densifier.get_all_nodes()
        
        keyword_connections = densifier.find_keyword_connections(nodes)
        concept_flows = densifier.find_concept_flows(nodes)
        seed_connections = densifier.connect_seeds_to_references(nodes)
        
        print(f"\nPotential connections:")
        print(f"  Keyword-based: {len(keyword_connections)}")
        print(f"  Concept flows: {len(concept_flows)}")
        print(f"  Seed connections: {len(seed_connections)}")
        
        total_potential = len(keyword_connections) + len(concept_flows) + len(seed_connections)
        print(f"  Total potential edges: {total_potential}")
        
        # Estimate final orphan count (rough)
        estimated_final = max(0, orphan_count - len(seed_connections) - len(concept_flows)//2)
        print(f"  Estimated final orphans: {estimated_final}")
    
    else:
        result = densifier.densify_graph()
        
        print(f"\n📊 Summary:")
        for key, value in result.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())