#!/usr/bin/env python3
"""
Quick Edge Densification for Cashew
Targeted approach to reduce orphan nodes by focusing on high-confidence connections
"""

import sqlite3
import re
from typing import List, Dict, Set
import hashlib

DB_PATH = "/Users/bunny/.openclaw/workspace/cashew/data/graph.db"

class QuickDensifier:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        
    def get_connection(self):
        return sqlite3.connect(self.db_path)
        
    def get_high_confidence_nodes(self, limit: int = 500) -> List[Dict]:
        """Get top high-confidence nodes for focused processing"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, content, node_type, confidence, source_file
            FROM thought_nodes 
            WHERE confidence >= 0.7
            ORDER BY confidence DESC, LENGTH(content) DESC
            LIMIT ?
        """, (limit,))
        
        nodes = []
        for row in cursor.fetchall():
            nodes.append({
                'id': row[0],
                'content': row[1],
                'node_type': row[2],
                'confidence': row[3],
                'source_file': row[4]
            })
        
        conn.close()
        return nodes
        
    def get_orphan_nodes(self, limit: int = 200) -> List[Dict]:
        """Get orphan nodes that need connections"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT tn.id, tn.content, tn.node_type, tn.confidence
            FROM thought_nodes tn
            LEFT JOIN derivation_edges de ON tn.id = de.child_id
            WHERE de.child_id IS NULL 
            AND tn.node_type != 'seed'
            AND tn.confidence >= 0.6
            ORDER BY tn.confidence DESC
            LIMIT ?
        """, (limit,))
        
        orphans = []
        for row in cursor.fetchall():
            orphans.append({
                'id': row[0],
                'content': row[1],
                'node_type': row[2],
                'confidence': row[3]
            })
        
        conn.close()
        return orphans
        
    def create_edge(self, parent_id: str, child_id: str, relation: str, weight: float, reasoning: str) -> bool:
        """Create edge if it doesn't exist"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO derivation_edges (parent_id, child_id, relation, weight, reasoning)
                VALUES (?, ?, ?, ?, ?)
            """, (parent_id, child_id, relation, weight, reasoning))
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            conn.close()
            return False
            
    def find_keyword_matches(self, content1: str, content2: str) -> Set[str]:
        """Find significant keyword matches between two contents"""
        # Extract meaningful words (3+ chars, not common words)
        words1 = set(re.findall(r'\b[a-zA-Z]{3,}\b', content1.lower()))
        words2 = set(re.findall(r'\b[a-zA-Z]{3,}\b', content2.lower()))
        
        # Filter common words
        stopwords = {'the', 'and', 'that', 'this', 'with', 'for', 'was', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'}
        
        words1 = {w for w in words1 if w not in stopwords and len(w) > 3}
        words2 = {w for w in words2 if w not in stopwords and len(w) > 3}
        
        return words1.intersection(words2)
        
    def connect_related_concepts(self):
        """Connect nodes with shared religious/philosophical concepts"""
        print("🔗 Connecting related religious/philosophical concepts...")
        
        # Define concept groups
        concept_groups = {
            'religious': ['god', 'jesus', 'christ', 'christianity', 'christian', 'bible', 'prayer', 'faith', 'church', 'salvation'],
            'philosophical': ['truth', 'evidence', 'reasoning', 'logic', 'morality', 'system', 'belief', 'reality'],
            'emotional': ['love', 'compassion', 'guilt', 'shame', 'family', 'mother', 'father', 'child', 'authenticity'],
            'intellectual': ['pattern', 'depth', 'complexity', 'analysis', 'understanding', 'insight', 'breakthrough']
        }
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        edges_created = 0
        
        for group_name, concepts in concept_groups.items():
            for concept in concepts:
                # Find nodes containing this concept
                cursor.execute("""
                    SELECT id, content, node_type, confidence
                    FROM thought_nodes 
                    WHERE LOWER(content) LIKE ? AND confidence >= 0.7
                    ORDER BY confidence DESC
                    LIMIT 20
                """, (f'%{concept}%',))
                
                concept_nodes = cursor.fetchall()
                
                if len(concept_nodes) > 1:
                    # Connect the highest confidence node to others
                    primary = concept_nodes[0]
                    
                    for secondary in concept_nodes[1:]:
                        if primary[0] != secondary[0]:  # Different nodes
                            success = self.create_edge(
                                primary[0], secondary[0], 
                                'supports',
                                0.6,
                                f"Shared {group_name} concept: {concept}"
                            )
                            if success:
                                edges_created += 1
                                print(f"  ✓ {concept}: {primary[1][:30]}... → {secondary[1][:30]}...")
        
        conn.close()
        print(f"✅ Created {edges_created} concept-based edges")
        return edges_created
        
    def connect_same_source_files(self):
        """Connect high-confidence nodes from the same source files"""
        print("📄 Connecting nodes from same source files...")
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Group by source file
        cursor.execute("""
            SELECT source_file, COUNT(*) as node_count
            FROM thought_nodes
            WHERE confidence >= 0.8
            GROUP BY source_file
            HAVING node_count > 1
            ORDER BY node_count DESC
        """)
        
        source_files = cursor.fetchall()
        edges_created = 0
        
        for source_file, count in source_files[:10]:  # Top 10 source files
            cursor.execute("""
                SELECT id, content, confidence
                FROM thought_nodes
                WHERE source_file = ? AND confidence >= 0.8
                ORDER BY confidence DESC
                LIMIT 10
            """, (source_file,))
            
            nodes = cursor.fetchall()
            
            if len(nodes) > 1:
                # Connect the highest confidence node to others in same file
                primary = nodes[0]
                
                for secondary in nodes[1:]:
                    success = self.create_edge(
                        primary[0], secondary[0],
                        'supports',
                        0.5,
                        f"Same source context: {source_file}"
                    )
                    if success:
                        edges_created += 1
                        print(f"  ✓ {source_file}: {primary[1][:25]}... → {secondary[1][:25]}...")
        
        conn.close()
        print(f"✅ Created {edges_created} same-source edges")
        return edges_created
        
    def connect_orphans_to_keywords(self):
        """Connect orphan nodes to existing nodes via keyword matching"""
        print("🎯 Connecting orphans via keyword matching...")
        
        orphans = self.get_orphan_nodes(100)  # Top 100 orphans
        high_conf = self.get_high_confidence_nodes(200)  # Top 200 high-confidence
        
        edges_created = 0
        
        for orphan in orphans:
            for parent in high_conf:
                if orphan['id'] == parent['id']:
                    continue
                    
                # Find keyword matches
                matches = self.find_keyword_matches(orphan['content'], parent['content'])
                
                if len(matches) >= 2:  # At least 2 keyword matches
                    success = self.create_edge(
                        parent['id'], orphan['id'],
                        'supports',
                        min(0.8, len(matches) * 0.15 + 0.3),
                        f"Keyword matches: {', '.join(list(matches)[:3])}"
                    )
                    if success:
                        edges_created += 1
                        print(f"  ✓ {', '.join(list(matches)[:2])}: {parent['content'][:25]}... → {orphan['content'][:25]}...")
                        break  # Only connect each orphan once
        
        print(f"✅ Connected {edges_created} orphans via keywords")
        return edges_created
        
    def get_stats(self):
        """Get current graph stats"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM thought_nodes")
        total_nodes = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM derivation_edges")
        total_edges = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT COUNT(*)
            FROM thought_nodes tn
            LEFT JOIN derivation_edges de ON tn.id = de.child_id
            WHERE de.child_id IS NULL AND tn.node_type != 'seed'
        """)
        orphans = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'nodes': total_nodes,
            'edges': total_edges,
            'orphans': orphans,
            'orphan_ratio': orphans / total_nodes if total_nodes > 0 else 0
        }
        
    def quick_densify(self):
        """Run quick densification process"""
        print("🚀 Starting quick densification...")
        
        # Initial stats
        initial_stats = self.get_stats()
        print(f"Initial: {initial_stats['nodes']} nodes, {initial_stats['edges']} edges, {initial_stats['orphans']} orphans ({initial_stats['orphan_ratio']:.1%})")
        
        total_edges = 0
        
        # 1. Connect related concepts
        total_edges += self.connect_related_concepts()
        
        # 2. Connect same source files
        total_edges += self.connect_same_source_files()
        
        # 3. Connect orphans via keywords
        total_edges += self.connect_orphans_to_keywords()
        
        # Final stats
        final_stats = self.get_stats()
        
        print(f"\n📊 QUICK DENSIFICATION COMPLETE")
        print(f"Edges created: {total_edges}")
        print(f"Final: {final_stats['nodes']} nodes, {final_stats['edges']} edges, {final_stats['orphans']} orphans ({final_stats['orphan_ratio']:.1%})")
        print(f"Orphans reduced: {initial_stats['orphans']} → {final_stats['orphans']} (-{initial_stats['orphans'] - final_stats['orphans']})")
        
        return final_stats

def main():
    densifier = QuickDensifier()
    densifier.quick_densify()

if __name__ == "__main__":
    main()