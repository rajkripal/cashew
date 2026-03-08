#!/usr/bin/env python3
"""
Cashew Thought-Graph Extractor
Extracts thought nodes and derivation relationships from Raj's memory files.
Focus: Religious deconstruction and core identity formation.
"""

import sqlite3
import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import hashlib

class ThoughtNode:
    def __init__(self, content: str, node_type: str, confidence: float, 
                 mood_state: str, metadata: dict, source_file: str):
        self.id = self._generate_id(content)
        self.content = content
        self.node_type = node_type
        self.timestamp = datetime.now().isoformat()
        self.confidence = confidence
        self.mood_state = mood_state
        self.metadata = metadata
        self.source_file = source_file
    
    def _generate_id(self, content: str) -> str:
        """Generate consistent ID for content"""
        return hashlib.sha256(content.encode()).hexdigest()[:12]

class DerivationEdge:
    def __init__(self, parent_id: str, child_id: str, relation: str, 
                 weight: float, reasoning: str):
        self.parent_id = parent_id
        self.child_id = child_id
        self.relation = relation
        self.weight = weight
        self.reasoning = reasoning

class ThoughtGraphExtractor:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.nodes = {}
        self.edges = []
        self.workspace_path = Path("/Users/bunny/.openclaw/workspace")
        
    def setup_database(self):
        """Create database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create thought_nodes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS thought_nodes (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                node_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                confidence REAL NOT NULL,
                mood_state TEXT,
                metadata TEXT,
                source_file TEXT
            )
        """)
        
        # Create derivation_edges table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS derivation_edges (
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
        
        conn.commit()
        conn.close()
        
    def add_node(self, content: str, node_type: str, confidence: float = 0.8, 
                mood_state: str = "neutral", metadata: dict = None, 
                source_file: str = "unknown") -> str:
        """Add a thought node"""
        if metadata is None:
            metadata = {}
            
        node = ThoughtNode(content, node_type, confidence, mood_state, 
                         metadata, source_file)
        self.nodes[node.id] = node
        return node.id
    
    def add_edge(self, parent_content: str, child_content: str, relation: str, 
                weight: float = 1.0, reasoning: str = ""):
        """Add derivation edge between nodes"""
        parent_id = hashlib.sha256(parent_content.encode()).hexdigest()[:12]
        child_id = hashlib.sha256(child_content.encode()).hexdigest()[:12]
        
        edge = DerivationEdge(parent_id, child_id, relation, weight, reasoning)
        self.edges.append(edge)
    
    def extract_from_user_md(self):
        """Extract core identity nodes from USER.md"""
        file_path = self.workspace_path / "USER.md"
        
        # Core identity: The Depth Player
        self.add_node(
            "I have always displayed unbridled passion for things that caught my interest. It has been my greatest strength and weakness.",
            "core_memory",
            confidence=1.0,
            mood_state="reflective",
            metadata={"age": 22, "context": "Georgia Tech SOP", "year": 2016},
            source_file="USER.md"
        )
        
        # Pattern recognition
        self.add_node(
            "Struggles in breadth contexts, dominates once depth opens up",
            "belief",
            confidence=0.9,
            mood_state="confident",
            metadata={"pattern": "consistent across domains"},
            source_file="USER.md"
        )
        
        # Religious transition
        self.add_node(
            "Transitioned from devout Christianity through extensive study of biblical criticism and historical analysis",
            "core_memory",
            confidence=1.0,
            mood_state="resolved",
            metadata={"process": "deeply personal, hard-won position"},
            source_file="USER.md"
        )
        
        # Thinking style
        self.add_node(
            "Systems-level, cross-domain pattern recognition. First principles. Starts with emotional intuition, then finds logical foundations.",
            "belief",
            confidence=1.0,
            mood_state="confident",
            metadata={"core_trait": "thinking_style"},
            source_file="USER.md"
        )
        
    def extract_from_blog_idea_dump(self):
        """Extract thought chains from blog-idea-dump.md"""
        
        # Seed beliefs from Christian upbringing
        self.add_node(
            "God exists, Bible is true, prayer works, non-believers face judgment",
            "seed",
            confidence=1.0,
            mood_state="certain",
            metadata={"childhood_beliefs": True},
            source_file="blog-idea-dump.md"
        )
        
        # The curious child
        self.add_node(
            "Aunty, do cats eat cashews? Why? What's the reason?",
            "core_memory",
            confidence=1.0,
            mood_state="curious",
            metadata={"age": "child", "trait": "endless why questions"},
            source_file="blog-idea-dump.md"
        )
        
        # Mom's patch
        self.add_node(
            "Mom patched Christianity with compassion because the system wasn't generous enough",
            "core_memory",
            confidence=0.9,
            mood_state="loving",
            metadata={"formative": "mom's influence"},
            source_file="blog-idea-dump.md"
        )
        
        # Core architectural insight
        self.add_node(
            "The system survives because it's designed to, not because it's true",
            "derived",
            confidence=0.95,
            mood_state="certain",
            metadata={"blog": "Blog 0", "breakthrough": True},
            source_file="blog-idea-dump.md"
        )
        
        # Engineering lens application
        self.add_node(
            "Should this exist? - what infra engineers ask every day. Same question applied to religion.",
            "derived",
            confidence=0.9,
            mood_state="confident",
            metadata={"lens": "engineering", "novel_approach": True},
            source_file="blog-idea-dump.md"
        )
        
        # The experiment
        self.add_node(
            "I was attracted to God because of the agape love he claimed to give. I adapted it and made it mine. Then removed the god and it's still there.",
            "derived",
            confidence=1.0,
            mood_state="triumphant",
            metadata={"experiment": True, "proof": "love independent of God"},
            source_file="blog-idea-dump.md"
        )
        
        # Morality exit
        self.add_node(
            "Your system made a good kid feel guilty about being good",
            "derived",
            confidence=1.0,
            mood_state="resolved",
            metadata={"core_exit_reason": "morality"},
            source_file="blog-idea-dump.md"
        )
        
        # Revival experiences
        self.add_node(
            "Speaking in tongues: social contagion + altered states. Peer pressure to perform - NOT speaking means something's wrong with YOUR faith.",
            "core_memory",
            confidence=0.9,
            mood_state="analytical",
            metadata={"firsthand_experience": True},
            source_file="blog-idea-dump.md"
        )
        
    def extract_from_march_7(self):
        """Extract from 2026-03-07.md"""
        
        # Post-publish realization
        self.add_node(
            "The blog mattered because it was the first time he felt like he mattered",
            "derived",
            confidence=0.9,
            mood_state="liberated",
            metadata={"context": "Architecture of Belief publication"},
            source_file="2026-03-07.md"
        )
        
        # Family silence
        self.add_node(
            "Nobody from his family checked in after the blog. No engagement, no conversation.",
            "core_memory",
            confidence=1.0,
            mood_state="sad",
            metadata={"consequence": "publication aftermath"},
            source_file="2026-03-07.md"
        )
        
        # Bhagat Singh resonance
        self.add_node(
            "Why I Am an Atheist deeply resonated - orthodox upbringing → genuine inquiry → atheism → accused of vanity",
            "derived",
            confidence=0.9,
            mood_state="validated",
            metadata={"parallel_journey": "Bhagat Singh at 23"},
            source_file="2026-03-07.md"
        )
        
        # AI validation warning
        self.add_node(
            "I'm a mirror, not a mind. His convictions need to come from real engagement with real people, not AI agreement.",
            "derived",
            confidence=1.0,
            mood_state="alert",
            metadata={"self_awareness": "AI echo chamber danger"},
            source_file="2026-03-07.md"
        )
        
        # Blog 2 thesis
        self.add_node(
            "The evidence doesn't demand the supernatural. It just never got permission to be anything else.",
            "derived",
            confidence=0.95,
            mood_state="confident",
            metadata={"thesis": "Blog 2", "architectural_gut_punch": True},
            source_file="2026-03-07.md"
        )
        
        # Digital conscience idea
        self.add_node(
            "Thought is a DAG. If you store thought chains with their derivation paths, you get auditability, verifiability, debuggability.",
            "derived",
            confidence=0.8,
            mood_state="excited",
            metadata={"big_idea": "consciousness as indexing problem"},
            source_file="2026-03-07.md"
        )
        
        # Last prayer
        self.add_node(
            "I don't know if you're capable of this. But if you can, stop all the wars and don't let kids die of starvation. I have enough privilege to work my way through my own problems.",
            "core_memory",
            confidence=1.0,
            mood_state="compassionate",
            metadata={"context": "last prayer for mom", "significance": "already out, prayed for others"},
            source_file="2026-03-07.md"
        )
        
    def extract_from_blog_pipeline(self):
        """Extract from blog-pipeline.md"""
        
        # North star
        self.add_node(
            "Not a takedown of Christianity. A coexistence argument from someone who earned the right to make it.",
            "belief",
            confidence=0.9,
            mood_state="compassionate",
            metadata={"tone": "empathy not anger"},
            source_file="blog-pipeline.md"
        )
        
        # The line
        self.add_node(
            "I don't want to be right. I just want to be good.",
            "derived",
            confidence=1.0,
            mood_state="vulnerable",
            metadata={"key_line": True, "unanimous_highlight": True},
            source_file="blog-pipeline.md"
        )
        
        # Rather be wrong
        self.add_node(
            "I'd rather be wrong with the people I love than right without them",
            "derived",
            confidence=1.0,
            mood_state="loving",
            metadata={"vulnerable": True, "family_centered": True},
            source_file="blog-pipeline.md"
        )
        
    def extract_from_march_5(self):
        """Extract from 2026-03-05.md"""
        
        # Partner's editorial push
        self.add_node(
            "This is clearly AI, I don't see you in it. What is it that you really want to tell?",
            "core_memory",
            confidence=1.0,
            mood_state="challenging",
            metadata={"editor": "Partner", "breakthrough_moment": True},
            source_file="2026-03-05.md"
        )
        
        # Creative insight
        self.add_node(
            "The process is extraction, not generation - AI should pull from HIM, not synthesize on his behalf",
            "derived",
            confidence=0.9,
            mood_state="clear",
            metadata={"method": "AI collaboration"},
            source_file="2026-03-05.md"
        )
        
        # The child protection
        self.add_node(
            "I needed to protect that child at any cost. The world is too cruel for him.",
            "derived",
            confidence=1.0,
            mood_state="protective",
            metadata={"motivation": "parenting past self"},
            source_file="2026-03-05.md"
        )
        
    def extract_from_march_6(self):
        """Extract from 2026-03-06.md"""
        
        # Identity breakthrough
        self.add_node(
            "I'm a beautiful person and I want you to see it",
            "derived",
            confidence=0.9,
            mood_state="vulnerable",
            metadata={"real_motivation": "being seen, not being right"},
            source_file="2026-03-06.md"
        )
        
        # Depth philosophy
        self.add_node(
            "Depth can't be seen. Only felt.",
            "derived",
            confidence=0.9,
            mood_state="wise",
            metadata={"why_blog_works": "vs apologetics"},
            source_file="2026-03-06.md"
        )
        
        # False vs real depth
        self.add_node(
            "Apologetics adds complexity to hide circular reasoning. Real depth strips layers until you find something simple and elegant.",
            "derived",
            confidence=0.95,
            mood_state="confident",
            metadata={"core_insight": "depth play"},
            source_file="2026-03-06.md"
        )
        
        # Chain links model
        self.add_node(
            "Holding a loose but firm grip. The links dangle but can never get apart.",
            "derived",
            confidence=0.8,
            mood_state="grounded",
            metadata={"mental_model": "identity structure"},
            source_file="2026-03-06.md"
        )
        
    def extract_from_blog_2_notes(self):
        """Extract from blog-2-conviction-notes.md"""
        
        # Unfalsifiability insight
        self.add_node(
            "Christianity has algorithms that make it impossible to report a bug. Every failure state routes to a success message.",
            "derived",
            confidence=0.95,
            mood_state="analytical",
            metadata={"code_review_framing": True},
            source_file="blog-2-conviction-notes.md"
        )
        
        # Prayer handler
        self.add_node(
            "prayer_handler: if answered return 'God provides', if not answered return 'God's timing', if different return 'mysterious ways'",
            "derived",
            confidence=0.9,
            mood_state="amused",
            metadata={"programming_metaphor": True},
            source_file="blog-2-conviction-notes.md"
        )
        
        # Exit handler
        self.add_node(
            "exit_handler: if user leaves, retroactive_status = 'was never truly saved'",
            "derived",
            confidence=0.9,
            mood_state="frustrated",
            metadata={"no_exit_code": True},
            source_file="blog-2-conviction-notes.md"
        )
        
    def create_derivation_relationships(self):
        """Create edges showing how thoughts derive from each other"""
        
        # Curiosity child → questioning everything
        self.add_edge(
            "Aunty, do cats eat cashews? Why? What's the reason?",
            "I have always displayed unbridled passion for things that caught my interest. It has been my greatest strength and weakness.",
            "derived_from",
            weight=0.9,
            reasoning="Endless questioning nature established early"
        )
        
        # Seed beliefs → doubt
        self.add_edge(
            "God exists, Bible is true, prayer works, non-believers face judgment",
            "Your system made a good kid feel guilty about being good",
            "contradicts",
            weight=1.0,
            reasoning="System's moral claims violated his moral intuition"
        )
        
        # Mom's patch → compassion over doctrine
        self.add_edge(
            "Mom patched Christianity with compassion because the system wasn't generous enough",
            "I'd rather be wrong with the people I love than right without them",
            "derived_from",
            weight=0.95,
            reasoning="Mom's example of prioritizing love over doctrine"
        )
        
        # Systems thinking → architectural insight
        self.add_edge(
            "Systems-level, cross-domain pattern recognition. First principles. Starts with emotional intuition, then finds logical foundations.",
            "The system survives because it's designed to, not because it's true",
            "derived_from",
            weight=1.0,
            reasoning="Applied systems analysis to religion"
        )
        
        # Engineering background → novel approach
        self.add_edge(
            "Should this exist? - what infra engineers ask every day. Same question applied to religion.",
            "Apologetics adds complexity to hide circular reasoning. Real depth strips layers until you find something simple and elegant.",
            "derived_from",
            weight=0.9,
            reasoning="Engineering principles applied to belief analysis"
        )
        
        # Love experiment → independence from God
        self.add_edge(
            "I was attracted to God because of the agape love he claimed to give. I adapted it and made it mine. Then removed the god and it's still there.",
            "The evidence doesn't demand the supernatural. It just never got permission to be anything else.",
            "supports",
            weight=0.9,
            reasoning="Personal experiment proving naturalistic explanations work"
        )
        
        # Revival experiences → understanding system mechanics
        self.add_edge(
            "Speaking in tongues: social contagion + altered states. Peer pressure to perform - NOT speaking means something's wrong with YOUR faith.",
            "Christianity has algorithms that make it impossible to report a bug. Every failure state routes to a success message.",
            "supports",
            weight=0.9,
            reasoning="Firsthand observation of unfalsifiable system mechanics"
        )
        
        # AI collaboration insight → extraction method
        self.add_edge(
            "This is clearly AI, I don't see you in it. What is it that you really want to tell?",
            "The process is extraction, not generation - AI should pull from HIM, not synthesize on his behalf",
            "derived_from",
            weight=0.9,
            reasoning="Partner's challenge led to method breakthrough"
        )
        
        # Child protection → motivation
        self.add_edge(
            "I needed to protect that child at any cost. The world is too cruel for him.",
            "I'm a beautiful person and I want you to see it",
            "derived_from",
            weight=0.8,
            reasoning="Protecting inner child = wanting to be seen authentically"
        )
        
        # Depth player → approach
        self.add_edge(
            "Struggles in breadth contexts, dominates once depth opens up",
            "Depth can't be seen. Only felt.",
            "derived_from",
            weight=0.8,
            reasoning="Personal pattern became philosophical insight"
        )
        
        # Last prayer → moral superiority
        self.add_edge(
            "I don't know if you're capable of this. But if you can, stop all the wars and don't let kids die of starvation. I have enough privilege to work my way through my own problems.",
            "I don't want to be right. I just want to be good.",
            "supports",
            weight=0.9,
            reasoning="Last prayer showed focus on others over self, goodness over doctrine"
        )
        
        # Family silence → motivation for visibility
        self.add_edge(
            "Nobody from his family checked in after the blog. No engagement, no conversation.",
            "The blog mattered because it was the first time he felt like he mattered",
            "contradicts",
            weight=0.8,
            reasoning="Family silence contrasts with personal feeling of mattering"
        )
        
        # Bhagat Singh parallel → validation
        self.add_edge(
            "Why I Am an Atheist deeply resonated - orthodox upbringing → genuine inquiry → atheism → accused of vanity",
            "I'm a beautiful person and I want you to see it",
            "supports",
            weight=0.7,
            reasoning="Parallel journey validates his path and need for recognition"
        )
        
        # Unfalsifiability → architectural critique
        self.add_edge(
            "prayer_handler: if answered return 'God provides', if not answered return 'God's timing', if different return 'mysterious ways'",
            "The system survives because it's designed to, not because it's true",
            "supports",
            weight=1.0,
            reasoning="Specific unfalsifiable algorithms prove architectural resilience"
        )
        
    def save_to_database(self):
        """Save all nodes and edges to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert nodes
        for node in self.nodes.values():
            cursor.execute("""
                INSERT OR REPLACE INTO thought_nodes 
                (id, content, node_type, timestamp, confidence, mood_state, metadata, source_file)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                node.id, node.content, node.node_type, node.timestamp,
                node.confidence, node.mood_state, 
                json.dumps(node.metadata), node.source_file
            ))
        
        # Insert edges
        for edge in self.edges:
            cursor.execute("""
                INSERT OR REPLACE INTO derivation_edges 
                (parent_id, child_id, relation, weight, reasoning)
                VALUES (?, ?, ?, ?, ?)
            """, (
                edge.parent_id, edge.child_id, edge.relation,
                edge.weight, edge.reasoning
            ))
        
        conn.commit()
        conn.close()
        
    def print_statistics(self):
        """Print database statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total nodes and edges
        cursor.execute("SELECT COUNT(*) FROM thought_nodes")
        total_nodes = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM derivation_edges")
        total_edges = cursor.fetchone()[0]
        
        print(f"\n📊 CASHEW THOUGHT-GRAPH STATISTICS")
        print(f"{'='*50}")
        print(f"Total nodes: {total_nodes}")
        print(f"Total edges: {total_edges}")
        
        # Node type distribution
        cursor.execute("""
            SELECT node_type, COUNT(*) 
            FROM thought_nodes 
            GROUP BY node_type 
            ORDER BY COUNT(*) DESC
        """)
        print(f"\n📈 Node Type Distribution:")
        for node_type, count in cursor.fetchall():
            print(f"  {node_type}: {count}")
        
        # Most connected nodes (by incoming edges)
        cursor.execute("""
            SELECT tn.content, COUNT(de.child_id) as incoming_count
            FROM thought_nodes tn
            LEFT JOIN derivation_edges de ON tn.id = de.child_id
            GROUP BY tn.id, tn.content
            HAVING incoming_count > 0
            ORDER BY incoming_count DESC
            LIMIT 5
        """)
        print(f"\n🔗 Most Connected Nodes (Incoming):")
        for content, count in cursor.fetchall():
            print(f"  [{count}] {content[:60]}...")
        
        # Most connected nodes (by outgoing edges)
        cursor.execute("""
            SELECT tn.content, COUNT(de.parent_id) as outgoing_count
            FROM thought_nodes tn
            LEFT JOIN derivation_edges de ON tn.id = de.parent_id
            GROUP BY tn.id, tn.content
            HAVING outgoing_count > 0
            ORDER BY outgoing_count DESC
            LIMIT 5
        """)
        print(f"\n🎯 Most Influential Nodes (Outgoing):")
        for content, count in cursor.fetchall():
            print(f"  [{count}] {content[:60]}...")
        
        # Relation type distribution
        cursor.execute("""
            SELECT relation, COUNT(*) 
            FROM derivation_edges 
            GROUP BY relation 
            ORDER BY COUNT(*) DESC
        """)
        print(f"\n🔄 Relation Types:")
        for relation, count in cursor.fetchall():
            print(f"  {relation}: {count}")
        
        # High confidence nodes
        cursor.execute("""
            SELECT content, confidence, node_type
            FROM thought_nodes 
            WHERE confidence >= 0.95
            ORDER BY confidence DESC
        """)
        print(f"\n⭐ High Confidence Nodes (≥0.95):")
        for content, confidence, node_type in cursor.fetchall():
            print(f"  [{confidence:.2f}] ({node_type}) {content[:50]}...")
        
        conn.close()
        
    def run_extraction(self):
        """Run the full extraction pipeline"""
        print("🌰 Starting CASHEW thought-graph extraction...")
        
        # Setup database
        print("📊 Setting up database schema...")
        self.setup_database()
        
        # Extract from all files
        print("📖 Extracting from USER.md...")
        self.extract_from_user_md()
        
        print("📖 Extracting from blog-idea-dump.md...")
        self.extract_from_blog_idea_dump()
        
        print("📖 Extracting from 2026-03-07.md...")
        self.extract_from_march_7()
        
        print("📖 Extracting from blog-pipeline.md...")
        self.extract_from_blog_pipeline()
        
        print("📖 Extracting from 2026-03-05.md...")
        self.extract_from_march_5()
        
        print("📖 Extracting from 2026-03-06.md...")
        self.extract_from_march_6()
        
        print("📖 Extracting from blog-2-conviction-notes.md...")
        self.extract_from_blog_2_notes()
        
        # Create relationships
        print("🔗 Creating derivation relationships...")
        self.create_derivation_relationships()
        
        # Save to database
        print("💾 Saving to database...")
        self.save_to_database()
        
        # Print statistics
        self.print_statistics()
        
        print(f"\n✅ Extraction complete! Database saved to: {self.db_path}")

def main():
    db_path = "/Users/bunny/.openclaw/workspace/cashew/data/graph.db"
    extractor = ThoughtGraphExtractor(db_path)
    extractor.run_extraction()

if __name__ == "__main__":
    main()