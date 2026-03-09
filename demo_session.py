#!/usr/bin/env python3
"""
Demo of Cashew Session Integration Layer
"""

import sqlite3
import json
import tempfile
import os
from datetime import datetime, timezone

# Add current directory to path for imports
import sys
sys.path.append('.')

from core.session import start_session, end_session, think_cycle
from core.embeddings import embed_nodes

def create_demo_db():
    """Create a demo database with some test data"""
    fd, path = tempfile.mkstemp(suffix='_demo.db')
    os.close(fd)
    
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    
    # Create schema
    cursor.execute("""
        CREATE TABLE thought_nodes (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            node_type TEXT NOT NULL,
            timestamp TEXT,
            confidence REAL DEFAULT 0.5,
            source_file TEXT,
            metadata TEXT DEFAULT '{}',
            decayed INTEGER DEFAULT 0,
            last_accessed TEXT,
            access_count INTEGER DEFAULT 0
        )
    """)
    
    cursor.execute("""
        CREATE TABLE derivation_edges (
            parent_id TEXT NOT NULL,
            child_id TEXT NOT NULL,
            relation TEXT NOT NULL,
            weight REAL DEFAULT 1.0,
            reasoning TEXT,
            UNIQUE(parent_id, child_id)
        )
    """)
    
    # Add some demo data
    now = datetime.now(timezone.utc).isoformat()
    demo_nodes = [
        ("work1", "Team meeting scheduled for Friday at 2pm", "fact", "work"),
        ("work2", "Project deadline is next Tuesday", "fact", "work"),
        ("work3", "Need to finish code review before deadline", "decision", "work"),
        ("health1", "Started exercising regularly", "observation", "health"),
        ("health2", "Exercise improves focus and energy", "belief", "health"),
        ("learn1", "Learning Python for data analysis", "decision", "learning"),
        ("learn2", "Python has excellent ML libraries", "fact", "learning")
    ]
    
    for node_id, content, node_type, domain in demo_nodes:
        metadata = json.dumps({"domain": domain})
        cursor.execute("""
            INSERT INTO thought_nodes 
            (id, content, node_type, timestamp, confidence, metadata, source_file, access_count)
            VALUES (?, ?, ?, ?, 0.8, ?, 'demo', 0)
        """, (node_id, content, node_type, now, metadata))
    
    # Add some edges
    cursor.execute("""
        INSERT INTO derivation_edges (parent_id, child_id, relation, weight, reasoning)
        VALUES ('work1', 'work3', 'motivates', 0.7, 'Meeting creates urgency for code review')
    """)
    
    conn.commit()
    conn.close()
    
    return path

def mock_model_fn(prompt: str) -> str:
    """Mock LLM function that returns realistic extractions"""
    if "conversation" in prompt.lower():
        return json.dumps([
            {"content": "User is interested in machine learning applications", "type": "observation", "confidence": 0.8},
            {"content": "Will explore scikit-learn tutorials this weekend", "type": "decision", "confidence": 0.9},
            {"content": "ML knowledge will help with current data project", "type": "insight", "confidence": 0.7}
        ])
    else:
        return json.dumps([
            {"content": "Work productivity increases with structured learning", "type": "insight", "confidence": 0.8},
            {"content": "Technical skills and health habits reinforce each other", "type": "insight", "confidence": 0.7}
        ])

def main():
    """Run the demo"""
    print("🌿 Cashew Session Integration Demo")
    print("=" * 50)
    
    # Create demo database
    print("\n1. Creating demo database with test data...")
    db_path = create_demo_db()
    
    # Embed the demo nodes
    print("2. Creating embeddings for demo nodes...")
    embed_stats = embed_nodes(db_path)
    print(f"   Embedded {embed_stats['embedded']} nodes")
    
    # Test session start
    print("\n3. Starting session with work-related hints...")
    session_context = start_session(db_path, "demo_session", ["work", "meetings", "deadline"])
    print(f"   Found {len(session_context.nodes_used)} relevant context nodes")
    print(f"   Token estimate: {session_context.token_estimate}")
    
    if session_context.context_str:
        print("\n   Context provided:")
        print(session_context.context_str)
    
    # Test session end with knowledge extraction
    print("\n4. Ending session and extracting new knowledge...")
    conversation = """
    User: I've been thinking about learning machine learning to help with my data project at work.
    Assistant: That's a great idea! What specific areas interest you?
    User: Probably scikit-learn for now. I think it will help me analyze the customer data better.
    Assistant: Excellent choice. Scikit-learn is perfect for getting started.
    User: I'll plan to work through some tutorials this weekend.
    """
    
    extraction_result = end_session(db_path, "demo_session", conversation, mock_model_fn)
    print(f"   Extracted {len(extraction_result.new_nodes)} new nodes")
    print(f"   Created {len(extraction_result.new_edges)} new edges")
    
    if extraction_result.new_nodes:
        print("   New nodes created:")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        for node_id in extraction_result.new_nodes:
            cursor.execute("SELECT content, node_type FROM thought_nodes WHERE id = ?", (node_id,))
            row = cursor.fetchone()
            if row:
                content, node_type = row
                print(f"     [{node_type.upper()}] {content}")
        conn.close()
    
    # Test think cycle
    print("\n5. Running think cycle to generate insights...")
    think_result = think_cycle(db_path, mock_model_fn, focus_domain="work")
    print(f"   Think cycle on: {think_result.cluster_topic}")
    print(f"   Generated {len(think_result.new_nodes)} insights")
    print(f"   Created {len(think_result.new_edges)} derivation edges")
    
    if think_result.new_nodes:
        print("   Generated insights:")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        for node_id in think_result.new_nodes:
            cursor.execute("SELECT content FROM thought_nodes WHERE id = ?", (node_id,))
            row = cursor.fetchone()
            if row:
                print(f"     💡 {row[0]}")
        conn.close()
    
    # Show final statistics
    print("\n6. Final database statistics...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM thought_nodes WHERE decayed IS NULL OR decayed = 0")
    total_nodes = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM derivation_edges")
    total_edges = cursor.fetchone()[0]
    
    cursor.execute("SELECT node_type, COUNT(*) FROM thought_nodes GROUP BY node_type")
    node_types = dict(cursor.fetchall())
    
    conn.close()
    
    print(f"   Total active nodes: {total_nodes}")
    print(f"   Total edges: {total_edges}")
    print(f"   Node types: {node_types}")
    
    print(f"\n✅ Demo complete! Database saved at: {db_path}")
    print("🧠 The session integration layer successfully:")
    print("   • Retrieved relevant context for session start")
    print("   • Extracted structured knowledge from conversation")
    print("   • Generated cross-domain insights via think cycles")
    print("   • Maintained graph connectivity with automatic edge creation")

if __name__ == "__main__":
    main()