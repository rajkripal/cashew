#!/usr/bin/env python3
"""
Demo script to showcase the cashew persistence layer
"""

import os
import tempfile
import sqlite3
from core.embeddings import embed_nodes, search, get_embedding_stats
from core.retrieval import retrieve, format_context, explain_retrieval

def create_demo_db():
    """Create a small demo database for testing"""
    fd, db_path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute("""
        CREATE TABLE thought_nodes (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            node_type TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            confidence REAL NOT NULL,
            mood_state TEXT,
            metadata TEXT,
            source_file TEXT,
            decayed INTEGER DEFAULT 0
        )
    """)
    
    cursor.execute("""
        CREATE TABLE derivation_edges (
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
    
    # Insert demo thoughts
    demo_thoughts = [
        ("thought1", "I love learning about artificial intelligence and machine learning", "interest", "2023-01-01T10:00:00", 0.9, "excited", '{"domain": "technology"}', "demo"),
        ("thought2", "Python is my favorite programming language for AI projects", "preference", "2023-01-01T11:00:00", 0.8, "confident", '{"domain": "programming"}', "demo"),
        ("thought3", "Working on neural networks gives me a sense of accomplishment", "feeling", "2023-01-01T12:00:00", 0.7, "fulfilled", '{"domain": "personal"}', "demo"),
        ("thought4", "I should build a chatbot to practice my skills", "goal", "2023-01-01T13:00:00", 0.8, "motivated", '{"domain": "projects"}', "demo"),
        ("thought5", "Reading research papers helps me understand new techniques", "learning", "2023-01-01T14:00:00", 0.9, "curious", '{"domain": "education"}', "demo"),
        ("thought6", "Collaboration with other AI enthusiasts is valuable", "insight", "2023-01-01T15:00:00", 0.7, "social", '{"domain": "community"}', "demo"),
    ]
    
    cursor.executemany("""
        INSERT INTO thought_nodes 
        (id, content, node_type, timestamp, confidence, mood_state, metadata, source_file, decayed)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
    """, demo_thoughts)
    
    # Add some connections
    edges = [
        ("thought1", "thought2", "supports", 0.9, "AI interest supports Python preference"),
        ("thought2", "thought3", "enables", 0.8, "Python enables neural network work"),
        ("thought3", "thought4", "motivates", 0.7, "Accomplishment motivates new goals"),
        ("thought1", "thought5", "drives", 0.8, "AI interest drives learning"),
        ("thought5", "thought6", "leads_to", 0.6, "Learning leads to community appreciation"),
    ]
    
    cursor.executemany("""
        INSERT INTO derivation_edges 
        (parent_id, child_id, relation, weight, reasoning)
        VALUES (?, ?, ?, ?, ?)
    """, edges)
    
    conn.commit()
    conn.close()
    
    return db_path

def main():
    print("🔮 Cashew Persistence Layer Demo")
    print("=" * 50)
    
    # Create demo database
    print("Creating demo database...")
    db_path = create_demo_db()
    
    try:
        # Step 1: Embed all nodes
        print("\n📊 Step 1: Embedding nodes...")
        stats = embed_nodes(db_path)
        print(f"Embedded {stats['embedded']} nodes")
        
        # Step 2: Show embedding stats
        print("\n📈 Step 2: Embedding statistics...")
        embedding_stats = get_embedding_stats(db_path)
        print(f"Total nodes: {embedding_stats['total_nodes']}")
        print(f"Coverage: {embedding_stats['coverage_percentage']:.1f}%")
        print(f"Models: {list(embedding_stats['models_used'].keys())}")
        
        # Step 3: Test semantic search
        print("\n🔍 Step 3: Semantic search...")
        search_results = search(db_path, "artificial intelligence programming", top_k=3)
        print("Top 3 results for 'artificial intelligence programming':")
        for i, (node_id, score) in enumerate(search_results, 1):
            # Get node content
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT content FROM thought_nodes WHERE id = ?", (node_id,))
            content = cursor.fetchone()[0]
            conn.close()
            print(f"  {i}. [{score:.3f}] {content}")
        
        # Step 4: Test hybrid retrieval
        print("\n🚀 Step 4: Hybrid retrieval...")
        retrieval_results = retrieve(db_path, "machine learning projects", top_k=4, walk_depth=2)
        print("Hybrid retrieval results for 'machine learning projects':")
        for i, result in enumerate(retrieval_results, 1):
            path_info = f" (path: {' → '.join(result.path[-2:])})" if len(result.path) > 1 else ""
            print(f"  {i}. [{result.score:.3f}] {result.content[:60]}...{path_info}")
        
        # Step 5: Format context
        print("\n📝 Step 5: Formatted context...")
        context = format_context(retrieval_results, include_paths=True)
        print(context[:500] + "..." if len(context) > 500 else context)
        
        print("\n✅ Demo completed successfully!")
        print(f"Demo database created at: {db_path}")
        print("You can explore it further using the CLI tools:")
        print(f"  python3 -m core.embeddings search --db {db_path} --query 'your query'")
        print(f"  python3 -m core.retrieval retrieve --db {db_path} --query 'your query'")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        try:
            os.unlink(db_path)
            print(f"\nCleaned up demo database: {db_path}")
        except:
            pass

if __name__ == "__main__":
    main()