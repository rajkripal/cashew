#!/usr/bin/env python3
"""
Cashew Embeddings Module
Local sentence-transformer embeddings for semantic search
"""

import sqlite3
import numpy as np
from typing import List, Tuple, Optional
from datetime import datetime
import logging
import sys
import argparse

# Lazy import pattern - only load when first called
_model = None

def _get_model():
    """Lazy-load the sentence transformer model (singleton pattern)"""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer('all-MiniLM-L6-v2')
            logging.info("Loaded sentence-transformer model: all-MiniLM-L6-v2")
        except ImportError:
            raise ImportError("sentence-transformers package not installed. Run: pip install sentence-transformers")
    return _model

def embed_text(text: str) -> List[float]:
    """
    Embed a single text string into a vector
    
    Args:
        text: Input text to embed
        
    Returns:
        384-dimensional embedding vector as list of floats
    """
    if not text or not text.strip():
        # Return zero vector for empty text
        return [0.0] * 384
    
    model = _get_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()

def _ensure_embeddings_table(db_path: str):
    """Ensure the embeddings table exists in the database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            node_id TEXT PRIMARY KEY,
            vector BLOB NOT NULL,
            model TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (node_id) REFERENCES thought_nodes(id)
        )
    """)
    
    conn.commit()
    conn.close()

def embed_nodes(db_path: str, batch_size: int = 100) -> dict:
    """
    Embed all nodes that don't have embeddings yet
    
    Args:
        db_path: Path to SQLite database
        batch_size: Number of nodes to process at once
        
    Returns:
        Dictionary with statistics about the embedding process
    """
    _ensure_embeddings_table(db_path)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get nodes that need embedding (don't have decayed flag or are not decayed)
    cursor.execute("""
        SELECT tn.id, tn.content 
        FROM thought_nodes tn
        LEFT JOIN embeddings e ON tn.id = e.node_id
        WHERE e.node_id IS NULL 
        AND (tn.decayed IS NULL OR tn.decayed = 0)
        ORDER BY tn.timestamp DESC
    """)
    
    nodes_to_embed = cursor.fetchall()
    total_nodes = len(nodes_to_embed)
    
    if total_nodes == 0:
        conn.close()
        return {"total_nodes": 0, "embedded": 0, "skipped": 0}
    
    logging.info(f"Found {total_nodes} nodes to embed")
    
    embedded_count = 0
    model = _get_model()  # Load model once
    
    # Process in batches
    for i in range(0, total_nodes, batch_size):
        batch = nodes_to_embed[i:i + batch_size]
        batch_texts = [content for _, content in batch]
        batch_ids = [node_id for node_id, _ in batch]
        
        # Embed the batch
        try:
            embeddings = model.encode(batch_texts, convert_to_numpy=True)
            
            # Store embeddings
            for j, (node_id, content) in enumerate(batch):
                vector_bytes = embeddings[j].tobytes()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO embeddings 
                    (node_id, vector, model, updated_at)
                    VALUES (?, ?, ?, ?)
                """, (node_id, vector_bytes, "all-MiniLM-L6-v2", datetime.now().isoformat()))
                
                embedded_count += 1
            
            conn.commit()
            logging.info(f"Embedded batch {i//batch_size + 1}/{(total_nodes + batch_size - 1)//batch_size}")
            
        except Exception as e:
            logging.error(f"Error embedding batch starting at {i}: {e}")
            continue
    
    conn.close()
    
    return {
        "total_nodes": total_nodes,
        "embedded": embedded_count,
        "skipped": total_nodes - embedded_count
    }

def search(db_path: str, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
    """
    Semantic search using cosine similarity
    
    Args:
        db_path: Path to SQLite database
        query: Search query text
        top_k: Number of results to return
        
    Returns:
        List of (node_id, similarity_score) tuples, sorted by similarity descending
    """
    if not query or not query.strip():
        return []
    
    _ensure_embeddings_table(db_path)
    
    # Embed the query
    query_embedding = np.array(embed_text(query))
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all embeddings
    cursor.execute("""
        SELECT e.node_id, e.vector 
        FROM embeddings e
        JOIN thought_nodes tn ON e.node_id = tn.id
        WHERE tn.decayed IS NULL OR tn.decayed = 0
    """)
    
    results = []
    
    for node_id, vector_bytes in cursor.fetchall():
        try:
            # Convert bytes back to numpy array
            stored_embedding = np.frombuffer(vector_bytes, dtype=np.float32)
            
            # Calculate cosine similarity
            dot_product = np.dot(query_embedding, stored_embedding)
            query_norm = np.linalg.norm(query_embedding)
            stored_norm = np.linalg.norm(stored_embedding)
            
            if query_norm > 0 and stored_norm > 0:
                similarity = dot_product / (query_norm * stored_norm)
                results.append((node_id, float(similarity)))
        
        except Exception as e:
            logging.warning(f"Error processing embedding for node {node_id}: {e}")
            continue
    
    conn.close()
    
    # Sort by similarity (descending) and return top_k
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]

def get_embedding_stats(db_path: str) -> dict:
    """
    Get statistics about embeddings in the database
    
    Args:
        db_path: Path to SQLite database
        
    Returns:
        Dictionary with embedding statistics
    """
    _ensure_embeddings_table(db_path)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Count total nodes
    cursor.execute("SELECT COUNT(*) FROM thought_nodes WHERE decayed IS NULL OR decayed = 0")
    total_nodes = cursor.fetchone()[0]
    
    # Count embedded nodes
    cursor.execute("""
        SELECT COUNT(*) FROM embeddings e
        JOIN thought_nodes tn ON e.node_id = tn.id
        WHERE tn.decayed IS NULL OR tn.decayed = 0
    """)
    embedded_nodes = cursor.fetchone()[0]
    
    # Count by model
    cursor.execute("""
        SELECT model, COUNT(*) FROM embeddings e
        JOIN thought_nodes tn ON e.node_id = tn.id
        WHERE tn.decayed IS NULL OR tn.decayed = 0
        GROUP BY model
    """)
    models = dict(cursor.fetchall())
    
    # Most recent embedding
    cursor.execute("""
        SELECT MAX(updated_at) FROM embeddings e
        JOIN thought_nodes tn ON e.node_id = tn.id
        WHERE tn.decayed IS NULL OR tn.decayed = 0
    """)
    last_updated = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        "total_nodes": total_nodes,
        "embedded_nodes": embedded_nodes,
        "missing_embeddings": total_nodes - embedded_nodes,
        "coverage_percentage": (embedded_nodes / total_nodes * 100) if total_nodes > 0 else 0.0,
        "models_used": models,
        "last_updated": last_updated
    }

def main():
    """CLI interface for embeddings module"""
    parser = argparse.ArgumentParser(description="Cashew Embeddings Module")
    parser.add_argument("command", choices=["embed", "search", "stats"], help="Command to run")
    parser.add_argument("--db", default="/Users/bunny/.openclaw/workspace/cashew/data/graph.db", help="Database path")
    parser.add_argument("--query", help="Search query (for search command)")
    parser.add_argument("--top-k", type=int, default=10, help="Number of search results")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for embedding")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    if args.command == "embed":
        print("🔮 Embedding nodes...")
        stats = embed_nodes(args.db, args.batch_size)
        print(f"✅ Embedded {stats['embedded']} nodes (total: {stats['total_nodes']}, skipped: {stats['skipped']})")
    
    elif args.command == "search":
        if not args.query:
            print("Error: --query required for search command")
            return 1
        
        print(f"🔍 Searching for: {args.query}")
        results = search(args.db, args.query, args.top_k)
        
        if not results:
            print("No results found")
            return 0
        
        # Load node details for display
        conn = sqlite3.connect(args.db)
        cursor = conn.cursor()
        
        print(f"\nTop {len(results)} results:")
        for i, (node_id, score) in enumerate(results):
            cursor.execute("SELECT content, node_type FROM thought_nodes WHERE id = ?", (node_id,))
            row = cursor.fetchone()
            if row:
                content, node_type = row
                print(f"{i+1:2}. [{score:.3f}] {content[:80]}..." + (f" [{node_type}]" if len(content) > 80 else f" [{node_type}]"))
            else:
                print(f"{i+1:2}. [{score:.3f}] {node_id} [missing]")
        
        conn.close()
    
    elif args.command == "stats":
        print("📊 Embedding Statistics")
        stats = get_embedding_stats(args.db)
        
        print(f"Total nodes: {stats['total_nodes']}")
        print(f"Embedded nodes: {stats['embedded_nodes']}")
        print(f"Missing embeddings: {stats['missing_embeddings']}")
        print(f"Coverage: {stats['coverage_percentage']:.1f}%")
        print(f"Models used: {dict(stats['models_used'])}")
        print(f"Last updated: {stats['last_updated']}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())