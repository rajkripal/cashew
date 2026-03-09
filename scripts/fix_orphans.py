#!/usr/bin/env python3
"""
Orphan Cleanup Script for cashew thought-graph engine.
Finds orphan nodes (no edges) and connects them to similar nodes or decays them.
"""

import sqlite3
import numpy as np
import pickle
import sys
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

def load_embedding(blob_data):
    """Load embedding from BLOB data."""
    if not blob_data:
        return None
    
    try:
        # First try pickle
        return pickle.loads(blob_data)
    except:
        try:
            # Try as numpy array
            return np.frombuffer(blob_data, dtype=np.float32).copy()
        except:
            try:
                # Try as float64
                return np.frombuffer(blob_data, dtype=np.float64).copy()
            except:
                return None

def find_orphans(cursor):
    """Find all orphan nodes (nodes with no edges)."""
    cursor.execute("""
        SELECT tn.id, tn.content, tn.node_type, tn.confidence, tn.timestamp 
        FROM thought_nodes tn 
        WHERE tn.decayed = 0 
        AND tn.id NOT IN (
            SELECT DISTINCT parent_id FROM derivation_edges 
            UNION 
            SELECT DISTINCT child_id FROM derivation_edges
        )
    """)
    return cursor.fetchall()

def find_non_orphans_with_embeddings(cursor):
    """Get all non-orphan nodes that have embeddings."""
    cursor.execute("""
        SELECT DISTINCT tn.id, e.vector, tn.confidence, tn.timestamp
        FROM thought_nodes tn 
        JOIN embeddings e ON tn.id = e.node_id
        WHERE tn.decayed = 0 
        AND tn.id IN (
            SELECT DISTINCT parent_id FROM derivation_edges 
            UNION 
            SELECT DISTINCT child_id FROM derivation_edges
        )
    """)
    return cursor.fetchall()

def get_orphan_embedding(cursor, node_id):
    """Get embedding for an orphan node."""
    cursor.execute("SELECT vector FROM embeddings WHERE node_id = ?", (node_id,))
    result = cursor.fetchone()
    if result:
        return load_embedding(result[0])
    return None

def find_most_similar_node(orphan_embedding, non_orphan_embeddings):
    """Find the most similar non-orphan node via cosine similarity."""
    if orphan_embedding is None or len(non_orphan_embeddings) == 0:
        return None, 0.0
    
    try:
        # Reshape orphan embedding for sklearn
        orphan_vec = orphan_embedding.reshape(1, -1)
        
        # Find valid non-orphan embeddings with same dimension
        valid_embeddings = []
        valid_node_ids = []
        
        for node_id, embedding, confidence, timestamp in non_orphan_embeddings:
            if embedding is not None and len(embedding) == len(orphan_embedding):
                valid_embeddings.append(embedding)
                valid_node_ids.append((node_id, confidence, timestamp))
        
        if len(valid_embeddings) == 0:
            return None, 0.0
        
        # Calculate similarities
        embeddings_matrix = np.array(valid_embeddings)
        similarities = cosine_similarity(orphan_vec, embeddings_matrix)[0]
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        best_node_info = valid_node_ids[best_idx]
        
        return best_node_info[0], float(best_similarity)
    
    except Exception as e:
        print(f"Warning: Could not compute similarity: {e}", file=sys.stderr)
        return None, 0.0

def create_edge(cursor, parent_id, child_id, relation="similarity_link"):
    """Create an edge between parent and child nodes."""
    try:
        cursor.execute("""
            INSERT OR IGNORE INTO derivation_edges 
            (parent_id, child_id, relation, weight, reasoning)
            VALUES (?, ?, ?, ?, ?)
        """, (parent_id, child_id, relation, 0.7, "Connected orphan via similarity"))
        return True
    except Exception as e:
        print(f"Error creating edge {parent_id} -> {child_id}: {e}", file=sys.stderr)
        return False

def decay_node(cursor, node_id):
    """Mark a node as decayed."""
    try:
        cursor.execute("""
            UPDATE thought_nodes 
            SET decayed = 1, last_updated = ?
            WHERE id = ?
        """, (datetime.now().isoformat(), node_id))
        return True
    except Exception as e:
        print(f"Error decaying node {node_id}: {e}", file=sys.stderr)
        return False

def fix_orphans(db_path):
    """Main orphan cleanup function."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Find all orphans
    orphans = find_orphans(cursor)
    print(f"Found {len(orphans)} orphan nodes", file=sys.stderr)
    
    # Get non-orphans with embeddings
    non_orphan_data = find_non_orphans_with_embeddings(cursor)
    print(f"Found {len(non_orphan_data)} non-orphan nodes with embeddings", file=sys.stderr)
    
    # Load non-orphan embeddings
    non_orphan_embeddings = []
    for node_id, vector_blob, confidence, timestamp in non_orphan_data:
        embedding = load_embedding(vector_blob)
        if embedding is not None:
            non_orphan_embeddings.append((node_id, embedding, confidence, timestamp))
    
    print(f"Loaded {len(non_orphan_embeddings)} valid non-orphan embeddings", file=sys.stderr)
    
    # Process each orphan
    stats = {
        'connected': 0,
        'decayed': 0,
        'left_alone': 0,
        'no_embedding': 0,
        'actions': []
    }
    
    for orphan_id, content, node_type, confidence, timestamp in orphans:
        # Get orphan embedding
        orphan_embedding = get_orphan_embedding(cursor, orphan_id)
        
        if orphan_embedding is None:
            stats['no_embedding'] += 1
            stats['actions'].append({
                'orphan_id': orphan_id,
                'action': 'no_embedding',
                'content': content[:100]
            })
            continue
        
        # Find most similar non-orphan
        best_match_id, similarity = find_most_similar_node(
            orphan_embedding, non_orphan_embeddings
        )
        
        if best_match_id is None:
            stats['no_embedding'] += 1
            stats['actions'].append({
                'orphan_id': orphan_id,
                'action': 'no_match_found',
                'content': content[:100]
            })
            continue
        
        if similarity > 0.7:
            # Connect them
            if create_edge(cursor, best_match_id, orphan_id):
                stats['connected'] += 1
                stats['actions'].append({
                    'orphan_id': orphan_id,
                    'action': 'connected',
                    'parent_id': best_match_id,
                    'similarity': similarity,
                    'content': content[:100]
                })
            
        elif similarity < 0.5:
            # Decay the orphan
            if decay_node(cursor, orphan_id):
                stats['decayed'] += 1
                stats['actions'].append({
                    'orphan_id': orphan_id,
                    'action': 'decayed',
                    'similarity': similarity,
                    'content': content[:100]
                })
        
        else:
            # Leave alone (0.5 <= similarity <= 0.7)
            stats['left_alone'] += 1
            stats['actions'].append({
                'orphan_id': orphan_id,
                'action': 'left_alone',
                'similarity': similarity,
                'content': content[:100]
            })
    
    # Commit changes
    conn.commit()
    conn.close()
    
    return stats

def main():
    """Main entry point."""
    db_path = "data/graph.db"
    
    try:
        print("Starting orphan cleanup...", file=sys.stderr)
        stats = fix_orphans(db_path)
        
        # Print summary
        print("\n" + "="*60, file=sys.stderr)
        print("ORPHAN CLEANUP RESULTS", file=sys.stderr)
        print("="*60, file=sys.stderr)
        print(f"Connected: {stats['connected']}", file=sys.stderr)
        print(f"Decayed: {stats['decayed']}", file=sys.stderr)
        print(f"Left alone: {stats['left_alone']}", file=sys.stderr)
        print(f"No embedding: {stats['no_embedding']}", file=sys.stderr)
        print("="*60, file=sys.stderr)
        
        # Print detailed log
        print("DETAILED LOG:")
        for action in stats['actions']:
            if action['action'] == 'connected':
                print(f"CONNECTED: {action['orphan_id']} -> {action['parent_id']} (sim: {action['similarity']:.3f}) | {action['content']}")
            elif action['action'] == 'decayed':
                print(f"DECAYED: {action['orphan_id']} (sim: {action['similarity']:.3f}) | {action['content']}")
            elif action['action'] == 'left_alone':
                print(f"LEFT_ALONE: {action['orphan_id']} (sim: {action['similarity']:.3f}) | {action['content']}")
            else:
                print(f"{action['action'].upper()}: {action['orphan_id']} | {action['content']}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    exit(main())