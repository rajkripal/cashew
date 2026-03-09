#!/usr/bin/env python3
"""
Semantic Deduplication Script for cashew thought-graph engine.
Finds active node pairs with high embedding similarity and merges duplicates.
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

def get_active_nodes_with_embeddings(cursor):
    """Get all active nodes with their embeddings."""
    cursor.execute("""
        SELECT tn.id, tn.content, tn.confidence, tn.timestamp, e.vector
        FROM thought_nodes tn 
        JOIN embeddings e ON tn.id = e.node_id
        WHERE tn.decayed = 0
        ORDER BY tn.timestamp ASC
    """)
    
    results = []
    for node_id, content, confidence, timestamp, vector_blob in cursor.fetchall():
        embedding = load_embedding(vector_blob)
        if embedding is not None:
            results.append({
                'id': node_id,
                'content': content,
                'confidence': confidence,
                'timestamp': timestamp,
                'embedding': embedding
            })
    
    return results

def find_duplicate_pairs(nodes, similarity_threshold=0.95):
    """Find pairs of nodes with high similarity."""
    if len(nodes) < 2:
        return []
    
    # Group nodes by embedding dimension to avoid comparison errors
    dimension_groups = {}
    for node in nodes:
        dim = len(node['embedding'])
        if dim not in dimension_groups:
            dimension_groups[dim] = []
        dimension_groups[dim].append(node)
    
    duplicate_pairs = []
    
    for dim, group_nodes in dimension_groups.items():
        if len(group_nodes) < 2:
            continue
        
        try:
            # Build similarity matrix for this dimension group
            embeddings_matrix = np.array([node['embedding'] for node in group_nodes])
            similarity_matrix = cosine_similarity(embeddings_matrix)
            
            # Find high-similarity pairs
            for i in range(len(group_nodes)):
                for j in range(i + 1, len(group_nodes)):
                    similarity = similarity_matrix[i][j]
                    if not np.isnan(similarity) and similarity > similarity_threshold:
                        duplicate_pairs.append({
                            'node1': group_nodes[i],
                            'node2': group_nodes[j],
                            'similarity': float(similarity)
                        })
        
        except Exception as e:
            print(f"Warning: Could not compute similarities for dimension {dim}: {e}", file=sys.stderr)
            continue
    
    # Sort by similarity (highest first)
    duplicate_pairs.sort(key=lambda x: x['similarity'], reverse=True)
    return duplicate_pairs

def choose_node_to_keep(node1, node2):
    """Choose which node to keep based on confidence and timestamp."""
    # First priority: higher confidence
    if node1['confidence'] != node2['confidence']:
        return node1 if node1['confidence'] > node2['confidence'] else node2
    
    # Second priority: older timestamp (first one created)
    try:
        ts1 = datetime.fromisoformat(node1['timestamp'].replace('Z', '+00:00'))
        ts2 = datetime.fromisoformat(node2['timestamp'].replace('Z', '+00:00'))
        return node1 if ts1 < ts2 else node2
    except:
        # If timestamp parsing fails, default to node1
        return node1

def get_node_edges(cursor, node_id):
    """Get all edges where node is parent or child."""
    cursor.execute("""
        SELECT parent_id, child_id, relation, weight, reasoning
        FROM derivation_edges
        WHERE parent_id = ? OR child_id = ?
    """, (node_id, node_id))
    return cursor.fetchall()

def transfer_edges(cursor, from_node_id, to_node_id):
    """Transfer edges from one node to another."""
    edges = get_node_edges(cursor, from_node_id)
    transferred = 0
    
    for parent_id, child_id, relation, weight, reasoning in edges:
        try:
            if parent_id == from_node_id:
                # Node was parent, make target node the new parent
                new_parent_id = to_node_id
                new_child_id = child_id
            else:
                # Node was child, make target node the new child
                new_parent_id = parent_id
                new_child_id = to_node_id
            
            # Skip self-edges
            if new_parent_id == new_child_id:
                continue
            
            # Insert new edge (ignore if already exists)
            cursor.execute("""
                INSERT OR IGNORE INTO derivation_edges 
                (parent_id, child_id, relation, weight, reasoning)
                VALUES (?, ?, ?, ?, ?)
            """, (new_parent_id, new_child_id, relation, weight, 
                  f"Transferred from duplicate {from_node_id}: {reasoning}"))
            transferred += 1
            
        except Exception as e:
            print(f"Warning: Could not transfer edge {parent_id}->{child_id}: {e}", file=sys.stderr)
    
    # Delete old edges
    cursor.execute("""
        DELETE FROM derivation_edges 
        WHERE parent_id = ? OR child_id = ?
    """, (from_node_id, from_node_id))
    
    return transferred

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

def deduplicate_graph(db_path, similarity_threshold=0.95):
    """Main deduplication function."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all active nodes with embeddings
    print("Loading active nodes with embeddings...", file=sys.stderr)
    nodes = get_active_nodes_with_embeddings(cursor)
    print(f"Loaded {len(nodes)} nodes", file=sys.stderr)
    
    # Find duplicate pairs
    print(f"Finding duplicate pairs (similarity > {similarity_threshold})...", file=sys.stderr)
    duplicate_pairs = find_duplicate_pairs(nodes, similarity_threshold)
    print(f"Found {len(duplicate_pairs)} potential duplicate pairs", file=sys.stderr)
    
    if len(duplicate_pairs) == 0:
        conn.close()
        return {
            'merged': 0,
            'edges_transferred': 0,
            'merges': []
        }
    
    # Process each duplicate pair
    stats = {
        'merged': 0,
        'edges_transferred': 0,
        'merges': []
    }
    
    processed_nodes = set()  # Track nodes already processed to avoid double-processing
    
    for pair in duplicate_pairs:
        node1 = pair['node1']
        node2 = pair['node2']
        similarity = pair['similarity']
        
        # Skip if either node was already processed
        if node1['id'] in processed_nodes or node2['id'] in processed_nodes:
            continue
        
        # Choose which node to keep
        keep_node = choose_node_to_keep(node1, node2)
        discard_node = node2 if keep_node == node1 else node1
        
        print(f"Merging: {discard_node['id']} -> {keep_node['id']} (similarity: {similarity:.4f})", file=sys.stderr)
        
        # Transfer edges
        edges_transferred = transfer_edges(cursor, discard_node['id'], keep_node['id'])
        
        # Decay the duplicate node
        if decay_node(cursor, discard_node['id']):
            stats['merged'] += 1
            stats['edges_transferred'] += edges_transferred
            stats['merges'].append({
                'kept_node': keep_node['id'],
                'discarded_node': discard_node['id'],
                'similarity': similarity,
                'kept_content': keep_node['content'][:100] + '...' if len(keep_node['content']) > 100 else keep_node['content'],
                'discarded_content': discard_node['content'][:100] + '...' if len(discard_node['content']) > 100 else discard_node['content'],
                'edges_transferred': edges_transferred,
                'kept_confidence': keep_node['confidence'],
                'discarded_confidence': discard_node['confidence']
            })
            
            # Mark both nodes as processed
            processed_nodes.add(keep_node['id'])
            processed_nodes.add(discard_node['id'])
    
    # Commit changes
    conn.commit()
    conn.close()
    
    return stats

def main():
    """Main entry point."""
    db_path = "data/graph.db"
    similarity_threshold = 0.95
    
    try:
        print("Starting semantic deduplication...", file=sys.stderr)
        stats = deduplicate_graph(db_path, similarity_threshold)
        
        # Print summary
        print("\n" + "="*60, file=sys.stderr)
        print("SEMANTIC DEDUPLICATION RESULTS", file=sys.stderr)
        print("="*60, file=sys.stderr)
        print(f"Merged pairs: {stats['merged']}", file=sys.stderr)
        print(f"Edges transferred: {stats['edges_transferred']}", file=sys.stderr)
        print("="*60, file=sys.stderr)
        
        # Print detailed log
        print("\nDETAILED LOG:")
        for merge in stats['merges']:
            print(f"MERGED: {merge['discarded_node']} -> {merge['kept_node']} (sim: {merge['similarity']:.4f}, edges: {merge['edges_transferred']})")
            print(f"  KEPT (conf: {merge['kept_confidence']:.2f}): {merge['kept_content']}")
            print(f"  DISCARDED (conf: {merge['discarded_confidence']:.2f}): {merge['discarded_content']}")
            print()
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    exit(main())