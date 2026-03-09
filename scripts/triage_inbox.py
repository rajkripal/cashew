#!/usr/bin/env python3
"""
Inbox Triage Script - Sleep Cycle for Cashew
Triages nodes from the inbox hotspot into appropriate domain hotspots.

Target: Move nodes from inbox (07621e89a08e) with similarity >0.35 to proper clusters.
Goal: Reduce inbox from 289 nodes to under 100.
"""

import sqlite3
import sys
import time
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# Add the parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.embeddings import embed_text
from core.complete_clustering import load_embeddings_with_metadata


def get_connection(db_path: str) -> sqlite3.Connection:
    """Get database connection"""
    return sqlite3.connect(db_path)


def get_inbox_members(db_path: str, inbox_hotspot_id: str = "07621e89a08e") -> List[str]:
    """Get all member node IDs from the inbox hotspot"""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT de.child_id
        FROM derivation_edges de
        JOIN thought_nodes tn ON de.child_id = tn.id
        WHERE de.parent_id = ? AND de.relation = 'summarizes'
        AND (tn.decayed IS NULL OR tn.decayed = 0)
    """, (inbox_hotspot_id,))
    
    member_ids = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    return member_ids


def get_target_hotspots(db_path: str, exclude_inbox_id: str = "07621e89a08e") -> List[Tuple[str, str]]:
    """Get all non-inbox hotspot IDs and their domain"""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, COALESCE(domain, 'general') as domain
        FROM thought_nodes
        WHERE node_type = 'hotspot' 
        AND id != ?
        AND (decayed IS NULL OR decayed = 0)
    """, (exclude_inbox_id,))
    
    hotspots = [(row[0], row[1]) for row in cursor.fetchall()]
    conn.close()
    
    return hotspots


def compute_node_similarities(db_path: str, inbox_member_ids: List[str], 
                            target_hotspot_ids: List[str]) -> Dict[str, List[Tuple[str, float]]]:
    """
    Compute embedding similarities between inbox members and target hotspots.
    
    Returns:
        Dict mapping inbox_member_id -> [(hotspot_id, similarity), ...] sorted by similarity desc
    """
    # Load all embeddings
    node_ids, vectors, node_meta = load_embeddings_with_metadata(db_path)
    
    if not node_ids:
        return {}
    
    # Build lookup
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    
    # Get indices for inbox members and target hotspots that have embeddings
    inbox_indices = [id_to_idx[mid] for mid in inbox_member_ids if mid in id_to_idx]
    target_indices = [id_to_idx[hid] for hid in target_hotspot_ids if hid in id_to_idx]
    
    if not inbox_indices or not target_indices:
        print("Warning: No embeddings found for inbox members or target hotspots")
        return {}
    
    # Get vectors
    inbox_vectors = vectors[inbox_indices]
    target_vectors = vectors[target_indices]
    
    # Compute cosine similarities (inbox_members x target_hotspots)
    similarities = np.dot(inbox_vectors, target_vectors.T) / (
        np.linalg.norm(inbox_vectors, axis=1, keepdims=True) * 
        np.linalg.norm(target_vectors, axis=1, keepdims=False)
    )
    
    # Build results
    results = {}
    inbox_ids_with_embeddings = [inbox_member_ids[i] for i in range(len(inbox_member_ids)) 
                                if inbox_member_ids[i] in id_to_idx]
    target_ids_with_embeddings = [target_hotspot_ids[i] for i in range(len(target_hotspot_ids))
                                 if target_hotspot_ids[i] in id_to_idx]
    
    for i, inbox_id in enumerate(inbox_ids_with_embeddings):
        if i < similarities.shape[0]:  # Safety check
            # Get similarities for this inbox member to all targets
            member_similarities = [(target_ids_with_embeddings[j], float(similarities[i, j]))
                                 for j in range(similarities.shape[1])]
            
            # Sort by similarity descending
            member_similarities.sort(key=lambda x: x[1], reverse=True)
            results[inbox_id] = member_similarities
    
    return results


def move_node_to_hotspot(db_path: str, node_id: str, old_hotspot_id: str, 
                        new_hotspot_id: str, similarity: float) -> bool:
    """
    Move a node from one hotspot to another.
    
    Returns:
        True if successful, False otherwise
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    try:
        # Delete old edge
        cursor.execute("""
            DELETE FROM derivation_edges 
            WHERE parent_id = ? AND child_id = ? AND relation = 'summarizes'
        """, (old_hotspot_id, node_id))
        
        # Create new edge
        cursor.execute("""
            INSERT OR IGNORE INTO derivation_edges (parent_id, child_id, relation, weight, reasoning)
            VALUES (?, ?, 'summarizes', ?, ?)
        """, (new_hotspot_id, node_id, 0.8, f"Sleep cycle reclassification (similarity: {similarity:.3f})"))
        
        conn.commit()
        conn.close()
        return True
        
    except Exception as e:
        print(f"Error moving node {node_id}: {e}")
        conn.rollback()
        conn.close()
        return False


def triage_inbox(db_path: str, similarity_threshold: float = 0.35, 
                inbox_hotspot_id: str = "07621e89a08e", dry_run: bool = False) -> Dict:
    """
    Main inbox triage function.
    
    Args:
        db_path: Path to graph database
        similarity_threshold: Minimum similarity to move a node (default 0.35)
        inbox_hotspot_id: ID of the inbox hotspot
        dry_run: If True, don't actually move nodes
        
    Returns:
        Dict with triage results and statistics
    """
    print(f"🗂️  Starting inbox triage (threshold: {similarity_threshold}, dry_run: {dry_run})")
    
    # Get inbox members
    print("📋 Getting inbox members...")
    inbox_members = get_inbox_members(db_path, inbox_hotspot_id)
    print(f"   Found {len(inbox_members)} nodes in inbox")
    
    if not inbox_members:
        return {"success": False, "error": "No inbox members found", "moved": 0}
    
    # Get target hotspots
    print("🎯 Getting target hotspots...")
    target_hotspots = get_target_hotspots(db_path, inbox_hotspot_id)
    target_hotspot_ids = [h[0] for h in target_hotspots]
    print(f"   Found {len(target_hotspots)} target hotspots")
    
    if not target_hotspots:
        return {"success": False, "error": "No target hotspots found", "moved": 0}
    
    # Compute similarities
    print("🧮 Computing similarities...")
    similarities = compute_node_similarities(db_path, inbox_members, target_hotspot_ids)
    
    if not similarities:
        return {"success": False, "error": "No similarities computed (missing embeddings?)", "moved": 0}
    
    print(f"   Computed similarities for {len(similarities)} inbox nodes")
    
    # Find nodes to move
    moves_to_make = []
    
    for inbox_node_id, node_similarities in similarities.items():
        if node_similarities:  # Has at least one target similarity
            best_hotspot_id, best_similarity = node_similarities[0]  # Already sorted desc
            
            if best_similarity >= similarity_threshold:
                moves_to_make.append((inbox_node_id, best_hotspot_id, best_similarity))
    
    print(f"📦 Found {len(moves_to_make)} nodes to move (similarity >= {similarity_threshold})")
    
    # Execute moves
    successful_moves = 0
    failed_moves = 0
    move_details = []
    
    for inbox_node_id, target_hotspot_id, similarity in moves_to_make:
        if dry_run:
            move_details.append({
                "node_id": inbox_node_id,
                "from_hotspot": inbox_hotspot_id,
                "to_hotspot": target_hotspot_id,
                "similarity": similarity,
                "status": "dry_run"
            })
            successful_moves += 1
        else:
            success = move_node_to_hotspot(db_path, inbox_node_id, inbox_hotspot_id, 
                                         target_hotspot_id, similarity)
            if success:
                successful_moves += 1
                move_details.append({
                    "node_id": inbox_node_id,
                    "from_hotspot": inbox_hotspot_id,
                    "to_hotspot": target_hotspot_id,
                    "similarity": similarity,
                    "status": "moved"
                })
                print(f"   ✅ Moved {inbox_node_id} to {target_hotspot_id[:8]} (sim: {similarity:.3f})")
            else:
                failed_moves += 1
                move_details.append({
                    "node_id": inbox_node_id,
                    "from_hotspot": inbox_hotspot_id,
                    "to_hotspot": target_hotspot_id,
                    "similarity": similarity,
                    "status": "failed"
                })
                print(f"   ❌ Failed to move {inbox_node_id}")
    
    # Final stats
    remaining_in_inbox = len(inbox_members) - successful_moves
    
    result = {
        "success": True,
        "total_inbox_members": len(inbox_members),
        "nodes_evaluated": len(similarities),
        "moves_attempted": len(moves_to_make),
        "successful_moves": successful_moves,
        "failed_moves": failed_moves,
        "remaining_in_inbox": remaining_in_inbox,
        "similarity_threshold": similarity_threshold,
        "dry_run": dry_run,
        "move_details": move_details,
        "target_hotspots_available": len(target_hotspots)
    }
    
    print(f"\n📊 Triage Results:")
    print(f"   Total inbox members: {len(inbox_members)}")
    print(f"   Nodes successfully moved: {successful_moves}")
    print(f"   Failed moves: {failed_moves}")
    print(f"   Remaining in inbox: {remaining_in_inbox}")
    
    if not dry_run and successful_moves > 0:
        print(f"\n🎉 Inbox reduced from {len(inbox_members)} to {remaining_in_inbox} nodes!")
    
    return result


def main():
    """CLI interface for inbox triage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Triage inbox nodes to appropriate clusters")
    parser.add_argument("--db", default="/Users/bunny/.openclaw/workspace/cashew/data/graph.db",
                       help="Database path")
    parser.add_argument("--threshold", type=float, default=0.35,
                       help="Similarity threshold for moving nodes (default: 0.35)")
    parser.add_argument("--inbox-id", default="07621e89a08e",
                       help="Inbox hotspot ID (default: 07621e89a08e)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be moved without making changes")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    start_time = time.time()
    
    result = triage_inbox(
        db_path=args.db,
        similarity_threshold=args.threshold,
        inbox_hotspot_id=args.inbox_id,
        dry_run=args.dry_run
    )
    
    elapsed = time.time() - start_time
    
    if result["success"]:
        print(f"\n✅ Triage completed in {elapsed:.1f}s")
        if args.dry_run:
            print("   (This was a dry run - no changes made)")
    else:
        print(f"\n❌ Triage failed: {result.get('error', 'Unknown error')}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())