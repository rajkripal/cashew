#!/usr/bin/env python3
"""
Cashew Embeddings Module
Local sentence-transformer embeddings for semantic search
"""

import sqlite3
import numpy as np
import time
from typing import List, Tuple, Optional, Dict
from datetime import datetime
import logging
import sys
import argparse

from .metrics import record_metric, is_metrics_enabled

# sqlite-vec for O(log N) vector search
_vec_available = False
try:
    import sqlite_vec
    _vec_available = True
except ImportError:
    logging.info("sqlite-vec not installed — falling back to brute-force search")

def embed_text(text: str) -> List[float]:
    """
    Embed a single text string into a vector.

    Routes through the embedding service: content-hash cache, then warm
    daemon, then in-process model as a last resort. Every caller benefits
    transparently without touching call sites.
    """
    from .embedding_service import embed as _embed
    return _embed(text)

def ensure_schema(db_path: str):
    """Ensure all tables have the correct schema (call before any DB writes)"""
    _ensure_embeddings_table(db_path)


def _load_vec(conn: sqlite3.Connection):
    """Load sqlite-vec extension into a connection"""
    if _vec_available:
        try:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        except Exception as e:
            logging.warning(f"Failed to load sqlite-vec: {e}")


def _has_vec_table(conn: sqlite3.Connection) -> bool:
    """Check if vec_embeddings virtual table exists"""
    try:
        row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='vec_embeddings'").fetchone()
        return row is not None
    except Exception:
        return False


def _ensure_embeddings_table(db_path: str):
    """Ensure the embeddings table exists with the correct schema"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='embeddings'")
    if cursor.fetchone() is None:
        cursor.execute("""
            CREATE TABLE embeddings (
                node_id TEXT PRIMARY KEY,
                vector BLOB NOT NULL,
                model TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (node_id) REFERENCES thought_nodes(id)
            )
        """)
    else:
        # Migrate: ensure 'vector' column exists (old init created 'embedding' instead)
        cursor.execute("PRAGMA table_info(embeddings)")
        columns = {row[1] for row in cursor.fetchall()}
        if 'vector' not in columns:
            cursor.execute("ALTER TABLE embeddings ADD COLUMN vector BLOB")
        if 'updated_at' not in columns:
            cursor.execute("ALTER TABLE embeddings ADD COLUMN updated_at TEXT")
    
    # Also ensure thought_nodes has metadata + last_updated columns
    cursor.execute("PRAGMA table_info(thought_nodes)")
    tn_columns = {row[1] for row in cursor.fetchall()}
    if 'metadata' not in tn_columns:
        cursor.execute("ALTER TABLE thought_nodes ADD COLUMN metadata TEXT")
    if 'last_updated' not in tn_columns:
        cursor.execute("ALTER TABLE thought_nodes ADD COLUMN last_updated TEXT")
    if 'mood_state' not in tn_columns:
        cursor.execute("ALTER TABLE thought_nodes ADD COLUMN mood_state TEXT")
    
    # Ensure derivation_edges has weight + reasoning columns
    cursor.execute("PRAGMA table_info(derivation_edges)")
    de_columns = {row[1] for row in cursor.fetchall()}
    if 'weight' not in de_columns:
        cursor.execute("ALTER TABLE derivation_edges ADD COLUMN weight REAL")
    if 'reasoning' not in de_columns:
        cursor.execute("ALTER TABLE derivation_edges ADD COLUMN reasoning TEXT")
    
    # sqlite-vec virtual table for O(log N) nearest neighbor search
    if _vec_available:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='vec_embeddings'")
        if cursor.fetchone() is None:
            try:
                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                conn.enable_load_extension(False)
                from .embedding_service import EMBEDDING_DIM as _DIM
                cursor.execute(f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings
                    USING vec0(node_id text primary key, embedding float[{_DIM}] distance_metric=cosine)
                """)
                logging.info("Created vec_embeddings virtual table")
            except Exception as e:
                logging.warning(f"Could not create vec_embeddings table: {e}")
    
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
    start_time = time.perf_counter() if is_metrics_enabled() else None
    
    _ensure_embeddings_table(db_path)
    
    conn = sqlite3.connect(db_path)
    _load_vec(conn)
    cursor = conn.cursor()
    
    # Get nodes that need embedding (not decayed, non-empty content).
    # Empty content would produce a zero-norm vector, which poisons sqlite-vec
    # cosine distance (returns NULL) and corrupts nearest-neighbor queries.
    cursor.execute("""
        SELECT tn.id, tn.content
        FROM thought_nodes tn
        LEFT JOIN embeddings e ON tn.id = e.node_id
        WHERE e.node_id IS NULL
        AND (tn.decayed IS NULL OR tn.decayed = 0)
        AND tn.content IS NOT NULL
        AND TRIM(tn.content) != ''
        ORDER BY tn.timestamp DESC
    """)

    nodes_to_embed = cursor.fetchall()
    total_nodes = len(nodes_to_embed)
    
    if total_nodes == 0:
        conn.close()
        return {"total_nodes": 0, "embedded": 0, "skipped": 0}
    
    logging.info(f"Found {total_nodes} nodes to embed")
    
    embedded_count = 0
    from .embedding_service import get_default_service
    service = get_default_service()  # cache + daemon + local fallback

    # Process in batches
    for i in range(0, total_nodes, batch_size):
        batch = nodes_to_embed[i:i + batch_size]
        batch_texts = [content for _, content in batch]
        batch_ids = [node_id for node_id, _ in batch]

        # Embed the batch
        try:
            embeddings = service.embed_np(batch_texts)
            
            # Store embeddings
            has_vec = _vec_available and _has_vec_table(conn)
            for j, (node_id, content) in enumerate(batch):
                vector_bytes = embeddings[j].astype(np.float32).tobytes()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO embeddings 
                    (node_id, vector, model, updated_at)
                    VALUES (?, ?, ?, ?)
                """, (node_id, vector_bytes, service.model, datetime.now().isoformat()))
                
                # Dual-write to vec_embeddings for O(log N) search
                if has_vec:
                    try:
                        cursor.execute("DELETE FROM vec_embeddings WHERE node_id = ?", (node_id,))
                        cursor.execute(
                            "INSERT INTO vec_embeddings(node_id, embedding) VALUES (?, ?)",
                            (node_id, vector_bytes)
                        )
                    except Exception as e:
                        logging.warning(f"vec_embeddings write failed for {node_id}: {e}")
                
                embedded_count += 1
            
            conn.commit()
            logging.info(f"Embedded batch {i//batch_size + 1}/{(total_nodes + batch_size - 1)//batch_size}")
            
        except Exception as e:
            logging.error(f"Error embedding batch starting at {i}: {e}")
            continue
    
    conn.close()
    
    # Record metrics
    if is_metrics_enabled() and start_time is not None:
        duration = (time.perf_counter() - start_time) * 1000
        
        # Check dual-write success
        has_vec = _vec_available and _has_vec_table(conn)
        dual_write_success = has_vec and (embedded_count > 0)
        
        record_metric(db_path, 'embed', duration,
                      nodes_embedded=embedded_count,
                      total_nodes=total_nodes,
                      dual_write_success=dual_write_success)
    
    return {
        "total_nodes": total_nodes,
        "embedded": embedded_count,
        "skipped": total_nodes - embedded_count
    }

def search(db_path: str, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
    """
    Semantic search using cosine similarity.
    Uses sqlite-vec for O(log N) search when available, falls back to brute force.
    
    Args:
        db_path: Path to SQLite database
        query: Search query text
        top_k: Number of results to return
        
    Returns:
        List of (node_id, similarity_score) tuples, sorted by similarity descending
    """
    if not query or not query.strip():
        return []
    
    start_time = time.perf_counter() if is_metrics_enabled() else None
    used_vec = False
    
    _ensure_embeddings_table(db_path)
    
    # Embed the query
    query_embedding = np.array(embed_text(query), dtype=np.float32)
    
    conn = sqlite3.connect(db_path)
    
    # Try sqlite-vec first (O(log N))
    if _vec_available and _has_vec_table(conn):
        try:
            _load_vec(conn)
            cursor = conn.cursor()
            query_bytes = query_embedding.tobytes()
            
            # sqlite-vec returns cosine distance (0 = identical, 2 = opposite)
            # Convert to similarity: sim = 1 - distance
            rows = cursor.execute("""
                SELECT node_id, distance FROM vec_embeddings 
                WHERE embedding MATCH ?
                ORDER BY distance LIMIT ?
            """, (query_bytes, top_k)).fetchall()
            
            conn.close()
            results = [(node_id, 1.0 - distance) for node_id, distance in rows]
            used_vec = True
            
            # Record metrics
            if is_metrics_enabled() and start_time is not None:
                duration = (time.perf_counter() - start_time) * 1000
                record_metric(db_path, 'search', duration,
                              used_sqlite_vec=True,
                              result_count=len(results))
            
            return results
        except Exception as e:
            logging.warning(f"sqlite-vec search failed, falling back to brute force: {e}")
    
    # Brute-force fallback (O(N))
    cursor = conn.cursor()
    cursor.execute("""
        SELECT e.node_id, e.vector 
        FROM embeddings e
        JOIN thought_nodes tn ON e.node_id = tn.id
        WHERE tn.decayed IS NULL OR tn.decayed = 0
    """)
    
    results = []
    query_norm = np.linalg.norm(query_embedding)
    
    for node_id, vector_bytes in cursor.fetchall():
        try:
            stored_embedding = np.frombuffer(vector_bytes, dtype=np.float32)
            stored_norm = np.linalg.norm(stored_embedding)
            
            if query_norm > 0 and stored_norm > 0:
                similarity = float(np.dot(query_embedding, stored_embedding) / (query_norm * stored_norm))
                results.append((node_id, similarity))
        except Exception as e:
            logging.warning(f"Error processing embedding for node {node_id}: {e}")
            continue
    
    conn.close()
    results.sort(key=lambda x: x[1], reverse=True)
    final_results = results[:top_k]
    
    # Record metrics
    if is_metrics_enabled() and start_time is not None:
        duration = (time.perf_counter() - start_time) * 1000
        record_metric(db_path, 'search', duration,
                      used_sqlite_vec=False,
                      result_count=len(final_results))
    
    return final_results

# Quality gate parameters
# MiniLM-L6 cosine similarity distribution: mean ~0.13, P99 ~0.49, true dupes peak ~0.85-0.90
NOVELTY_THRESHOLD = 0.82  # reject if nearest neighbor similarity > this


def load_all_embeddings(db_path: str) -> Dict[str, np.ndarray]:
    """Load all non-decayed node embeddings from DB. Call once, pass to check_novelty."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT e.node_id, e.vector 
        FROM embeddings e
        JOIN thought_nodes tn ON e.node_id = tn.id
        WHERE tn.decayed IS NULL OR tn.decayed = 0
    """)
    embeddings = {}
    for node_id, vector_bytes in cursor.fetchall():
        try:
            embeddings[node_id] = np.frombuffer(vector_bytes, dtype=np.float32)
        except Exception:
            continue
    conn.close()
    return embeddings


def check_novelty(db_path: str, content: str, threshold: float = NOVELTY_THRESHOLD,
                  preloaded_embeddings: Optional[Dict[str, np.ndarray]] = None) -> Tuple[bool, float, Optional[str]]:
    """
    Check if content is sufficiently novel compared to existing graph.
    Uses sqlite-vec when available for O(log N) nearest-neighbor lookup.
    
    Returns:
        (is_novel, max_similarity, nearest_node_id)
    """
    try:
        candidate_embedding = np.array(embed_text(content), dtype=np.float32)
    except Exception as e:
        logging.warning(f"Failed to embed candidate for novelty check: {e}")
        return True, 0.0, None  # fail open
    
    # Fast path: sqlite-vec nearest neighbor
    if preloaded_embeddings is None:
        conn = sqlite3.connect(db_path)
        if _vec_available and _has_vec_table(conn):
            try:
                _load_vec(conn)
                row = conn.execute(
                    "SELECT node_id, distance FROM vec_embeddings WHERE embedding MATCH ? ORDER BY distance LIMIT 1",
                    (candidate_embedding.tobytes(),)
                ).fetchone()
                conn.close()
                if row:
                    max_sim = 1.0 - row[1]  # cosine distance → similarity
                    return max_sim < threshold, max_sim, row[0]
                return True, 0.0, None
            except Exception:
                pass
        conn.close()
    
    # Brute-force fallback (or when preloaded_embeddings provided)
    if preloaded_embeddings is None:
        preloaded_embeddings = load_all_embeddings(db_path)
    
    max_sim = 0.0
    nearest_id = None
    cn = np.linalg.norm(candidate_embedding)
    if cn == 0:
        return True, 0.0, None
    
    for node_id, stored in preloaded_embeddings.items():
        sn = np.linalg.norm(stored)
        if sn > 0:
            sim = float(np.dot(candidate_embedding, stored) / (cn * sn))
            if sim > max_sim:
                max_sim = sim
                nearest_id = node_id
    
    return max_sim < threshold, max_sim, nearest_id


def backfill_vec_index(db_path: str) -> dict:
    """
    Backfill vec_embeddings from existing embeddings table.
    Call once after adding sqlite-vec to an existing database.
    """
    if not _vec_available:
        return {"error": "sqlite-vec not installed"}
    
    _ensure_embeddings_table(db_path)
    
    conn = sqlite3.connect(db_path)
    _load_vec(conn)
    cursor = conn.cursor()
    
    if not _has_vec_table(conn):
        conn.close()
        return {"error": "vec_embeddings table not found"}
    
    # Get all active embeddings
    cursor.execute("""
        SELECT e.node_id, e.vector FROM embeddings e
        JOIN thought_nodes tn ON e.node_id = tn.id
        WHERE tn.decayed IS NULL OR tn.decayed = 0
    """)
    rows = cursor.fetchall()
    
    inserted = 0
    for node_id, vector_bytes in rows:
        try:
            cursor.execute("DELETE FROM vec_embeddings WHERE node_id = ?", (node_id,))
            cursor.execute(
                "INSERT INTO vec_embeddings(node_id, embedding) VALUES (?, ?)",
                (node_id, vector_bytes)
            )
            inserted += 1
        except Exception as e:
            logging.warning(f"Failed to backfill {node_id}: {e}")
    
    conn.commit()
    conn.close()
    return {"backfilled": inserted, "total": len(rows)}


def get_embedding_stats(db_path: str) -> dict:
    """
    Get statistics about embeddings in the database
    
    Args:
        db_path: Path to SQLite database
        
    Returns:
        Dictionary with embedding statistics
    """
    _ensure_embeddings_table(db_path)

    from .stats import get_embedding_coverage, get_connection
    conn = get_connection(db_path)
    cursor = conn.cursor()

    # Count embedded and total nodes
    embedded_nodes, total_nodes = get_embedding_coverage(cursor)
    
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
    parser.add_argument("--db", default="./data/graph.db", help="Database path")
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