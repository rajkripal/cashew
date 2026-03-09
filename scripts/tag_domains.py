#!/usr/bin/env python3
"""
Tag existing nodes with domain information
All existing nodes are tagged as 'raj' domain
"""

import sqlite3
import json
import argparse
import sys
from datetime import datetime, timezone

def get_connection(db_path: str) -> sqlite3.Connection:
    """Get database connection"""
    return sqlite3.connect(db_path)

def ensure_domain_column(db_path: str):
    """Ensure the domain column exists in thought_nodes table"""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    # Check if domain column exists
    cursor.execute("PRAGMA table_info(thought_nodes)")
    columns = [row[1] for row in cursor.fetchall()]
    
    if 'domain' not in columns:
        print("Adding domain column to thought_nodes table...")
        cursor.execute("ALTER TABLE thought_nodes ADD COLUMN domain TEXT")
        conn.commit()
        print("✅ Domain column added")
    else:
        print("✅ Domain column already exists")
    
    conn.close()

def tag_existing_nodes(db_path: str, domain: str = 'raj') -> dict:
    """
    Tag all existing nodes that don't have a domain as the specified domain
    
    Returns:
        Dictionary with statistics about the tagging process
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    # Get nodes that don't have a domain set
    cursor.execute("""
        SELECT id, metadata 
        FROM thought_nodes 
        WHERE domain IS NULL OR domain = ''
        AND (decayed IS NULL OR decayed = 0)
    """)
    
    nodes_to_tag = cursor.fetchall()
    total_nodes = len(nodes_to_tag)
    
    if total_nodes == 0:
        conn.close()
        return {"total_nodes": 0, "tagged": 0}
    
    print(f"Found {total_nodes} nodes to tag as '{domain}' domain")
    
    tagged_count = 0
    
    for node_id, metadata_str in nodes_to_tag:
        try:
            # Parse existing metadata
            if metadata_str:
                try:
                    metadata = json.loads(metadata_str)
                except (json.JSONDecodeError, TypeError):
                    metadata = {}
            else:
                metadata = {}
            
            # Update metadata with domain
            metadata['domain'] = domain
            
            # Update both the domain column and metadata JSON
            cursor.execute("""
                UPDATE thought_nodes 
                SET domain = ?, metadata = ?
                WHERE id = ?
            """, (domain, json.dumps(metadata), node_id))
            
            tagged_count += 1
            
        except Exception as e:
            print(f"Error tagging node {node_id}: {e}")
            continue
    
    conn.commit()
    conn.close()
    
    return {
        "total_nodes": total_nodes,
        "tagged": tagged_count
    }

def get_domain_stats(db_path: str) -> dict:
    """Get statistics about domains in the database"""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    # Count nodes by domain
    cursor.execute("""
        SELECT 
            COALESCE(domain, 'unknown') as domain_name,
            COUNT(*) as count
        FROM thought_nodes 
        WHERE (decayed IS NULL OR decayed = 0)
        GROUP BY domain_name
        ORDER BY count DESC
    """)
    
    domain_counts = dict(cursor.fetchall())
    
    # Total nodes
    cursor.execute("""
        SELECT COUNT(*) 
        FROM thought_nodes 
        WHERE (decayed IS NULL OR decayed = 0)
    """)
    total_nodes = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        "domain_counts": domain_counts,
        "total_nodes": total_nodes
    }

def main():
    """CLI interface for domain tagging"""
    parser = argparse.ArgumentParser(description="Tag existing nodes with domain information")
    parser.add_argument("--db", required=True, help="Path to SQLite database")
    parser.add_argument("--domain", default="raj", help="Domain to tag existing nodes with")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--stats", action="store_true", help="Show current domain statistics")
    
    args = parser.parse_args()
    
    # Check if database exists
    try:
        conn = sqlite3.connect(args.db)
        conn.close()
    except sqlite3.Error as e:
        print(f"❌ Error connecting to database: {e}")
        return 1
    
    if args.stats:
        print("📊 Current domain statistics:")
        stats = get_domain_stats(args.db)
        for domain, count in stats["domain_counts"].items():
            percentage = (count / stats["total_nodes"] * 100) if stats["total_nodes"] > 0 else 0
            print(f"  {domain}: {count} nodes ({percentage:.1f}%)")
        print(f"  Total: {stats['total_nodes']} nodes")
        return 0
    
    if args.dry_run:
        print(f"🔍 DRY RUN: Would tag existing nodes as '{args.domain}' domain")
        
        # Show current stats
        stats = get_domain_stats(args.db)
        print(f"Current state:")
        for domain, count in stats["domain_counts"].items():
            print(f"  {domain}: {count} nodes")
        
        return 0
    
    print(f"🏷️  Tagging existing nodes as '{args.domain}' domain...")
    
    # Ensure domain column exists
    ensure_domain_column(args.db)
    
    # Tag existing nodes
    result = tag_existing_nodes(args.db, args.domain)
    
    print(f"✅ Tagged {result['tagged']} out of {result['total_nodes']} nodes")
    
    # Show final stats
    print("\n📊 Final domain statistics:")
    stats = get_domain_stats(args.db)
    for domain, count in stats["domain_counts"].items():
        percentage = (count / stats["total_nodes"] * 100) if stats["total_nodes"] > 0 else 0
        print(f"  {domain}: {count} nodes ({percentage:.1f}%)")
    print(f"  Total: {stats['total_nodes']} nodes")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())