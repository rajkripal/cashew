#!/usr/bin/env python3
"""
End-to-end test for cashew OpenClaw integration
This script demonstrates what cashew would actually inject as context.

This is a READ-ONLY test - it does not modify the brain graph.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from integration.openclaw import generate_session_context


def test_work_context():
    """Test work-related context generation"""
    print("🏢 Testing Work Context Generation")
    print("=" * 50)
    
    db_path = "/Users/bunny/.openclaw/workspace/cashew/data/graph.db"
    hints = ["work", "promotion"]
    
    context = generate_session_context(db_path, hints)
    
    if context:
        print("✅ Successfully generated work context:")
        print()
        print(context)
        print()
        print(f"📏 Context length: {len(context)} characters")
        print(f"📝 Lines: {context.count(chr(10)) + 1}")
    else:
        print("❌ No work context generated")
    
    return bool(context)


def test_personal_context():
    """Test personal-related context generation"""
    print("👤 Testing Personal Context Generation") 
    print("=" * 50)
    
    db_path = "/Users/bunny/.openclaw/workspace/cashew/data/graph.db"
    hints = ["personal", "fitness", "health"]
    
    context = generate_session_context(db_path, hints)
    
    if context:
        print("✅ Successfully generated personal context:")
        print()
        print(context)
        print()
        print(f"📏 Context length: {len(context)} characters")
        print(f"📝 Lines: {context.count(chr(10)) + 1}")
    else:
        print("❌ No personal context generated")
    
    return bool(context)


def test_general_context():
    """Test general context generation without specific hints"""
    print("🌍 Testing General Context Generation")
    print("=" * 50)
    
    db_path = "/Users/bunny/.openclaw/workspace/cashew/data/graph.db"
    hints = None  # No specific hints
    
    context = generate_session_context(db_path, hints)
    
    if context:
        print("✅ Successfully generated general context:")
        print()
        print(context)
        print()
        print(f"📏 Context length: {len(context)} characters")
        print(f"📝 Lines: {context.count(chr(10)) + 1}")
    else:
        print("❌ No general context generated")
    
    return bool(context)


def test_technical_context():
    """Test technical/engineering context generation"""
    print("🔧 Testing Technical Context Generation")
    print("=" * 50)
    
    db_path = "/Users/bunny/.openclaw/workspace/cashew/data/graph.db"
    hints = ["engineering", "technical", "software", "architecture"]
    
    context = generate_session_context(db_path, hints)
    
    if context:
        print("✅ Successfully generated technical context:")
        print()
        print(context)
        print()
        print(f"📏 Context length: {len(context)} characters")
        print(f"📝 Lines: {context.count(chr(10)) + 1}")
    else:
        print("❌ No technical context generated")
    
    return bool(context)


def show_database_stats():
    """Show basic statistics about the brain graph"""
    print("📊 Brain Graph Statistics")
    print("=" * 50)
    
    try:
        import sqlite3
        db_path = "/Users/bunny/.openclaw/workspace/cashew/data/graph.db"
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Count nodes
        cursor.execute("SELECT COUNT(*) FROM thought_nodes WHERE decayed IS NULL OR decayed = 0")
        total_nodes = cursor.fetchone()[0]
        
        # Count by type
        cursor.execute("""
            SELECT node_type, COUNT(*) 
            FROM thought_nodes 
            WHERE decayed IS NULL OR decayed = 0
            GROUP BY node_type 
            ORDER BY COUNT(*) DESC
        """)
        node_types = cursor.fetchall()
        
        # Count edges
        cursor.execute("SELECT COUNT(*) FROM derivation_edges")
        total_edges = cursor.fetchone()[0]
        
        # Check embeddings
        cursor.execute("""
            SELECT COUNT(*) FROM embeddings e
            JOIN thought_nodes tn ON e.node_id = tn.id
            WHERE tn.decayed IS NULL OR tn.decayed = 0
        """)
        embedded_nodes = cursor.fetchone()[0]
        
        conn.close()
        
        print(f"Total active nodes: {total_nodes}")
        print(f"Total edges: {total_edges}")
        print(f"Embedded nodes: {embedded_nodes}")
        print(f"Embedding coverage: {embedded_nodes/total_nodes*100:.1f}%" if total_nodes > 0 else "No nodes")
        print()
        print("Node types:")
        for node_type, count in node_types:
            print(f"  {node_type}: {count}")
        print()
        
        return total_nodes > 0
        
    except Exception as e:
        print(f"❌ Error reading database: {e}")
        return False


def main():
    """Run all end-to-end tests"""
    print("🧠 Cashew OpenClaw Integration - End-to-End Test")
    print("=" * 60)
    print("This test demonstrates what cashew would inject as context")
    print("READ-ONLY: Does not modify the brain graph")
    print()
    
    # Check database exists
    db_path = "/Users/bunny/.openclaw/workspace/cashew/data/graph.db"
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return 1
    
    print(f"📁 Using database: {db_path}")
    print()
    
    # Show database stats first
    if not show_database_stats():
        print("❌ Failed to read database statistics")
        return 1
    
    print()
    
    # Run context generation tests
    tests = [
        ("Work Context", test_work_context),
        ("Personal Context", test_personal_context), 
        ("Technical Context", test_technical_context),
        ("General Context", test_general_context)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print()
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Error in {test_name}: {e}")
            results.append((test_name, False))
        
        print()
        print("-" * 60)
    
    # Summary
    print()
    print("📋 Test Summary")
    print("=" * 30)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print()
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Cashew integration is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return 1


if __name__ == "__main__":
    # Set environment variable for embeddings
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    sys.exit(main())