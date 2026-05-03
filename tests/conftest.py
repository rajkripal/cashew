#!/usr/bin/env python3
"""
Shared test configuration and fixtures
"""

import pytest
import sqlite3
import tempfile
import os
import json
from datetime import datetime, timezone
from pathlib import Path

@pytest.fixture
def temp_db():
    """Create a temporary database with proper schema for testing"""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    
    # Initialize with cashew schema
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    
    # Core schema matching cashew_context.py cmd_init + legacy columns for compatibility
    cursor.execute('''
        CREATE TABLE thought_nodes (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            node_type TEXT NOT NULL,
            domain TEXT,
            timestamp TEXT,
            access_count INTEGER DEFAULT 0,
            last_accessed TEXT,
            source_file TEXT,
            decayed INTEGER DEFAULT 0,
            metadata TEXT DEFAULT '{}',
            last_updated TEXT,
            mood_state TEXT,
            permanent INTEGER DEFAULT 0,
            referent_time TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE derivation_edges (
            parent_id TEXT,
            child_id TEXT,
            weight REAL,
            reasoning TEXT,
            timestamp TEXT,
            PRIMARY KEY (parent_id, child_id),
            FOREIGN KEY (parent_id) REFERENCES thought_nodes(id),
            FOREIGN KEY (child_id) REFERENCES thought_nodes(id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE embeddings (
            node_id TEXT PRIMARY KEY,
            vector BLOB NOT NULL,
            model TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (node_id) REFERENCES thought_nodes(id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE hotspots (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            status TEXT,
            domain TEXT,
            file_pointers TEXT,
            cluster_node_ids TEXT,
            tags TEXT,
            created TEXT,
            last_updated TEXT
        )
    ''')
    
    # Create indexes
    cursor.execute('CREATE INDEX idx_nodes_timestamp ON thought_nodes(timestamp)')
    cursor.execute('CREATE INDEX idx_nodes_domain ON thought_nodes(domain)')
    cursor.execute('CREATE INDEX idx_nodes_type ON thought_nodes(node_type)')
    cursor.execute('CREATE INDEX idx_edges_parent ON derivation_edges(parent_id)')
    cursor.execute('CREATE INDEX idx_edges_child ON derivation_edges(child_id)')
    
    conn.commit()
    conn.close()
    
    yield path
    
    # Cleanup
    os.unlink(path)

@pytest.fixture  
def temp_db_with_data():
    """Create a temporary database with sample data for testing"""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    
    # Initialize with schema and sample data
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    
    # Core schema with compatibility columns
    cursor.execute('''
        CREATE TABLE thought_nodes (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            node_type TEXT NOT NULL,
            domain TEXT,
            timestamp TEXT,
            access_count INTEGER DEFAULT 0,
            last_accessed TEXT,
            source_file TEXT,
            decayed INTEGER DEFAULT 0,
            metadata TEXT DEFAULT '{}',
            last_updated TEXT,
            mood_state TEXT,
            permanent INTEGER DEFAULT 0,
            referent_time TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE derivation_edges (
            parent_id TEXT,
            child_id TEXT,
            weight REAL,
            reasoning TEXT,
            timestamp TEXT,
            PRIMARY KEY (parent_id, child_id),
            FOREIGN KEY (parent_id) REFERENCES thought_nodes(id),
            FOREIGN KEY (child_id) REFERENCES thought_nodes(id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE embeddings (
            node_id TEXT PRIMARY KEY,
            vector BLOB NOT NULL,
            model TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (node_id) REFERENCES thought_nodes(id)
        )
    ''')
    
    # Add sample data
    now = datetime.now(timezone.utc).isoformat()
    sample_nodes = [
        ("node1", "Machine learning algorithms improve with data", "fact", "tech"),
        ("node2", "Python is good for data science", "observation", "tech"),
        ("node3", "Exercise boosts cognitive function", "belief", "health"),
        ("node4", "God exists and created the universe", "belief", "philosophy"),
        ("node5", "Systems thinking reveals interconnections", "insight", "meta")
    ]

    for node_id, content, node_type, domain in sample_nodes:
        cursor.execute("""
            INSERT INTO thought_nodes
            (id, content, node_type, domain, timestamp, source_file, access_count, metadata)
            VALUES (?, ?, ?, ?, ?, 'test', 0, '{}')
        """, (node_id, content, node_type, domain, now))

    # Add sample edges
    sample_edges = [
        ("node1", "node2", 0.7, "supports - Both about tech/programming"),
        ("node3", "node5", 0.5, "related_to - Both about mental processes")
    ]

    for parent_id, child_id, weight, reasoning in sample_edges:
        cursor.execute("""
            INSERT INTO derivation_edges
            (parent_id, child_id, weight, reasoning, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (parent_id, child_id, weight, reasoning, now))
    
    conn.commit()
    conn.close()
    
    yield path
    
    # Cleanup
    os.unlink(path)