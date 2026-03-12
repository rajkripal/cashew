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
    
    # Core schema from cashew_context.py cmd_init
    cursor.execute('''
        CREATE TABLE thought_nodes (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            node_type TEXT NOT NULL,
            domain TEXT,
            timestamp TEXT,
            access_count INTEGER DEFAULT 0,
            last_accessed TEXT,
            confidence REAL,
            source_file TEXT,
            decayed INTEGER DEFAULT 0
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE derivation_edges (
            parent_id TEXT,
            child_id TEXT,

            confidence REAL,
            timestamp TEXT,
            PRIMARY KEY (parent_id, child_id, relation),
            FOREIGN KEY (parent_id) REFERENCES thought_nodes(id),
            FOREIGN KEY (child_id) REFERENCES thought_nodes(id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE embeddings (
            node_id TEXT PRIMARY KEY,
            embedding BLOB,
            model TEXT,
            timestamp TEXT,
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
    
    # Core schema
    cursor.execute('''
        CREATE TABLE thought_nodes (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            node_type TEXT NOT NULL,
            domain TEXT,
            timestamp TEXT,
            access_count INTEGER DEFAULT 0,
            last_accessed TEXT,
            confidence REAL,
            source_file TEXT,
            decayed INTEGER DEFAULT 0
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE derivation_edges (
            parent_id TEXT,
            child_id TEXT,

            confidence REAL,
            timestamp TEXT,
            PRIMARY KEY (parent_id, child_id, relation),
            FOREIGN KEY (parent_id) REFERENCES thought_nodes(id),
            FOREIGN KEY (child_id) REFERENCES thought_nodes(id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE embeddings (
            node_id TEXT PRIMARY KEY,
            embedding BLOB,
            model TEXT,
            timestamp TEXT,
            FOREIGN KEY (node_id) REFERENCES thought_nodes(id)
        )
    ''')
    
    # Add sample data
    now = datetime.now(timezone.utc).isoformat()
    sample_nodes = [
        ("node1", "Machine learning algorithms improve with data", "fact", "tech", 0.8),
        ("node2", "Python is good for data science", "observation", "tech", 0.7),
        ("node3", "Exercise boosts cognitive function", "belief", "health", 0.9),
        ("node4", "God exists and created the universe", "belief", "philosophy", 0.6),
        ("node5", "Systems thinking reveals interconnections", "insight", "meta", 0.8)
    ]
    
    for node_id, content, node_type, domain, confidence in sample_nodes:
        cursor.execute("""
            INSERT INTO thought_nodes 
            (id, content, node_type, domain, timestamp, confidence, source_file, access_count)
            VALUES (?, ?, ?, ?, ?, ?, 'test', 0)
        """, (node_id, content, node_type, domain, now, confidence))
    
    # Add sample edges
    sample_edges = [
        ("node1", "node2", "supports", 0.7, "Both about tech/programming"),
        ("node3", "node5", "related_to", 0.5, "Both about mental processes")
    ]
    
    for parent_id, child_id, relation, confidence, reasoning in sample_edges:
        cursor.execute("""
            INSERT INTO derivation_edges 
            (parent_id, child_id, relation, confidence, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (parent_id, child_id, relation, confidence, now))
    
    conn.commit()
    conn.close()
    
    yield path
    
    # Cleanup
    os.unlink(path)