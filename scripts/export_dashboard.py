#!/usr/bin/env python3
"""
Export any cashew DB to dashboard-compatible JSON.
Usage: python3 scripts/export_dashboard.py <db_path> <output_json_path>

Also generates a dashboard HTML page if --html <name> is provided.
"""

import sqlite3
import json
import sys
import os
from datetime import datetime

DASHBOARD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dashboard")

def export(db_path: str, output_path: str, title: str = "cashew"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Export nodes
    c.execute("""
        SELECT id, content, node_type, confidence, source_file, timestamp, mood_state
        FROM thought_nodes 
        WHERE decayed = 0 OR decayed IS NULL
    """)
    nodes = []
    for row in c.fetchall():
        nodes.append({
            "id": row[0],
            "content": row[1],
            "node_type": row[2],
            "confidence": row[3],
            "source_file": row[4] or "",
            "timestamp": row[5] or "",
            "mood_state": row[6] or ""
        })
    
    # Export edges — use source/target format for dashboard compatibility
    c.execute("SELECT parent_id, child_id, relation, weight, reasoning FROM derivation_edges")
    edges = []
    node_ids = {n["id"] for n in nodes}
    for row in c.fetchall():
        # Only include edges where both nodes exist
        if row[0] in node_ids and row[1] in node_ids:
            edges.append({
                "source": row[0],
                "target": row[1],
                "relation": row[2] or "derived_from",
                "weight": row[3] or 0.5,
                "reasoning": row[4] or ""
            })
    
    dashboard = {
        "metadata": {
            "title": title,
            "exported": datetime.now().isoformat(),
            "db_path": db_path
        },
        "statistics": {
            "total_nodes": len(nodes),
            "total_edges": len(edges)
        },
        "nodes": nodes,
        "edges": edges,
        "clusters": []
    }
    
    with open(output_path, 'w') as f:
        json.dump(dashboard, f, indent=2)
    
    print(f"Exported: {len(nodes)} nodes, {len(edges)} edges → {output_path}")
    return dashboard


def create_dashboard_html(data_filename: str, title: str):
    """Create a dashboard HTML page pointing to a specific data file."""
    template_path = os.path.join(DASHBOARD_DIR, "index.html")
    with open(template_path) as f:
        html = f.read()
    
    html = html.replace("./data/graph.json", f"./data/{data_filename}")
    html = html.replace("🥜 cashew", f"🥜 {title}")
    
    safe_name = data_filename.replace(".json", "")
    output_path = os.path.join(DASHBOARD_DIR, f"{safe_name}.html")
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"Dashboard page: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 scripts/export_dashboard.py <db_path> <output_json> [--html <title>]")
        sys.exit(1)
    
    db_path = sys.argv[1]
    output_path = sys.argv[2]
    
    title = "cashew"
    if "--html" in sys.argv:
        idx = sys.argv.index("--html")
        if idx + 1 < len(sys.argv):
            title = sys.argv[idx + 1]
    
    export(db_path, output_path, title)
    
    # Auto-generate dashboard HTML if output is in dashboard/data/
    if "dashboard/data/" in output_path:
        data_filename = os.path.basename(output_path)
        create_dashboard_html(data_filename, title)
