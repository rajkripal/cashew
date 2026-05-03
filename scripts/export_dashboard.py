#!/usr/bin/env python3
"""
Export any cashew DB to dashboard-compatible JSON.
Usage: python3 scripts/export_dashboard.py <db_path> <output_json_path>

Also generates a dashboard HTML page if --html <name> is provided.
"""

import json
import sys
import os
from datetime import datetime
from pathlib import Path

import networkx as nx

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.config import get_user_domain, get_ai_domain
from core import db as cdb

DASHBOARD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dashboard")

def export(db_path: str, output_path: str, title: str = "cashew"):
    conn = cdb.connect(db_path)
    c = conn.cursor()

    # Export nodes
    c.execute(f"""
        SELECT id, content, node_type, source_file, timestamp, mood_state, domain, tags
        FROM {cdb.NODES_TABLE}
        WHERE decayed = 0 OR decayed IS NULL
    """)
    nodes = []
    # Map DB domains to dashboard display names
    DOMAIN_MAP = {
        "raj": "raj",
        "bunny": "bunny",
        "user": "raj",
        "ai": "bunny",
        "default": "raj"
    }
    for row in c.fetchall():
        db_domain = row[6] or "default"
        nodes.append({
            "id": row[0],
            "content": row[1],
            "node_type": row[2],
            "source_file": row[3] or "",
            "timestamp": row[4] or "",
            "mood_state": row[5] or "",
            "domain": DOMAIN_MAP.get(db_domain, "raj"),
            "tags": (row[7] or "").split(",") if row[7] else []
        })
    
    # Export edges — use source/target format for dashboard compatibility
    c.execute(f"SELECT parent_id, child_id, weight, reasoning FROM {cdb.EDGES_TABLE}")
    edges = []
    node_ids = {n["id"] for n in nodes}
    for row in c.fetchall():
        # Only include edges where both nodes exist
        if row[0] in node_ids and row[1] in node_ids:
            # Extract relation type from reasoning for dashboard compatibility
            reasoning = row[3] or ""
            relation = "derived_from"  # default
            if "summarizes" in reasoning.lower():
                relation = "summarizes"
            elif "cross_link" in reasoning.lower():
                relation = "cross_link"
            elif "contradict" in reasoning.lower():
                relation = "contradicts"
            
            edges.append({
                "source": row[0],
                "target": row[1],
                "relation": relation,
                "weight": row[2] or 0.5,
                "reasoning": reasoning
            })
    
    # Pre-compute layout positions using networkx
    G = nx.Graph()
    for n in nodes:
        G.add_node(n["id"])
    for e in edges:
        G.add_edge(e["source"], e["target"], weight=e.get("weight", 0.5))

    if len(G.nodes) > 0:
        print(f"Computing layout for {len(G.nodes)} nodes...")
        pos = nx.spring_layout(G, k=1.0/max(1, len(G.nodes)**0.3), iterations=100, seed=42)
        # Normalize positions to 0-1 range
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        x_range = x_max - x_min or 1
        y_range = y_max - y_min or 1
        for n in nodes:
            if n["id"] in pos:
                n["x"] = (pos[n["id"]][0] - x_min) / x_range
                n["y"] = (pos[n["id"]][1] - y_min) / y_range
            else:
                n["x"] = 0.5
                n["y"] = 0.5
        print("Layout computed.")

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
