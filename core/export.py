#!/usr/bin/env python3
"""
Cashew Graph Export
Export graph data to JSON for visualization dashboard
"""

import sqlite3
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
import argparse
import sys
from datetime import datetime

# Database path is now configurable via environment variable or CLI
from .config import get_db_path
DEFAULT_EXPORT_PATH = "./data/graph_export.json"

@dataclass
class ExportedNode:
    id: str
    content: str
    type: str
    mood_state: str
    metadata: dict
    source_file: str
    timestamp: str
    decayed: bool

@dataclass
class ExportedEdge:
    source: str  # parent_id
    target: str  # child_id
    weight: float
    reasoning: str

class GraphExporter:
    """Export graph data for visualization"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = get_db_path()
        self.db_path = db_path
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def export_nodes(self) -> List[Dict]:
        """Export all nodes"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, content, node_type, mood_state,
                   metadata, source_file, timestamp,
                   COALESCE(decayed, 0) as decayed
            FROM thought_nodes
            ORDER BY timestamp
        """)

        nodes = []
        for row in cursor.fetchall():
            node = {
                "id": row[0],
                "content": row[1],
                "type": row[2],
                "mood_state": row[3],
                "metadata": json.loads(row[4]) if row[4] else {},
                "source_file": row[5],
                "timestamp": row[6],
                "decayed": bool(row[7])
            }
            nodes.append(node)
        
        conn.close()
        return nodes
    
    def export_edges(self) -> List[Dict]:
        """Export all edges"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT parent_id, child_id, weight, reasoning
            FROM derivation_edges
            ORDER BY weight DESC
        """)
        
        edges = []
        for row in cursor.fetchall():
            edge = {
                "source": row[0],  # parent_id becomes source
                "target": row[1],  # child_id becomes target
                "weight": row[2],
                "reasoning": row[3] or ""
            }
            edges.append(edge)
        
        conn.close()
        return edges
    
    def calculate_graph_stats(self, nodes: List[Dict], edges: List[Dict]) -> Dict:
        """Calculate graph statistics"""
        # Node type counts
        node_types = {}
        for node in nodes:
            node_type = node["type"]
            node_types[node_type] = node_types.get(node_type, 0) + 1

        # Edge reasoning patterns (simplified analysis)
        reasoning_keywords = {}
        for edge in edges:
            reasoning = edge["reasoning"].lower()
            # Extract key patterns from reasoning
            if "summariz" in reasoning:
                reasoning_keywords["summarizes"] = reasoning_keywords.get("summarizes", 0) + 1
            elif "cross" in reasoning:
                reasoning_keywords["cross_links"] = reasoning_keywords.get("cross_links", 0) + 1
            elif "contradict" in reasoning or "conflict" in reasoning:
                reasoning_keywords["contradictions"] = reasoning_keywords.get("contradictions", 0) + 1
            else:
                reasoning_keywords["other"] = reasoning_keywords.get("other", 0) + 1
        
        # Source file distribution
        source_files = {}
        for node in nodes:
            source = node["source_file"]
            source_files[source] = source_files.get(source, 0) + 1
        
        # Connectivity stats
        node_degrees = {}  # node_id -> {in_degree, out_degree}
        for node in nodes:
            node_degrees[node["id"]] = {"in": 0, "out": 0}
        
        for edge in edges:
            source, target = edge["source"], edge["target"]
            if source in node_degrees:
                node_degrees[source]["out"] += 1
            if target in node_degrees:
                node_degrees[target]["in"] += 1
        
        # Calculate degree statistics
        in_degrees = [d["in"] for d in node_degrees.values()]
        out_degrees = [d["out"] for d in node_degrees.values()]
        
        # Find hub nodes (high degree)
        hub_nodes = []
        for node_id, degrees in node_degrees.items():
            total_degree = degrees["in"] + degrees["out"]
            if total_degree >= 3:  # Threshold for hub
                node = next((n for n in nodes if n["id"] == node_id), None)
                if node:
                    hub_nodes.append({
                        "id": node_id,
                        "content": node["content"][:60] + "...",
                        "type": node["type"],
                        "in_degree": degrees["in"],
                        "out_degree": degrees["out"],
                        "total_degree": total_degree
                    })
        
        hub_nodes.sort(key=lambda x: x["total_degree"], reverse=True)
        
        return {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "node_types": node_types,
            "reasoning_patterns": reasoning_keywords,
            "source_files": source_files,
            "avg_in_degree": sum(in_degrees) / len(in_degrees) if in_degrees else 0,
            "avg_out_degree": sum(out_degrees) / len(out_degrees) if out_degrees else 0,
            "max_in_degree": max(in_degrees) if in_degrees else 0,
            "max_out_degree": max(out_degrees) if out_degrees else 0,
            "hub_nodes": hub_nodes[:10],  # Top 10 hubs
            "export_timestamp": datetime.now().isoformat()
        }
    
    def generate_clusters(self, nodes: List[Dict], edges: List[Dict]) -> List[Dict]:
        """Generate cluster information for visualization"""
        # Simple clustering based on source file and node type
        clusters = {}
        
        for node in nodes:
            # Create cluster key based on source file and type
            source = node["source_file"]
            node_type = node["type"]
            cluster_key = f"{source}_{node_type}"
            
            if cluster_key not in clusters:
                clusters[cluster_key] = {
                    "id": cluster_key,
                    "label": f"{source} ({node_type})",
                    "source_file": source,
                    "node_type": node_type,
                    "nodes": [],
                    "color": self._get_cluster_color(source, node_type)
                }
            
            clusters[cluster_key]["nodes"].append(node["id"])
        
        # Convert to list and add size info
        cluster_list = []
        for cluster in clusters.values():
            cluster["size"] = len(cluster["nodes"])
            cluster_list.append(cluster)
        
        # Sort by size (largest first)
        cluster_list.sort(key=lambda c: c["size"], reverse=True)
        
        return cluster_list
    
    def _get_cluster_color(self, source_file: str, node_type: str) -> str:
        """Get color for cluster visualization"""
        # Color by node type primarily
        type_colors = {
            "seed": "#FF6B6B",          # Red
            "core_memory": "#FFD93D",   # Gold
            "derived": "#74C0FC",       # Blue
            "belief": "#51CF66",        # Green
            "question": "#DA77F2",      # Purple
            "dream": "#22D3EE",         # Cyan
            "decayed": "#868E96"        # Gray
        }
        
        return type_colors.get(node_type, "#ADB5BD")  # Default gray
    
    def export_full_graph(self, output_path: str = DEFAULT_EXPORT_PATH) -> Dict:
        """Export complete graph data"""
        print("📊 Exporting graph data...")
        
        # Export nodes and edges
        nodes = self.export_nodes()
        edges = self.export_edges()
        
        print(f"Exported {len(nodes)} nodes and {len(edges)} edges")
        
        # Calculate statistics
        stats = self.calculate_graph_stats(nodes, edges)
        
        # Generate clusters
        clusters = self.generate_clusters(nodes, edges)
        
        # Create full export data
        export_data = {
            "metadata": {
                "version": "1.0",
                "export_timestamp": datetime.now().isoformat(),
                "source_database": self.db_path
            },
            "statistics": stats,
            "nodes": nodes,
            "edges": edges,
            "clusters": clusters
        }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"✅ Graph exported to: {output_path}")
        return export_data
    
    def export_summary_report(self) -> str:
        """Generate a text summary of the graph"""
        nodes = self.export_nodes()
        edges = self.export_edges()
        stats = self.calculate_graph_stats(nodes, edges)
        
        report = []
        report.append("🌰 CASHEW GRAPH SUMMARY")
        report.append("=" * 50)
        
        # Basic stats
        report.append(f"\n📊 Overview:")
        report.append(f"  Total nodes: {stats['total_nodes']}")
        report.append(f"  Total edges: {stats['total_edges']}")
        report.append(f"  Average connectivity: {stats['avg_in_degree']:.1f} in, {stats['avg_out_degree']:.1f} out")
        
        # Node types
        report.append(f"\n🏷️  Node Types:")
        for node_type, count in sorted(stats['node_types'].items()):
            report.append(f"  {node_type}: {count} nodes")
        
        # Reasoning patterns
        report.append(f"\n🔗 Reasoning Patterns:")
        for pattern, count in sorted(stats['reasoning_patterns'].items(), key=lambda x: x[1], reverse=True):
            report.append(f"  {pattern}: {count} edges")
        
        # Hub nodes
        if stats['hub_nodes']:
            report.append(f"\n⭐ Hub Nodes (most connected):")
            for hub in stats['hub_nodes'][:5]:
                report.append(f"  [{hub['total_degree']}] {hub['content']} ({hub['type']})")
        
        # Source files
        report.append(f"\n📁 Source Files:")
        for source, count in sorted(stats['source_files'].items(), key=lambda x: x[1], reverse=True):
            report.append(f"  {source}: {count} nodes")
        
        return "\n".join(report)


def main():
    """CLI interface for graph export"""
    parser = argparse.ArgumentParser(description="Cashew Graph Exporter")
    parser.add_argument("command", choices=["export", "summary"], help="Command to run")
    parser.add_argument("--output", help="Output file path", default=DEFAULT_EXPORT_PATH)
    
    args = parser.parse_args()
    
    exporter = GraphExporter()
    
    if args.command == "export":
        export_data = exporter.export_full_graph(args.output)
        
        print(f"\n📈 Export Summary:")
        print(f"  Nodes: {len(export_data['nodes'])}")
        print(f"  Edges: {len(export_data['edges'])}")
        print(f"  Clusters: {len(export_data['clusters'])}")
        print(f"  Hub nodes: {len(export_data['statistics']['hub_nodes'])}")
    
    elif args.command == "summary":
        summary = exporter.export_summary_report()
        print(summary)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())