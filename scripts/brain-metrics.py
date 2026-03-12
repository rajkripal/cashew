#!/usr/bin/env python3
"""
Brain Quality Metrics Tracking for cashew thought-graph engine.
Daily tracking of retrieval quality, graph health, and performance metrics.
"""

import sqlite3
import json
import numpy as np
import os
import sys
import subprocess
import argparse
import time
import re
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from pathlib import Path
import pickle
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

def run_retrieval_queries(db_path):
    """Run standardized retrieval quality queries and measure performance."""
    test_queries = [
        "E5 promotion simulation testing manager",
        "Partner Chiki family personal", 
        "cashew Dagger blog series projects",
        "silence pattern perfectionism streak mentality",
        "Nag Friend-S Cristian Vinny pastor people",
        "Electrons in a Box guitar Slash music",
        "Vienna anchor emotional patterns happiness",
        "untyped edges ablation graph architecture",
        "blog reactions weight loss Microsoft lunch",
        "Mac Mini email accounts cron infrastructure"
    ]
    
    results = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    context_script = os.path.join(script_dir, 'cashew_context.py')
    
    if not os.path.exists(context_script):
        print(f"Error: cashew_context.py not found at {context_script}", file=sys.stderr)
        return None
    
    for query in test_queries:
        try:
            # Run context query with timing
            start_time = time.time()
            
            env = os.environ.copy()
            env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
            
            cmd = [sys.executable, context_script, 'context', '--hints', query]
            cashew_root = os.path.dirname(os.path.dirname(context_script))
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  cwd=cashew_root, env=env, timeout=60)
            
            end_time = time.time()
            latency_ms = int((end_time - start_time) * 1000)
            
            if result.returncode == 0:
                # Parse output to count nodes and tokens
                output = result.stdout
                # Count lines that match context format: "1. [TYPE] content..."
                node_lines = [line for line in output.split('\n') if re.match(r'^\d+\. \[', line)]
                node_count = len(node_lines)
                
                # Extract token count from footer line, fallback to chars/4 estimate
                token_match = re.search(r'\(~(\d+) tokens\)', output)
                if token_match:
                    token_count = int(token_match.group(1))
                else:
                    token_count = len(output) // 4  # Fallback estimate
                
                results.append({
                    'query': query,
                    'node_count': node_count,
                    'token_count': token_count,
                    'latency_ms': latency_ms,
                    'success': True
                })
            else:
                results.append({
                    'query': query,
                    'node_count': 0,
                    'token_count': 0,
                    'latency_ms': latency_ms,
                    'success': False,
                    'error': result.stderr
                })
                
        except subprocess.TimeoutExpired:
            results.append({
                'query': query,
                'node_count': 0,
                'token_count': 0,
                'latency_ms': 60000,  # timeout
                'success': False,
                'error': 'Timeout'
            })
        except Exception as e:
            results.append({
                'query': query,
                'node_count': 0,
                'token_count': 0,
                'latency_ms': 0,
                'success': False,
                'error': str(e)
            })
    
    return results

def get_graph_stats(db_path):
    """Get comprehensive graph-wide statistics."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Basic counts
    cursor.execute("SELECT COUNT(*) FROM thought_nodes WHERE decayed = 0")
    total_nodes = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM derivation_edges")
    total_edges = cursor.fetchone()[0]
    
    # Embedding coverage
    cursor.execute("""
        SELECT COUNT(*) FROM thought_nodes tn 
        JOIN embeddings e ON tn.id = e.node_id 
        WHERE tn.decayed = 0
    """)
    nodes_with_embeddings = cursor.fetchone()[0]
    embedding_coverage = nodes_with_embeddings / total_nodes if total_nodes > 0 else 0
    
    # Hotspot count
    cursor.execute("""
        SELECT COUNT(*) FROM thought_nodes 
        WHERE node_type = 'hotspot' AND decayed = 0
    """)
    hotspot_count = cursor.fetchone()[0]
    
    # Orphan count
    cursor.execute("""
        SELECT COUNT(*) FROM thought_nodes tn 
        WHERE tn.decayed = 0 
        AND tn.id NOT IN (
            SELECT DISTINCT parent_id FROM derivation_edges 
            UNION 
            SELECT DISTINCT child_id FROM derivation_edges
        )
    """)
    orphan_count = cursor.fetchone()[0]
    
    # Near-duplicates analysis (cosine similarity > 0.82)
    cursor.execute("""
        SELECT e.node_id, e.vector 
        FROM embeddings e 
        JOIN thought_nodes tn ON e.node_id = tn.id 
        WHERE tn.decayed = 0
    """)
    
    embeddings_data = cursor.fetchall()
    near_duplicate_count = 0
    
    if len(embeddings_data) > 1:
        try:
            vectors = []
            node_ids = []
            
            for node_id, vector_blob in embeddings_data:
                embedding = load_embedding(vector_blob)
                if embedding is not None and len(embedding.shape) == 1:
                    vectors.append(embedding)
                    node_ids.append(node_id)
            
            if len(vectors) > 1:
                # Check if all vectors have same length
                lengths = [len(v) for v in vectors]
                if len(set(lengths)) == 1:
                    vectors_array = np.array(vectors)
                    similarity_matrix = cosine_similarity(vectors_array)
                    
                    # Count pairs with similarity > 0.82
                    for i in range(len(node_ids)):
                        for j in range(i+1, len(node_ids)):
                            sim = similarity_matrix[i][j]
                            if not np.isnan(sim) and sim > 0.82:
                                near_duplicate_count += 1
        except Exception as e:
            print(f"Warning: Could not compute near-duplicates: {e}", file=sys.stderr)
    
    conn.close()
    
    return {
        'total_nodes': total_nodes,
        'total_edges': total_edges,
        'embedding_coverage': round(embedding_coverage, 4),
        'hotspot_count': hotspot_count,
        'orphan_count': orphan_count,
        'near_duplicate_count': near_duplicate_count
    }

def save_metrics(retrieval_results, graph_stats, metrics_file):
    """Save metrics to JSONL file."""
    timestamp = datetime.now().isoformat()
    
    # Calculate aggregated retrieval metrics
    successful_queries = [r for r in retrieval_results if r['success']]
    
    retrieval_metrics = {
        'total_queries': len(retrieval_results),
        'successful_queries': len(successful_queries),
        'avg_nodes': round(sum(r['node_count'] for r in successful_queries) / max(1, len(successful_queries)), 2),
        'avg_tokens': round(sum(r['token_count'] for r in successful_queries) / max(1, len(successful_queries)), 2),
        'avg_latency_ms': round(sum(r['latency_ms'] for r in successful_queries) / max(1, len(successful_queries)), 2),
        'max_latency_ms': max([r['latency_ms'] for r in retrieval_results], default=0),
        'failed_queries': [r['query'] for r in retrieval_results if not r['success']]
    }
    
    metrics_entry = {
        'timestamp': timestamp,
        'retrieval': retrieval_metrics,
        'graph': graph_stats,
        'raw_queries': retrieval_results  # Include raw data for detailed analysis
    }
    
    # Append to JSONL file
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    with open(metrics_file, 'a') as f:
        f.write(json.dumps(metrics_entry) + '\n')
    
    return metrics_entry

def load_metrics_history(metrics_file):
    """Load historical metrics from JSONL file."""
    if not os.path.exists(metrics_file):
        return []
    
    history = []
    try:
        with open(metrics_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    history.append(json.loads(line))
    except Exception as e:
        print(f"Warning: Error reading metrics file: {e}", file=sys.stderr)
    
    return history

def analyze_trends(history, days=7):
    """Analyze trends over the specified number of days."""
    if len(history) < 2:
        return {
            'status': 'insufficient_data',
            'message': 'Need at least 2 data points for trend analysis'
        }
    
    # Filter to last N days
    cutoff = datetime.now() - timedelta(days=days)
    recent_history = []
    
    for entry in history:
        try:
            entry_time = datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00'))
            if entry_time.tzinfo:
                entry_time = entry_time.replace(tzinfo=None)
            if entry_time > cutoff:
                recent_history.append(entry)
        except:
            continue
    
    if len(recent_history) < 2:
        return {
            'status': 'insufficient_recent_data',
            'message': f'Need at least 2 data points in last {days} days'
        }
    
    # Calculate trends
    def calculate_trend(values):
        if len(values) < 2:
            return 0
        return (values[-1] - values[0]) / values[0] if values[0] != 0 else 0
    
    # Extract time series
    avg_nodes = [entry['retrieval']['avg_nodes'] for entry in recent_history]
    avg_tokens = [entry['retrieval']['avg_tokens'] for entry in recent_history]
    avg_latency = [entry['retrieval']['avg_latency_ms'] for entry in recent_history]
    orphan_counts = [entry['graph']['orphan_count'] for entry in recent_history]
    
    # Calculate percentage changes
    nodes_trend = calculate_trend(avg_nodes)
    tokens_trend = calculate_trend(avg_tokens) 
    latency_trend = calculate_trend(avg_latency)
    orphan_trend = calculate_trend(orphan_counts) if orphan_counts[0] > 0 else 0
    
    # Detect anomalies (sudden spikes/drops >30%)
    anomalies = []
    
    if abs(nodes_trend) > 0.3:
        direction = "increased" if nodes_trend > 0 else "decreased"
        anomalies.append(f"Average nodes per query {direction} by {abs(nodes_trend)*100:.1f}%")
    
    if abs(latency_trend) > 0.5:  # 50% threshold for latency
        direction = "increased" if latency_trend > 0 else "decreased"
        anomalies.append(f"Query latency {direction} by {abs(latency_trend)*100:.1f}%")
    
    if abs(orphan_trend) > 0.3:
        direction = "increased" if orphan_trend > 0 else "decreased"
        anomalies.append(f"Orphan nodes {direction} by {abs(orphan_trend)*100:.1f}%")
    
    # Overall assessment
    concerning_trends = []
    if nodes_trend < -0.15:  # 15% drop in retrieval quality
        concerning_trends.append("declining retrieval quality")
    if latency_trend > 0.3:   # 30% increase in latency
        concerning_trends.append("performance degradation")
    if orphan_trend > 0.2:    # 20% increase in orphans
        concerning_trends.append("graph fragmentation")
    
    status = "concerning" if concerning_trends or anomalies else "stable"
    
    return {
        'status': status,
        'period_days': days,
        'data_points': len(recent_history),
        'trends': {
            'avg_nodes_change_pct': round(nodes_trend * 100, 1),
            'avg_tokens_change_pct': round(tokens_trend * 100, 1),
            'avg_latency_change_pct': round(latency_trend * 100, 1),
            'orphan_count_change_pct': round(orphan_trend * 100, 1)
        },
        'current_values': {
            'avg_nodes': round(avg_nodes[-1], 1),
            'avg_tokens': round(avg_tokens[-1], 1),
            'avg_latency_ms': round(avg_latency[-1], 1),
            'orphan_count': orphan_counts[-1]
        },
        'anomalies': anomalies,
        'concerning_trends': concerning_trends
    }

def print_report(history):
    """Print comprehensive metrics report."""
    if not history:
        print("📊 BRAIN METRICS REPORT")
        print("=" * 50)
        print("No historical data available.")
        return
    
    latest = history[-1]
    analysis = analyze_trends(history, days=7)
    
    print("📊 BRAIN METRICS REPORT")
    print("=" * 50)
    print(f"Latest measurement: {latest['timestamp']}")
    print(f"Data points: {len(history)} total")
    print()
    
    # Current performance
    print("🔍 CURRENT RETRIEVAL PERFORMANCE:")
    retrieval = latest['retrieval']
    print(f"  • Success rate: {retrieval['successful_queries']}/{retrieval['total_queries']} queries")
    print(f"  • Average nodes per query: {retrieval['avg_nodes']}")
    print(f"  • Average tokens per query: {retrieval['avg_tokens']}")
    print(f"  • Average latency: {retrieval['avg_latency_ms']}ms")
    print(f"  • Max latency: {retrieval['max_latency_ms']}ms")
    
    if retrieval['failed_queries']:
        print(f"  • Failed queries: {', '.join(retrieval['failed_queries'])}")
    print()
    
    # Current graph health  
    print("🧠 CURRENT GRAPH HEALTH:")
    graph = latest['graph']
    print(f"  • Total nodes: {graph['total_nodes']:,}")
    print(f"  • Total edges: {graph['total_edges']:,}")
    print(f"  • Embedding coverage: {graph['embedding_coverage']*100:.1f}%")
    print(f"  • Hotspots: {graph['hotspot_count']}")
    print(f"  • Orphan nodes: {graph['orphan_count']}")
    print(f"  • Near-duplicate pairs: {graph['near_duplicate_count']}")
    print()
    
    # Trend analysis
    print("📈 TREND ANALYSIS (Last 7 days):")
    if analysis['status'] == 'insufficient_data':
        print(f"  {analysis['message']}")
    elif analysis['status'] == 'insufficient_recent_data':
        print(f"  {analysis['message']}")
    else:
        print(f"  • {analysis['data_points']} measurements over {analysis['period_days']} days")
        
        trends = analysis['trends']
        def format_trend(pct):
            if abs(pct) < 1:
                return "stable"
            direction = "↑" if pct > 0 else "↓"
            return f"{direction} {abs(pct):.1f}%"
        
        print(f"  • Average nodes: {format_trend(trends['avg_nodes_change_pct'])}")
        print(f"  • Average tokens: {format_trend(trends['avg_tokens_change_pct'])}")
        print(f"  • Query latency: {format_trend(trends['avg_latency_change_pct'])}")
        print(f"  • Orphan count: {format_trend(trends['orphan_count_change_pct'])}")
        
        if analysis['anomalies']:
            print()
            print("⚠️  ANOMALIES DETECTED:")
            for anomaly in analysis['anomalies']:
                print(f"  • {anomaly}")
        
        if analysis['concerning_trends']:
            print()
            print("🚨 CONCERNING TRENDS:")
            for trend in analysis['concerning_trends']:
                print(f"  • {trend}")
        
        if analysis['status'] == 'stable' and not analysis['anomalies']:
            print()
            print("✅ All metrics are stable and healthy")
    
    print("=" * 50)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Brain quality metrics tracking')
    parser.add_argument('--db-path', default='data/graph.db',
                      help='Path to graph database (default: data/graph.db)')
    parser.add_argument('--metrics-file', default='data/brain-metrics.jsonl',
                      help='Path to metrics JSONL file (default: data/brain-metrics.jsonl)')
    parser.add_argument('--report', action='store_true',
                      help='Show trend report without collecting new metrics')
    
    args = parser.parse_args()
    
    if args.report:
        # Just show the report
        history = load_metrics_history(args.metrics_file)
        print_report(history)
        return 0
    
    try:
        # Collect metrics
        print("🧠 Collecting brain quality metrics...", file=sys.stderr)
        
        # Run retrieval queries
        print("  Running retrieval queries...", file=sys.stderr)
        retrieval_results = run_retrieval_queries(args.db_path)
        if retrieval_results is None:
            print("Error: Failed to run retrieval queries", file=sys.stderr)
            return 1
        
        # Collect graph stats
        print("  Analyzing graph structure...", file=sys.stderr)
        graph_stats = get_graph_stats(args.db_path)
        
        # Save metrics
        print("  Saving metrics...", file=sys.stderr)
        metrics_entry = save_metrics(retrieval_results, graph_stats, args.metrics_file)
        
        print("✅ Metrics collection complete", file=sys.stderr)
        print(f"📊 Results: {metrics_entry['retrieval']['successful_queries']}/{metrics_entry['retrieval']['total_queries']} queries successful, avg {metrics_entry['retrieval']['avg_nodes']} nodes/query", file=sys.stderr)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    exit(main())