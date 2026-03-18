#!/usr/bin/env python3
"""
Graph Health Check Script for cashew thought-graph engine.
Reports various health metrics and outputs JSON with human-readable summary.
Enhanced with regression detection and retrieval quality testing.
"""

import sqlite3
import json
import numpy as np
from datetime import datetime, timedelta
import pickle
import sys
import os
import subprocess
import argparse
from collections import Counter
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
            return np.frombuffer(blob_data, dtype=np.float32).copy()  # copy to avoid buffer issues
        except:
            try:
                # Try as float64
                return np.frombuffer(blob_data, dtype=np.float64).copy()
            except:
                return None

def load_baseline(baseline_path):
    """Load baseline metrics from JSON file."""
    if not os.path.exists(baseline_path):
        return None
    
    try:
        with open(baseline_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load baseline from {baseline_path}: {e}", file=sys.stderr)
        return None

def save_baseline(stats, baseline_path):
    """Save key metrics as baseline for future comparisons."""
    baseline_data = {
        'timestamp': stats['timestamp'],
        'orphan_count': stats['orphans']['count'],
        'active_nodes': stats['nodes']['active'],
        'total_edges': stats['edges']['total'],
        'dedup_pairs': stats['near_duplicates']['count'],
        'think_diversity': stats['think_cycle_diversity']
    }
    
    try:
        with open(baseline_path, 'w') as f:
            json.dump(baseline_data, f, indent=2)
        print(f"Saved baseline to {baseline_path}", file=sys.stderr)
    except Exception as e:
        print(f"Warning: Could not save baseline to {baseline_path}: {e}", file=sys.stderr)

def detect_regressions(stats, baseline):
    """Compare current stats against baseline and detect regressions."""
    regressions = []
    
    if baseline is None:
        return False, []
    
    # Check orphan count increase >20%
    current_orphans = stats['orphans']['count']
    baseline_orphans = baseline['orphan_count']
    if baseline_orphans > 0 and (current_orphans - baseline_orphans) / baseline_orphans > 0.2:
        regressions.append(f"Orphan count increased {((current_orphans - baseline_orphans) / baseline_orphans * 100):.1f}% ({baseline_orphans} → {current_orphans})")
    
    # Check dedup pairs increase >20%
    current_dedups = stats['near_duplicates']['count']
    baseline_dedups = baseline['dedup_pairs']
    if baseline_dedups > 0 and (current_dedups - baseline_dedups) / baseline_dedups > 0.2:
        regressions.append(f"Duplicate pairs increased {((current_dedups - baseline_dedups) / baseline_dedups * 100):.1f}% ({baseline_dedups} → {current_dedups})")
    
    # Check think diversity drop (avg similarity increase >0.15)
    if (stats['think_cycle_diversity'] and baseline['think_diversity'] and 
        stats['think_cycle_diversity']['avg_similarity'] and baseline['think_diversity']['avg_similarity']):
        current_sim = stats['think_cycle_diversity']['avg_similarity']
        baseline_sim = baseline['think_diversity']['avg_similarity']
        sim_increase = current_sim - baseline_sim
        if sim_increase > 0.15:
            regressions.append(f"Think cycle diversity dropped (avg similarity increased {sim_increase:.3f}: {baseline_sim:.3f} → {current_sim:.3f})")
    
    # Check active node count drop >10%
    current_active = stats['nodes']['active']
    baseline_active = baseline['active_nodes']
    if baseline_active > 0 and (baseline_active - current_active) / baseline_active > 0.1:
        regressions.append(f"Active nodes dropped {((baseline_active - current_active) / baseline_active * 100):.1f}% ({baseline_active} → {current_active})")
    
    regression_detected = len(regressions) > 0
    return regression_detected, regressions

def run_retrieval_regression_test(db_path):
    """Run standardized retrieval quality test queries."""
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
        print(f"Warning: cashew_context.py not found at {context_script}", file=sys.stderr)
        return None
    
    for query in test_queries:
        try:
            # Run context query
            env = os.environ.copy()
            env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
            
            cmd = [sys.executable, context_script, 'context', '--hints', query]
            # Run from the parent directory of the scripts directory (cashew root)
            cashew_root = os.path.dirname(os.path.dirname(context_script))
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  cwd=cashew_root, env=env, timeout=30)
            
            if result.returncode == 0:
                # Parse output to count nodes and tokens
                output = result.stdout
                # Count lines that match the actual context format: "1. [TYPE] content..."
                import re
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
                    'success': True
                })
            else:
                results.append({
                    'query': query,
                    'node_count': 0,
                    'token_count': 0,
                    'success': False,
                    'error': result.stderr
                })
                
        except subprocess.TimeoutExpired:
            results.append({
                'query': query,
                'node_count': 0,
                'token_count': 0,
                'success': False,
                'error': 'Timeout'
            })
        except Exception as e:
            results.append({
                'query': query,
                'node_count': 0,
                'token_count': 0,
                'success': False,
                'error': str(e)
            })
    
    return results

def load_retrieval_baseline(baseline_path):
    """Load retrieval test baseline."""
    if not os.path.exists(baseline_path):
        return None
    
    try:
        with open(baseline_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load retrieval baseline: {e}", file=sys.stderr)
        return None

def save_retrieval_baseline(results, baseline_path):
    """Save retrieval test results as baseline."""
    baseline_data = {
        'timestamp': datetime.now().isoformat(),
        'results': results
    }
    
    try:
        with open(baseline_path, 'w') as f:
            json.dump(baseline_data, f, indent=2)
        print(f"Saved retrieval baseline to {baseline_path}", file=sys.stderr)
    except Exception as e:
        print(f"Warning: Could not save retrieval baseline: {e}", file=sys.stderr)

def detect_retrieval_regressions(current_results, baseline_results):
    """Detect retrieval quality regressions."""
    if not baseline_results:
        return False, []
    
    regressions = []
    baseline_map = {r['query']: r for r in baseline_results['results']}
    
    # Calculate averages
    current_avg_nodes = sum(r['node_count'] for r in current_results if r['success']) / max(1, len([r for r in current_results if r['success']]))
    baseline_avg_nodes = sum(r['node_count'] for r in baseline_map.values() if r['success']) / max(1, len([r for r in baseline_map.values() if r['success']]))
    
    # Check for avg nodes drop >30%
    if baseline_avg_nodes > 0 and (baseline_avg_nodes - current_avg_nodes) / baseline_avg_nodes > 0.3:
        regressions.append(f"Average nodes returned dropped {((baseline_avg_nodes - current_avg_nodes) / baseline_avg_nodes * 100):.1f}% ({baseline_avg_nodes:.1f} → {current_avg_nodes:.1f})")
    
    # Check for any query returning 0 nodes
    zero_result_queries = [r['query'] for r in current_results if r['node_count'] == 0]
    if zero_result_queries:
        regressions.append(f"Queries returning 0 nodes: {', '.join(zero_result_queries)}")
    
    regression_detected = len(regressions) > 0
    return regression_detected, regressions

def get_health_stats(db_path):
    """Generate comprehensive health statistics for the graph."""
    from core.stats import get_active_node_count, get_edge_count

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Basic node/edge counts
    active_nodes = get_active_node_count(cursor)

    cursor.execute("SELECT COUNT(*) FROM thought_nodes WHERE decayed = 1")
    decayed_nodes = cursor.fetchone()[0]

    total_edges = get_edge_count(cursor)
    
    # Orphan nodes (nodes with no edges)
    cursor.execute("""
        SELECT tn.id, tn.content, tn.node_type, tn.timestamp 
        FROM thought_nodes tn 
        WHERE tn.decayed = 0 
        AND tn.id NOT IN (
            SELECT DISTINCT parent_id FROM derivation_edges 
            UNION 
            SELECT DISTINCT child_id FROM derivation_edges
        )
    """)
    orphans = cursor.fetchall()
    
    # Recent decay stats (7 and 30 days)
    now = datetime.now()
    seven_days_ago = now - timedelta(days=7)
    thirty_days_ago = now - timedelta(days=30)
    
    cursor.execute("""
        SELECT COUNT(*) FROM thought_nodes 
        WHERE decayed = 1 AND last_updated > ?
    """, (seven_days_ago.isoformat(),))
    decayed_7d = cursor.fetchone()[0]
    
    cursor.execute("""
        SELECT COUNT(*) FROM thought_nodes 
        WHERE decayed = 1 AND last_updated > ?
    """, (thirty_days_ago.isoformat(),))
    decayed_30d = cursor.fetchone()[0]
    
    # Node type distribution
    cursor.execute("""
        SELECT node_type, COUNT(*) FROM thought_nodes 
        WHERE decayed = 0 GROUP BY node_type
    """)
    node_types = dict(cursor.fetchall())
    
    # Get embeddings for similarity analysis
    cursor.execute("""
        SELECT e.node_id, e.vector, tn.node_type, tn.source_file, tn.timestamp
        FROM embeddings e 
        JOIN thought_nodes tn ON e.node_id = tn.id 
        WHERE tn.decayed = 0
    """)
    embedding_data = cursor.fetchall()
    
    # Load embeddings and check for near-duplicates
    embeddings = {}
    node_info = {}
    for node_id, vector_blob, node_type, source_file, timestamp in embedding_data:
        embedding = load_embedding(vector_blob)
        if embedding is not None:
            embeddings[node_id] = embedding
            node_info[node_id] = {
                'node_type': node_type,
                'source_file': source_file,
                'timestamp': timestamp
            }
    
    # Find near-duplicate clusters (cosine similarity > 0.92)
    near_duplicates = []
    node_ids = list(embeddings.keys())
    if len(node_ids) > 1:
        try:
            # Ensure all embeddings have same dimension
            vectors = []
            valid_node_ids = []
            for nid in node_ids:
                vec = embeddings[nid]
                if vec is not None and len(vec.shape) == 1:
                    vectors.append(vec)
                    valid_node_ids.append(nid)
            
            if len(vectors) > 1:
                # Check if all vectors have same length
                lengths = [len(v) for v in vectors]
                if len(set(lengths)) == 1:  # All same length
                    vectors_array = np.array(vectors)
                    similarity_matrix = cosine_similarity(vectors_array)
                    
                    for i in range(len(valid_node_ids)):
                        for j in range(i+1, len(valid_node_ids)):
                            sim = similarity_matrix[i][j]
                            if not np.isnan(sim) and sim > 0.92:
                                near_duplicates.append({
                                    'node1': valid_node_ids[i],
                                    'node2': valid_node_ids[j],
                                    'similarity': float(sim)
                                })
        except Exception as e:
            print(f"Warning: Could not compute similarities: {e}", file=sys.stderr)
    
    # Think cycle topic diversity check
    think_cycle_nodes = []
    recent_cutoff = now - timedelta(days=7)
    for node_id, info in node_info.items():
        try:
            node_timestamp = datetime.fromisoformat(info['timestamp'].replace('Z', '+00:00'))
            # Make timezone-naive for comparison
            if node_timestamp.tzinfo:
                node_timestamp = node_timestamp.replace(tzinfo=None)
            if (info['source_file'] == 'system_generated' and 
                node_timestamp > recent_cutoff):
                think_cycle_nodes.append(node_id)
        except:
            continue  # Skip problematic timestamps
    
    think_cycle_diversity = None
    if len(think_cycle_nodes) > 1:
        try:
            valid_think_vectors = []
            for nid in think_cycle_nodes:
                if nid in embeddings and embeddings[nid] is not None:
                    valid_think_vectors.append(embeddings[nid])
            
            if len(valid_think_vectors) > 1:
                think_vectors = np.array(valid_think_vectors)
                think_similarity = cosine_similarity(think_vectors)
                # Get average similarity (excluding diagonal)
                mask = ~np.eye(think_similarity.shape[0], dtype=bool)
                avg_similarity = np.mean(think_similarity[mask])
                think_cycle_diversity = {
                    'node_count': len(valid_think_vectors),
                    'avg_similarity': float(avg_similarity),
                    'high_cluster_risk': bool(avg_similarity > 0.8)
                }
        except Exception as e:
            print(f"Warning: Could not compute think cycle diversity: {e}", file=sys.stderr)
    
    conn.close()
    
    return {
        'timestamp': now.isoformat(),
        'nodes': {
            'active': active_nodes,
            'decayed': decayed_nodes,
            'total': active_nodes + decayed_nodes
        },
        'edges': {
            'total': total_edges
        },
        'orphans': {
            'count': len(orphans),
            'samples': [
                {
                    'id': orp[0],
                    'content': orp[1][:100] + '...' if len(orp[1]) > 100 else orp[1],
                    'type': orp[2],
                    'timestamp': orp[3]
                }
                for orp in orphans[:5]  # Show first 5 samples
            ]
        },
        'decay_stats': {
            'last_7_days': decayed_7d,
            'last_30_days': decayed_30d
        },
        'node_types': node_types,
        'near_duplicates': {
            'count': len(near_duplicates),
            'clusters': near_duplicates[:10]  # Show first 10
        },
        'think_cycle_diversity': think_cycle_diversity,
        'embeddings_available': len(embeddings)
    }

def print_human_summary(stats, regressions=None, retrieval_results=None, retrieval_regressions=None):
    """Print human-readable summary of health stats."""
    print("\n" + "="*60, file=sys.stderr)
    print("CASHEW GRAPH HEALTH REPORT", file=sys.stderr)
    print("="*60, file=sys.stderr)
    print(f"Generated: {stats['timestamp']}", file=sys.stderr)
    print("", file=sys.stderr)
    
    # Print regression warnings first if detected
    if regressions:
        print("🚨 REGRESSIONS DETECTED:", file=sys.stderr)
        for regression in regressions:
            print(f"   ❌ {regression}", file=sys.stderr)
        print("", file=sys.stderr)
    elif regressions is not None:
        print("✅ NO BASELINE REGRESSIONS DETECTED", file=sys.stderr)
        print("", file=sys.stderr)
    
    print(f"📊 NODES: {stats['nodes']['active']:,} active, {stats['nodes']['decayed']:,} decayed", file=sys.stderr)
    print(f"🔗 EDGES: {stats['edges']['total']:,} total", file=sys.stderr)
    print(f"🏝️  ORPHANS: {stats['orphans']['count']:,} nodes with no connections", file=sys.stderr)
    
    if stats['orphans']['samples']:
        print("\n   Sample orphans:", file=sys.stderr)
        for sample in stats['orphans']['samples']:
            print(f"   • {sample['id']}: {sample['content']}", file=sys.stderr)
    
    print(f"\n🗑️  DECAY: {stats['decay_stats']['last_7_days']} nodes in last 7 days, {stats['decay_stats']['last_30_days']} in last 30 days", file=sys.stderr)
    
    print(f"\n📝 NODE TYPES:", file=sys.stderr)
    for node_type, count in stats['node_types'].items():
        print(f"   • {node_type}: {count:,}", file=sys.stderr)
    
    print(f"\n🎭 DUPLICATES: {stats['near_duplicates']['count']} near-duplicate pairs found", file=sys.stderr)
    if stats['near_duplicates']['clusters']:
        print("   Top clusters:", file=sys.stderr)
        for dup in stats['near_duplicates']['clusters'][:3]:
            print(f"   • {dup['similarity']:.3f} similarity: {dup['node1']} ↔ {dup['node2']}", file=sys.stderr)
    
    if stats['think_cycle_diversity']:
        div = stats['think_cycle_diversity']
        print(f"\n🤖 THINK CYCLES: {div['node_count']} recent nodes, avg similarity {div['avg_similarity']:.3f}", file=sys.stderr)
        if div['high_cluster_risk']:
            print("   ⚠️  HIGH CLUSTERING RISK - think cycles may be stuck in topic loop", file=sys.stderr)
        else:
            print("   ✅ Good topic diversity", file=sys.stderr)
    else:
        print("\n🤖 THINK CYCLES: No recent system-generated nodes found", file=sys.stderr)
    
    print(f"\n🧠 EMBEDDINGS: {stats['embeddings_available']:,} nodes have embeddings", file=sys.stderr)
    
    # Print retrieval test results if available
    if retrieval_results:
        successful_queries = [r for r in retrieval_results if r['success']]
        failed_queries = [r for r in retrieval_results if not r['success']]
        
        if retrieval_regressions:
            print(f"\n🔍 RETRIEVAL TEST: {len(successful_queries)}/{len(retrieval_results)} queries successful", file=sys.stderr)
            print("🚨 RETRIEVAL REGRESSIONS DETECTED:", file=sys.stderr)
            for regression in retrieval_regressions:
                print(f"   ❌ {regression}", file=sys.stderr)
        else:
            print(f"\n🔍 RETRIEVAL TEST: ✅ {len(successful_queries)}/{len(retrieval_results)} queries successful", file=sys.stderr)
        
        if failed_queries:
            print("   Failed queries:", file=sys.stderr)
            for failed in failed_queries[:3]:  # Show first 3 failures
                print(f"   • {failed['query']}: {failed.get('error', 'Unknown error')}", file=sys.stderr)
        
        if successful_queries:
            avg_nodes = sum(r['node_count'] for r in successful_queries) / len(successful_queries)
            print(f"   Average nodes returned: {avg_nodes:.1f}", file=sys.stderr)
    
    print("="*60, file=sys.stderr)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Graph health check with regression detection')
    parser.add_argument('--regression-test', action='store_true', 
                      help='Run retrieval quality regression tests')
    parser.add_argument('--db-path', default='data/graph.db',
                      help='Path to graph database (default: data/graph.db)')
    
    args = parser.parse_args()
    db_path = args.db_path
    
    try:
        stats = get_health_stats(db_path)
        
        # Baseline comparison
        baseline_path = os.path.join(os.path.dirname(db_path), 'health-baseline.json')
        baseline = load_baseline(baseline_path)
        regression_detected, regressions = detect_regressions(stats, baseline)
        
        # Add regression info to stats
        stats['regression_detected'] = regression_detected
        if regressions:
            stats['regressions'] = regressions
        
        # Run retrieval regression test if requested
        retrieval_results = None
        retrieval_regressions = None
        if args.regression_test:
            print("Running retrieval quality regression test...", file=sys.stderr)
            retrieval_results = run_retrieval_regression_test(db_path)
            
            if retrieval_results:
                # Load retrieval baseline
                retrieval_baseline_path = os.path.join(os.path.dirname(db_path), 'retrieval-baseline.json')
                retrieval_baseline = load_retrieval_baseline(retrieval_baseline_path)
                
                # Detect retrieval regressions
                retrieval_regression_detected, retrieval_regressions = detect_retrieval_regressions(
                    retrieval_results, retrieval_baseline
                )
                
                # Save current results as new baseline if no baseline exists
                if retrieval_baseline is None:
                    save_retrieval_baseline(retrieval_results, retrieval_baseline_path)
                
                # Add to stats
                stats['retrieval_test'] = {
                    'results': retrieval_results,
                    'regression_detected': retrieval_regression_detected
                }
                if retrieval_regressions:
                    stats['retrieval_test']['regressions'] = retrieval_regressions
                
                # Update overall regression flag
                stats['regression_detected'] = stats['regression_detected'] or retrieval_regression_detected
        
        # Save current stats as new baseline if no baseline exists
        if baseline is None:
            save_baseline(stats, baseline_path)
        
        # Output JSON to stdout
        print(json.dumps(stats, indent=2))
        
        # Print human summary to stderr so JSON output stays clean
        print_human_summary(stats, regressions, retrieval_results, retrieval_regressions)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())