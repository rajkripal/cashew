#!/usr/bin/env python3
"""
Cashew Metrics Module
Lightweight performance tracking and instrumentation for the cashew knowledge graph
"""

import os
import json
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from functools import wraps
import time


def is_metrics_enabled() -> bool:
    """Check if metrics collection is enabled via environment variable"""
    return os.getenv('CASHEW_METRICS', '0') == '1'


def ensure_metrics_table(db_path: str):
    """Ensure the metrics table exists in the database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            metric_type TEXT NOT NULL,
            duration_ms REAL,
            metadata TEXT
        )
    """)
    
    conn.commit()
    conn.close()


def record_metric(db_path: str, metric_type: str, duration_ms: float, **kwargs):
    """
    Record a metric to the database
    
    Args:
        db_path: Path to SQLite database
        metric_type: Type of metric ('retrieval', 'extraction', 'search', 'embed', 'sleep')
        duration_ms: Duration in milliseconds
        **kwargs: Additional metadata to store as JSON
    """
    if not is_metrics_enabled():
        return
    
    try:
        ensure_metrics_table(db_path)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO metrics (timestamp, metric_type, duration_ms, metadata)
            VALUES (?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            metric_type,
            duration_ms,
            json.dumps(kwargs) if kwargs else None
        ))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        # Fail silently to avoid breaking the main functionality
        logging.debug(f"Error recording metric: {e}")


def timing_decorator(metric_type: str, db_path_key: str = 'db_path'):
    """
    Decorator to time function calls and record metrics
    
    Args:
        metric_type: Type of metric to record
        db_path_key: Name of function parameter containing db_path
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not is_metrics_enabled():
                return func(*args, **kwargs)
            
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start_time) * 1000
                
                # Extract db_path from function arguments
                db_path = None
                if db_path_key in kwargs:
                    db_path = kwargs[db_path_key]
                elif len(args) > 0:
                    # Assume first argument is db_path if not in kwargs
                    db_path = args[0]
                
                if db_path:
                    # Extract relevant metadata from result if available
                    metadata = {}
                    if hasattr(result, '__len__'):
                        metadata['result_count'] = len(result)
                    
                    record_metric(db_path, metric_type, duration_ms, **metadata)
                
                return result
                
            except Exception as e:
                # Still record the metric even if function failed
                duration_ms = (time.perf_counter() - start_time) * 1000
                if 'db_path' in locals():
                    record_metric(db_path, metric_type, duration_ms, error=str(e))
                raise
                
        return wrapper
    return decorator


def get_metrics_summary(db_path: str, hours: int = 24) -> Dict[str, Any]:
    """
    Get aggregated metrics summary for dashboard
    
    Args:
        db_path: Path to SQLite database
        hours: Hours of history to include
        
    Returns:
        Dictionary with summary statistics
    """
    try:
        ensure_metrics_table(db_path)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Calculate time threshold
        since = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        # Overall stats
        cursor.execute("""
            SELECT metric_type, COUNT(*) as count, AVG(duration_ms) as avg_duration,
                   MIN(duration_ms) as min_duration, MAX(duration_ms) as max_duration
            FROM metrics 
            WHERE timestamp >= ?
            GROUP BY metric_type
        """, (since,))
        
        by_type = {}
        for row in cursor.fetchall():
            metric_type, count, avg_duration, min_duration, max_duration = row
            by_type[metric_type] = {
                'count': count,
                'avg_duration': round(avg_duration, 2) if avg_duration else 0,
                'min_duration': round(min_duration, 2) if min_duration else 0,
                'max_duration': round(max_duration, 2) if max_duration else 0
            }
        
        # Total queries
        total_queries = sum(stats['count'] for stats in by_type.values())
        
        # Average retrieval time
        retrieval_stats = by_type.get('retrieval', {})
        avg_retrieval_time = retrieval_stats.get('avg_duration', 0)
        
        # Recent activity
        cursor.execute("""
            SELECT COUNT(*) FROM metrics 
            WHERE timestamp >= ? AND metric_type = 'retrieval'
        """, ((datetime.now() - timedelta(hours=1)).isoformat(),))
        
        recent_queries = cursor.fetchone()[0]
        
        # System health - get graph stats
        cursor.execute("SELECT COUNT(*) FROM thought_nodes WHERE decayed IS NULL OR decayed = 0")
        node_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM derivation_edges")
        edge_count = cursor.fetchone()[0]
        
        # Check vec_embeddings sync status
        try:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='vec_embeddings'")
            has_vec_table = cursor.fetchone() is not None
            
            if has_vec_table:
                cursor.execute("SELECT COUNT(*) FROM vec_embeddings")
                vec_count = cursor.fetchone()[0]
                vec_sync_ratio = vec_count / max(node_count, 1)
            else:
                vec_sync_ratio = 0
        except Exception as e:
            # Handle sqlite-vec loading issues gracefully
            logging.debug(f"Error checking vec_embeddings: {e}")
            vec_sync_ratio = 0
        
        conn.close()
        
        return {
            'total_queries': total_queries,
            'avg_retrieval_time': avg_retrieval_time,
            'recent_queries_1h': recent_queries,
            'by_type': by_type,
            'system_health': {
                'node_count': node_count,
                'edge_count': edge_count,
                'vec_sync_ratio': round(vec_sync_ratio, 2)
            },
            'uptime_hours': hours
        }
        
    except Exception as e:
        logging.error(f"Error getting metrics summary: {e}")
        return {}


def get_metrics_timeseries(db_path: str, metric_type: str, hours: int = 24) -> List[Dict[str, Any]]:
    """
    Get time series data for a specific metric type
    
    Args:
        db_path: Path to SQLite database
        metric_type: Type of metric to retrieve
        hours: Hours of history to include
        
    Returns:
        List of data points with timestamp and duration
    """
    try:
        ensure_metrics_table(db_path)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        since = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        cursor.execute("""
            SELECT timestamp, duration_ms, metadata
            FROM metrics 
            WHERE metric_type = ? AND timestamp >= ?
            ORDER BY timestamp
        """, (metric_type, since))
        
        results = []
        for row in cursor.fetchall():
            timestamp, duration_ms, metadata_json = row
            
            metadata = {}
            if metadata_json:
                try:
                    metadata = json.loads(metadata_json)
                except json.JSONDecodeError:
                    pass
            
            results.append({
                'timestamp': timestamp,
                'duration_ms': duration_ms,
                'metadata': metadata
            })
        
        conn.close()
        return results
        
    except Exception as e:
        logging.error(f"Error getting metrics timeseries: {e}")
        return []


def get_retrieval_stats(db_path: str, hours: int = 24) -> Dict[str, Any]:
    """
    Get detailed retrieval statistics for dashboard
    
    Args:
        db_path: Path to SQLite database
        hours: Hours of history to include
        
    Returns:
        Dictionary with retrieval-specific breakdown
    """
    try:
        ensure_metrics_table(db_path)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        since = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        cursor.execute("""
            SELECT metadata FROM metrics 
            WHERE metric_type = 'retrieval' AND timestamp >= ? AND metadata IS NOT NULL
        """, (since,))
        
        embed_times = []
        search_times = []
        bfs_times = []
        seeds_found = []
        bfs_explored = []
        results_returned = []
        overlap_ratios = []
        
        for (metadata_json,) in cursor.fetchall():
            try:
                metadata = json.loads(metadata_json)
                
                if 'embed_time_ms' in metadata:
                    embed_times.append(metadata['embed_time_ms'])
                if 'search_time_ms' in metadata:
                    search_times.append(metadata['search_time_ms'])
                if 'bfs_time_ms' in metadata:
                    bfs_times.append(metadata['bfs_time_ms'])
                if 'seeds_found' in metadata:
                    seeds_found.append(metadata['seeds_found'])
                if 'bfs_explored' in metadata:
                    bfs_explored.append(metadata['bfs_explored'])
                if 'results_returned' in metadata:
                    results_returned.append(metadata['results_returned'])
                if 'overlap_ratio' in metadata:
                    overlap_ratios.append(metadata['overlap_ratio'])
                    
            except json.JSONDecodeError:
                continue
        
        def safe_avg(lst):
            return round(sum(lst) / len(lst), 2) if lst else 0
        
        conn.close()
        
        return {
            'timing_breakdown': {
                'avg_embed_time': safe_avg(embed_times),
                'avg_search_time': safe_avg(search_times),
                'avg_bfs_time': safe_avg(bfs_times)
            },
            'exploration_stats': {
                'avg_seeds_found': safe_avg(seeds_found),
                'avg_bfs_explored': safe_avg(bfs_explored),
                'avg_results_returned': safe_avg(results_returned)
            },
            'bfs_value': {
                'avg_overlap_ratio': safe_avg(overlap_ratios),
                'bfs_bonus': safe_avg([max(0, e - s) for e, s in zip(bfs_explored, seeds_found) if e and s])
            }
        }
        
    except Exception as e:
        logging.error(f"Error getting retrieval stats: {e}")
        return {}


def get_recent_metrics(db_path: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Get recent metrics for dashboard table
    
    Args:
        db_path: Path to SQLite database
        limit: Number of recent metrics to return
        
    Returns:
        List of recent metric records
    """
    try:
        ensure_metrics_table(db_path)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT timestamp, metric_type, duration_ms, metadata
            FROM metrics 
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        
        results = []
        for row in cursor.fetchall():
            timestamp, metric_type, duration_ms, metadata_json = row
            
            metadata = {}
            if metadata_json:
                try:
                    metadata = json.loads(metadata_json)
                except json.JSONDecodeError:
                    pass
            
            results.append({
                'timestamp': timestamp,
                'metric_type': metric_type,
                'duration_ms': round(duration_ms, 2) if duration_ms else 0,
                'metadata': metadata
            })
        
        conn.close()
        return results
        
    except Exception as e:
        logging.error(f"Error getting recent metrics: {e}")
        return []


def clear_metrics(db_path: str):
    """Clear all metrics from the database"""
    try:
        ensure_metrics_table(db_path)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM metrics")
        conn.commit()
        conn.close()
        
        logging.info("All metrics cleared")
        
    except Exception as e:
        logging.error(f"Error clearing metrics: {e}")


def export_metrics(db_path: str, hours: int = 24) -> Dict[str, Any]:
    """
    Export metrics as JSON
    
    Args:
        db_path: Path to SQLite database
        hours: Hours of history to export
        
    Returns:
        Dictionary with all metrics data
    """
    try:
        return {
            'summary': get_metrics_summary(db_path, hours),
            'retrieval_stats': get_retrieval_stats(db_path, hours),
            'recent_metrics': get_recent_metrics(db_path, 100),
            'timeseries': {
                'retrieval': get_metrics_timeseries(db_path, 'retrieval', hours),
                'extraction': get_metrics_timeseries(db_path, 'extraction', hours),
                'search': get_metrics_timeseries(db_path, 'search', hours),
                'embed': get_metrics_timeseries(db_path, 'embed', hours)
            }
        }
        
    except Exception as e:
        logging.error(f"Error exporting metrics: {e}")
        return {}