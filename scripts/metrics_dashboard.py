#!/usr/bin/env python3
"""
Cashew Metrics Dashboard
Live web dashboard for monitoring cashew performance
"""

import os
import sys
import json
import argparse
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import socketserver
from pathlib import Path

# Add parent directory to path so we can import cashew modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.metrics import (
    get_metrics_summary, get_metrics_timeseries, 
    get_retrieval_stats, get_recent_metrics, is_metrics_enabled
)
from core.config import get_db_path


class MetricsHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, db_path=None, **kwargs):
        self.db_path = db_path or get_db_path()
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/':
            self.serve_dashboard()
        elif parsed_path.path == '/api/summary':
            self.serve_summary()
        elif parsed_path.path == '/api/timeseries':
            self.serve_timeseries(parsed_path.query)
        elif parsed_path.path == '/api/recent':
            self.serve_recent()
        else:
            self.send_error(404)
    
    def serve_dashboard(self):
        """Serve the main dashboard HTML"""
        html = self.generate_dashboard_html()
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))
    
    def serve_summary(self):
        """API endpoint for summary stats"""
        try:
            summary = get_metrics_summary(self.db_path, hours=24)
            self.send_json(summary)
        except Exception as e:
            self.send_json({'error': str(e)}, status=500)
    
    def serve_timeseries(self, query_string):
        """API endpoint for timeseries data"""
        try:
            params = parse_qs(query_string)
            metric_type = params.get('type', ['retrieval'])[0]
            hours = int(params.get('hours', ['24'])[0])
            
            data = get_metrics_timeseries(self.db_path, metric_type, hours)
            self.send_json(data)
        except Exception as e:
            self.send_json({'error': str(e)}, status=500)
    
    def serve_recent(self):
        """API endpoint for recent metrics"""
        try:
            data = get_recent_metrics(self.db_path, limit=20)
            self.send_json(data)
        except Exception as e:
            self.send_json({'error': str(e)}, status=500)
    
    def send_json(self, data, status=200):
        """Send JSON response"""
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))
    
    def generate_dashboard_html(self):
        """Generate the dashboard HTML with embedded CSS/JS"""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cashew Metrics Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        {self.get_css()}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🥜 Cashew Metrics Dashboard</h1>
            <div class="status" id="status">
                <span class="status-indicator" id="metrics-status">●</span>
                <span id="status-text">Loading...</span>
            </div>
        </header>
        
        <div class="grid">
            <!-- Overview Cards -->
            <div class="card overview-cards" id="overview">
                <h2>Overview (24h)</h2>
                <div class="cards-grid">
                    <div class="metric-card">
                        <div class="metric-value" id="total-queries">-</div>
                        <div class="metric-label">Total Queries</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="avg-retrieval">-</div>
                        <div class="metric-label">Avg Retrieval (ms)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="recent-queries">-</div>
                        <div class="metric-label">Recent (1h)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="bfs-bonus">-</div>
                        <div class="metric-label">BFS Bonus</div>
                    </div>
                </div>
            </div>
            
            <!-- Retrieval Breakdown -->
            <div class="card">
                <h3>Retrieval Timing Breakdown</h3>
                <canvas id="breakdown-chart" ></canvas>
            </div>
            
            <!-- BFS Value Over Time -->
            <div class="card">
                <h3>BFS Value (Overlap Ratio)</h3>
                <canvas id="bfs-chart" ></canvas>
            </div>
            
            <!-- Latency Over Time -->
            <div class="card">
                <h3>Retrieval Latency Over Time</h3>
                <canvas id="latency-chart" ></canvas>
            </div>
            
            <!-- System Health -->
            <div class="card">
                <h3>System Health</h3>
                <div class="health-grid" id="health">
                    <div>Nodes: <span id="node-count">-</span></div>
                    <div>Edges: <span id="edge-count">-</span></div>
                    <div>Vec Sync: <span id="vec-sync">-</span></div>
                    <div>Uptime: <span id="uptime">-</span></div>
                </div>
            </div>
            
            <!-- Recent Activity -->
            <div class="card full-width">
                <h3>Recent Activity</h3>
                <table id="recent-table">
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>Type</th>
                            <th>Duration (ms)</th>
                            <th>Details</th>
                        </tr>
                    </thead>
                    <tbody id="recent-tbody">
                        <tr><td colspan="4">Loading...</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>
    
    <script>
        {self.get_javascript()}
    </script>
</body>
</html>"""

    def get_css(self):
        """Return the CSS styles"""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            background: #0a0a0a;
            color: #e0e0e0;
            line-height: 1.4;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 1px solid #333;
        }
        
        h1 {
            color: #c9a227;
            font-size: 2em;
            font-weight: 600;
        }
        
        .status {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.9em;
        }
        
        .status-indicator {
            font-size: 1.2em;
            color: #4CAF50;
        }
        
        .status-indicator.error {
            color: #f44336;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }
        
        .card {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 20px;
        }
        
        .card.full-width {
            grid-column: 1 / -1;
        }
        
        .card h2, .card h3 {
            color: #c9a227;
            margin-bottom: 15px;
            font-size: 1.1em;
        }
        
        .overview-cards .cards-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 15px;
        }
        
        .metric-card {
            background: #2a2a2a;
            padding: 15px;
            border-radius: 6px;
            text-align: center;
        }
        
        .metric-value {
            font-size: 1.8em;
            font-weight: bold;
            color: #c9a227;
            margin-bottom: 5px;
        }
        
        .metric-label {
            font-size: 0.8em;
            color: #aaa;
        }
        
        .health-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 10px;
            font-size: 0.9em;
        }
        
        .health-grid > div {
            padding: 10px;
            background: #2a2a2a;
            border-radius: 4px;
        }
        
        .health-grid span {
            color: #c9a227;
            font-weight: bold;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85em;
        }
        
        th, td {
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid #333;
        }
        
        th {
            background: #2a2a2a;
            color: #c9a227;
            font-weight: 600;
        }
        
        tr:hover {
            background: #1a1a1a;
        }
        
        canvas {
            max-width: 100%;
            height: 300px !important; min-height: 200px;
        }
        
        .error {
            color: #f44336;
            padding: 10px;
            background: rgba(244, 67, 54, 0.1);
            border-radius: 4px;
            margin: 10px 0;
        }
        """
    
    def get_javascript(self):
        """Return the JavaScript code"""
        return """
        let charts = {};
        
        // Chart.js default configuration
        Chart.defaults.color = '#e0e0e0';
        Chart.defaults.borderColor = '#333';
        Chart.defaults.backgroundColor = 'rgba(201, 162, 39, 0.1)';
        Chart.defaults.font.family = 'Monaco, Menlo, Ubuntu Mono, monospace';
        Chart.defaults.font.size = 11;
        Chart.defaults.responsive = true;
        Chart.defaults.maintainAspectRatio = false;
        
        async function fetchData(endpoint) {
            try {
                const response = await fetch(endpoint);
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                return await response.json();
            } catch (error) {
                console.error('Fetch error:', error);
                updateStatus('error', `Connection failed: ${error.message}`);
                throw error;
            }
        }
        
        function updateStatus(type, message) {
            const indicator = document.getElementById('metrics-status');
            const text = document.getElementById('status-text');
            
            indicator.className = `status-indicator ${type}`;
            text.textContent = message;
        }
        
        async function updateOverview() {
            try {
                const data = await fetchData('/api/summary');
                
                document.getElementById('total-queries').textContent = data.total_queries || 0;
                document.getElementById('avg-retrieval').textContent = 
                    data.avg_retrieval_time ? Math.round(data.avg_retrieval_time) : '-';
                document.getElementById('recent-queries').textContent = data.recent_queries_1h || 0;
                
                // BFS bonus calculation
                const retrievalStats = await fetchData('/api/timeseries?type=retrieval&hours=24');
                const bfsBonus = calculateBfsBonus(retrievalStats);
                document.getElementById('bfs-bonus').textContent = bfsBonus;
                
                // System health
                const health = data.system_health || {};
                document.getElementById('node-count').textContent = health.node_count || '-';
                document.getElementById('edge-count').textContent = health.edge_count || '-';
                document.getElementById('vec-sync').textContent = 
                    health.vec_sync_ratio ? `${Math.round(health.vec_sync_ratio * 100)}%` : '-';
                document.getElementById('uptime').textContent = `${data.uptime_hours || 24}h`;
                
            } catch (error) {
                console.error('Failed to update overview:', error);
            }
        }
        
        function calculateBfsBonus(retrievalData) {
            if (!retrievalData.length) return '-';
            
            let totalBonus = 0;
            let count = 0;
            
            for (const point of retrievalData) {
                if (point.metadata && point.metadata.bfs_explored && point.metadata.seeds_found) {
                    const bonus = Math.max(0, point.metadata.bfs_explored - point.metadata.seeds_found);
                    totalBonus += bonus;
                    count++;
                }
            }
            
            return count > 0 ? Math.round(totalBonus / count) : '-';
        }
        
        async function updateBreakdownChart() {
            try {
                const data = await fetchData('/api/summary');
                const byType = data.by_type || {};
                
                const ctx = document.getElementById('breakdown-chart');
                
                if (charts.breakdown) {
                    charts.breakdown.destroy();
                }
                
                charts.breakdown = new Chart(ctx, {
                    type: 'doughnut',
                    data: {
                        labels: Object.keys(byType),
                        datasets: [{
                            data: Object.values(byType).map(t => t.avg_duration),
                            backgroundColor: [
                                'rgba(201, 162, 39, 0.8)',
                                'rgba(76, 175, 80, 0.8)',
                                'rgba(33, 150, 243, 0.8)',
                                'rgba(156, 39, 176, 0.8)',
                                'rgba(255, 152, 0, 0.8)'
                            ]
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: {
                                position: 'bottom'
                            }
                        }
                    }
                });
            } catch (error) {
                console.error('Failed to update breakdown chart:', error);
            }
        }
        
        async function updateLatencyChart() {
            try {
                const data = await fetchData('/api/timeseries?type=retrieval&hours=24');
                
                const ctx = document.getElementById('latency-chart');
                
                if (charts.latency) {
                    charts.latency.destroy();
                }
                
                charts.latency = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: data.map(d => new Date(d.timestamp).toLocaleTimeString()),
                        datasets: [{
                            label: 'Retrieval Time (ms)',
                            data: data.map(d => d.duration_ms),
                            borderColor: 'rgba(201, 162, 39, 1)',
                            backgroundColor: 'rgba(201, 162, 39, 0.1)',
                            tension: 0.1,
                            fill: true
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                beginAtZero: true,
                                title: {
                                    display: true,
                                    text: 'Duration (ms)'
                                }
                            }
                        }
                    }
                });
            } catch (error) {
                console.error('Failed to update latency chart:', error);
            }
        }
        
        async function updateBfsChart() {
            try {
                const data = await fetchData('/api/timeseries?type=retrieval&hours=24');
                
                const bfsData = data
                    .filter(d => d.metadata && d.metadata.overlap_ratio !== undefined)
                    .map(d => ({
                        x: new Date(d.timestamp).toLocaleTimeString(),
                        y: d.metadata.overlap_ratio * 100
                    }));
                
                const ctx = document.getElementById('bfs-chart');
                
                if (charts.bfs) {
                    charts.bfs.destroy();
                }
                
                charts.bfs = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: bfsData.map(d => d.x),
                        datasets: [{
                            label: 'Seeds vs Total (%)',
                            data: bfsData.map(d => d.y),
                            borderColor: 'rgba(76, 175, 80, 1)',
                            backgroundColor: 'rgba(76, 175, 80, 0.1)',
                            tension: 0.1,
                            fill: true
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                beginAtZero: true,
                                max: 100,
                                title: {
                                    display: true,
                                    text: 'Overlap Ratio (%)'
                                }
                            }
                        }
                    }
                });
            } catch (error) {
                console.error('Failed to update BFS chart:', error);
            }
        }
        
        async function updateRecentTable() {
            try {
                const data = await fetchData('/api/recent');
                const tbody = document.getElementById('recent-tbody');
                
                if (data.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="4">No recent activity</td></tr>';
                    return;
                }
                
                tbody.innerHTML = data.map(row => {
                    const time = new Date(row.timestamp).toLocaleTimeString();
                    const details = formatMetadata(row.metadata);
                    
                    return `<tr>
                        <td>${time}</td>
                        <td>${row.metric_type}</td>
                        <td>${row.duration_ms}</td>
                        <td>${details}</td>
                    </tr>`;
                }).join('');
                
            } catch (error) {
                console.error('Failed to update recent table:', error);
                document.getElementById('recent-tbody').innerHTML = 
                    '<tr><td colspan="4">Error loading data</td></tr>';
            }
        }
        
        function formatMetadata(metadata) {
            if (!metadata || Object.keys(metadata).length === 0) return '-';
            
            const parts = [];
            if (metadata.result_count !== undefined) parts.push(`${metadata.result_count} results`);
            if (metadata.nodes_created) parts.push(`${metadata.nodes_created} nodes`);
            if (metadata.edges_created) parts.push(`${metadata.edges_created} edges`);
            if (metadata.used_sqlite_vec !== undefined) {
                parts.push(metadata.used_sqlite_vec ? 'vec' : 'brute-force');
            }
            if (metadata.bfs_explored) parts.push(`${metadata.bfs_explored} explored`);
            
            return parts.join(', ') || JSON.stringify(metadata);
        }
        
        async function updateDashboard() {
            try {
                updateStatus('ok', 'Refreshing...');
                
                await Promise.all([
                    updateOverview(),
                    updateBreakdownChart(),
                    updateLatencyChart(),
                    updateBfsChart(),
                    updateRecentTable()
                ]);
                
                updateStatus('ok', `Updated ${new Date().toLocaleTimeString()}`);
            } catch (error) {
                updateStatus('error', 'Update failed');
            }
        }
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            updateDashboard();
            
            // Auto-refresh every 10 seconds
            setInterval(updateDashboard, 10000);
        });
        """

    def log_message(self, format, *args):
        """Override to suppress HTTP request logging"""
        return


class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    """Handle requests in separate threads"""
    pass


def create_handler_class(db_path):
    """Create a handler class with the db_path bound"""
    class Handler(MetricsHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, db_path=db_path, **kwargs)
    return Handler


def main():
    parser = argparse.ArgumentParser(description='Cashew Metrics Dashboard')
    parser.add_argument('--port', type=int, default=8787, help='Port to serve on')
    parser.add_argument('--host', default='localhost', help='Host to bind to')
    parser.add_argument('--db', help='Database path (defaults to config)')
    
    args = parser.parse_args()
    
    # Check if metrics are enabled
    if not is_metrics_enabled():
        print("⚠️  Metrics collection is disabled. Set CASHEW_METRICS=1 to enable.")
        print("   The dashboard will show empty data until metrics are enabled.")
        print()
    
    db_path = args.db or get_db_path()
    
    if not os.path.exists(db_path):
        print(f"❌ Database not found at: {db_path}")
        print("   Run 'cashew init' first or specify --db path")
        sys.exit(1)
    
    handler_class = create_handler_class(db_path)
    
    try:
        server = ThreadedHTTPServer((args.host, args.port), handler_class)
        print(f"🥜 Cashew Metrics Dashboard")
        print(f"   Database: {db_path}")
        print(f"   Metrics:  {'✅ Enabled' if is_metrics_enabled() else '⚠️  Disabled (set CASHEW_METRICS=1)'}")
        print(f"   Server:   http://{args.host}:{args.port}")
        print()
        print("Press Ctrl+C to stop...")
        
        server.serve_forever()
        
    except KeyboardInterrupt:
        print("\n👋 Stopping dashboard...")
        server.shutdown()
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()