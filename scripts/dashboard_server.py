#!/usr/bin/env python3
"""Cashew dashboard HTTP server.

Serves the dashboard UI plus a small JSON/SSE API so the dashboard can run
live recursive-BFS queries against a real database instead of a static
snapshot.

Routes
  GET  /                        → dashboard/v2.html (minimalist shell)
  GET  /legacy                  → dashboard/index.html (old dashboard, still works)
  GET  /dashboard/<path>        → static files under dashboard/
  GET  /api/graph               → full graph JSON (nodes + edges + metadata)
  GET  /api/search?q=...        → Server-Sent Events stream of a recursive-BFS walk

Usage
  python3 scripts/dashboard_server.py --db /path/to/graph.db [--port 8765]
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
DASHBOARD_DIR = ROOT / "dashboard"

from core import db as cdb  # noqa: E402
from core.retrieval import retrieve_bfs_streaming  # noqa: E402


def load_graph_json(db_path: str) -> dict:
    conn = cdb.connect(db_path)
    c = conn.cursor()
    c.execute(
        f"SELECT id, content, node_type, confidence, source_file, timestamp, "
        f"mood_state, domain, tags FROM {cdb.NODES_TABLE} "
        f"WHERE decayed = 0 OR decayed IS NULL"
    )
    nodes = []
    for row in c.fetchall():
        nodes.append({
            "id": row[0], "content": row[1], "node_type": row[2],
            "confidence": row[3], "source_file": row[4] or "",
            "timestamp": row[5] or "", "mood_state": row[6] or "",
            "domain": row[7] or "",
            "tags": (row[8] or "").split(",") if row[8] else [],
        })
    node_ids = {n["id"] for n in nodes}
    c.execute(f"SELECT parent_id, child_id, weight, reasoning FROM {cdb.EDGES_TABLE}")
    edges = []
    for p, ch, w, reason in c.fetchall():
        if p in node_ids and ch in node_ids:
            edges.append({"source": p, "target": ch, "weight": w, "reasoning": reason or ""})
    conn.close()
    return {
        "metadata": {"db_path": db_path, "exported": datetime.utcnow().isoformat()},
        "statistics": {"total_nodes": len(nodes), "total_edges": len(edges)},
        "nodes": nodes, "edges": edges,
    }


class Handler(BaseHTTPRequestHandler):
    db_path: str = ""

    def log_message(self, fmt, *args):  # quiet default logging
        sys.stderr.write(f"[dashboard] {fmt % args}\n")

    def _write_head(self, status: int, ctype: str, extra: dict | None = None):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()

    def _send_json(self, payload: dict, status: int = 200):
        body = json.dumps(payload).encode()
        self._write_head(status, "application/json", {"Content-Length": str(len(body))})
        self.wfile.write(body)

    def _send_file(self, path: Path):
        if not path.exists() or not path.is_file():
            self._send_json({"error": "not found"}, 404)
            return
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        data = path.read_bytes()
        self._write_head(200, ctype, {"Content-Length": str(len(data))})
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        path = url.path
        q = parse_qs(url.query)

        if path == "/" or path == "/index.html":
            return self._send_file(DASHBOARD_DIR / "v2.html")
        if path == "/legacy":
            return self._send_file(DASHBOARD_DIR / "index.html")
        if path.startswith("/dashboard/"):
            rel = path[len("/dashboard/"):]
            return self._send_file(DASHBOARD_DIR / rel)
        if path == "/api/graph":
            try:
                return self._send_json(load_graph_json(self.db_path))
            except Exception as e:
                return self._send_json({"error": f"{type(e).__name__}: {e}"}, 500)
        if path == "/api/search":
            query = (q.get("q", [""])[0] or "").strip()
            n_seeds = int(q.get("seeds", ["5"])[0])
            picks = int(q.get("picks", ["3"])[0])
            depth = int(q.get("depth", ["3"])[0])
            if not query:
                return self._send_json({"error": "missing q"}, 400)
            self._write_head(200, "text/event-stream", {
                "Cache-Control": "no-cache", "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            })
            try:
                for ev in retrieve_bfs_streaming(self.db_path, query,
                                                 n_seeds=n_seeds, picks_per_hop=picks,
                                                 max_depth=depth):
                    chunk = f"data: {json.dumps(ev)}\n\n".encode()
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except Exception as e:
                err = f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n".encode()
                try:
                    self.wfile.write(err)
                    self.wfile.flush()
                except Exception:
                    pass
            return
        return self._send_json({"error": "not found", "path": path}, 404)


def run(db_path: str, host: str = "127.0.0.1", port: int = 8765) -> None:
    Handler.db_path = db_path
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"🥜 cashew dashboard: http://{host}:{port}  (db: {db_path})", file=sys.stderr)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("shutting down", file=sys.stderr)
    finally:
        srv.server_close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True, help="Path to cashew SQLite DB")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    run(args.db, args.host, args.port)


if __name__ == "__main__":
    main()
