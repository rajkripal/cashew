#!/usr/bin/env python3
"""
Cashew warm daemon — keeps the embedding model and sqlite-vec connection
loaded so repeated queries don't pay the ~3s cold-start cost.

Wire protocol: newline-delimited JSON over a unix socket.

  request:  {"op": "ping"}                                    → {"ok": true, "result": "pong"}
  request:  {"op": "context", "db": str, "hints": [str, ...],
             "tags": [str]|null, "exclude_tags": [str]|null}  → {"ok": true, "result": "<context string>"}
  request:  {"op": "embed", "text": str}                      → {"ok": true, "result": [float, ...]}

On any error: {"ok": false, "error": str}. One request per connection; the
server closes the socket after writing the response.

Defaults to ${XDG_RUNTIME_DIR:-$HOME/.cashew}/daemon.sock. Override with
CASHEW_SOCKET.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import socketserver
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("cashew.daemon")


def default_socket_path() -> str:
    env = os.environ.get("CASHEW_SOCKET")
    if env:
        return env
    base = os.environ.get("XDG_RUNTIME_DIR") or str(Path.home() / ".cashew")
    Path(base).mkdir(parents=True, exist_ok=True)
    return str(Path(base) / "daemon.sock")


def _read_request(rfile) -> Optional[dict]:
    line = rfile.readline()
    if not line:
        return None
    return json.loads(line.decode("utf-8"))


def _write_response(wfile, payload: dict) -> None:
    wfile.write((json.dumps(payload) + "\n").encode("utf-8"))
    wfile.flush()


def _handle(req: dict) -> dict:
    op = req.get("op")
    if op == "ping":
        return {"ok": True, "result": "pong"}

    if op == "embed":
        from .embeddings import embed_text
        text = req.get("text", "")
        return {"ok": True, "result": embed_text(text)}

    if op == "context":
        from integration.openclaw import generate_session_context
        db = req.get("db")
        if not db:
            return {"ok": False, "error": "missing 'db'"}
        hints = req.get("hints") or None
        tags = req.get("tags") or None
        exclude_tags = req.get("exclude_tags") or None
        result = generate_session_context(db, hints, tags=tags, exclude_tags=exclude_tags)
        return {"ok": True, "result": result}

    return {"ok": False, "error": f"unknown op: {op!r}"}


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            req = _read_request(self.rfile)
            if req is None:
                return
            resp = _handle(req)
        except Exception as e:
            logger.exception("request failed")
            resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        try:
            _write_response(self.wfile, resp)
        except BrokenPipeError:
            pass


class _ThreadingUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True


def serve(socket_path: Optional[str] = None, warm: bool = True) -> None:
    """Run the daemon in the foreground. Blocks until interrupted."""
    path = socket_path or default_socket_path()

    # Clean up stale socket if no one is listening on it.
    if os.path.exists(path):
        try:
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            probe.settimeout(0.2)
            probe.connect(path)
            probe.close()
            raise RuntimeError(
                f"cashew daemon socket already in use: {path}. "
                f"Another daemon is running, or set CASHEW_SOCKET to a different path."
            )
        except (ConnectionRefusedError, FileNotFoundError, socket.timeout, OSError):
            # Stale socket file (or a regular file at that path with no listener).
            os.unlink(path)

    if warm:
        logger.info("warming embedding model...")
        t0 = time.perf_counter()
        from .embeddings import embed_text
        embed_text("warmup")
        logger.info(f"model warm in {(time.perf_counter()-t0)*1000:.0f}ms")

    server = _ThreadingUnixServer(path, _Handler)
    try:
        os.chmod(path, 0o600)  # owner-only
    except OSError:
        pass
    logger.info(f"cashew daemon listening on {path}")
    try:
        server.serve_forever()
    finally:
        server.server_close()
        try:
            os.unlink(path)
        except OSError:
            pass


def client_request(req: dict, socket_path: Optional[str] = None, timeout: float = 0.5) -> Optional[dict]:
    """Send one request to the daemon. Returns parsed response dict, or None
    if the daemon isn't reachable (so callers can fall back to in-process)."""
    path = socket_path or default_socket_path()
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect(path)
    except (FileNotFoundError, ConnectionRefusedError, socket.timeout, OSError):
        return None
    try:
        s.sendall((json.dumps(req) + "\n").encode("utf-8"))
        # Responses can be large (context strings) — drop timeout to a longer read budget.
        s.settimeout(30.0)
        buf = bytearray()
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            buf.extend(chunk)
            if b"\n" in chunk:
                break
        line = bytes(buf).split(b"\n", 1)[0]
        if not line:
            return None
        return json.loads(line.decode("utf-8"))
    except (socket.timeout, OSError, json.JSONDecodeError):
        return None
    finally:
        try:
            s.close()
        except OSError:
            pass


def serve_in_thread(socket_path: Optional[str] = None, warm: bool = False) -> threading.Thread:
    """Start the daemon on a background thread. Used by tests."""
    t = threading.Thread(
        target=lambda: serve(socket_path=socket_path, warm=warm),
        daemon=True,
    )
    t.start()
    # Wait briefly for the socket to appear.
    path = socket_path or default_socket_path()
    for _ in range(50):
        if os.path.exists(path):
            return t
        time.sleep(0.02)
    return t
