#!/usr/bin/env python3
"""Tests for the cashew warm daemon (core/daemon.py)."""

import json
import os
import socket
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import daemon as cashew_daemon


def _free_socket_path() -> str:
    tmp = tempfile.mkdtemp(prefix="cashew-daemon-test-")
    return str(Path(tmp) / "d.sock")


def _start(socket_path: str):
    thread = cashew_daemon.serve_in_thread(socket_path=socket_path, warm=False)
    for _ in range(100):
        if os.path.exists(socket_path):
            break
        time.sleep(0.01)
    return thread


class DaemonProtocolTests(unittest.TestCase):
    def setUp(self):
        self.sock = _free_socket_path()
        _start(self.sock)

    def tearDown(self):
        # Daemon threads die with the process; clean the socket file.
        try:
            os.unlink(self.sock)
        except OSError:
            pass

    def test_ping_roundtrip(self):
        resp = cashew_daemon.client_request({"op": "ping"}, socket_path=self.sock, timeout=1.0)
        self.assertEqual(resp, {"ok": True, "result": "pong"})

    def test_unknown_op_returns_error_not_crash(self):
        resp = cashew_daemon.client_request({"op": "bogus"}, socket_path=self.sock, timeout=1.0)
        self.assertEqual(resp["ok"], False)
        self.assertIn("unknown op", resp["error"])
        # Daemon should still answer after a bad op.
        resp2 = cashew_daemon.client_request({"op": "ping"}, socket_path=self.sock, timeout=1.0)
        self.assertEqual(resp2, {"ok": True, "result": "pong"})

    def test_malformed_json_does_not_crash_daemon(self):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(1.0)
        s.connect(self.sock)
        s.sendall(b"not-json\n")
        try:
            data = s.recv(4096)
        except socket.timeout:
            data = b""
        s.close()
        if data:
            resp = json.loads(data.decode().splitlines()[0])
            self.assertEqual(resp["ok"], False)
        # Still alive after a bad payload.
        resp = cashew_daemon.client_request({"op": "ping"}, socket_path=self.sock, timeout=1.0)
        self.assertEqual(resp, {"ok": True, "result": "pong"})

    def test_context_op_delegates(self):
        # Stub out generate_session_context so tests don't need a populated DB.
        captured = {}

        def fake(db, hints, tags=None, exclude_tags=None):
            captured["args"] = (db, hints, tags, exclude_tags)
            return "FAKE_CONTEXT"

        import integration.openclaw as oc
        with mock.patch.object(oc, "generate_session_context", side_effect=fake):
            resp = cashew_daemon.client_request(
                {"op": "context", "db": "/tmp/x.db", "hints": ["a", "b"]},
                socket_path=self.sock,
                timeout=5.0,
            )
        self.assertEqual(resp, {"ok": True, "result": "FAKE_CONTEXT"})
        self.assertEqual(captured["args"], ("/tmp/x.db", ["a", "b"], None, None))

    def test_context_op_requires_db(self):
        resp = cashew_daemon.client_request({"op": "context"}, socket_path=self.sock, timeout=1.0)
        self.assertEqual(resp["ok"], False)
        self.assertIn("missing 'db'", resp["error"])


class DaemonClientFallbackTests(unittest.TestCase):
    def test_client_returns_none_when_socket_missing(self):
        path = _free_socket_path()  # exists as dir, file doesn't
        self.assertFalse(os.path.exists(path))
        resp = cashew_daemon.client_request({"op": "ping"}, socket_path=path, timeout=0.2)
        self.assertIsNone(resp)

    def test_client_returns_none_when_connection_refused(self):
        # Create a stale socket file that nothing is listening on.
        path = _free_socket_path()
        Path(path).touch()
        resp = cashew_daemon.client_request({"op": "ping"}, socket_path=path, timeout=0.2)
        self.assertIsNone(resp)


class DaemonStartupTests(unittest.TestCase):
    def test_serve_refuses_when_socket_already_in_use(self):
        path = _free_socket_path()
        _start(path)
        # Second serve() call on the same path should raise — the first daemon owns it.
        with self.assertRaises(RuntimeError):
            cashew_daemon.serve(socket_path=path, warm=False)
        try:
            os.unlink(path)
        except OSError:
            pass

    def test_serve_cleans_stale_socket(self):
        path = _free_socket_path()
        Path(path).touch()  # stale file, no listener
        self.assertTrue(os.path.exists(path))
        # serve() should unlink and bind; run in a thread and ping.
        t = cashew_daemon.serve_in_thread(socket_path=path, warm=False)
        try:
            for _ in range(100):
                resp = cashew_daemon.client_request({"op": "ping"}, socket_path=path, timeout=0.2)
                if resp:
                    break
                time.sleep(0.01)
            self.assertEqual(resp, {"ok": True, "result": "pong"})
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass


if __name__ == "__main__":
    unittest.main()
