"""LLM backend abstraction for cashew.

Extraction, think cycles, and dream cycles all need to call a language model.
The rest of the codebase talks to a callable `model_fn(prompt) -> str` with an
attached `.usage` dict — this module produces those callables from a backend.

Adding a new backend: subclass `LLMBackend`, implement `_generate`, register it
in `build_backend`. Default is `ClaudeCodeBackend` (headless `claude -p` under
the user's Max plan — no API keys, no gateways, no extra-usage billing).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional


_EMPTY_MCP_CONFIG = '{"mcpServers":{}}'


def _ensure_empty_mcp_config() -> Optional[str]:
    """Write a one-time empty MCP config to tempdir.

    Headless `claude -p` invocations launched from inside a parent Claude
    Code session inherit the parent's plugin set. Plugins that own a
    long-lived external resource (telegram bot poller, etc.) cannot tolerate
    a second instance running concurrently. Passing `--strict-mcp-config`
    plus this empty config suppresses plugin-supplied MCP servers in the
    headless child without disturbing the parent.

    Returns the absolute path, or None if the file cannot be created.
    """
    try:
        path = Path(tempfile.gettempdir()) / "cashew-empty-mcp.json"
        if not path.exists():
            path.write_text(_EMPTY_MCP_CONFIG, encoding="utf-8")
        return str(path)
    except OSError:
        return None


class LLMBackend(ABC):
    """Callable LLM. Subclasses implement `_generate`; the base handles usage
    accounting and exposes the conventional `model_fn(prompt) -> str` shape."""

    model: str

    def __init__(self, model: str):
        self.model = model
        self.usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "calls": 0,
            "model": model,
        }

    @abstractmethod
    def _generate(self, prompt: str) -> tuple[str, int, int]:
        """Return (response_text, prompt_tokens, completion_tokens)."""

    def __call__(self, prompt: str) -> str:
        text, p_tok, c_tok = self._generate(prompt)
        if p_tok <= 0:
            p_tok = len(prompt) // 4
        if c_tok <= 0:
            c_tok = len(text) // 4
        self.usage["prompt_tokens"] += p_tok
        self.usage["completion_tokens"] += c_tok
        self.usage["total_tokens"] += p_tok + c_tok
        self.usage["calls"] += 1
        return text


class ClaudeCodeBackend(LLMBackend):
    """Shell out to headless `claude -p`. Runs under the Claude Code subscription."""

    def __init__(self, model: Optional[str] = None):
        super().__init__(model or os.environ.get("CASHEW_CLAUDE_MODEL", "claude-opus-4-7"))
        self._bin = shutil.which("claude")
        if not self._bin:
            raise RuntimeError("`claude` CLI not found on PATH")

    def _generate(self, prompt: str) -> tuple[str, int, int]:
        cmd = [self._bin, "-p", prompt,
               "--model", self.model,
               "--output-format", "json",
               "--permission-mode", "bypassPermissions"]
        empty_mcp = _ensure_empty_mcp_config()
        if empty_mcp:
            cmd += ["--strict-mcp-config", "--mcp-config", empty_mcp]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"claude -p failed (rc={proc.returncode}): {proc.stderr[:500]}")
        try:
            envelope = json.loads(proc.stdout)
        except json.JSONDecodeError:
            last = [line for line in proc.stdout.splitlines() if line.strip()][-1]
            envelope = json.loads(last)
        text = envelope.get("result") or envelope.get("text") or ""
        usage = envelope.get("usage", {}) or {}
        return text, int(usage.get("input_tokens", 0) or 0), int(usage.get("output_tokens", 0) or 0)


def build_backend(name: Optional[str] = None) -> Optional[Callable[[str], str]]:
    """Build a backend by name. Returns a callable with `.usage` attached,
    or None if the backend is unavailable (e.g. `claude` CLI missing).

    Name defaults to `$CASHEW_LLM_BACKEND` or `claude_code`.
    """
    name = (name or os.environ.get("CASHEW_LLM_BACKEND") or "claude_code").lower()
    try:
        if name == "claude_code":
            return ClaudeCodeBackend()
    except RuntimeError as e:
        print(f"⚠️  {name} backend unavailable: {e}")
        return None
    print(f"⚠️  Unknown LLM backend: {name!r}")
    return None
