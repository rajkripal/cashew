"""Tests for core.llm backend abstraction."""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.llm import LLMBackend, ClaudeCodeBackend, build_backend


class FakeBackend(LLMBackend):
    """Test backend with controllable response + fake usage."""

    def __init__(self, response="ok", p_tok=10, c_tok=20, model="fake"):
        super().__init__(model=model)
        self._response = response
        self._p = p_tok
        self._c = c_tok

    def _generate(self, prompt):
        return self._response, self._p, self._c


class TestLLMBackend:
    def test_call_returns_response(self):
        b = FakeBackend(response="hello")
        assert b("prompt") == "hello"

    def test_usage_accumulates_across_calls(self):
        b = FakeBackend(p_tok=5, c_tok=7)
        b("one")
        b("two")
        assert b.usage["prompt_tokens"] == 10
        assert b.usage["completion_tokens"] == 14
        assert b.usage["total_tokens"] == 24
        assert b.usage["calls"] == 2

    def test_usage_falls_back_to_length_estimate_when_tokens_zero(self):
        b = FakeBackend(response="xxxx" * 4, p_tok=0, c_tok=0)
        prompt = "y" * 40
        b(prompt)
        # ~len/4 estimate
        assert b.usage["prompt_tokens"] == len(prompt) // 4
        assert b.usage["completion_tokens"] == len("xxxx" * 4) // 4

    def test_model_stored_in_usage(self):
        b = FakeBackend(model="sonnet-9")
        assert b.usage["model"] == "sonnet-9"
        assert b.model == "sonnet-9"


class TestClaudeCodeBackend:
    def test_init_requires_claude_on_path(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _: None)
        with pytest.raises(RuntimeError, match="claude"):
            ClaudeCodeBackend()

    def test_default_model_env_override(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _: "/fake/claude")
        monkeypatch.setenv("CASHEW_CLAUDE_MODEL", "haiku-5")
        b = ClaudeCodeBackend()
        assert b.model == "haiku-5"

    def test_explicit_model_wins_over_env(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _: "/fake/claude")
        monkeypatch.setenv("CASHEW_CLAUDE_MODEL", "env-model")
        b = ClaudeCodeBackend(model="explicit")
        assert b.model == "explicit"

    def test_generate_passes_strict_mcp_config(self, monkeypatch):
        """Headless `claude -p` must isolate MCP config so plugins from a
        surrounding Claude Code session do not double-spawn in the child."""
        import json as _json
        import subprocess as _subprocess

        monkeypatch.setattr("shutil.which", lambda _: "/fake/claude")

        captured: dict = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            class R:
                returncode = 0
                stdout = _json.dumps({"result": "ok",
                                      "usage": {"input_tokens": 1,
                                                "output_tokens": 1}})
                stderr = ""
            return R()

        monkeypatch.setattr(_subprocess, "run", fake_run)

        b = ClaudeCodeBackend()
        b("hi")

        cmd = captured["cmd"]
        assert "--strict-mcp-config" in cmd, (
            "ClaudeCodeBackend must pass --strict-mcp-config to suppress "
            "plugin-supplied MCP servers in the headless subprocess"
        )
        idx = cmd.index("--mcp-config")
        cfg_path = cmd[idx + 1]
        # The config file must exist and be a valid empty MCP config.
        with open(cfg_path) as fh:
            data = _json.load(fh)
        assert data == {"mcpServers": {}}


class TestBuildBackend:
    def test_unknown_backend_returns_none(self):
        assert build_backend("nope-not-real") is None

    def test_claude_code_unavailable_returns_none(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _: None)
        assert build_backend("claude_code") is None

    def test_default_reads_env(self, monkeypatch):
        monkeypatch.setenv("CASHEW_LLM_BACKEND", "nope")
        assert build_backend() is None
