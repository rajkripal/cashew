"""Tests for the cashew structural context-injection hook."""
import importlib.util
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "inject_context", str(Path(__file__).parent / "inject_context.py"))
hook = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hook)


def test_clean_strips_wrapper_tags():
    assert hook._clean("<system-reminder>noise</system-reminder>real ask") == "real ask"
    assert "chan" not in hook._clean('<channel x="y">the message</channel>')
    assert hook._clean('<channel x="y">the message</channel>') == "the message"


def test_trivial_and_short_inject_nothing():
    for t in ["ok thanks", "yes", "got it", "sure", "makes sense", "👍", "hi"]:
        assert hook.build_context(t) == ""


def test_no_db_injects_nothing(monkeypatch):
    monkeypatch.delenv("CASHEW_DB", raising=False)
    monkeypatch.delenv("CASHEW_DB_PATH", raising=False)
    assert hook.build_context("a genuinely substantive question about the project") == ""


def test_relevance_gate_and_non_authoritative_labeling(monkeypatch, tmp_path):
    db = tmp_path / "g.db"; db.write_text("x")
    monkeypatch.setenv("CASHEW_DB", str(db))
    monkeypatch.setattr(hook, "_retrieve", lambda d, m: [
        {"type": "insight", "domain": "ai", "content": "genuinely relevant node", "score": 0.91},
        {"type": "fact", "domain": "raj", "content": "barely related noise", "score": 0.70},
    ])
    out = hook.build_context("a substantive question worth retrieving on")
    assert "genuinely relevant node" in out       # above 0.83 gate
    assert "barely related noise" not in out       # below gate -> dropped
    assert "LEADS to verify" in out and "NOT authority" in out


def test_all_below_threshold_injects_nothing(monkeypatch, tmp_path):
    db = tmp_path / "g.db"; db.write_text("x")
    monkeypatch.setenv("CASHEW_DB", str(db))
    monkeypatch.setattr(hook, "_retrieve", lambda d, m: [
        {"type": "fact", "domain": "ai", "content": "noise", "score": 0.79}])
    assert hook.build_context("a substantive question worth retrieving on") == ""


def test_main_is_failsafe(monkeypatch):
    def _boom(_):
        raise RuntimeError("brain exploded")
    monkeypatch.setattr(hook, "build_context", _boom)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"prompt": "anything substantive"})))
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = hook.main()
    assert rc == 0 and buf.getvalue() == ""   # no crash, no output
