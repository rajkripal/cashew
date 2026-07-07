#!/usr/bin/env python3
"""Cashew structural context injection — UserPromptSubmit hook.

Makes brain-lookup STRUCTURAL instead of volitional. The skill's CLAUDE.md
protocol asks Claude to "query the brain first" — but that's a decision the model
has to remember, and in practice it's skipped. This hook removes the decision:
on every substantive prompt it queries cashew and injects the relevant context
directly, so the brain is consulted whether or not anyone remembered to.

Design (each addresses a real failure mode):
  1. Deterministic trigger — runs on every non-trivial prompt (a length + trivial
     pre-gate skips "ok", "thanks", etc. so trivia never even queries).
  2. The embedding of the raw prompt IS the query — no brittle keyword-guessing.
     An explicit `cashew context` call remains available as a refinement layer.
  3. Relevance GATE — inject only nodes whose similarity clears a threshold, so
     trivia and noise inject nothing. The default sits above a typical model's
     unrelated-pair cosine floor; tune per model via CASHEW_HOOK_RELEVANCE.
  4. Non-authoritative framing — injected as clearly-labeled, score-tagged LEADS
     with an explicit "verify, not authority" instruction, so a plausible-but-
     wrong hit can't quietly misguide.

Fail-safe: any error, timeout, or missing brain injects nothing and exits 0 —
the hook must never block a turn. Wire it via .claude/settings.json:

  "hooks": {"UserPromptSubmit": [{"hooks": [{"type": "command",
    "command": "python3 /path/to/skills/claude-code/hooks/inject_context.py"}]}]}}

(scripts/install-structural-mode.sh does this for you.) Requires cashew installed
and CASHEW_DB pointing at the brain. A warm daemon (`cashew serve`) keeps it fast
(~0.2s); without one the first call pays a model load.
"""

from __future__ import annotations

import json
import os
import re
import sys

RELEVANCE_THRESHOLD = float(os.environ.get("CASHEW_HOOK_RELEVANCE", "0.83"))
TOP_K = int(os.environ.get("CASHEW_HOOK_TOP_K", "6"))
MAX_INJECTED = int(os.environ.get("CASHEW_HOOK_MAX_NODES", "5"))
MAX_CHARS = int(os.environ.get("CASHEW_HOOK_MAX_CHARS", "240"))

_TRIVIAL_RE = re.compile(
    r"^\s*(ok(ay)?|k+|thanks?|thank you|thx|ty|yes|yep|yeah|no|nope|sure|got it|"
    r"cool|nice|great|perfect|done|good|makes sense|sounds good|👍|🙏|❤️|👀)[\s!.…]*$",
    re.IGNORECASE,
)


def _resolve_db() -> str | None:
    return os.environ.get("CASHEW_DB") or os.environ.get("CASHEW_DB_PATH")


def _clean(prompt: str) -> str:
    """Reduce the prompt to human text: strip harness wrappers so the embedding
    reflects the actual request, not surrounding metadata."""
    t = re.sub(r"<system-reminder>.*?</system-reminder>", " ", prompt, flags=re.S | re.I)
    t = re.sub(r"</?[a-zA-Z][^>]*>", " ", t)  # drop any XML-ish wrapper tags
    return re.sub(r"\s+", " ", t).strip()


def _retrieve(db: str, message: str) -> list[dict]:
    """Scored retrieval via cashew's own engine (shipped together, so importing
    internals is fine). Returns [] on any failure."""
    try:
        from core.retrieval import retrieve_recursive_bfs
    except Exception:
        # Source checkout (cashew not pip-installed): let CASHEW_DIR point at the
        # repo so `core` is importable. Pip installs put `core` on the path already.
        cashew_dir = os.environ.get("CASHEW_DIR")
        if cashew_dir and cashew_dir not in sys.path:
            sys.path.insert(0, cashew_dir)
        try:
            from core.retrieval import retrieve_recursive_bfs
        except Exception:
            return []
    try:
        results = retrieve_recursive_bfs(db, message, top_k=TOP_K)
    except Exception:
        return []
    out = []
    for r in results:
        try:
            out.append({
                "type": getattr(r, "node_type", "?"),
                "domain": getattr(r, "domain", "?"),
                "content": getattr(r, "content", "") or "",
                "score": round(float(getattr(r, "score", 0.0)), 3),
            })
        except Exception:
            continue
    return out


def build_context(prompt: str) -> str:
    """Return a labeled, relevance-gated context block, or "" (inject nothing)."""
    message = _clean(prompt)
    if len(message) < 15 or _TRIVIAL_RE.match(message):
        return ""
    db = _resolve_db()
    if not db or not os.path.exists(db):
        return ""
    hits = [n for n in _retrieve(db, message[:1000]) if n["score"] >= RELEVANCE_THRESHOLD]
    if not hits:
        return ""
    lines = [
        "## Cashew brain context (auto-retrieved, relevance-gated)",
        "Retrieved by similarity to this message. Treat as LEADS to verify, NOT "
        "authority — scores shown; ignore any that are off-topic and confirm "
        "against source before relying. Does not replace an explicit `cashew context` query.",
    ]
    for n in hits[:MAX_INJECTED]:
        content = " ".join(n["content"].split())[:MAX_CHARS]
        lines.append(f"- [{n['type']}·{n['domain']}·{n['score']:.2f}] {content}")
    return "\n".join(lines)


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except Exception:
        return 0
    prompt = payload.get("prompt") or ""
    if not prompt:
        return 0
    try:
        ctx = build_context(prompt)
        if ctx:
            print(ctx)  # stdout on UserPromptSubmit is added to the model's context
    except Exception:
        pass  # fail-safe: never block the turn
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        sys.exit(0)
