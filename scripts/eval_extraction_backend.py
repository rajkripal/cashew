#!/usr/bin/env python3
"""Sovereignty-tax harness: compare extraction quality/speed, API backend vs local Ollama.

Background
----------
Cashew's default extraction backend is `ClaudeCodeBackend` (headless `claude -p`,
see core/llm.py) — no API keys, but it depends on network + subscription access.
The open question this script answers with real numbers: what do we give up
(quality, latency) by running extraction against a fully local model instead?
That delta is the "sovereignty tax."

This is a harness, not a finished study. It runs extraction twice per sample
conversation — once via the existing backend (`core.llm.build_backend()`),
once via a local Ollama model — and reports cheap, structural metrics only:
node count, success/error, latency. It does NOT judge semantic quality; that
needs a follow-up with an LLM-judge or manual review.

Model choice
------------
Default local model is `qwen2.5:7b-instruct` served by Ollama on
`http://localhost:11434`. This is a placeholder pick (small enough to run on
the Mac Mini, instruction-tuned, JSON-capable) — swap via --local-model /
--ollama-host or $CASHEW_EVAL_OLLAMA_MODEL / $CASHEW_EVAL_OLLAMA_HOST if Raj
picks something else (a Llama variant, etc).

Usage
-----
    python3 scripts/eval_extraction_backend.py
    python3 scripts/eval_extraction_backend.py --input samples.jsonl
    python3 scripts/eval_extraction_backend.py --local-model llama3.1:8b --ollama-host http://localhost:11434

Input format (--input, optional): JSONL, one object per line:
    {"id": "sample-1", "text": "raj: ...\\nbunny: ..."}
If --input is omitted, a handful of built-in sample conversations are used.

Each sample is extracted into its own throwaway sqlite db (created via
`scripts/cashew_context.py init`) so runs never touch the real brain.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.llm import build_backend  # noqa: E402
from integration.session import extract_from_conversation  # noqa: E402

DEFAULT_SAMPLES = [
    {
        "id": "sample-1-decision",
        "text": (
            "raj: I've decided to stick with SQLite for cashew instead of migrating "
            "to Postgres. The ops overhead isn't worth it at this scale.\n"
            "bunny: noted, extracting that as a decision.\n"
        ),
    },
    {
        "id": "sample-2-todo",
        "text": (
            "raj: remind me to follow up with the recruiter by Friday about the E5 packet.\n"
            "bunny: got it, I'll track that as a TODO and ping you Thursday.\n"
        ),
    },
    {
        "id": "sample-3-correction",
        "text": (
            "raj: actually no, Harika isn't traveling next week, I mixed up the dates. "
            "The trip is the week after.\n"
            "bunny: correcting that now.\n"
        ),
    },
]


class OllamaBackend:
    """Minimal local backend hitting Ollama's /api/generate. Mirrors the
    model_fn(prompt) -> str contract used by core.llm.LLMBackend subclasses."""

    def __init__(self, model: str, host: str, timeout: float = 120.0):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0, "model": model}

    def __call__(self, prompt: str) -> str:
        payload = json.dumps({"model": self.model, "prompt": prompt, "stream": False}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/generate", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        text = body.get("response", "")
        self.usage["calls"] += 1
        self.usage["prompt_tokens"] += len(prompt) // 4
        self.usage["completion_tokens"] += len(text) // 4
        self.usage["total_tokens"] = self.usage["prompt_tokens"] + self.usage["completion_tokens"]
        return text


def check_ollama_reachable(host: str, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(f"{host.rstrip('/')}/api/tags", timeout=timeout):
            return True
    except (urllib.error.URLError, OSError):
        return False


def make_temp_db() -> str:
    fd, path = tempfile.mkstemp(suffix=".db", prefix="cashew-eval-")
    os.close(fd)
    os.remove(path)  # cmd_init refuses to overwrite; let it create fresh
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "cashew_context.py"), "init", "--db", path],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"failed to init eval db: {result.stderr[:500]}")
    return path


def run_one(backend_name: str, model_fn: Optional[Callable[[str], str]], text: str, sample_id: str) -> dict[str, Any]:
    db_path = None
    try:
        db_path = make_temp_db()
        t0 = time.perf_counter()
        result = extract_from_conversation(db_path, text, f"eval-{backend_name}-{sample_id}", model_fn=model_fn)
        elapsed = time.perf_counter() - t0
        return {
            "sample": sample_id,
            "backend": backend_name,
            "success": bool(result.get("success")),
            "node_count": result.get("new_nodes", 0) or 0,
            "latency_s": elapsed,
            "error": None if result.get("success") else str(result.get("error") or "extraction returned success=False"),
        }
    except Exception as e:  # noqa: BLE001 — this harness must never crash mid-run
        return {"sample": sample_id, "backend": backend_name, "success": False, "node_count": 0, "latency_s": None, "error": str(e)}
    finally:
        if db_path and os.path.exists(db_path):
            os.remove(db_path)


def load_samples(input_path: Optional[str]) -> list[dict[str, str]]:
    if not input_path:
        return DEFAULT_SAMPLES
    samples = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            samples.append({"id": obj.get("id", f"sample-{len(samples)}"), "text": obj["text"]})
    return samples or DEFAULT_SAMPLES


def summarize(rows: list[dict[str, Any]]) -> None:
    backends = sorted(set(r["backend"] for r in rows))
    print()
    print(f"{'backend':<20} {'avg_nodes':<12} {'avg_latency_s':<16} {'errors':<8} {'runs':<6}")
    print("-" * 66)
    for backend in backends:
        sub = [r for r in rows if r["backend"] == backend]
        latencies = [r["latency_s"] for r in sub if r["latency_s"] is not None]
        avg_nodes = sum(r["node_count"] for r in sub) / len(sub) if sub else 0.0
        avg_latency = sum(latencies) / len(latencies) if latencies else float("nan")
        errors = sum(1 for r in sub if not r["success"])
        print(f"{backend:<20} {avg_nodes:<12.2f} {avg_latency:<16.2f} {errors:<8} {len(sub):<6}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", help="JSONL file of {id, text} sample conversations")
    parser.add_argument("--local-model", default=os.environ.get("CASHEW_EVAL_OLLAMA_MODEL", "qwen2.5:7b-instruct"))
    parser.add_argument("--ollama-host", default=os.environ.get("CASHEW_EVAL_OLLAMA_HOST", "http://localhost:11434"))
    parser.add_argument("--skip-api", action="store_true", help="skip the existing (claude_code) backend")
    parser.add_argument("--skip-local", action="store_true", help="skip the local Ollama backend")
    args = parser.parse_args()

    samples = load_samples(args.input)
    print(f"Loaded {len(samples)} sample(s).")

    rows: list[dict[str, Any]] = []

    if not args.skip_api:
        api_backend = build_backend()
        if api_backend is None:
            print("API backend unavailable (claude CLI not found or misconfigured) — skipping.")
        else:
            for s in samples:
                rows.append(run_one("api (claude_code)", api_backend, s["text"], s["id"]))

    if not args.skip_local:
        if not check_ollama_reachable(args.ollama_host):
            print(f"Local backend unreachable: Ollama not responding at {args.ollama_host}. "
                  f"Start it with `ollama serve` and `ollama pull {args.local_model}` to include it.")
        else:
            local_backend = OllamaBackend(args.local_model, args.ollama_host)
            for s in samples:
                rows.append(run_one(f"local ({args.local_model})", local_backend, s["text"], s["id"]))

    if not rows:
        print("No backends ran — nothing to compare.")
        return 0

    summarize(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
