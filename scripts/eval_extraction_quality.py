#!/usr/bin/env python3
"""Extraction-quality eval harness: API backend vs. a local model backend.

Runs `extract_from_conversation` once per configured backend against a fixed
set of sample transcripts, into a fresh throwaway database per backend, and
reports objective, countable stats (nodes kept, edges created, empty
extractions, wall-clock time, token usage where the backend reports it).

This does NOT compute a "quality" score that requires human or LLM judgment.
See docs/extraction-eval-harness.md for scope and what's not yet wired up
(there is currently no local-model LLMBackend in core/llm.py to compare
against -- see that doc for what needs to be added first).

Usage:
    python scripts/eval_extraction_quality.py --backends claude_code
    python scripts/eval_extraction_quality.py --backends claude_code,ollama_qwen
    python scripts/eval_extraction_quality.py --transcripts-dir path/to/dir --output results.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

CASHEW_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CASHEW_ROOT))

from core.llm import build_backend  # noqa: E402
from scripts.cashew_init import create_database_schema  # noqa: E402
from integration.session import extract_from_conversation  # noqa: E402

DEFAULT_TRANSCRIPTS_DIR = CASHEW_ROOT / "eval" / "sample_transcripts"


def load_transcripts(transcripts_dir: Path) -> list[tuple[str, str]]:
    """Return [(name, text), ...] for every .txt file in transcripts_dir."""
    files = sorted(transcripts_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(
            f"No .txt transcripts found in {transcripts_dir}. "
            "Add sample conversation files (one per .txt) before running."
        )
    return [(f.stem, f.read_text()) for f in files]


def make_scratch_db() -> str:
    """Create a fresh, empty cashew DB in a tempdir and return its path."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="cashew_eval_"))
    db_path = str(tmp_dir / "eval.db")
    create_database_schema(db_path)
    return db_path


def run_backend(backend_name: str, transcripts: list[tuple[str, str]]) -> dict[str, Any]:
    """Run extraction for every transcript against one backend into a scratch DB."""
    model_fn: Optional[Callable[[str], str]] = build_backend(backend_name)
    if model_fn is None:
        return {"backend": backend_name, "error": "backend unavailable, see stderr above"}

    db_path = make_scratch_db()
    per_transcript = []
    t_start = time.time()
    for name, text in transcripts:
        t0 = time.time()
        result = extract_from_conversation(db_path, text, session_id=f"eval_{name}", model_fn=model_fn)
        elapsed = time.time() - t0
        per_transcript.append({
            "transcript": name,
            "success": result.get("success", False),
            "new_nodes": result.get("new_nodes", 0),
            "new_edges": result.get("new_edges", 0),
            "llm_kept_nothing": result.get("llm_kept_nothing", False),
            "elapsed_sec": round(elapsed, 2),
        })
    total_elapsed = time.time() - t_start

    return {
        "backend": backend_name,
        "model": getattr(model_fn, "model", None),
        "db_path": db_path,
        "num_transcripts": len(transcripts),
        "total_new_nodes": sum(r["new_nodes"] for r in per_transcript),
        "total_new_edges": sum(r["new_edges"] for r in per_transcript),
        "empty_extractions": sum(1 for r in per_transcript if r["llm_kept_nothing"]),
        "failures": sum(1 for r in per_transcript if not r["success"]),
        "total_elapsed_sec": round(total_elapsed, 2),
        "token_usage": dict(getattr(model_fn, "usage", {})),
        "per_transcript": per_transcript,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backends", default="claude_code",
        help="Comma-separated backend names to run and compare (see core/llm.py build_backend). "
             "Default: claude_code only (no local backend exists yet).",
    )
    parser.add_argument("--transcripts-dir", type=Path, default=DEFAULT_TRANSCRIPTS_DIR)
    parser.add_argument("--output", type=Path, default=None, help="Write JSON results here")
    args = parser.parse_args()

    transcripts = load_transcripts(args.transcripts_dir)
    print(f"Loaded {len(transcripts)} sample transcript(s) from {args.transcripts_dir}")

    results = []
    for backend_name in [b.strip() for b in args.backends.split(",") if b.strip()]:
        print(f"\n=== Running backend: {backend_name} ===")
        results.append(run_backend(backend_name, transcripts))

    print("\n=== Summary ===")
    for r in results:
        if "error" in r:
            print(f"{r['backend']}: ERROR - {r['error']}")
            continue
        print(
            f"{r['backend']} ({r['model']}): "
            f"{r['total_new_nodes']} nodes, {r['total_new_edges']} edges, "
            f"{r['empty_extractions']} empty extractions, "
            f"{r['failures']} failures, {r['total_elapsed_sec']}s total"
        )

    if len(results) == 2 and all("error" not in r for r in results):
        a, b = results
        print(
            f"\nnode-count delta ({a['backend']} vs {b['backend']}): "
            f"{a['total_new_nodes'] - b['total_new_nodes']:+d}; "
            f"wall-time delta: {a['total_elapsed_sec'] - b['total_elapsed_sec']:+.2f}s"
        )
        print(
            "(This is a raw count/time delta, not a validated quality score. "
            "See docs/extraction-eval-harness.md for what would be needed to "
            "turn this into a real 'sovereignty tax' number.)"
        )

    if args.output:
        args.output.write_text(json.dumps(results, indent=2))
        print(f"\nWrote results to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
