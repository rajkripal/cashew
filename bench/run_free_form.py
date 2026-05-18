"""Skeleton runner for the MemBench free-form synthesis track.

Loads free-form questions from a JSON file, retrieves context from a cashew DB,
generates a short answer via Claude, judges with the AttrScore-style judge, and
writes results to a JSONL file.

Question JSON schema:
{
  "questions": [
    {
      "qid": "string",
      "corpus": "movie|food|book|spanning",
      "question": "string",
      "gold_sketch": "string (empty until filled in)",
      "authoring_note": "string"
    }
  ]
}

Usage:
  python run_free_form.py --db cashew.db --output results.jsonl
  python run_free_form.py --db cashew.db --output results.jsonl --smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

LOCOMO_DIR = Path("/Users/bunny/.openclaw/workspace/benchmarks/locomo")
sys.path.insert(0, str(LOCOMO_DIR))
from cashew_adapter import (  # noqa: E402
    claude_p,
    make_model_fn,
    looks_like_rate_limit,
    RateLimitError,
    question_to_hints,
)

CASHEW_ROOT = Path(os.environ.get("CASHEW_ROOT", "/Users/bunny/.openclaw/workspace/cashew"))
sys.path.insert(0, str(CASHEW_ROOT))
from integration.session import generate_session_context  # noqa: E402

from free_form_judge import judge_free_form  # noqa: E402

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_QUESTIONS_FILE = (
    Path("/Users/bunny/.openclaw/workspace/cashew/papers/locomo-run")
    / "membench-free-form-questions.json"
)
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
ANSWER_MODEL = "claude-haiku-4-5-20251001"

# ---------------------------------------------------------------------------
# Answer prompt
# ---------------------------------------------------------------------------

_ANSWER_TMPL = """\
You are answering a question about someone's personal history based solely on the
provided context. Answer in 2-3 sentences. Do not add information that is not
present in the context. If the context does not contain enough information to
answer, say so briefly.

Context:
{context}

Question: {question}

Answer:"""


def retrieve_context(question: str, db_path: str, top_k: int = 10) -> str:
    """Pull relevant context from a cashew DB for a question."""
    hints = question_to_hints(question)
    try:
        ctx = generate_session_context(db_path, hints=hints, max_nodes=top_k)
        return ctx or ""
    except Exception as exc:
        return f"[context retrieval error: {exc}]"


def generate_answer(question: str, context: str, model: str = ANSWER_MODEL) -> tuple[str, dict]:
    """Generate a free-form answer using retrieved context."""
    prompt = _ANSWER_TMPL.format(context=context or "(no context found)", question=question)
    return claude_p(prompt, model=model)


def run(
    questions_file: Path,
    db_path: str,
    output_file: Path,
    smoke: bool = False,
    judge_model: str = DEFAULT_MODEL,
    answer_model: str = ANSWER_MODEL,
) -> None:
    if not questions_file.exists():
        print(f"ERROR: questions file not found: {questions_file}", file=sys.stderr)
        print(
            "Create it by converting the markdown draft at "
            "papers/locomo-run/membench-questions-draft.md to JSON.",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(questions_file) as f:
        data = json.load(f)

    questions = data.get("questions", [])
    if not questions:
        print("ERROR: no questions found in JSON file.", file=sys.stderr)
        sys.exit(1)

    if smoke:
        questions = questions[:3]
        print(f"--smoke: running first {len(questions)} questions only")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    total = len(questions)
    scores = []

    with open(output_file, "w") as out_f:
        for i, q in enumerate(questions, 1):
            qid = q.get("qid", f"q{i}")
            question_text = q["question"]
            gold_sketch = q.get("gold_sketch", "")
            corpus = q.get("corpus", "unknown")

            print(f"[{i}/{total}] qid={qid} corpus={corpus}")

            context = retrieve_context(question_text, db_path)
            answer, ans_usage = generate_answer(question_text, context, model=answer_model)
            judgment = judge_free_form(
                question=question_text,
                system_answer=answer,
                retrieved_context=context,
                gold_sketch=gold_sketch,
                model=judge_model,
            )

            result = {
                "qid": qid,
                "corpus": corpus,
                "question": question_text,
                "gold_sketch": gold_sketch,
                "retrieved_context": context,
                "system_answer": answer,
                "score": judgment["score"],
                "supported": judgment["supported"],
                "reasoning": judgment["reasoning"],
                "usage_answer": ans_usage,
                "usage_judge": judgment["usage"],
            }
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()

            scores.append(judgment["score"])
            print(
                f"  score={judgment['score']:.1f}  supported={judgment['supported']}  "
                f"{judgment['reasoning'][:80]}"
            )

    if scores:
        avg = sum(scores) / len(scores)
        print(f"\nDone. {len(scores)} questions. Mean score: {avg:.3f}")
        print(f"Results written to: {output_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MemBench free-form synthesis track against a cashew DB."
    )
    parser.add_argument(
        "--questions-file",
        type=Path,
        default=DEFAULT_QUESTIONS_FILE,
        help="Path to questions JSON file (default: %(default)s)",
    )
    parser.add_argument(
        "--db",
        required=True,
        help="Path to cashew SQLite DB to query for context.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("free_form_results.jsonl"),
        help="Path to output JSONL file (default: %(default)s)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run only the first 3 questions (for quick sanity checks).",
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_MODEL,
        help="Claude model for judging (default: %(default)s)",
    )
    parser.add_argument(
        "--answer-model",
        default=ANSWER_MODEL,
        help="Claude model for generating answers (default: %(default)s)",
    )

    args = parser.parse_args()
    run(
        questions_file=args.questions_file,
        db_path=args.db,
        output_file=args.output,
        smoke=args.smoke,
        judge_model=args.judge_model,
        answer_model=args.answer_model,
    )


if __name__ == "__main__":
    main()
