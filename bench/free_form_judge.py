"""AttrScore-style LLM judge for MemBench free-form synthesis questions.

AttrScore measures citation precision: does the system's response faithfully
reflect what's in the retrieved context? We adapt that idea here.

Score rubric:
  1.0 -- fully supported by the retrieved context
  0.5 -- partially supported (some claims supported, some not verifiable)
  0.0 -- not supported or actively contradicts the context / hallucinated
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

LOCOMO_DIR = Path("/Users/bunny/.openclaw/workspace/benchmarks/locomo")
sys.path.insert(0, str(LOCOMO_DIR))
from cashew_adapter import claude_p, looks_like_rate_limit, RateLimitError  # noqa: E402

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = """\
You are a strict factual-attribution judge. Your task is to assess whether a
system's answer is faithfully supported by the provided retrieved context.

Scoring rubric:
  1.0 -- Every claim in the answer is directly supported by the retrieved context.
  0.5 -- Some claims are supported, but others are not verifiable from the context
         (missing information) or are weakly supported.
  0.0 -- The answer contradicts the context, hallucinates facts not present in
         the context, or the context contains no relevant information at all.

If a gold sketch is provided, use it only as a hint about what a good answer
looks like -- do NOT score based on string similarity to the gold sketch.
Attribution to context is the only criterion.

Respond with valid JSON only, no prose outside JSON:
{
  "score": <float 0.0|0.5|1.0>,
  "supported": <true|false>,
  "reasoning": "<one or two sentences>"
}
"""

_JUDGE_USER_TMPL = """\
## Question
{question}

## Retrieved Context
{retrieved_context}

## Gold Sketch (optional hint)
{gold_sketch}

## System Answer
{system_answer}

Evaluate the system answer strictly against the retrieved context. Return JSON.
"""


def judge_free_form(
    question: str,
    system_answer: str,
    retrieved_context: str,
    gold_sketch: str = "",
    model: str = "claude-haiku-4-5-20251001",
) -> Dict[str, Any]:
    """Judge one free-form answer against retrieved context.

    Returns:
        {
            "score": float,       # 0.0, 0.5, or 1.0
            "supported": bool,
            "reasoning": str,
            "usage": dict,
        }
    """
    user_msg = _JUDGE_USER_TMPL.format(
        question=question,
        retrieved_context=retrieved_context or "(none)",
        gold_sketch=gold_sketch or "(none provided)",
        system_answer=system_answer,
    )
    full_prompt = f"<system>\n{_JUDGE_SYSTEM}\n</system>\n\n{user_msg}"

    text, usage = claude_p(full_prompt, model=model)

    # Parse JSON out of the response.
    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError:
        # Try to extract a JSON object substring.
        import re
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            parsed = json.loads(m.group())
        else:
            # Fallback: conservative score
            parsed = {
                "score": 0.0,
                "supported": False,
                "reasoning": f"Judge parse error. Raw: {text[:200]}",
            }

    score = float(parsed.get("score", 0.0))
    return {
        "score": score,
        "supported": bool(parsed.get("supported", score >= 0.5)),
        "reasoning": str(parsed.get("reasoning", "")),
        "usage": usage,
    }


def batch_judge(
    results: List[Dict[str, Any]],
    model: str = "claude-haiku-4-5-20251001",
) -> List[Dict[str, Any]]:
    """Judge a batch of result dicts.

    Each dict must have keys: question, system_answer, retrieved_context.
    Optional key: gold_sketch.

    Returns a parallel list of judgment dicts (same shape as judge_free_form).
    """
    judgments = []
    for item in results:
        judgment = judge_free_form(
            question=item["question"],
            system_answer=item["system_answer"],
            retrieved_context=item.get("retrieved_context", ""),
            gold_sketch=item.get("gold_sketch", ""),
            model=model,
        )
        judgments.append(judgment)
    return judgments
