#!/usr/bin/env python3
"""depth-bench v1: cashew vs Mem0 on synthesis questions, judge-graded provenance.

Pipeline:
  1) Load 10 hand-written depth questions for conv-26.
  2) For each question:
       - cashew: retrieve(conv-26-B.db) -> answer_with_citations
       - mem0:   search(conv-conv-26 store) -> answer_with_citations
  3) Provenance check: for each cited (session_N, quote), feed the
     locomo10.json session text + the cited quote to a judge LLM and
     record yes/no + reason.
  4) Aggregate per-system precision; write DEPTH-BENCH-V1.md.

Time budget ~3h. Designed to be resumable; writes JSONL incrementally.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

THIS = Path(__file__).resolve().parent
LOCOMO_HARNESS = Path("/Users/bunny/.openclaw/workspace/benchmarks/locomo")
sys.path.insert(0, str(LOCOMO_HARNESS))

DATA_FILE = LOCOMO_HARNESS / "data" / "locomo10.json"
CONV_ID = "conv-26"
CASHEW_SRC_DB = THIS / "dbs" / "conv-26-B.db"   # sleep+think applied (MiniLM)
MEM0_STORE_ROOT = THIS / "mem0-stores"
MEM0_CONV_KEY = "conv-26"  # init_mem0 builds path = store_root / f"conv-{conv_id}"
OUT_DIR = THIS / "depth-bench-v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ANSWERS_PATH = OUT_DIR / "answers.jsonl"
JUDGEMENTS_PATH = OUT_DIR / "judgements.jsonl"
REPORT_PATH = THIS / "DEPTH-BENCH-V1.md"

JUDGE_MODEL = os.environ.get("DEPTH_JUDGE_MODEL", "claude-sonnet-4-6")
ANSWER_MODEL = os.environ.get("DEPTH_ANSWER_MODEL", "claude-sonnet-4-6")

# ---------------------------------------------------------------------------
# 10 depth questions for conv-26 (Caroline + Melanie)
# Probes synthesis, not factoid recall. Avoid LoCoMo "when/where" shape.
# ---------------------------------------------------------------------------
DEPTH_QUESTIONS = [
    {
        "id": "Q1",
        "text": "What is Caroline's emotional architecture for processing her gender transition? Describe the recurring inner moves she uses, not just the outward steps.",
    },
    {
        "id": "Q2",
        "text": "How does Melanie's parenting philosophy show up in seemingly unrelated topics (her art, her friendships, her advice to Caroline)?",
    },
    {
        "id": "Q3",
        "text": "Where are the tensions between Caroline's public activism and her personal vulnerability? Point to moments where these collide.",
    },
    {
        "id": "Q4",
        "text": "What recurring pattern surfaces in how both Caroline and Melanie discuss the idea of community or belonging?",
    },
    {
        "id": "Q5",
        "text": "What does Melanie struggle with that she doesn't say directly? Look for indirect expressions, deflections, or things she circles around.",
    },
    {
        "id": "Q6",
        "text": "How does Caroline use creative work (writing, art, projects) as a tool for processing identity? Is it expression, externalization, or something else?",
    },
    {
        "id": "Q7",
        "text": "What is the implicit contract of this friendship? What does each woman get from the other that she can't get elsewhere in her life?",
    },
    {
        "id": "Q8",
        "text": "How do Caroline and Melanie differ in their relationship to time and the future? Whose orientation is more forward-looking, and what does that reveal?",
    },
    {
        "id": "Q9",
        "text": "Where does Caroline's confidence wobble? Identify the specific topics or contexts where she hedges, second-guesses, or seeks validation.",
    },
    {
        "id": "Q10",
        "text": "What does this pair NOT discuss that you would expect close friends to discuss? What are the negative-space topics?",
    },
]

# ---------------------------------------------------------------------------
# Citation-forcing prompts
# ---------------------------------------------------------------------------
CITED_ANSWER_PROMPT = """You are answering a synthesis-style question about a long
multi-session conversation between two people. You have ONLY the retrieved
memory excerpts below. The conversation is split into numbered sessions
(session_1 through session_19).

# Retrieved memory
{ctx}

# Question
{q}

Write a SHORT synthesis (3-6 sentences) drawing only on the retrieved memory.
Then list 2-5 supporting evidence citations. Each citation MUST name the
session number it came from and quote (exact or paraphrased) the supporting
fragment.

Output EXACTLY in this format, with no extra preamble:

Answer: <your synthesis>

Supporting evidence:
1. [session N] "<quote or paraphrase>"
2. [session N] "<quote or paraphrase>"
...

If a memory excerpt does not clearly come from a numbered session, infer the
most plausible session from the content; if you cannot, write [session ?].
You MUST emit at least one citation.
"""

JUDGE_PROMPT = """You are a fact-grounding judge. You will see the transcript
of a multi-session conversation between two people, and a CLAIM that another
system asserted is supported by that conversation. Decide whether the
transcript explicitly states or clearly implies the claim.

Be strict about facts (names, events, specific assertions about a person)
but accept reasonable paraphrase or inference where the meaning is clearly
present. Mark "no" if the claim is fabricated, contradicts the transcript,
or stretches well beyond what is said.

# Conversation transcript
{transcript}

# Claim
"{claim}"

Reply with exactly one word ("yes" or "no") on the first line, then a brief
one-sentence reason on the second line.
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_conv_sample() -> dict:
    samples = json.loads(DATA_FILE.read_text())
    s = next((x for x in samples if x["sample_id"] == CONV_ID), None)
    if s is None:
        raise SystemExit(f"conv {CONV_ID} not found in {DATA_FILE}")
    return s


def session_transcripts(sample: dict) -> Dict[int, str]:
    """Return {session_idx: 'Speaker: text\\n...'} for all sessions."""
    conv = sample["conversation"]
    out: Dict[int, str] = {}
    for k, v in conv.items():
        if not k.startswith("session_") or k.endswith("date_time"):
            continue
        try:
            idx = int(k.split("_")[1])
        except Exception:
            continue
        if not isinstance(v, list):
            continue
        lines = [f"{t.get('speaker','?')}: {t.get('text','')}" for t in v]
        date = conv.get(f"{k}_date_time", "")
        header = f"# {k} ({date})\n" if date else f"# {k}\n"
        out[idx] = header + "\n".join(lines)
    return out


def parse_cited_answer(text: str) -> Tuple[str, List[Tuple[int, str]]]:
    """Return (answer_text, [(session_idx, quote), ...])."""
    answer = ""
    citations: List[Tuple[int, str]] = []
    if not text:
        return "", []
    # split at "Supporting evidence"
    low = text.lower()
    se_idx = low.find("supporting evidence")
    if se_idx < 0:
        # No citations found; return whole as answer.
        ans_part = text
        cite_part = ""
    else:
        ans_part = text[:se_idx]
        cite_part = text[se_idx:]
    # extract answer body
    a = ans_part.strip()
    if a.lower().startswith("answer:"):
        a = a[len("answer:"):].strip()
    answer = a
    # parse citations
    import re
    for line in cite_part.splitlines():
        line = line.strip()
        if not line:
            continue
        # match  "1. [session 8] "quote"" or  "[session 8] quote"
        m = re.match(r"^[-\d\.\)\*\s]*\[\s*session\s*([0-9?]+)\s*\]\s*[:\-]?\s*(.*)$", line, re.IGNORECASE)
        if not m:
            continue
        sid_raw = m.group(1)
        try:
            sid = int(sid_raw)
        except Exception:
            sid = -1
        quote = m.group(2).strip().strip('"').strip("'").strip()
        if quote:
            citations.append((sid, quote))
    return answer, citations


# ---------------------------------------------------------------------------
# Cashew retrieval + answer
# ---------------------------------------------------------------------------

def cashew_retrieve_and_answer(question: str, work_db: Path) -> Dict[str, Any]:
    from cashew_adapter import retrieve, claude_p
    ctx, retr_lat = retrieve(str(work_db), question)
    prompt = CITED_ANSWER_PROMPT.format(ctx=ctx[:60000], q=question)
    t0 = time.time()
    text, usage = claude_p(prompt, model=ANSWER_MODEL)
    ans_lat = time.time() - t0
    ans, cites = parse_cited_answer(text)
    return {
        "raw": text,
        "answer": ans,
        "citations": [{"session": s, "quote": q} for s, q in cites],
        "ctx_chars": len(ctx),
        "retrieval_latency_s": retr_lat,
        "answer_latency_s": ans_lat,
        "usage": usage,
    }


# ---------------------------------------------------------------------------
# Mem0 retrieval + answer
# ---------------------------------------------------------------------------

def mem0_retrieve_and_answer(question: str) -> Dict[str, Any]:
    from mem0_adapter import init_mem0, retrieve as mem0_retrieve
    from cashew_adapter import claude_p
    mem = init_mem0(MEM0_STORE_ROOT, MEM0_CONV_KEY, model=ANSWER_MODEL)
    ctx, retr_lat, n_hits = mem0_retrieve(mem, MEM0_CONV_KEY, question, limit=30)
    prompt = CITED_ANSWER_PROMPT.format(ctx=ctx[:60000], q=question)
    t0 = time.time()
    text, usage = claude_p(prompt, model=ANSWER_MODEL)
    ans_lat = time.time() - t0
    ans, cites = parse_cited_answer(text)
    return {
        "raw": text,
        "answer": ans,
        "citations": [{"session": s, "quote": q} for s, q in cites],
        "ctx_chars": len(ctx),
        "n_hits": n_hits,
        "retrieval_latency_s": retr_lat,
        "answer_latency_s": ans_lat,
        "usage": usage,
    }


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

def judge_citation(transcripts: Dict[int, str], sid: int, quote: str, full_transcript: str) -> Dict[str, Any]:
    """Judge a claim against the transcript.

    Cashew/Mem0 retrieved memory does not carry session-of-origin metadata, so
    cited session numbers are typically inferred or "?". To avoid penalizing
    both systems for that missing metadata (the methodology question is whether
    the claim is GROUNDED, not whether the system labelled the right session),
    we judge against the FULL conversation transcript. If the model emitted a
    plausible session number we still pass that session FIRST as a
    higher-priority fragment, then the rest, but we do not gate on session
    correctness.
    """
    from cashew_adapter import claude_p
    if sid in transcripts:
        # Put the cited session first, then the rest, truncated.
        head = transcripts[sid]
        rest = "\n\n".join(t for k, t in sorted(transcripts.items()) if k != sid)
        t_full = (head + "\n\n" + rest)[:80000]
    else:
        t_full = full_transcript[:80000]
    prompt = JUDGE_PROMPT.format(transcript=t_full, claim=quote)
    text, usage = claude_p(prompt, model=JUDGE_MODEL)
    text_s = (text or "").strip()
    first = text_s.split("\n", 1)[0].strip().lower()
    verdict = "yes" if first.startswith("y") else ("no" if first.startswith("n") else "unclear")
    reason = text_s.split("\n", 1)[1].strip() if "\n" in text_s else ""
    return {"verdict": verdict, "reason": reason, "raw": text_s, "judge_called": True, "usage": usage}


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def load_jsonl(p: Path) -> List[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def append_jsonl(p: Path, rec: dict) -> None:
    with p.open("a") as f:
        f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    sample = load_conv_sample()
    transcripts = session_transcripts(sample)
    full_transcript = "\n\n".join(transcripts[k] for k in sorted(transcripts))
    print(f"loaded conv-26: {len(transcripts)} sessions, {len(full_transcript)} chars")

    # cashew DB: copy to a working location so we don't touch the cron's DB
    work_db = OUT_DIR / "conv-26-B.workcopy.db"
    if not work_db.exists():
        if not CASHEW_SRC_DB.exists():
            raise SystemExit(f"cashew db missing: {CASHEW_SRC_DB}")
        shutil.copy(CASHEW_SRC_DB, work_db)
        print(f"copied {CASHEW_SRC_DB} -> {work_db}")

    # phase 1: answers
    done_keys = {(r["q_id"], r["system"]) for r in load_jsonl(ANSWERS_PATH)}
    for q in DEPTH_QUESTIONS:
        for system in ("cashew", "mem0"):
            if (q["id"], system) in done_keys:
                continue
            t0 = time.time()
            try:
                if system == "cashew":
                    out = cashew_retrieve_and_answer(q["text"], work_db)
                else:
                    out = mem0_retrieve_and_answer(q["text"])
            except Exception as e:
                print(f"[{q['id']} {system}] FAIL: {e!r}")
                traceback.print_exc()
                out = {"error": repr(e), "answer": "", "citations": []}
            rec = {"q_id": q["id"], "question": q["text"], "system": system, "wall_s": time.time() - t0, **out}
            append_jsonl(ANSWERS_PATH, rec)
            n_cit = len(out.get("citations", []))
            print(f"[{q['id']} {system}] cites={n_cit} chars={out.get('ctx_chars','?')} wall={time.time()-t0:.1f}s")

    # phase 2: judge each citation
    answers = load_jsonl(ANSWERS_PATH)
    judged = load_jsonl(JUDGEMENTS_PATH)
    judged_keys = {(r["q_id"], r["system"], r["cite_idx"]) for r in judged}
    for r in answers:
        for i, c in enumerate(r.get("citations", [])):
            key = (r["q_id"], r["system"], i)
            if key in judged_keys:
                continue
            try:
                v = judge_citation(transcripts, int(c["session"]), c["quote"], full_transcript)
            except Exception as e:
                v = {"verdict": "error", "reason": repr(e), "judge_called": False}
            rec = {
                "q_id": r["q_id"],
                "system": r["system"],
                "cite_idx": i,
                "session": c["session"],
                "quote": c["quote"],
                **v,
            }
            append_jsonl(JUDGEMENTS_PATH, rec)
            print(f"[judge {r['q_id']} {r['system']} c{i}] sess={c['session']} -> {v.get('verdict')}")

    # phase 3: aggregate
    judged = load_jsonl(JUDGEMENTS_PATH)
    answers = load_jsonl(ANSWERS_PATH)
    by_qsys: Dict[Tuple[str, str], Dict[str, int]] = {}
    for r in judged:
        key = (r["q_id"], r["system"])
        d = by_qsys.setdefault(key, {"cites": 0, "verified": 0})
        d["cites"] += 1
        if r.get("verdict") == "yes":
            d["verified"] += 1
    # Per-system aggregate
    sys_agg: Dict[str, Dict[str, Any]] = {}
    for sysname in ("cashew", "mem0"):
        per_q_prec: List[float] = []
        total_cites = 0
        total_verified = 0
        for q in DEPTH_QUESTIONS:
            d = by_qsys.get((q["id"], sysname), {"cites": 0, "verified": 0})
            total_cites += d["cites"]
            total_verified += d["verified"]
            if d["cites"] > 0:
                per_q_prec.append(d["verified"] / d["cites"])
            else:
                per_q_prec.append(0.0)
        sys_agg[sysname] = {
            "total_citations": total_cites,
            "total_verified": total_verified,
            "micro_precision": (total_verified / total_cites) if total_cites else 0.0,
            "macro_precision": sum(per_q_prec) / len(per_q_prec) if per_q_prec else 0.0,
            "questions_above_0_8": sum(1 for p in per_q_prec if p > 0.8),
            "per_q_precision": dict(zip([q["id"] for q in DEPTH_QUESTIONS], per_q_prec)),
        }

    # Build report
    write_report(answers, judged, by_qsys, sys_agg)
    print("DONE. Report at", REPORT_PATH)
    print(json.dumps(sys_agg, indent=2))


def write_report(answers, judgements, by_qsys, sys_agg) -> None:
    ans_by_key = {(r["q_id"], r["system"]): r for r in answers}
    judge_by_key: Dict[Tuple[str, str], List[dict]] = {}
    for j in judgements:
        judge_by_key.setdefault((j["q_id"], j["system"]), []).append(j)

    lines: List[str] = []
    L = lines.append
    L("# DEPTH-BENCH-V1 — cashew vs Mem0 on conv-26 (synthesis questions)\n")
    L("Author: depth-bench-v1 (locomo-cashew team).  ")
    L("Methodology: 10 hand-written depth questions; both systems retrieve and answer with REQUIRED citations; an LLM judge grades each citation against the actual session transcript (provenance, not subjective quality).\n")
    L("- conv: conv-26 (Caroline + Melanie, 19 sessions)")
    L(f"- cashew DB: `papers/locomo-run/dbs/conv-26-B.db` (sleep+think applied, MiniLM embeddings) — chosen because no DB has BOTH gte-large embeddings AND think cycles; documented per task.")
    L(f"- mem0 store: `papers/locomo-run/mem0-stores/conv-conv-26/` (sonnet backbone)")
    L(f"- answer model: `{ANSWER_MODEL}` ; judge model: `{JUDGE_MODEL}`")
    L(f"- precision = verified_citations / emitted_citations (no recall — no gold reference)")
    L("")
    L("## Aggregate precision\n")
    L("| system | total citations | verified | micro precision | macro precision (mean per-Q) | Qs with prec > 0.8 |")
    L("|---|---:|---:|---:|---:|---:|")
    for sysname in ("cashew", "mem0"):
        a = sys_agg[sysname]
        L(f"| **{sysname}** | {a['total_citations']} | {a['total_verified']} | {a['micro_precision']:.3f} | {a['macro_precision']:.3f} | {a['questions_above_0_8']} / {len(DEPTH_QUESTIONS)} |")
    L("")
    L("## The 10 depth questions\n")
    for q in DEPTH_QUESTIONS:
        L(f"- **{q['id']}** — {q['text']}")
    L("")
    L("## Per-question side-by-side\n")
    for q in DEPTH_QUESTIONS:
        L(f"### {q['id']}: {q['text']}\n")
        for sysname in ("cashew", "mem0"):
            r = ans_by_key.get((q["id"], sysname), {})
            judges = judge_by_key.get((q["id"], sysname), [])
            d = by_qsys.get((q["id"], sysname), {"cites": 0, "verified": 0})
            prec = (d["verified"] / d["cites"]) if d["cites"] else 0.0
            L(f"**{sysname}** — citations: {d['cites']}, verified: {d['verified']}, precision: {prec:.2f}")
            ans_text = (r.get("answer") or "").strip().replace("\n", " ")
            L("")
            L(f"> {ans_text[:1200]}")
            L("")
            if judges:
                L("Citations:")
                for j in sorted(judges, key=lambda x: x["cite_idx"]):
                    mark = "VERIFIED" if j.get("verdict") == "yes" else ("UNVERIFIED" if j.get("verdict") == "no" else f"({j.get('verdict')})")
                    L(f"- [session {j['session']}] [{mark}] \"{j['quote'][:200]}\" — judge: {j.get('reason','')[:200]}")
                L("")
        L("")
    L("## Qualitative observations\n")
    L("(See per-question side-by-side above. Below, narrative patterns visible after running.)\n")
    L("- _filled in by hand after the run_\n")
    L("\n## Verdict\n- _filled in by hand after the run_\n")
    REPORT_PATH.write_text("\n".join(lines))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("FATAL\n" + traceback.format_exc())
        sys.exit(1)
