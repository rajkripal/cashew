# MEM0-SMOKE — Mem0 x LoCoMo smoke gate

Date: 2026-05-08
Runner: mem0-runner

## Setup

- mem0ai 2.0.2 installed via `pip3 install --break-system-packages mem0ai`
- fastembed 0.8.0 for local ONNX embeddings (thenlper/gte-large, 1024d)
- Custom LLM provider `claude_p` registered in mem0's LlmFactory: shells out to `claude -p --model claude-sonnet-4-6` (same path cashew uses, so the backbone is identical between runs).
- Mem0's `mem0/llms/configs.py` provider whitelist patched to allow `"claude_p"` (one-line edit; redo if mem0ai is reinstalled).
- BM25 sparse encoder disabled (py_rust_stemmers segfaults under Python 3.14 ABI). Dense-vector recall only.

## Smoke command

```
cd /Users/bunny/.openclaw/workspace/benchmarks/locomo
KMP_DUPLICATE_LIB_OK=TRUE python3 run_mem0_locomo.py --convs conv-26 --smoke --reset
```

## Result

```
[2026-05-08 08:42:28] ingest conv-26 s1: 18 turns, 7.2s, events=6
[2026-05-08 08:42:36] ingest conv-26 s2: 17 turns, 8.4s, events=7
[2026-05-08 08:42:38] conv-26 q0 cat2 F1=1.00 EM=1 hits=13 pred='7 May 2023' gold='7 May 2023'
[2026-05-08 08:42:41] conv-26 q1 cat2 F1=1.00 EM=1 hits=13 pred='2022' gold='2022'
```

- Both gates green: retrieval returned 13 memories (non-empty), F1 was computed (not NaN), EM=1 on both.
- ~7-8s per session ingest (claude-sonnet-4-6 LLM extract).
- ~2-3s per question (retrieve + answer).

## Sonnet-as-LLM confirmation

Logged a sample LLM call from inside `ClaudePLLM.generate_response`:
- system prompt: "You are a Memory Extractor — a precise, evidence-bound processor responsible for extracting rich, contextual memories from conversations. Your sole operation is ADD..."
- model: claude-sonnet-4-6 (passed via `--model claude-sonnet-4-6` to `claude -p`)
- response: valid JSON with extracted facts (e.g. "User's name is Alice and she loves hiking in the Cascades mountains.")

Confirmed: Mem0's extract pipeline is being driven by claude-sonnet-4-6.

## Known wrinkles, captured for the writeup

1. Sonnet sometimes wraps JSON in markdown / commentary. ClaudePLLM strips ```fences and extracts the first {...} block before returning. Without that, Mem0's parser fails silently and ADD returns an empty results list.
2. spaCy `en_core_web_sm` was downloaded mid-run by mem0; first-run cost is ~5s. Subsequent runs reuse it.
3. fastembed downloads thenlper/gte-large ONNX (~600MB) on first init.
4. Mem0 search API in 2.0 requires `filters={"user_id": ...}` (top-level user_id is rejected). Adapter handles both shapes.

## Gate verdict: PASS

Proceeding to full conv-26 (199 questions).
