# Extraction-prompt fix — LoCoMo adapter framing

**Author:** phase3-fixer
**Scope:** `benchmarks/locomo/cashew_adapter.py::session_to_text`
**Cashew core prompt: NOT changed.** This is a benchmark-adapter wrapper, not a cashew prompt edit.

## Problem (from SMOKE.md)

Pre-fix smoke produced **25 zero-node events** across 19 sessions of conv-26. All had `raw_response = "[]"` — the extraction LLM was returning a syntactically valid empty array, not erroring. ~42% of sessions (8/19) silently produced no nodes.

Inspection of `core/session.py:731` confirmed the extraction prompt is tuned for engineering/decision content:
- Tells the LLM to extract "ONLY genuinely new, specific, **substantive** knowledge — not summaries or meta-comments"
- All examples are engineering oriented ("embeddings", "extraction triggered by context fullness")
- Has a binary `keep` gate: "should this node exist in the graph forever? If you would not want future-you to read it, set keep=false"
- Single-user framing (`get_user_domain()` is the only user)

LoCoMo conversations are casual two-friend chats about hobbies, food, family, travel, painting. By the prompt's own bar, the LLM correctly judged most of those exchanges as not "permanent-graph-worthy" and returned `[]`. **This is the prompt working as intended for its target user (Raj's personal graph), not a cashew bug.**

Editing `core/session.py` would alter behavior for every cashew user. The fix belongs at the benchmark-adapter boundary.

## Change

`session_to_text()` now prepends a benchmark-adapter preamble before the raw LoCoMo transcript. The preamble:

1. Names this as a "transcript of a chat between two friends" (not a personal note dump)
2. Explicitly grants permission for both speakers' facts to be extracted
3. Explicitly tells the LLM that "everyday-life details (food, hobbies, family, travel, work, health, relationships, emotions) ARE substantive when they are specific and standalone — set keep=true for them"
4. Asks for `speaker:<name>` tags so downstream retrieval can attribute facts

The cashew system prompt then reads: `<benchmark preamble> + DATE/SPEAKERS/turns`. The model now interprets "substantive" through the benchmark's lens, not the engineering-graph default.

### Sample preamble (conv-26)

> This is a transcript of a chat between two friends, Caroline and Melanie, on 1:14 pm on 25 May, 2023. Extract concrete observations, facts, plans, preferences, beliefs, events, and feelings about either speaker. Treat this as a personal journal of both people: everyday-life details (food, hobbies, family, travel, work, health, relationships, emotions) ARE substantive when they are specific and standalone — set keep=true for them. Tag each node with the relevant speaker's name (e.g. "speaker:caroline" or "speaker:melanie") plus topical tags. Use the user domain for facts about either speaker.

## Before / after — conv-26 (19 sessions)

| metric                            | before fix             | after fix              |
|-----------------------------------|------------------------|------------------------|
| sessions with 0 new nodes         | 8 / 19 (~42%)          | **0 / 19 (0%)**        |
| `extraction-debug.jsonl` entries  | 25 (all `raw_response="[]"`) | 0 (file not created) |
| total nodes in conv-26 DB         | 119                    | **192**                |
| total edges in conv-26 DB         | (not measured)         | 576                    |
| smoke F1 q0 (cat-2 date)          | 0.40                   | **1.00**               |
| smoke F1 q1 (cat-2 date)          | 0.00                   | 0.00 (different failure: gold=2022 not in conv) |
| smoke F1 q2 (cat-3)               | 0.17                   | 0.25                   |
| smoke F1 mean (3 questions, A)    | 0.19                   | **0.42**               |

Goal was "drop the 0-node rate from ~42% of sessions to <5%." **Achieved 0%** on the smoke conv. Will re-verify on larger N during the medium test.

## Files touched

- `/Users/bunny/.openclaw/workspace/benchmarks/locomo/cashew_adapter.py` — `session_to_text()` only.

No edits to cashew core, no edits to the runner.

## Caveats

1. **Tag format compliance is unverified.** The preamble asks for `speaker:<name>` tags but does not enforce them. If the LLM ignores that instruction, downstream speaker-attribution retrieval is unaffected (we don't query by speaker tag yet) but a future improvement is to verify and parse them.
2. **Conv-26 only.** The 0% empty-extract rate is on one conversation. Other LoCoMo convs may have different content profiles. The medium test (1 full conv across both variants, all questions) will surface any per-conv variance.
3. **Prompt size grew by ~700 chars.** Negligible vs the ~5,000-char average session prompt; no rate-limit impact observed.
