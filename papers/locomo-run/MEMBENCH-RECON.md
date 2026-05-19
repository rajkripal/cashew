# MemBench Recon

**Date:** 2026-05-07
**Scope:** Decide whether MemBench (arXiv:2506.21605) + AttrScore (arXiv:2305.06311) / AIS (arXiv:2112.12870) is a drop-in eval for cashew's "think-cycle insight beats flat retrieval on synthesis" claim, or whether we need our own bench.

## TL;DR

**Verdict: EXTEND.** MemBench's reflective track is the right *shape* of question (preference/emotion synthesis from multi-session dialogue) and is the closest published precedent, but three blockers prevent a clean "just run MemBench + AttrScore" plan:

1. **No pluggable memory API.** MemBench is implemented against 7 specific memory mechanisms (FullMemory, RetrievalMemory, RecentMemory, GenerativeAgent, MemoryBank, MemGPT, SCM) on a Qwen2.5-7B base. There is no documented `Memory` ABC with `ingest/retrieve/answer` that an external system plugs into. Wiring cashew + Mem0 means writing the adapter layer ourselves.
2. **Reflective questions are multiple-choice with accuracy as the metric.** Synthesis answers are scored by MCQ accuracy, not free-form generation. This eliminates the need for attribution scoring at all on MemBench-as-shipped, and also kills our citation-faithfulness story on this benchmark — there is nothing to cite, only an option to pick.
3. **Reflective subtypes are narrow:** preference-based and emotion-based only. Our cashew claim is broader synthesis ("inner process around career risk", cross-domain pattern recognition). MemBench's reflective track maps to a *subset* of what we want to probe, not the whole.

AttrScore + AIS are pure methodology, freely portable to any free-form generation eval. They are not bound to MemBench. So the methodology gap is on MemBench, not on the attribution side.

## Per-track summary (MemBench v1, ACL Findings 2025)

Source corpus: 500 user relation graphs, multi-turn dialogues synthesized from MovieLens / Food / Goodreads recommendation datasets + a news dataset. Entirely synthetic. Public, MIT-licensed, downloadable via Google Drive / Baidu Cloud at https://github.com/import-myself/Membench.

| Track | Size | What it tests | Question style |
|---|---|---|---|
| PS-FM (Participation Factual) | ~39k Q / 8k sessions | Single-/multi-hop, comparative, aggregative, post-processing, knowledge-updating, single-/multi-session-assistant. Agent is 1st-person participant. | MCQ |
| PS-RM (**Participation Reflective**) | **~3.5k Q / 3.5k sessions** | Inferring user preferences (movies/food/books) and emotion states from dialogue history. **This is the synthesis track.** | MCQ |
| OS-FM (Observation Factual) | ~8.5k Q / 8.5k sessions | Same factual subtypes but agent observes 3rd-party messages. | MCQ |
| OS-RM (Observation Reflective) | ~2k Q / 2k sessions | Reflective inference in observer mode. | MCQ |

Reflective questions are LLM-generated (GPT-4o-mini) by mapping low-level item ratings to high-level taste/emotion summaries. So they are synthetic-but-grounded: there *is* a deterministic "ground-truth preference" derived from the rating pairs, which is why MCQ works.

Metrics shipped: accuracy (MCQ), Recall@10 (key-evidence retrieval), capacity (accuracy degradation as memory grows), temporal efficiency (read/write seconds). **No attribution / citation faithfulness metric.**

## AttrScore + AIS

- **AttrScore (Yue 2023, arXiv:2305.06311).** Methodology + dataset (AttrEval-Simulation, AttrEval-GenSearch, 12 domains from New Bing). The methodology — LLM judge classifies each statement-citation pair as attributable / extrapolatory / contradictory — is fully portable. Code, FLAN-T5-Large checkpoints on HuggingFace (osunlp/AttrScore). Drop-in if (a) our system emits citations and (b) we have a source corpus to verify against.
- **AIS (Rashkin 2023, arXiv:2112.12870).** Pure framework. Defines "according to source P, statement s holds" with a 2-stage human annotation pipeline. Methodology only; no required dataset. Adopt freely.

Conclusion on attribution: not the bottleneck. Both are unencumbered. The bottleneck is whether the underlying QA dataset elicits free-form, citation-bearing answers in the first place — MemBench-as-shipped does not.

## Has anyone run cashew-shaped (graph + think-cycle) systems on MemBench?

Not found in search. The seven baselines in the paper are all flat or summarization-style memories. A-MEM (arXiv:2502.12110) is the closest graph-style published, but its evaluation is on its own QA, not MemBench. No leaderboard exists. We would be the first graph-think-cycle system on this bench.

## Public availability

- Paper: arXiv:2506.21605, ACL Findings 2025. Open access.
- Code + data: https://github.com/import-myself/Membench, MIT license. Dataset on Google Drive / Baidu Cloud, no gating.
- Hardware: Qwen2.5-7B baseline ran locally; the bench itself is data + harness, no required GPU for our use case if we use API-based answer generation.

## Path options

| Option | Effort | What we get | What we lose |
|---|---|---|---|
| (a) Just run MemBench reflective | ~2 days adapter + 1 day eval | Direct comparison cashew vs Mem0 vs MemGPT etc. on PS-RM/OS-RM (5.5k MCQ) | No attribution story; MCQ collapses synthesis quality to a single bit; preference/emotion only |
| (b) MemBench + add AttrScore | ~1 week | (a) plus citation-faithfulness scoring on a free-form variant we generate ourselves | Have to author the free-form variant — at that point it's our bench in disguise |
| (c) MemBench reflective + own ~50 synthesis Qs | ~1 week | MCQ comparison on shared ground + free-form synthesis comparison with attribution. **Strongest paper.** | Two evals to maintain; reviewers may push back on the small custom set |
| (d) Skip MemBench, design own | 2-3 weeks | Full control over question shape | Loses positioning vs an established bench; reviewer ammo: "why not just MemBench?" |

## Estimated effort to wire cashew + Mem0 into MemBench

- Read the harness source and write a `MemorySystem` shim that maps MemBench's per-session message list → cashew's ingest, MemBench's question → cashew context retrieval → answer-with-MCQ-letter prompt: **8-12 hours.**
- Same shim for Mem0 (we already have it from LoCoMo work, mostly translation): **2-4 hours.**
- Run PS-RM + OS-RM end-to-end (~5.5k Q × 2 systems × ~$0.005/Q on Sonnet) ≈ **$55 + ~6 wall-clock hours** with current rate-limit-aware cron.
- Total to a first reflective-track number: **~2 days work + ~1 day compute.**

## Blockers

None hard. Soft:
- MCQ format means the cashew think-cycle's actual synthesis quality is invisible — the system wins or loses on whether retrieval surfaces the right preference cluster. That's still a real signal but it's not the synthesis-judging story.
- Preference/emotion narrowness means MemBench reflective doesn't probe the full surface we care about (e.g. behavioral patterns, decision-process synthesis).
- Synthetic dialogues from rec-system pairs are not human-authored; cashew's extraction prompts may behave differently on this distribution than on real conversation logs.

## Recommendation

Option (c). Run MemBench PS-RM + OS-RM as the externally-credible anchor (cashew vs Mem0 head-to-head on a published bench, no methodology novelty needed there), and supplement with ~50 hand-authored free-form synthesis questions where AttrScore-style citation faithfulness is the metric. The MemBench number defends "we tested on the standard bench and won/tied." The custom set defends "and on the synthesis questions MemBench can't ask, here's what citation-grounded judging shows."

This matches the NOVELTY-CHECK conclusion: methodology is recombination, not novel; the contribution is the application + the empirical claim that retrieval-strong systems score differently under provenance-anchored synthesis judging.

## Citations

- MemBench: Wei et al., arXiv:2506.21605 (ACL Findings 2025).
- AttrScore: Yue et al., arXiv:2305.06311 (EMNLP Findings 2023).
- AIS: Rashkin et al., arXiv:2112.12870 (Computational Linguistics 2023).
- Bohnet et al., arXiv:2212.08037 (attributed QA).
- Liu et al., arXiv:2304.09848 (citation precision/recall).
- A-MEM: arXiv:2502.12110.
- LoCoMo: arXiv:2402.17753.
