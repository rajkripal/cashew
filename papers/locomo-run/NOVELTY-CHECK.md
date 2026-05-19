# Novelty Check: cashew + provenance-anchored depth-question benchmark

Scope: pressure-test Raj's claim that our benchmark is novel along four axes. Method: WebSearch + WebFetch on the heaviest hits, abstract+method depth on the closest comparators. Time spent ~50min.

---

## Claim 1: Benchmark for proactive insight generation in agent memory

**What we want to claim.** A benchmark where the agent UNPROMPTED surfaces patterns/connections/themes from accumulated memory, and that act of synthesis is what gets scored — not "did the agent answer this question."

**Prior art surveyed:**

- **InsightBench** (arXiv:2407.06423) — evaluates "multi-step insight generation" but is for **business analytics agents over tabular data**. Agent is prompted with a goal (analyze this dataset) and produces an insight summary. Adjacent but different domain (BI on dataframes, not personal/episodic memory) and the agent is still prompted.
- **MemBench** (arXiv:2506.21605, ACL Findings 2025) — closest in spirit. Splits memory into *factual* (explicit) vs *reflective* (implicit, e.g. taste preferences inferred from behavior). Includes "reflective summarization" task. **However:** the agent is still prompted with an explicit query ("what is the user's preference for X?"). Not unprompted surfacing.
- **A-MEM** (arXiv:2502.12110) — its "memory evolution" mechanism *does* generate higher-order patterns autonomously (Zettelkasten-style link generation), but the paper evaluates downstream QA, not the insights themselves. The synthesis is a means, not the thing measured.
- **Generative Agents** (Park et al. 2023, UIST) — reflections are generated proactively. Evaluation is human believability ratings via interview; ablation isolates reflection's contribution. Not a benchmark in the leaderboard sense and not provenance-anchored.
- **Hindsight Agent Memory Benchmark Manifesto** (Mar 2026) — explicitly identifies the gap that current benchmarks (LoCoMo, LongMemEval) are stuck on retrieval QA, but does NOT itself propose insight-generation benchmarking. They want agentic-decision and scale dimensions instead.
- **LongMemEval** — five tasks, all reactive QA. LLM-judge for semantic correctness, no provenance grounding, no synthesis-of-self questions.

**Verdict: EXTENSION, not pure novel.** The conceptual seed (reflective memory, pattern emergence) is in MemBench, A-MEM, and Generative Agents. What we add: (a) the *thing being scored* is the pattern itself, not downstream QA performance, and (b) we anchor it on provenance to make the score deterministic. That second piece is what's actually fresh — see Claim 2.

**Must-cite to position honestly:** MemBench, A-MEM, Generative Agents (Park 2023), Hindsight manifesto, InsightBench.

---

## Claim 2: Provenance-anchored / source-citation-grounded LLM judging for synthesis questions

**What we want to claim.** Instead of asking the LLM judge "is this answer good?" (subjective), we ask "is this cited fact actually present in the source memory?" (deterministic). Score = citation precision/recall on provenance trail.

**Prior art surveyed (read carefully):**

- **Rashkin et al., "Measuring Attribution in NLG Models"** (arXiv:2112.12870, Computational Linguistics 2023) — defines AIS ("Attributable to Identified Sources"). Frames attribution as "according to source P, statement s holds." This is the foundational framework. Two-stage human annotation pipeline.
- **Bohnet et al., "Attributed Question Answering"** (arXiv:2212.08037) — operationalizes AIS for open-domain QA. Each generated answer must come with a passage citation; AIS-rated by humans, automated metric correlated. **This is essentially the move we're making, applied to open-domain factoid QA.**
- **Liu, Zhang, Liang, "Evaluating Verifiability in Generative Search Engines"** (arXiv:2304.09848, EMNLP Findings 2023) — defines **citation precision** (every citation supports its statement) and **citation recall** (every statement is supported). Audits Bing Chat / Perplexity / etc. **Exactly the metric framework we want.** Method is human eval, but the metric is the precedent.
- **Yue et al., "Automatic Evaluation of Attribution by LLMs"** (arXiv:2305.06311, AttrScore, EMNLP Findings 2023) — replaces humans with an LLM judge to classify each citation as attributable / extrapolatory / contradictory. **This is the LLM-as-judge-for-citation-faithfulness move, fully formed, in 2023.**
- **AttributionBench** (arXiv:2402.15089) — benchmarks how hard automatic attribution evaluation is. Confirms the framework is mature.
- In personal-memory / agent-memory benchmarks (LoCoMo, LongMemEval, MemBench, MemoryAgentBench): **none of them use provenance-anchored citation-faithfulness judging.** They use semantic-equivalence LLM judging on factoid answers.

**Verdict: NOT NOVEL as methodology, NOVEL as application.** The methodology — LLM judge scores citation faithfulness against retrieved sources, with precision/recall metrics — is fully developed by Bohnet 2022, Rashkin 2023, Liu 2023, Yue 2023. We are PORTING that methodology from open-domain QA / generative search to personal-memory synthesis questions.

This is honest framing. It is not table-stakes either, because **no one in the agent-memory benchmark line has done this port.** That's a real gap, just a smaller one than "novel methodology."

**Must-cite (load-bearing):** Rashkin 2023, Bohnet 2022, Liu 2023, Yue 2023. Failing to cite these would look ignorant or dishonest.

---

## Claim 3: Depth / synthesis questions in memory benchmarks

**What we want to claim.** Synthesis-style probes ("what's user X's inner process around career risk?") rather than factoid retrieval ("when did X visit Paris?").

**Prior art surveyed:**

- **LoCoMo** (arXiv:2402.17753) — five reasoning types: single-hop, multi-hop, temporal, commonsense, adversarial. Multi-hop is closest, but answers are still grounded factoids ("which event caused X"). Not "inner process" style.
- **MemBench** — reflective memory questions about preferences and emotional states. Closer to depth, but still single-axis ("what is X's preference").
- **EvolMem** (arXiv:2601.03543) — cognitive-psychology grounded, declarative + non-declarative memory split.
- **LongMemEval** — multi-session reasoning aggregates facts; not inner-process probes.
- **EPISODIC MEMORIES GENERATION AND EVALUATION** (ICLR 2025) — episodic memory eval, factual.
- **RESEARCHRUBRICS** (Scale, deep research benchmark) — rubric-based eval of long-form synthesis. Different domain (research reports) but methodologically adjacent: rubric items as anchored evaluation criteria.

**Verdict: EXTENSION.** MemBench's reflective memory is the closest precedent; we are pushing further toward genuinely abstractive "inner process" probes, but the category is not invented by us. The framing of "depth questions" as a question class is a useful contribution but should be positioned as an extension of MemBench's reflective memory axis, not as a wholly new question type.

---

## Claim 4: Closely-related prior systems — what do they benchmark?

| System | Insight generation? | Benchmarked on insights? |
|---|---|---|
| **Generative Agents** (Park 2023) | Yes (reflections proactively generated) | Human believability via interview; ablation. NOT a leaderboard. |
| **A-MEM** (2502.12110) | Yes (memory evolution, link generation) | Downstream QA performance only. Insights themselves not scored. |
| **Cognee Memify** | Yes (post-processing pipeline derives facts, prunes, reweights) | Benchmarked on HotPotQA multi-hop QA correctness. Not insight quality. |
| **MemGPT / Letta reflections** | Summarization-as-memory | Downstream task performance. |
| **Zep / Graphiti** | Temporal knowledge graph summarization | Mem0/Cognee comparison on QA correctness. |
| **SCM (Self-Controlled Memory)** | Summarization | QA. |
| **Hindsight** | Manifesto, no system+benchmark yet | N/A — explicitly calls for new benchmarks. |
| **MemBench** | Reflective memory questions | Reflective summarization is scored, but via prompted question, not unprompted surfacing. |

**Pattern:** Every system that *generates* insights evaluates itself via downstream QA, NOT via "are these insights correct/well-grounded." The honest gap.

---

## Synthesis: what's actually fresh

1. **Recombination.** The methodology (citation-faithfulness LLM judge: Bohnet/Rashkin/Liu/Yue 2022-2023) and the target (insight generation in agent memory: A-MEM, MemBench, Generative Agents) both exist. Combining them — using citation-grounded judging to score the insights an agent surfaces from personal memory — is the fresh move.
2. **The deterministic-judge angle for synthesis.** Synthesis QA has historically been judged subjectively because "is this insight good?" feels unfalsifiable. We're claiming you can make it falsifiable by demanding provenance and scoring citation precision. This framing is the strongest novel claim.
3. **A benchmark, not just a system.** Most adjacent work (A-MEM, Cognee) ships systems with QA benchmarks. We're shipping the benchmark itself.

## What is NOT fresh and we must NOT claim

- LLM-as-judge for citation faithfulness — Yue 2023 owns this.
- Citation precision / recall as evaluation metrics — Liu 2023 owns this.
- AIS framing — Rashkin 2023 owns this.
- Reflective / non-factoid memory questions — MemBench owns this.
- Proactive synthesis from memory — Generative Agents owns this.

## Per-claim verdicts

| Claim | Verdict |
|---|---|
| 1. Proactive insight benchmark | **EXTENSION** of MemBench/A-MEM/Park; what we add is scoring the insight itself, not downstream QA |
| 2. Provenance-anchored LLM judge | **RECOMBINATION** — methodology is Yue/Liu/Bohnet 2022-2023, application to personal-memory synthesis is novel |
| 3. Depth / synthesis questions | **EXTENSION** of MemBench reflective memory axis |
| 4. System landscape framing | **NOVEL OBSERVATION** — nobody benchmarks insight quality directly; we are |

## Most likely reviewer objection

> "Your evaluation methodology (LLM judge + citation precision) is Yue et al. 2023 / Liu et al. 2023 applied to a different dataset. Your synthesis-question class is MemBench's reflective memory with a new name. What's the contribution beyond porting?"

## Defensible answer to that objection

> The contribution is: (a) the first benchmark in the agent-memory line to score the *insights themselves* rather than downstream QA, (b) demonstrating that provenance-grounded judging — established for open-domain QA — can be ported to subjective-feeling synthesis tasks and made falsifiable, and (c) empirical evidence that systems which look strong on retrieval QA (Mem0, Zep) score very differently when judged on insight-citation faithfulness. Each piece individually is an extension; together they fill a gap the Hindsight manifesto explicitly named.

That answer holds up only if (c) is actually true in our results. Critical to validate before submission.
