# cashew paper — outline / spec

Status: pre-decision. Author reads tomorrow morning, decides whether to commit.
Scope assumed: cashew as an open-source persistent thought-graph memory system for personal-context AI agents.

**Direction (May 2026, post-clarification):** paper-rigorous. Do not design a new benchmark before exhausting existing ones. Lit scan now documents the *benchmarks* each system used; a new section evaluates which existing benchmarks cashew should run against; the new-benchmark section must justify itself before proposing anything. Section ordering reflects this priority.

---

## 1. Lit scan summary

For each system, capture: (a) architectural claim, (b) **what benchmarks they actually evaluated against**, (c) gap cashew exploits.

### 1.1 Foundational / direct competitors

- **MemGPT / Letta** — Packer et al., arXiv:2310.08560 (Oct 2023).
  - What: OS-style virtual context management. Hierarchical tiers (main context vs external), function-call-driven paging in/out.
  - Benchmarks they ran: Deep Memory Retrieval (DMR, custom synthetic), document QA over long docs (custom), MSC.
  - Wins on: making a fixed context window behave like it's larger via swap.
  - Doesn't address: graph structure, decay, cross-domain synthesis. Memory is doc-shaped, not relational. No forgetting beyond eviction.

- **Zep / Graphiti** — Rasmussen et al., arXiv:2501.13956 (Jan 2025).
  - What: bitemporal knowledge graph for agents. Tracks event time T and ingestion time T' for every node/edge. Typed entities + relations.
  - Benchmarks they ran: DMR (beats MemGPT), **LongMemEval**. Claims +18.5% accuracy, -90% latency vs baselines.
  - Wins on: temporal correctness, fact supersession, structured business data ingest.
  - Doesn't address: forgetting as fitness-driven (Zep keeps bitemporal history; cashew prunes). Heavy edge-semantics commitment — opposite of cashew's bet.

- **Mem0** — Chhikara et al., arXiv:2504.19413 (Apr 2025, ECAI 2025).
  - What: extraction + retrieval framework. Single-pass ADD-only extraction, multi-signal retrieval (semantic + BM25 + entity).
  - Benchmarks they ran: **LoCoMo** (headline result, 26% LLM-as-Judge gain over OpenAI Memory; Mem0-graph adds ~2%). Also **LongMemEval** (93.4% per Apr 2026 update), and BEAM-1M (62%).
  - Comparison set used: MemGPT, LangMem, OpenAI Memory, RAG (varied chunk size + k), full-context, proprietary.
  - Wins on: token efficiency, latency, production deployability. Strong on benchmarks.
  - Doesn't address: graph traversal beyond entity links; consolidation/decay; the "personal evolving context" frame. ADD-only is explicitly the anti-pattern cashew targets.

- **A-MEM** — Xu et al., arXiv:2502.12110 (NeurIPS 2025).
  - What: Zettelkasten-inspired agentic memory. Each note auto-generates keywords/tags, gets dynamically linked, and triggers updates to related historical notes (memory evolution).
  - Benchmarks they ran: **LoCoMo** and **LongMemEval** across six foundation models.
  - Wins on: dynamic linking + memory evolution (closest spiritual cousin to cashew's sleep cycle).
  - Doesn't address: organic decay (notes accumulate, evolve, but don't fade by fitness). Edge semantics are richer than cashew's. No survival gate.

- **MemoryBank / SiliconFriend** — Zhong et al., arXiv:2305.10250 (AAAI 2024).
  - What: vector store over past dialogue turns + Ebbinghaus forgetting curve as decay heuristic.
  - Benchmarks they ran: simulated dialogues (ChatGPT-as-user) + qualitative real-user transcripts. **No public hard benchmark** — this is a weakness reviewers note.
  - Relevant to cashew: closest existing system that *explicitly* models forgetting. Cashew's claim against MemoryBank: structural decay over a graph beats per-item Ebbinghaus over a flat store.

- **Cognee** — topoteretes/cognee, no arXiv (mostly product + OSS).
  - What: extract-cognify-load pipeline. Builds a typed knowledge graph (subject-relation-object triplets) + embeddings. "Memify" step prunes stale nodes, reweights edges.
  - Benchmarks they ran: claims 92.5% vs RAG's 60% on undisclosed eval (treat skeptically).
  - Wins on: integration ergonomics, ontology layering, multi-store backend.
  - Doesn't address: published evaluation, decay theory. Heavy ontology stance — cashew rejects this.

- **Generative Agents (Park et al.)** — arXiv:2304.03442 (Apr 2023).
  - What: memory stream of timestamped natural-language observations + reflection (periodic LLM summarization into higher-level insights) + retrieval scored by recency × importance × relevance.
  - Benchmarks they ran: human-believability evals in Smallville sandbox, not retrieval metrics.
  - Wins on: emergent behavior at the simulation level; established the recency/importance/relevance triple.
  - Doesn't address: scale, real personal context, real workload; flat stream, no graph; "importance" is LLM self-rated (cashew killed this exact mechanism in PR #25).

### 1.2 Recent academic work (2025–2026)

- **Memory in the Age of AI Agents (survey)** — arXiv:2512.13564 (Dec 2025). Taxonomy across forms (factual/experiential/working; token/parametric/latent), dynamics (formation/evolution/retrieval), and benchmarks. Useful framing for positioning cashew as token-level + factual + experiential, with explicit forgetting dynamics.
- **Memory in LLMs: Mechanisms, Evaluation and Evolution (survey)** — arXiv:2509.18868. Complementary survey.
- **SCM: Sleep-Consolidated Memory** — arXiv:2604.20943. NREM-style consolidation + REM-style novel-association generation for LLMs. Direct prior art for cashew's "dream nodes."
- **Learning to Forget (sleep-inspired consolidation)** — arXiv:2603.14517. Sleep-cycle approach to proactive interference. Same metaphor cluster.
- **GAM: Hierarchical Graph-based Agentic Memory** — arXiv:2604.12285. Semantic-Event-Triggered encoding/consolidation split. Hierarchical (cashew is flat) — useful contrast.
- **AMA-Bench** — arXiv:2602.22769. Long-horizon agent memory benchmark proposal. Possible target benchmark or direct competitor benchmark.
- **MemoryAgentBench** — arXiv:2507.05257. Defines four core competencies: accurate retrieval, test-time learning, long-range understanding, **selective forgetting**. First benchmark we found that *rewards* forgetting rather than only penalizing recall failure.
- **MemoryBench** — arXiv:2510.17281. Memory + continual-learning framing.
- **MEMTRACK** — arXiv:2510.01353. Long-term memory tracking eval.
- **Beyond a Million Tokens** — arXiv:2510.27246. Benchmarking and enhancing long-term memory in LLMs.
- **LongMemEval** — Wu et al., arXiv:2410.10813 (ICLR 2025). 500 manually authored Qs over five abilities (info extraction, multi-session reasoning, temporal, knowledge updates, abstention). 115K → 1.5M tokens. **De facto standard for chat-assist memory.** Public code + data.
- **LoCoMo** — Maharana et al., arXiv:2402.17753 (Snap Research). 50 dialogues, ~19 sessions each, 300 turns / 9k tokens. Multi-hop, temporal, adversarial QA. **Standard for very-long-term conversational memory.** Public.
- **PerLTQA** — arXiv:2402.16288. 8,593 Qs, 30 personas, semantic + episodic memory.
- **DialSim / LongDialQA** — arXiv:2406.13144. Real-time simulator over Friends/TBBT/Office, 1,300+ sessions, 352K+ tokens, 6-second response budget, anonymized names.
- **MSC** — Xu et al., arXiv:2107.07567 (ACL 2022). Beyond Goldfish Memory. Persona-aware multi-session, ≤5 sessions. Foundational, small-horizon by today's standards.
- **BABILong** — arXiv:2406.10149 (NeurIPS 2024). bAbI 20-task reasoning embedded in PG19. Up to 50M tokens.
- **RULER** — arXiv:2404.06654 (NVIDIA). 13 synthetic tasks: NIAH variants, multi-hop tracing, aggregation, QA. Configurable length.
- **∞Bench / InfiniteBench** — arXiv:2402.13718 (ACL 2024). 12 tasks across retrieval/code/math/novels/dialogue, avg >100K tokens.
- **NIAH variants:** NoLiMa (arXiv:2502.05167), Sequential-NIAH (arXiv:2504.04713), NeedleChain (arXiv:2507.22411), DENIAHL (arXiv:2411.19360), U-NIAH (arXiv:2503.00353), MM-NIAH.
- **LightMem** — arXiv:2510.18866. Lightweight memory-augmented generation. Possible runnable baseline.

### 1.3 Product-track (note as products, not papers)

- **LangGraph + LangMem** — Memory primitives (Memory Manager extracts/updates/consolidates; cross-thread namespaced JSON store). Production tooling, no published eval.
- **LlamaIndex Memory** — short/long-term agent memory components. Production tooling.
- **Letta** — productized MemGPT. Same comments.

### 1.4 Cognitive-science prior art (for the decay claim)

- Ebbinghaus 1885 forgetting curve: R = e^(-t/S). Exponential decay is the canonical model.
- Wixted & Ebbesen 1991: power-law fits empirical forgetting better than exponential across datasets.
- ACT-R declarative memory: base-level activation = log of decayed-summed-presentations. Decay is a first-class architectural primitive in cognitive architectures since the 1990s.
- Implication: organic decay is not novel as a *concept*. Cashew's claim has to be operational — survival gate + fitness function over a graph signal — not "we invented forgetting."

### 1.5 Field-level read on benchmark choice

- **LoCoMo is the de-facto comparison surface** for memory architectures (Mem0, A-MEM, MemGuide, EvolMem all use it).
- **LongMemEval is the academic-rigor surface** (ICLR 2025 pedigree; Zep, Mem0, A-MEM all reported on it).
- **BABILong / RULER / ∞Bench / NIAH** are long-context-LLM benchmarks. Memory-architecture papers cite them but rarely lead with them — running cashew there would conflate "is the memory architecture good?" with "is the underlying LLM good at long context?"
- **MemoryBank** is the only major prior system that explicitly modeled forgetting, and it ran no hard benchmark — a gap cashew should fill rather than recreate.

---

## 2. Contribution gap (pressure-tested)

### Claim A: "Survival gate + pure-graph-signal fitness as the sole forgetting mechanism."
- Specifically: `access_count > 0 OR edge_degree > 0` as the deterministic gate, plus `branching_factor + 0.5 * cross_links + 0.1 * derivation_depth` for the rest.
- Pressure test: ACT-R has activation-based decay; A-MEM has memory evolution; SCM has explicit forgetting. None publish a survival gate this minimal *that survives ablation*. Most use LLM-rated importance (which cashew killed in PR #25 for the right reason: uncalibrated self-report).
- Defensible if: ablation shows the gate alone matches or beats LLM-importance scoring on retrieval-quality-at-T-days. Otherwise this is just "we picked simpler hyperparameters."
- Verdict: **plausibly novel as an empirical claim**, weak as a conceptual claim. Needs the ablation to land.

### Claim B: "Untyped edges + node-types-as-hints beats typed-edge ontologies for personal memory."
- Cashew docs claim this was ablation-tested. Verify before publication. If real, this is the strongest single contribution because it directly contradicts Zep, Cognee, A-MEM design choices.
- Pressure test: most "no edge semantics" arguments rest on "the LLM does the reasoning." This is true for retrieval-reasoning, but typed edges could still help retrieval *targeting*. The ablation needs to compare on the same retrieval task, not just on author's gut.
- Defensible if: head-to-head retrieval benchmark with cashew (untyped) vs cashew-with-typed-edges variant on same data. Even a small effect in cashew's favor is publishable because it inverts the field's default.
- Verdict: **strongest candidate contribution**. Reproducible, falsifiable, contrarian. Worth the paper alone.

### Claim C: "Sleep cycle / dream nodes for personal-context graphs."
- Pressure test: SCM (arXiv:2604.20943), Learning to Forget (2603.14517), GAM (2604.12285), and Cognee's Memify all do consolidation. Generative Agents do reflection. The metaphor is crowded.
- Defensible if: cashew's specific recipe (dedup at >0.82, cross-link at 0.7–0.85, sqrt(N) core promotion) is shown to outperform a no-sleep-cycle ablation on a long-horizon task. But "we have a sleep cycle" is not novel.
- Verdict: **weak as headline contribution**. Better framed as a component, not the contribution.

### Claim D: "Constant context cost decoupled from graph size."
- Empirical observation in PHILOSOPHY.md §10: 270–430 words/topic query independent of graph size.
- Pressure test: this is a property of bounded BFS retrieval, not a cashew-unique property. MemGPT and Mem0 also bound their retrieval output. The interesting version: *retrieval quality* stays bounded as N grows, not just token count. Needs the curve over N ∈ {100, 1k, 10k, 100k}.
- Verdict: **interesting if measured**. Reframe as "retrieval quality scales sublinearly with graph size on a personal-memory workload."

### Claim E (philosophy track): "Dumb graph, smart reasoning layer."
- Pressure test: this is mostly framing. Every retrieval-augmented system is "dumb storage + smart LLM" at some level. The substantive version is the ablation in Claim B.
- Verdict: **not a standalone contribution**. Use as positioning, not as the headline.

### Recommended headline contributions (rank-ordered):
1. Untyped edges + decay-by-graph-signal as a personal-memory architecture, validated by ablation against typed-edge and LLM-importance variants on existing long-horizon benchmarks.
2. A long-horizon *personal*-memory benchmark (vs the current dialogue/QA-shaped benchmarks), **only if §4 establishes that existing benchmarks are insufficient**.
3. Open-source single-SQLite reference implementation reproducing the result.

---

## 3. Benchmark survey: what to run cashew against (existing benchmarks first)

For each benchmark: what it measures, horizon, public availability, whether it stresses cashew's specific claims, rough effort to wire cashew into it.

### 3.1 LongMemEval (arXiv:2410.10813, ICLR 2025)
- **What:** 500 questions over scalable user-assistant chat histories, probing five abilities: information extraction, multi-session reasoning, temporal reasoning, knowledge updates, abstention.
- **Horizon:** 115K tokens (LongMemEval_S) up to 1.5M tokens (LongMemEval_M).
- **Availability:** Public code + data ([github.com/xiaowu0162/LongMemEval](https://github.com/xiaowu0162/LongMemEval)).
- **Stresses cashew?** Yes, strongly. Knowledge-updates and temporal-reasoning categories directly probe correction and supersession (cashew's "extend / correct / pivot"). Abstention probes whether forgetting is principled.
- **Effort to wire in:** Low. Their harness expects `insert(turn)` + `query(question)`; cashew has both via `cashew_context.py extract` and `context`.

### 3.2 LoCoMo (arXiv:2402.17753, Snap Research)
- **What:** very-long conversations (avg 300 turns, 9K tokens, up to 35 sessions). QA, event summarization, multimodal dialogue.
- **Horizon:** per-conversation; modest by token count, long by session count.
- **Availability:** Public ([github.com/snap-research/locomo](https://github.com/snap-research/locomo)).
- **Stresses cashew?** Multi-session coherence — yes. Organic forgetting — no, LoCoMo only rewards recall, doesn't penalize bloat.
- **Effort to wire in:** Low. De-facto comparison surface (Mem0, A-MEM all run it). Mandatory for comparability.

### 3.3 BABILong (arXiv:2406.10149, NeurIPS 2024)
- **What:** bAbI 20-task reasoning embedded in PG19. Fact chaining, induction, deduction, counting.
- **Horizon:** Up to 50M tokens.
- **Availability:** Public ([github.com/booydar/babilong](https://github.com/booydar/babilong); HF dataset RMT-team/babilong).
- **Stresses cashew?** Stresses long-context retrieval; less aligned with conversational/temporal claims. More long-context-LLM benchmark than memory-system benchmark.
- **Effort to wire in:** Medium. Background-as-memory shape is unnatural for cashew.

### 3.4 RULER (arXiv:2404.06654, NVIDIA)
- **What:** 13 synthetic tasks: NIAH variants, multi-hop tracing, aggregation, QA. Configurable length.
- **Availability:** Public ([github.com/NVIDIA/RULER](https://github.com/NVIDIA/RULER)).
- **Stresses cashew?** Not really. Long-context-LLM benchmark, not a memory-architecture test.
- **Effort to wire in:** Medium; probably the wrong fight.

### 3.5 ∞Bench / InfiniteBench (arXiv:2402.13718, ACL 2024)
- **What:** 12 tasks across retrieval/code/math/novels/dialogue. Avg >100K tokens.
- **Availability:** Public ([github.com/OpenBMB/InfiniteBench](https://github.com/OpenBMB/InfiniteBench)).
- **Stresses cashew?** Dialogue + Novel-QA subsets probe cross-reference resolution over long spans; rest is long-context-LLM territory.
- **Effort to wire in:** Medium. Selectively running the dialogue subset is more honest than full ∞Bench.

### 3.6 MSC — Multi-Session Chat (arXiv:2107.07567, ACL 2022)
- **What:** Persona-aware multi-session human-human dialogue.
- **Availability:** Public via ParlAI.
- **Stresses cashew?** Foundational but small-horizon (≤5 sessions). Useful sanity check, not a headline.
- **Effort to wire in:** Low.

### 3.7 NIAH variants — RULER, NoLiMa, NeedleChain, Sequential-NIAH, DENIAHL, U-NIAH
- NoLiMa (arXiv:2502.05167) is the most interesting: minimal lexical overlap between needle and query, forces latent association — closer to cashew's "reasoning layer matters" claim.
- Sequential-NIAH (arXiv:2504.04713): temporal/logical ordering of needles.
- For our purposes: long-context-LLM benchmarks. Note and skip.

### 3.8 PerLTQA (arXiv:2402.16288)
- **What:** 8,593 Qs across 30 personas, semantic + episodic. Profile, social, events, dialogues.
- **Stresses cashew?** Yes — explicitly tests memory classification + retrieval + synthesis on personalized data, matching cashew's deployment shape (one user, one brain).
- **Effort to wire in:** Low-medium. Public, but smaller traction than LongMemEval/LoCoMo.

### 3.9 DialSim / LongDialQA (arXiv:2406.13144)
- **What:** Real-time simulator over Friends/TBBT/Office scripts. 1,300+ sessions, 352K+ tokens, 6s response time limit, anonymized names.
- **Stresses cashew?** Yes — long-running, multi-party, anonymized (defeats prior-knowledge cheating), and adds a *latency* dimension that most memory benchmarks ignore. Cashew's graph queries are fast; this is a fair fight.
- **Effort to wire in:** Medium. Real-time harness integration.

### 3.10 MemoryAgentBench (arXiv:2507.05257) and MemoryBench (arXiv:2510.17281)
- **MemoryAgentBench** explicitly defines four competencies, including **selective forgetting** as a positively-rewarded capability. The first benchmark we found that rewards forgetting rather than only penalizing recall failure.
- **MemoryBench** broader continual-learning framing.
- **Stresses cashew?** Yes, most directly of any benchmark surveyed. Selective forgetting is exactly cashew's organic-decay claim.
- **Effort to wire in:** Medium. Newer, harness less battle-tested.

### 3.11 Coverage map
| Cashew claim | LongMemEval | LoCoMo | BABILong | RULER | ∞Bench | DialSim | PerLTQA | MemoryAgentBench |
|---|---|---|---|---|---|---|---|---|
| Long-horizon retrieval | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| Multi-session coherence | Yes | Yes | No | No | Partial | Yes | Yes | Yes |
| Knowledge update / correction | **Yes** | Partial | No | No | No | Partial | Partial | Partial |
| Temporal reasoning | Yes | Partial | Partial | Partial | No | Yes | Partial | Partial |
| Selective forgetting (rewarded) | Abstention only | No | No | No | No | No | No | **Yes** |
| Latency / throughput dimension | No | No | No | No | No | **Yes** | No | No |

### 3.12 Tiering — what to run, ranked by signal-to-effort

**Tier 1 — must run (high signal, low-medium effort):**
1. **LoCoMo.** Mandatory: de-facto comparison surface for Mem0, A-MEM, MemGuide. Without a LoCoMo number, cashew is not in the conversation.
2. **LongMemEval.** Highest coverage of cashew's claims (knowledge updates, temporal reasoning, abstention). ICLR 2025 pedigree.
3. **MemoryAgentBench.** The only benchmark that directly rewards selective forgetting — cashew's most distinctive claim.

**Tier 2 — run if Tier 1 results justify:**
4. **PerLTQA.** Personalized memory matches cashew's deployment shape. Good for ablations.
5. **DialSim.** Adds latency axis and anonymization defense. Useful if Tier 1 shows accuracy parity and we want to argue on a different axis.

**Tier 3 — probably skip, justify in paper:**
- **BABILong, RULER, ∞Bench, NIAH variants:** long-context-LLM benchmarks; would conflate cashew quality with backbone-LLM quality. Cite and skip with a one-paragraph justification.
- **MSC:** historical, too short. Cite as foundational, skip empirically.

---

## 4. Do we need a new benchmark? (Justify before designing)

Default answer: **no.** Tier 1 + Tier 2 above cover:
- Multi-session conversational recall (LoCoMo)
- Knowledge update / temporal reasoning / abstention (LongMemEval)
- Selective forgetting as a positively-rewarded signal (MemoryAgentBench)
- Personalization (PerLTQA)
- Latency under realistic constraints (DialSim)

The only credible gap is **organic-decay-as-emergent-forgetting.** MemoryAgentBench rewards selective forgetting but doesn't distinguish between *explicit* forgetting policies (Ebbinghaus heuristics, TTLs, MemoryBank-style) and *emergent* forgetting (cashew's claim that orphaned subgraphs naturally lose salience without an explicit eviction rule). It also doesn't measure *cross-domain* hit rate — cashew's BFS-walk claim — at long horizons over a single user's evolving personal graph.

A new benchmark must clear three bars before being proposed:
1. No combination of existing benchmarks measures the property.
2. The property is load-bearing for a claim in the paper.
3. The proposed benchmark is reproducible and fair to non-cashew systems.

If cashew can clear Tier 1 + Tier 2 with strong numbers, the new benchmark may not be necessary at all — the paper becomes "untyped-edges + decay wins on existing benchmarks," which is a cleaner story than "we won on the benchmark we made."

### 4.1 Conditional benchmark design (only if §3 lands and §4's three bars are cleared)

If after running Tier 1 + Tier 2 we still need to substantiate (a) the mechanism claim for emergent decay, or (b) the cross-domain BFS-walk claim, then a small targeted probe is justified. Even then, prefer extending MemoryAgentBench's forgetting suite or contributing a probe to PerLTQA over inventing a new benchmark wholesale.

If we *do* go custom, the design that would clear the bars:

#### 4.1.1 Task setup
- Streaming multi-session input over a synthetic 12-month timeline. Mix of: notes (declarative facts), conversations (extraction targets), corrections (supersession events), commitments (TODO lifecycle).
- Each "day" has 5–20 input items. 365 days simulated.
- Periodic probe queries: "what is X's status?", "what did I decide about Y?", "summarize the Z thread", "is W still true?". Probes are timestamped — the gold answer changes as the timeline progresses.

#### 4.1.2 Memory load
- Synthesis: 50% signal (referenced again later, gold-relevant), 50% noise (one-shot facts that should fade).
- Adversarial: corrections that supersede earlier facts. System must not return the stale fact after the correction.
- Drift: same entity discussed across months with shifting attributes. Tests cross-domain edge surfacing.

#### 4.1.3 Metrics
- **Retrieval precision@k / recall@k** at probe time.
- **Stale-fact rate**: fraction of probes returning a fact superseded ≥ N days ago. Lower is better. The forgetting-quality metric.
- **Context efficiency**: tokens returned per gold-relevant token.
- **Forgetting calibration**: for noise items inserted at day t, fraction still retrievable at day t + Δ. Plot as a curve. Compare to Ebbinghaus target. Flat = append-only failure. Too-steep = aggressive-decay failure.
- **Cross-domain hit rate**: probes constructed to require traversing across domain tags. Tests the BFS-graph-walk claim.
- **Wall-clock + token cost** at retrieval time as graph grows.

#### 4.1.4 Falsifiability — what kills cashew
- Append-only (Mem0-style) matches/beats cashew on stale-fact rate at 90+ days → decay does nothing useful. Cashew loses.
- Typed-edge variant beats untyped on cross-domain hit rate → central architectural bet loses.
- Retrieval quality degrades with N → BFS-walk claim loses.
- Sleep-cycle-disabled cashew matches sleep-cycle-on cashew → consolidation story is decorative.

#### 4.1.5 Scale
- Graph sizes: 100, 1k, 10k, 100k nodes. The 100k tier matters for the constant-context-cost claim. Author's real graph is ~3k; synthetic required for 10k+. Be honest about which sizes are synthetic vs real.

#### 4.1.6 Comparison set
- **Mem0** (arXiv:2504.19413), **A-MEM** (arXiv:2502.12110), **Zep / Graphiti** (arXiv:2501.13956), optionally **MemGPT**.
- Internal ablations: cashew-no-decay, cashew-typed-edges, cashew-no-sleep, cashew-LLM-importance.

#### 4.1.7 Honesty rules
- Specify and freeze before running cashew on it. Otherwise it's benchmarking-to-win. Pre-register metric thresholds in the repo.
- Use existing benchmarks (LongMemEval, LoCoMo, MemoryAgentBench) as a sanity floor. If cashew can't compete on those, the new benchmark looks like venue-shopping.

---

## 5. First experiments to run

Concrete, ranked by signal-to-effort. Each entry: what to run, what we expect, what counts as positive vs negative.

### Experiment 1 — LoCoMo, cashew vs Mem0 / A-MEM / MemGPT / RAG / full-context
- **Why first:** lowest effort (their harness exists), highest comparability (everyone has run it).
- **Setup:** wire `cashew_context.py extract` to LoCoMo's session-insert hook, `cashew_context.py context` to its query hook. Lock the backbone LLM (e.g., GPT-4o-mini and Claude Haiku) across all systems.
- **Expected:** parity-or-better on single-hop and open-domain; advantage on multi-hop where graph structure should help; possible disadvantage on temporal where Mem0's explicit timestamping is purpose-built.
- **Positive result:** within ~5 points of Mem0-graph on LLM-as-Judge, with materially lower token cost (cashew's decay should keep working set smaller).
- **Negative result:** >10-point gap on multi-hop. Falsifies "graph structure helps reasoning" — forces redesign.

### Experiment 2 — LongMemEval (S, then M)
- **Why second:** highest coverage of cashew's distinctive claims.
- **Setup:** same wiring as LoCoMo. Start with LongMemEval_S (115K) before LongMemEval_M (1.5M).
- **Expected:** strong on knowledge-update and abstention (decay supports both); moderate on multi-session reasoning; weakest on temporal unless we add explicit timestamp queries to the reasoning layer.
- **Positive result:** beats long-context baseline on knowledge-update by >10 points (long-context can't update); competitive with Mem0 on temporal.
- **Negative result:** loses to long-context on abstention. That would mean decay isn't actually helping the model say "I don't know" — cashew's most philosophically load-bearing claim.

### Experiment 3 — MemoryAgentBench, focused on the selective-forgetting subtask
- **Why third:** the only benchmark that directly probes the mechanism claim.
- **Setup:** cashew with decay enabled vs cashew with decay disabled (ablation). Compare against MemoryBank (Ebbinghaus heuristic) and a no-forgetting RAG baseline.
- **Expected:** cashew-with-decay matches or beats MemoryBank on forgetting-quality metrics, with significantly less engineering (no per-item importance scores, no explicit decay schedule).
- **Positive result:** decay-on > decay-off on forgetting metrics, within noise on retrieval metrics. The dream: forgetting helps without hurting recall.
- **Negative result:** decay-on hurts recall measurably. Forces us to either make decay smarter or reframe it as cost-saving rather than quality-improving.

### Experiment 4 (conditional) — PerLTQA + DialSim
- Run only if Experiments 1-3 land roughly as expected. PerLTQA gives us persona-coherence numbers; DialSim adds latency under a 6-second budget.

### Experiment 5 (conditional) — custom personal-memory benchmark
- Run only if §4's three bars are cleared by Experiments 1-4. Most likely shape: an extension to MemoryAgentBench's forgetting suite plus a cross-domain probe, not a wholesale new benchmark.

### What we are explicitly NOT running first
- BABILong / RULER / ∞Bench / NIAH variants: long-context-LLM benchmarks. Cite and explain why they don't distinguish memory architectures. Running them risks looking like benchmark-shopping.
- A custom benchmark: not until §4's three bars are cleared.

---

## 6. Realistic publication path

### Option 1 (recommended): COLM 2027 main track, system + ablation paper.
- COLM is the right venue: language-model-focused, friendly to systems/architecture work, 9-page limit suits a system + ablation paper.
- Realistic target: **COLM 2027** (deadlines typically late March 2027). Gives ~10 months. Tight but achievable if benchmark wiring (Experiment 1) starts in May 2026.
- Story: untyped-edges + organic-decay validated on LoCoMo + LongMemEval + MemoryAgentBench, with internal ablations. No new benchmark needed.

### Option 2: NeurIPS 2026 D&B track (Datasets and Benchmarks).
- Frame as "long-horizon personal-memory benchmark," with cashew as one entry among many.
- Pro: D&B accepts pure-benchmark papers.
- Con: less room for the architectural ablation story; only viable if §4 produces a defensible new benchmark.
- Deadline: typically May/June for NeurIPS 2026. Likely too tight given the new "exhaust existing benchmarks first" priority.

### Option 3: arXiv preprint as a position + early-results paper, then workshop.
- Target an ICLR 2027 workshop (MemAgents-style). Lower bar, faster publication, fine venue for "here is the bet, here is early evidence on Tier 1 benchmarks."
- Lowest cost. ~3–4 weeks from a frozen Experiment-1 result to a preprint.
- Recommended fallback if main-track timing slips.

### Estimated work to publishable
- Experiment 1 (LoCoMo wiring + run): 2 weeks.
- Experiment 2 (LongMemEval wiring + run): 1–2 weeks once Exp 1 harness exists.
- Experiment 3 (MemoryAgentBench + ablation): 2–3 weeks.
- Experiment 4 (PerLTQA + DialSim): 2–3 weeks if conditional triggers.
- Internal ablations (typed-edge, no-decay, no-sleep, LLM-importance): 2 weeks.
- Writing: 3–4 weeks for COLM main-track, 1–2 weeks for workshop.
- Total: ~10–12 weeks of focused effort for main-track if no new benchmark is built; ~16+ weeks if Experiment 5 fires.

---

## 7. Open questions for the author

1. **Scope.** Three viable shapes: (a) system paper anchored on the typed-vs-untyped ablation against existing benchmarks, (b) benchmark paper with cashew as one entry (only if §4 fires), (c) position paper on personal-memory architecture. New direction strongly favors (a). Decide before writing.
2. **Hyperparameter discipline.** Cashew's current thresholds (0.82 dedup, 0.7–0.85 cross-link, sqrt(N) core promotion, gc_threshold) were tuned on the author's personal graph. For honest benchmarking, freeze them before running on LoCoMo / LongMemEval / MemoryAgentBench. If retuning is needed, disclose and use a held-out set.
3. **IP / employer.** Author works at Meta. Verify open-source / personal-project clearance before submitting under personal name. Non-trivial: cashew has been in active personal use for 13+ months — confirm it doesn't intersect with Meta-internal work.
4. **Personal data.** The author's real graph contains private vault-tagged nodes. Any real-graph evaluation must be either (a) on a sanitized export, or (b) only reported as aggregate metrics, never as content snippets. **All headline numbers should be on public benchmarks**, not the personal graph.
5. **Reproducibility floor.** Decide whether cashew ships with a deterministic synthetic-graph generator so reviewers can reproduce without the author's personal data. Strongly recommended yes.
6. **Co-authorship.** Solo paper or invite collaborators (e.g., someone at Letta/Zep for benchmark fairness, or an academic for reviewer credibility)? Solo is faster and matches the "personal context" framing; co-authored is safer for review.
7. **Defensive framing.** "Personal memory for one human" is unusual scoping for an ML paper. Either lean into it as the contribution (most papers benchmark on dialogue datasets, not on a single human's evolving context) or broaden to general agent memory and lose the differentiation. Pick one; don't waffle.
8. **LLM-as-Judge fragility.** LoCoMo and LongMemEval both lean on judges. Report human-evaluated subsets where feasible.
9. **Cost / latency reporting.** Most memory papers don't; cashew's pitch is partially about being cheap. Report tokens-per-query and p95 latency uniformly across all experiments.

---

## Sources

- [MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560)
- [Zep (arXiv:2501.13956)](https://arxiv.org/abs/2501.13956)
- [Mem0 (arXiv:2504.19413)](https://arxiv.org/abs/2504.19413)
- [A-MEM (arXiv:2502.12110)](https://arxiv.org/abs/2502.12110)
- [MemoryBank (arXiv:2305.10250)](https://arxiv.org/abs/2305.10250)
- [Generative Agents (arXiv:2304.03442)](https://arxiv.org/abs/2304.03442)
- [Memory in the Age of AI Agents survey (arXiv:2512.13564)](https://arxiv.org/abs/2512.13564)
- [Memory in LLMs survey (arXiv:2509.18868)](https://arxiv.org/abs/2509.18868)
- [LongMemEval (arXiv:2410.10813)](https://arxiv.org/abs/2410.10813)
- [LoCoMo (arXiv:2402.17753)](https://arxiv.org/abs/2402.17753)
- [BABILong (arXiv:2406.10149)](https://arxiv.org/abs/2406.10149)
- [RULER (arXiv:2404.06654)](https://arxiv.org/abs/2404.06654)
- [InfiniteBench (arXiv:2402.13718)](https://arxiv.org/abs/2402.13718)
- [MSC (arXiv:2107.07567)](https://arxiv.org/abs/2107.07567)
- [PerLTQA (arXiv:2402.16288)](https://arxiv.org/abs/2402.16288)
- [DialSim (arXiv:2406.13144)](https://arxiv.org/abs/2406.13144)
- [MemoryAgentBench (arXiv:2507.05257)](https://arxiv.org/abs/2507.05257)
- [MemoryBench (arXiv:2510.17281)](https://arxiv.org/abs/2510.17281)
- [MEMTRACK (arXiv:2510.01353)](https://arxiv.org/abs/2510.01353)
- [Beyond a Million Tokens (arXiv:2510.27246)](https://arxiv.org/abs/2510.27246)
- [NoLiMa (arXiv:2502.05167)](https://arxiv.org/abs/2502.05167)
- [Sequential-NIAH (arXiv:2504.04713)](https://arxiv.org/abs/2504.04713)
- [SCM Sleep-Consolidated Memory (arXiv:2604.20943)](https://arxiv.org/abs/2604.20943)
- [Learning to Forget (arXiv:2603.14517)](https://arxiv.org/abs/2603.14517)
- [GAM Hierarchical Graph Agentic Memory (arXiv:2604.12285)](https://arxiv.org/abs/2604.12285)
- [AMA-Bench (arXiv:2602.22769)](https://arxiv.org/abs/2602.22769)
- [LightMem (arXiv:2510.18866)](https://arxiv.org/abs/2510.18866)
- [COLM 2026 dates](https://colmweb.org/dates.html)
- [Cognee](https://github.com/topoteretes/cognee)
- [LangMem](https://github.com/langchain-ai/langmem)
- [LongMemEval repo](https://github.com/xiaowu0162/LongMemEval)
- [LoCoMo repo](https://github.com/snap-research/locomo)
- [BABILong repo](https://github.com/booydar/babilong)
- [RULER repo](https://github.com/NVIDIA/RULER)
- [InfiniteBench repo](https://github.com/OpenBMB/InfiniteBench)
