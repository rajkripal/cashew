# LONG-BENCH-RECON — cashew x LifeBench / AMA-Bench

Recon brief for scoping cashew runs on two long-horizon memory benchmarks beyond LoCoMo. Read-only survey, no code cloned, no runs started.

## TL;DR

| Benchmark | Verdict | Notes |
|---|---|---|
| **LifeBench** | NEEDS WORK | Data downloadable, but eval interface is undocumented in the README; only 10 users so wall-clock is small; ND and TKU categories are a fundamentally different task than recall and would stress cashew in interesting ways. |
| **AMA-Bench** | RUNNABLE (synthetic subset) / NEEDS WORK (real subset) | Clean two-call API (`memory_construction` + `memory_retrieve`) — easiest adapter we've seen. Real subset has 996K-token Open-World-QA trajectories that will blow our extract budget. Synthetic subset (8K–128K) is the right first cut. Judge is Qwen3-32B via vLLM, which is a deviation from our Sonnet-judge LoCoMo baseline. |

**Recommended order: AMA-Bench synthetic subset first**, then LifeBench, then AMA-Bench real subset only if the first two succeed. AMA's API contract maps 1:1 to cashew's `extract_from_conversation` + `generate_session_context`, while LifeBench requires us to invent the integration shape.

## 1. LifeBench (arXiv 2603.03781, Mar 2026)

### Data layout

- Repo: `https://github.com/1754955896/LifeBench` (public, English + Chinese versions).
- Format: per-user **JSON files** in `life_bench_data/` — `persona.json`, `daily_event.json`, `location.json`, `daily_draft.json`, `phone_data/`, plus QA file.
- Scale: **10 users**, **1 year per user**, **~5,149 events/user (~14 events/day)**, **~3.66M tokens of context per user**, **8,046 health records per user**.
- QA: **2,003 questions total** across 5 categories:
  - Information Extraction (IE): 718 (35.85%)
  - Multi-hop Reasoning (MR): 597 (29.81%)
  - Temporal & Knowledge Updating (TKU): 229 (11.43%)
  - Non-declarative Memory Reasoning (ND): 429 (21.42%)
  - Unanswerable (UA): 30 (1.50%)
- Sources are heterogeneous: chats, SMS, calls, calendar, photos, notes, push notifications, contacts, fitness/health records. **Not a dialogue stream** — multi-modal digital trace.

### Evaluation interface

- **Judge**: GPT-5.1-Mini, "standard LoCoMo evaluation prompt" (so we already have the judge wrapper from our LoCoMo work, just swap the model).
- **Embedding (for the baselines they ran)**: `text-embedding-3-small`.
- **Metric**: proportion of questions judged correct by the LLM judge. Per-category accuracy is the reported breakdown.
- **External-memory plug-in shape**: NOT documented in the README. The repo only documents data synthesis (`run_all.py`) and config; it does not ship a turnkey adapter API. We'd have to read source to determine how MemOS/Hindsight/MemU were wired in. **This is the LifeBench blocker.**

### Reported baselines

| System | Overall |
|---|---|
| MemOS | **55.22%** (SOTA) |
| Hindsight | 40.99% |
| MemU | "significantly lower" (brief-summary failure mode) |

Per-category: MemOS dominates IE/MR/TKU; Hindsight wins ND. Compare to LoCoMo where Hindsight reaches ~90% — LifeBench is genuinely harder, and the gap is in the non-declarative inference categories rather than recall.

### Cashew-fit assessment

- **This is the most cashew-shaped benchmark we've found.** ND and TKU explicitly test inferring from fragmented signals and updating over time, which is closer to cashew's "pattern emergence from decay-managed graph" claim than LoCoMo's verbatim-recall categories.
- **But:** the data is multi-modal traces, not turn-based dialogue. Our `session_to_text` adapter logic from LoCoMo doesn't transfer. We'd need a per-source serializer (chat block, calendar entry, health record bucketed by week, etc.) and probably a daily-session ingest cadence to keep prompts under budget.
- **Scale per user**: 5,149 events × ~700 tokens average = ~3.66M tokens. If we ingest daily (~14 events/day, ~365 sessions/user) that's roughly the same per-conversation extract count as LoCoMo conv-26 had sessions, just 30× more conversations. Expected node count per user DB: ~10K–20K (vs LoCoMo's ~200/conv). **This is the scale stress test the lead asked for.**
- 10 users total → 10 DBs. Manageable parallelism.

### Estimated effort to run

- Adapter: **~6–10 hours** (multi-source serializer, day-session boundary logic, port LoCoMo judge prompt, figure out what MemOS/Hindsight do for plug-in by reading their commits if undocumented).
- Wall clock for full run: **~30–50 hours** Sonnet under our current rate limits if we ingest daily for 10 users (10 × 365 sessions × ~25s/extract). The QA pass (2,003 questions × retrieve+answer) is small in comparison: ~4 hours.
- Cheaper alternative: ingest at week boundaries, not daily — cuts ingest cost ~7×.

### Honest blockers

1. **Eval-plug-in shape is undocumented.** README is silent on how external memory systems integrate. We'd need to read MemOS/Hindsight integration code (or just write our own answer loop using their judge prompt — fine for our paper, weaker for direct comparison to their leaderboard).
2. **Multi-modal trace serializer is non-trivial.** Health records bucketed how? Photos = caption strings? Calendar entries vs SMS context. Real design work.
3. **No public leaderboard or eval harness shell scripts** visible from the README — only synthesis scripts.
4. Per-category numbers for MemOS/Hindsight are reported in the paper but the paper does not provide a CSV — capturing exact comparison numbers needs the LaTeX tables.

## 2. AMA-Bench (arXiv 2602.22769, Feb 2026)

### Data layout

- Repo: `https://github.com/AMA-Bench/AMA-Bench`. HF dataset: `AMA-bench/AMA-bench`.
- Format: **JSONL**. Test split is `open_end_qa_set.jsonl` (no train split).
- Two subsets:

**Real-world subset** (208 trajectories, 2,496 QA pairs):

| Domain | Trajectories | QA | Avg tokens | Max tokens |
|---|---|---|---|---|
| Web Task Execution | 31 | 372 | 34,265 | 166,260 |
| Open World Tool QA | 30 | 360 | **288,651** | **996,826** |
| Text-to-SQL (spider2) | 51 | 612 | 6,049 | 10,718 |
| Software Engineering (swebench) | 36 | 432 | — | — |
| Gaming | 30 | 360 | — | — |
| Embodied AI | 30 | 360 | — | — |
| **Overall** | 208 | 2,496 | **57,506** | — |

**Synthetic subset** (1,200 QA pairs):
- 5 horizon buckets: **8K, 16K, 32K, 64K, 128K tokens**.
- 240 QA per bucket.
- Environments: BabyAI (~30K tokens avg), TextWorld (~31.7K tokens avg). 50 trajectories on BabyAI side.

### Evaluation interface (this is the clean part)

Two-stage unified API for all methods:
```python
memory = memory_construction(traj_text, task="")
context = memory_retrieve(memory, question)
```

That maps **directly** to our existing cashew adapter:
- `memory_construction` → `extract_from_conversation` over the trajectory (we'd treat the whole trajectory as one or N sessions).
- `memory_retrieve` → `generate_session_context(hints=…)` — the same call our LoCoMo `retrieve()` already wraps.

This is the easiest adapter shape we've seen.

### Judge / metric

- **Judge LLM**: Qwen3-32B (via vLLM server, shell scripts provided).
- Reported judge fidelity: 92.67% accuracy vs human, 96.45% precision, 94.53% F1.
- Judge alternates referenced: GPT-5.2, Claude-4.6, DeepSeek-v3.2 — so we could substitute Sonnet for the judge if we cite the swap and expect a small offset.
- Metrics: **Accuracy + F1**.

### Reported baselines (Qwen3-32B backbone)

| Method | Real-world Acc |
|---|---|
| **AMA-Agent** | **0.5722** (their proposed) |
| MemoRAG | 0.4606 |
| HippoRAG2 | 0.4480 |
| Qwen3-Emb-4B | 0.4227 |
| BM25 | 0.3436 |

AMA-Agent breakdown:
- Recall 0.6238, Causal Inference 0.6145, State Updating 0.5305, State Abstraction 0.4719.

Total: **15 memory methods** evaluated. AMA-Agent's pitch is causality graph + tool-augmented retrieval — close enough in spirit to cashew that a head-to-head is meaningful.

### Cashew-fit assessment

- **AMA-Agent is the spiritual cousin of cashew** (causality graph + retrieval). Direct comparison is the most defensible scientific claim we can make from this benchmark.
- The State Updating + State Abstraction dimensions test pattern emergence — cashew's actual claim. Recall is just a control axis.
- **Scale**: synthetic 128K bucket → ~128K-token trajectories → ~500 nodes per DB at our LoCoMo extraction density. Real subset Open-World-QA → 996K tokens → could exceed **3K–5K nodes per DB**, which is past anything we've stressed. Good test of decay/sleep behavior.
- **Risk for cashew**: machine-generated trajectories are dense action/observation pairs, not natural language dialogue. Our extract prompt is tuned for conversation; we'd see noisy or low-yield extraction without a domain-specific extract template. **This is the AMA-Bench gotcha.**
- **Backbone mismatch**: their Qwen3-32B judge means our reported numbers are not directly stack-rankable on their leaderboard unless we run their judge. Sonnet judge is fine for our paper if we say so; not fine for "we beat AMA-Agent."

### Estimated effort to run

- Adapter: **~3–5 hours** (clean two-call API, JSONL parsing, domain-aware trajectory serializer).
- Wall clock, **synthetic subset only**: 1,200 QA × ~30s/end-to-end ≈ **10–15 hours** Sonnet, plus ingest ~6 hours. Manageable in two days.
- Wall clock, **full real subset**: 2,496 QA + 208 trajectories of which 30 are at 288K-token average. Ingest budget alone is **~40–80 hours** Sonnet. Skip until synthetic shows signal.
- vLLM judge: requires a CUDA box. We don't have one. Either (a) swap to Sonnet judge and document, (b) rent a GPU just for the judge pass, (c) skip the judge-fidelity claim.

### Honest blockers

1. **Qwen3-32B judge requires GPU + vLLM.** Mac Mini won't run it. Need a Sonnet-judge-with-disclaimer plan or a cloud GPU pass at the end.
2. **Real-subset Open-World-QA at 996K tokens** would overflow our per-extract prompt budget several times over. Needs trajectory chunking that we haven't designed. **Skip for v1.**
3. **Extract-prompt mismatch**: our LoCoMo extract prompt is dialogue-tuned. AMA trajectories are `(x, a, o)` tool-call sequences. Low-quality extraction is the #1 risk. Spike a 1-trajectory smoke before the full run.

## Cross-cutting recommendation

1. **Start with AMA-Bench synthetic subset (8K + 16K buckets first)**: smallest, cleanest API, fastest to validate cashew on tool-call-style trajectories. ~480 QA pairs, ~6 hours wall clock. Decisive go/no-go signal in one day.
2. **Then LifeBench**: bigger scientific upside (cashew's actual claim is pattern emergence, and ND + TKU categories test exactly that), but adapter cost is real because the eval-plug shape is undocumented.
3. **Defer AMA-Bench real subset** until we know cashew's extract template handles tool-call trajectories at all.

## Most surprising finding

**LifeBench's hardest categories (ND, TKU) are exactly cashew's pitch, but the benchmark Hindsight wins ND while losing overall.** That suggests the systems with the cleanest pattern-inference machinery underperform on the easier recall categories — which matches our own LoCoMo experience where cashew's ingest-only arm trailed Mem0 on cat-4 recall while doing well on inference. LifeBench would let us run the same shape of argument at 30× the scale, but it also means **a "cashew wins LifeBench overall" headline is unlikely** — the right framing is per-category, specifically ND.

## Sources

- LifeBench paper: `https://arxiv.org/abs/2603.03781` / HTML `https://arxiv.org/html/2603.03781v1`
- LifeBench repo: `https://github.com/1754955896/LifeBench`
- AMA-Bench paper: `https://arxiv.org/abs/2602.22769` / HTML `https://arxiv.org/html/2602.22769`
- AMA-Bench repo: `https://github.com/AMA-Bench/AMA-Bench`
- AMA-Bench site: `https://ama-bench.github.io/`
