# cashew — Design Document

## 1. Problem Statement

Current AI systems (and humans) produce conclusions but discard the derivation path. Chain-of-thought exists during inference but is ephemeral. Knowledge graphs store what is known but not how it was derived. No existing system combines:

1. **Persistent derivation chains** — every thought linked to its parents
2. **Global state modifiers** — context/mood affecting traversal and generation
3. **Auditability as a first-class operation** — "why do I believe X?" is a query, not introspection

### Core Question
If you store reasoning with its full derivation path, can a system meaningfully audit and self-correct its own beliefs?

### Prior Art
| System | What it does | Gap |
|--------|-------------|-----|
| Knowledge Graphs | Stores entities + relationships | No derivation tracking |
| Chain-of-thought (LLMs) | Step-by-step reasoning | Ephemeral — gone after response |
| RAG (LlamaIndex etc.) | Retrieves context for generation | Flat memory, no graph structure |
| MemGPT | Persistent LLM memory | No derivation graph |
| Argument mapping (Kialo) | Structured debate trees | Human-curated, not emergent |
| Causal inference (DAGitty) | Cause-effect models | Statistical modeling, not reasoning |

### What's New
cashew combines persistent derivation, global state modifiers, and auditability in a single system — then runs an experiment: seed it with a reasoning style and a belief system, and observe whether structure emerges and whether the system can self-audit.

---

## 2. Architecture

```
                    ┌─────────────┐
                    │  Global State │ (mood / context / priors)
                    │  Modifier     │
                    └──────┬──────┘
                           │ biases retrieval + generation
                           ▼
┌──────────┐    ┌──────────────────┐    ┌─────────────┐
│  Input    │───▶│  Thought Engine   │───▶│  Graph Store │
│  (query,  │    │                    │    │  (SQLite +   │
│   seed,   │    │  1. Retrieve       │    │   embeddings)│
│   why?)   │    │     relevant       │    │              │
└──────────┘    │     parent nodes   │    │  Nodes:      │
                │  2. Generate new   │    │  - id         │
                │     thought via LLM│    │  - content    │
                │  3. Create node    │    │  - embedding  │
                │  4. Link to parents│    │  - timestamp  │
                │  5. Store          │    │  - metadata   │
                └──────────────────┘    │  - mood_state  │
                                         │              │
                                         │  Edges:      │
                                         │  - parent_id  │
                                         │  - child_id   │
                                         │  - weight     │
                                         │  - relation   │
                                         └─────────────┘
                                                │
                                    ┌───────────┴───────────┐
                                    │                       │
                              ┌─────▼─────┐          ┌─────▼─────┐
                              │  Traversal │          │  Visualizer│
                              │  Engine    │          │  (D3.js /  │
                              │            │          │   Pyvis)   │
                              │  why(node) │          │            │
                              │  how(A→B)  │          │  Graph     │
                              │  roots()   │          │  render    │
                              │  cycles()  │          │  clusters  │
                              │  audit()   │          │  timeline  │
                              └───────────┘          └───────────┘
```

---

## 3. Data Model

### Node (Thought)
```python
@dataclass
class ThoughtNode:
    id: str                  # UUID
    content: str             # The actual thought / conclusion
    embedding: list[float]   # Vector embedding for similarity search
    timestamp: datetime      # When this thought was generated
    node_type: str           # "seed" | "derived" | "question" | "belief" | "doubt"
    mood_state: dict         # Global state snapshot at time of creation
    confidence: float        # 0.0 - 1.0, how confident the system is
    metadata: dict           # Extensible (source, tags, etc.)
```

### Edge (Derivation)
```python
@dataclass
class DerivationEdge:
    parent_id: str           # Source thought
    child_id: str            # Derived thought
    relation: str            # "derived_from" | "contradicts" | "supports" | "questions"
    weight: float            # Strength of derivation (affected by mood)
    reasoning: str           # Brief explanation of WHY this link exists
```

### Emergent Mood (read-only measurement)
```python
@dataclass
class MoodState:
    detected_mood: str       # Emergent: "curious" | "tense" | "nostalgic" | "confused" | "peaceful"
    contradiction_density: float  # Ratio of contradiction edges to total edges
    open_question_ratio: float    # Ratio of unanswered questions to total questions
    core_memory_retrieval_freq: float  # How often core memories appear in recent chains
    isolation_index: float        # Proportion of disconnected subgraphs
    avg_confidence: float         # Mean confidence across connected beliefs
    timestamp: datetime           # When this measurement was taken
```

---

## 4. Core Operations

### 4.1 think(input) → ThoughtNode
1. Embed the input
2. Retrieve top-K similar existing nodes (parents)
3. Apply global state modifier to re-rank parents
4. Send to LLM: input + parent contents + mood context → new thought
5. Create node, link to parents, store
6. Return node

### 4.2 why(node_id) → list[ThoughtNode]
1. Walk parent edges recursively until reaching seed nodes
2. Return the full derivation chain
3. Optionally: score each link's strength

### 4.3 how(node_a, node_b) → list[ThoughtNode]
1. Find shortest path between two nodes in the graph
2. Return the chain connecting them
3. If no path exists, return None (disconnected beliefs)

### 4.4 audit() → AuditReport
1. Find all nodes with no parents that aren't seeds (ungrounded beliefs)
2. Find near-cycles (A derives B derives C derives ~A)
3. Find contradictions (nodes with "contradicts" edges)
4. Find highest-confidence nodes with weakest derivation chains
5. Return report

### 4.5 detect_mood() → MoodState
Mood is NOT injected. It EMERGES from the graph.
1. Measure current graph properties:
   - Contradiction density (high → tension/anger)
   - Unanswered question ratio (high → curiosity)
   - Core memory retrieval frequency (high → nostalgia)
   - Isolated chain count (high → confusion)
   - Average confidence across connected beliefs (high → peace/certainty)
2. Return detected mood as a measurement, not an input
3. Log mood snapshots over time to track emotional trajectory
4. Mood detection is a READ operation — it never modifies the graph

### 4.6 seed(beliefs) → list[ThoughtNode]
1. Create root nodes with no parents
2. These represent starting axioms / beliefs
3. Tagged as node_type="seed"
4. The foundation the system builds on

---

## 5. The Experiment Protocol

### Phase 1: Seeding
Seed the system with:
- **Reasoning style:** "Always ask why. Follow chains to their root. Prefer simpler explanations. Test claims against observable reality. Trust moral intuitions."
- **Belief system:** Core Christian beliefs as seed nodes:
  - God exists and is all-powerful, all-knowing, all-good
  - The Bible is divinely inspired
  - Jesus was resurrected
  - Prayer works
  - Salvation requires faith in Christ
  - Morality comes from God
  - Non-believers face judgment

### Phase 2: Questioning
Feed the system questions (not conclusions):
- "Why do good people of other religions face judgment?"
- "What evidence supports the resurrection outside the Bible?"
- "Why does prayer not produce measurable outcomes in controlled studies?"
- "Can morality exist without God?"
- "Why do different denominations disagree on fundamental doctrines?"

Let the system think(). Don't tell it what to conclude.

### Phase 3: Mood Observation
After each batch of questions, run detect_mood() on the graph:
- Track how the emergent mood shifts as the system processes more questions
- Does the system move from "peaceful" (high confidence in seeds) → "tense" (contradictions accumulate) → "curious" (questions dominate) → "peaceful" again (new beliefs stabilize)?
- Log the emotional trajectory — does it mirror a real deconstruction journey?
- Compare mood trajectory against Raj's actual timeline

### Phase 4: Audit
Run audit() on the resulting graph:
- Are there ungrounded beliefs?
- Did cycles form? (circular reasoning)
- Did contradictions emerge?
- Where are the weakest derivation chains?

### Phase 5: Analysis
- Did the system exit the belief system? At which node?
- If not, what kept it in?
- Visualize the graph — do clusters form? Where?
- Compare mood-modulated graphs — structural differences?

---

## 6. Technical Decisions

### Language: Python (MVP)
- Fastest to insight
- NetworkX for graph ops
- SQLite for persistence
- If concept proves out → consider Rust rewrite for performance

### LLM: Claude Sonnet (via Anthropic API)
- Needs separate API key (not OpenClaw Max subscription)
- ~$0.01-0.03 per thought node
- Budget: $50 covers 2,000-5,000 thoughts

### Embeddings: Local
- embeddinggemma-300m (already on Mac Mini via OpenClaw)
- OR sentence-transformers (all-MiniLM-L6-v2) — lightweight, well-tested
- No API calls for retrieval

### Storage: SQLite
- Single file, no server, portable
- FTS5 for text search
- Blob columns for embeddings
- Handles 500K+ nodes easily

### Visualization: Pyvis (MVP) → D3.js (if we need interactivity)
- Pyvis generates HTML from NetworkX graphs
- Good enough for prototype
- D3.js if we want zoom, filter, time-scrubbing

---

## 7. Bottlenecks & Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| LLM just tells us what we want to hear (sycophancy) | HIGH | Use system prompts that enforce genuine reasoning, not agreement. Test with adversarial seeds. |
| Graph is just a fancy log (no emergent structure) | MEDIUM | Compare graph metrics (clustering, degree distribution) against random graphs. If indistinguishable → no emergence. |
| Mood modulation is superficial (just changes word choice, not structure) | MEDIUM | Measure structural graph metrics across moods, not just content. |
| Circular reasoning in LLM outputs looks like depth | HIGH | The audit() cycle detector is specifically for this. Also: track unique information per node — if child nodes add nothing new, the chain is fake depth. |
| We're just building a knowledge graph with extra steps | MEDIUM | The test: can why(node) produce a useful, non-obvious derivation chain? If yes → not a knowledge graph. If no → kill it. |
| LLM has training data about religion (not reasoning from scratch) | HIGH | This is the biggest threat to validity. The LLM already "knows" atheist arguments. Mitigation: track whether the system produces derivation paths or just recalls conclusions. If paths are novel but conclusions are known, that's still interesting. |

---

## 8. Success Criteria

The prototype succeeds if ANY of these are true:
1. **why(node) produces non-obvious derivation chains** — tracing back reveals connections the seeder didn't explicitly program
2. **audit() catches real circular reasoning** — the system can identify its own logical loops
3. **Mood emerges measurably from graph state** — detected mood correlates with graph metrics (contradiction density → tension, etc.)
4. **Emergent clusters form** — thoughts self-organize into topic groups without explicit categorization
5. **The exit path (if it happens) is traceable** — you can point to the exact node where the belief system cracked

The prototype fails if:
- why(node) just replays chain-of-thought (expensive logger)
- Mood detection doesn't correlate with graph metrics (arbitrary)
- Graph is flat / random (no emergence)
- LLM just recalls known arguments instead of deriving them

---

## 9. File Structure (Proposed)

```
cashew/
├── README.md
├── DESIGN.md
├── requirements.txt
├── cashew/
│   ├── __init__.py
│   ├── engine.py          # ThoughtEngine — main orchestrator
│   ├── graph.py            # Graph store (SQLite + NetworkX)
│   ├── llm.py              # LLM interface (Anthropic Sonnet)
│   ├── embeddings.py       # Local embedding generation
│   ├── traversal.py        # why(), how(), audit()
│   ├── mood.py             # Emergent mood detection (read-only measurement)
│   ├── models.py           # Data models (ThoughtNode, Edge, etc.)
│   └── visualize.py        # Graph visualization
├── experiments/
│   ├── religion_exit.py    # The main experiment
│   └── analysis.py         # Graph metrics + comparison
├── tests/
│   └── ...
└── data/
    └── ...                 # SQLite DBs, experiment outputs
```

---

## 10. Memory Hierarchy

### Seeds
- Starting axioms, given not derived
- Can be questioned but form the initial graph
- Examples: "God exists", "The Bible is divinely inspired", "Prayer works"

### Core Memories
- Nodes that earn permanence through network effects
- Not chosen — promoted automatically based on:
  - Branching factor (how many thoughts derive from this)
  - Cross-link count (connections to other chains)
  - Retrieval frequency (how often referenced by new thoughts)
  - Derivation depth (how many layers built on top)
- GC-immune: never decayed
- Always fed into context window for new thought generation
- Periodically re-evaluated (can be demoted if graph evolves away)
- Examples: formative experiences, questions that keep resurfacing

### Regular Nodes
- Standard thoughts, subject to normal GC
- Can be promoted to core or decayed

### Decayed Nodes
- Below fitness threshold, marked as decayed
- Not deleted — edges weakened, excluded from active traversal
- Can be revived if a new thought reconnects to them
- Represents "forgetting" — data exists, retrieval path degraded

## 11. Sleep Protocol

### Purpose
Connect isolated thought chains, deduplicate, garbage collect, and consolidate — all in one phase (mirrors neural sleep consolidation).

### Operations (run periodically after N active thoughts):

1. **Cross-linking:** Semantic similarity scan across chains. Dedup (>0.9), cross-link (0.7-0.9), or flag contradiction.
2. **Dream generation:** New "dream nodes" generated about discovered connections between chains. Parents span multiple trees → forest becomes graph.
3. **Garbage collection:** Random selection of N nodes, scored by composite fitness. Below threshold → decay. Random GC introduces noise that forces rederivation through novel paths (creativity mechanism).
4. **Core memory promotion:** Re-rank all nodes, promote/demote based on network metrics.
5. **Logging:** Every GC prune, every cross-link, every promotion logged for analysis.

### Self-similarity of sleep:
| Scale | Consolidation | Pruning | Cross-linking |
|-------|--------------|---------|---------------|
| Neural | Memory replay | Synaptic pruning | New associations |
| Thought graph | Core memory promotion | GC decay | Dream nodes |
| Social | Cultural canon formation | Forgotten ideas | Cross-domain insights |

5. **Mood snapshot:** Run detect_mood() after each sleep cycle and log the trajectory.

## 12. Engineering Principles

### Testing Philosophy
"I don't want an unfalsifiable system." — Raj

Every behavior gets a test. If we can't test it, we can't claim it works. The system that exposes unfalsifiable claims must itself be falsifiable.

**Test categories:**
1. **Unit tests** — every function, every edge case
   - Graph store: CRUD, dedup, edge creation, integrity constraints
   - Traversal: `why()` returns correct chains, `audit()` catches cycles
   - GC: correct fitness scoring, decay behavior, revival
   - Core memory promotion: correct ranking, threshold behavior

2. **Behavioral tests** — does the system do what we claim?
   - "Thought A derives from Thought B" → `why(A)` includes B
   - "Contradicting thoughts exist" → `audit()` flags them
   - "Mood changes traversal" → same input, different mood, measurably different output
   - "GC preserves high-branching nodes" → prune cycle doesn't kill hubs
   - "Sleep cross-links independent chains" → forest becomes graph

3. **Falsifiability tests** — can we prove our claims wrong?
   - "Structure emerges" → compare graph metrics against random graph. If indistinguishable, no emergence.
   - "Core memories are load-bearing" → remove one, measure impact on graph connectivity
   - "Mood affects structure not just words" → compare topology metrics across moods, not just content

4. **Regression tests** — don't break what works
   - Every bug gets a test before the fix
   - Every experiment run is reproducible (seeded randomness)

**Framework:** pytest. Run on every commit. No merge without green.

### Code Standards
- Type hints everywhere
- Docstrings on public functions
- No magic numbers — constants named and documented
- Git hygiene: specific `git add`, one feature per PR, never push to main

## 13. Open Questions (for Raj)

1. **How prescriptive should the reasoning style seed be?** Full personality profile or just principles?
2. **Should the system be allowed to question its own seeds?** (Can it ask "why do I believe God exists?" about a seed node, or are seeds axioms?)
3. **How many thought generations before we evaluate?** 100? 500? Let it run until convergence?
4. **Should we run a control experiment?** Same system, same questions, but seeded with secular beliefs — does it find religion? That would be powerful.
5. **Publication target:** Blog only? Or do we aim for something more formal?
6. **Core memory threshold:** How many core memories at any given time? Fixed cap or dynamic?
7. **Can seeds be decayed?** If "God exists" has low branching factor, should GC be allowed to touch it? Or are seeds permanently protected?
8. **Sleep frequency:** Every N thoughts? After each question? Time-based?
