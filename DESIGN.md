# cashew — Design Document

## 1. Problem Statement

Current AI systems (and humans) produce conclusions but discard the derivation path. Chain-of-thought exists during inference but is ephemeral. Knowledge graphs store what is known but not how it was derived. No existing system combines:

1. **Persistent derivation chains** — every thought linked to its parents
2. **Emergent self-organization** — organic structure from cross-linking and decay  
3. **Auditability as a first-class operation** — "why do I believe X?" is a query, not introspection

### Core Question
If you store reasoning with its full derivation path, can a system meaningfully audit and self-correct its own beliefs while exhibiting emergent insights?

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
cashew combines persistent derivation, recursive BFS retrieval, and organic graph structure in a single system that exhibits power law properties and genuine insight generation through sleep cycle consolidation.

---

## 2. Architecture

```
┌──────────┐    ┌──────────────────┐    ┌─────────────┐
│  Input    │───▶│  Thought Engine   │───▶│  Graph Store │
│  (query,  │    │                    │    │  (SQLite +   │
│   seed,   │    │  1. BFS retrieval  │    │   sqlite-vec)│
│   context)│    │     via sqlite-vec │    │              │
└──────────┘    │  2. Generate new   │    │  Nodes:      │
                │     thought via LLM│    │  - id         │
                │  3. Create node    │    │  - content    │
                │  4. Link to parents│    │  - embedding  │
                │  5. Store + index  │    │  - timestamp  │
                └──────────────────┘    │  - metadata   │
                                         │              │
                                         │  Edges:      │
                                         │  - parent_id  │
                                         │  - child_id   │
                                         │  - weight     │
                                         │  - reasoning  │
                                         └─────────────┘
                                                │
┌───────────┬───────────────┬─────────────────┴───────────────┐
│           │               │                                 │
│  Traversal │    Sleep      │    sqlite-vec                  │
│  Engine    │   Protocol    │    Integration                 │
│            │               │                                │
│  why(node) │   - Decay     │   - O(log N) search            │
│  how(A→B)  │   - Promote   │   - Cosine distance            │
│  audit()   │   - Cross-link│   - Embedding dual-write       │
│  roots()   │   - Dedup     │   - Fallback to brute force    │
└───────────┘   - Core memory│                                │
                └───────────┘ └────────────────────────────────┘
```

---

## 3. Data Model

### Current Schema (SQLite)
```sql
CREATE TABLE thought_nodes (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    node_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    confidence REAL NOT NULL,
    metadata TEXT,
    source_file TEXT,
    decayed INTEGER DEFAULT 0,
    last_updated TEXT DEFAULT NULL,
    last_accessed TEXT,
    access_count INTEGER DEFAULT 0,
    domain TEXT,
    permanent INTEGER DEFAULT 0,
    tags TEXT
);

CREATE TABLE derivation_edges (
    parent_id TEXT NOT NULL,
    child_id TEXT NOT NULL,
    weight REAL NOT NULL,
    confidence REAL NOT NULL,
    reasoning TEXT,
    FOREIGN KEY (parent_id) REFERENCES thought_nodes(id),
    FOREIGN KEY (child_id) REFERENCES thought_nodes(id),
    PRIMARY KEY (parent_id, child_id)
);

CREATE TABLE embeddings (
    node_id TEXT PRIMARY KEY,
    vector BLOB NOT NULL,
    model TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (node_id) REFERENCES thought_nodes(id)
);

CREATE VIRTUAL TABLE vec_embeddings USING vec0(
    node_id TEXT PRIMARY KEY,
    embedding float[384] distance_metric=cosine
);
```

### Schema Ownership Contract

Cashew is embeddable as a library: downstream consumers (e.g. `hermes-cashew`) layer their own tables on top of cashew's database. This section defines what cashew owns and what extension points consumers can rely on.

**Cashew-owned tables** — schema is managed by `core.db.ensure_schema()`, may change across cashew versions under the rules below:

- `thought_nodes`
- `derivation_edges`
- `embeddings`
- `hotspots`
- `metrics`
- `vec_embeddings` (virtual, when sqlite-vec is available)

**Cashew-owned columns** on `thought_nodes`: `id`, `content`, `node_type`, `domain`, `timestamp`, `access_count`, `last_accessed`, `confidence`, `source_file`, `decayed`, `metadata`, `last_updated`, `mood_state`, `permanent`, `tags`, `referent_time`.

**Cashew-owned columns** on `derivation_edges`: `parent_id`, `child_id`, `weight`, `reasoning`, `confidence`, `timestamp`.

**Migration policy**

- Additive-only within a major version. Cashew will add columns or tables in minor releases; it will never drop or rename an owned column or table without a major version bump.
- `ensure_schema()` is idempotent and safe to call on empty databases, legacy databases missing columns, and fully-current databases.
- `PRAGMA user_version` carries the applied schema version. `core.db.get_schema_version(db)` reads it; `core.db.schema_version()` returns the version this build produces.

**Extension points for downstream consumers**

- Consumers may create their own tables (use a clear prefix, e.g. `hermes_*`) — cashew will not touch them.
- Consumers may add columns prefixed `ext_` to cashew-owned tables. Cashew will never introduce an `ext_`-prefixed column.
- Consumers should call `ensure_schema(db)` before running their own migrations, then branch on `get_schema_version(db)` to decide whether their own migration ladder needs to run.

Example downstream pattern:

```python
from core.db import ensure_schema, get_schema_version

ensure_schema(db_path)                  # cashew applies its migrations
if get_schema_version(db_path) >= 1:
    apply_my_layer(db_path)             # downstream layers on top
```

### Node Types (Current Implementation)
- **fact** — Data points and factual observations
- **observation** — Personal experiences and direct observations
- **insight** — Derived understanding from think cycles and analysis
- **decision** — Specific choices and their reasoning
- **belief** — Core principles and values
- **derived** — LLM-generated hypotheses and conclusions
- **meta** — Self-analysis and system reflection
- **core_memory** — High-importance, frequently accessed knowledge
- **cross_link** — Connections between disparate knowledge domains

---

## 4. Core Operations

### 4.1 Context Retrieval
**Recursive BFS with sqlite-vec seeding (O(log N) seeds + O(K) traversal)**
```python
def retrieve_recursive_bfs(hints: list[str], n_seeds=5, picks_per_hop=3, max_depth=3) -> list[ThoughtNode]:
    # 1. Embed hints, find top-k seeds via sqlite-vec O(log N) search
    # 2. BFS traversal from seeds up to max_depth hops
    # 3. Score neighbors by cosine similarity each hop, select picks_per_hop best
    # 4. Return ranked candidates by final similarity to original query
```

### 4.2 Knowledge Extraction  
**Convert conversations to graph nodes with derivation links**
```python
def extract(input_text: str) -> list[ThoughtNode]:
    # 1. Parse input for extractable knowledge
    # 2. Find derivation parents via similarity
    # 3. Generate new nodes with confidence scores
    # 4. Create edges with reasoning explanations
```

### 4.3 Think Cycles
**Cross-domain synthesis for insight generation**
```python
def think_cycle() -> list[ThoughtNode]:
    # 1. Query graph for recent high-novelty nodes
    # 2. Feed diverse context to LLM for synthesis
    # 3. Generate insights about connections and patterns
    # 4. Create derived nodes linked to source material
```

### 4.4 Traversal Operations
```python
def why(node_id: str) -> list[ThoughtNode]:
    # Walk parent edges recursively to seed nodes
    
def how(node_a: str, node_b: str) -> list[ThoughtNode]:
    # Find shortest path between nodes
    
def audit() -> AuditReport:
    # Find cycles, orphans, weak derivations
    
def roots() -> list[ThoughtNode]:
    # Return all nodes with no parents
```

### 4.5 Sleep Protocol
**Self-maintenance through cross-linking, decay, and deduplication**
```python
def run_sleep_cycle():
    # 1. Cross-link semantically similar nodes across domains (0.7-0.85 similarity)
    # 2. Decay low-fitness nodes (composite: access + confidence + age)
    # 3. Deduplicate near-identical content (>0.82 similarity, merge + redirect)
    # 4. Promote frequently accessed, high-confidence nodes to permanent=1
    # 5. Log all operations for auditability
```

---

## 5. Experiments

### Experiment 1: Organic Graph Growth ✅ ACHIEVED
**Goal:** Demonstrate emergent knowledge organization without clustering

**Method:** 
1. Seed graph with diverse knowledge from memory files
2. Run sleep cycles for cross-linking and decay
3. Observe organic connectivity patterns
4. Validate that retrieval finds relevant context via BFS

**Results:** Graph organically develops high-connectivity nodes and semantic neighborhoods via cross-linking. BFS traversal successfully retrieves contextually relevant knowledge without synthetic hierarchy.

**Scaled to:** 3,064 nodes, 6,122 edges, zero hotspots required (as of April 2026, author's personal graph)

### Experiment 2: General Domain Architecture ✅ ACHIEVED
**Goal:** Test domain-agnostic knowledge organization

**Method:**
1. Seed abstract knowledge domains (science, philosophy, engineering)
2. Run think cycles and sleep consolidation
3. Observe organic connectivity patterns via cross-linking
4. Test cross-domain insight generation

**Results:** Multi-domain graphs successfully maintain domain separation while enabling cross-domain synthesis. BFS retrieval finds relevant knowledge across domains. Sleep cycles create meaningful inter-domain connections without synthetic hierarchy.

**Scaled to:** Multiple domains (user/ai) with 3,064 nodes across knowledge areas (as of April 2026, author's personal graph)

### Experiment 3: Capable Agent Framework (Future)
**Goal:** Graph as agency engine for AI agents

**Method:**
1. Seed graph with engineer's reasoning patterns and principles
2. Agent makes decisions by querying graph for relevant principles
3. All decisions traceable back to human seed via why() traversal
4. Human audits and corrects graph, agent improves

**Success criteria:**
- Agent cognitive style matches human reasoning approach
- Every decision has auditable derivation chain
- Graph corrections improve agent performance

---

## 6. Technical Decisions

### Platform: Agent-native on OpenClaw
- No standalone app. cashew lives inside OpenClaw's agent infrastructure
- Main agent orchestrates, sub-agents handle think cycles
- Claude Sonnet for reasoning, local embeddings for retrieval

### Storage: SQLite + sqlite-vec
- Single file, no server, portable  
- sqlite-vec virtual table for O(log N) vector search
- Dual-write to both embeddings (BLOB) and vec_embeddings (float[384])
- Handles 2000+ nodes efficiently with cosine distance
- sentence-transformers for local embedding generation

### Language: Python
- NetworkX for in-memory graph operations
- pytest for comprehensive testing
- Type hints and docstrings throughout

### Visualization: vis.js Dashboard
- Real-time graph rendering with search and filtering
- Color-coded by node type (human vs system-generated)
- Deployable to Cloudflare Pages

---

## 7. Key Insights

### Power Law Properties
The graph exhibits natural power law behavior:
- **Preferential attachment** — New thoughts connect to high-connectivity hubs
- **Self-organized criticality** — Sleep cycles restructure accumulated knowledge
- **Fractal structure** — Same patterns at all scales
- **Emergent hierarchy** — Organization from simple connection rules

### Organic Retrieval Scaling
Traditional RAG systems use flat vector search (O(N) comparisons). cashew uses sqlite-vec for O(log N) seed selection followed by recursive BFS graph traversal, achieving efficient retrieval while preserving semantic relationships through organic connectivity.

### Think Cycles Generate Cross-Domain Synthesis
Cross-domain context synthesis produces insights that connect disparate knowledge areas. The graph's organic structure enables discovery of non-obvious relationships across different domains.

---

## 8. Success Criteria

### Phase 1: Personal Thought Graph ✅ ACHIEVED
1. ✅ **why(node) produces non-obvious derivation chains** — tracing reveals unplanned connections
2. ✅ **audit() catches real circular reasoning** — cycle detection works
3. ✅ **Organic connectivity emerges** — cross-linking creates natural pathways without synthetic structure  
4. ✅ **Think cycles produce cross-domain synthesis** — connections across knowledge areas
5. ✅ **Graph exhibits preferential attachment** — high-connectivity nodes via cross-linking  
6. ✅ **sqlite-vec retrieval scales** — O(log N) seed selection + BFS traversal

### Phase 2: Multi-Domain Knowledge ✅ ACHIEVED  
1. ✅ Multiple domains (user/ai) co-exist in single graph
2. ✅ Cross-domain sleep cycles create inter-domain connections
3. ✅ BFS retrieval finds relevant nodes across domains
4. ✅ System scales to 3000+ nodes across knowledge areas
5. ✅ Domain tags enable filtering while preserving cross-pollination

### Phase 3: Agency Engine (Future)
1. [ ] Graph drives agent decisions through principle retrieval
2. [ ] Every action traces back to human seed via why()
3. [ ] Agent cognitive style matches human reasoning approach
4. [ ] Human corrections improve graph and agent performance

### The prototype fails if:
- Think cycles only summarize existing content (expensive logger)
- Graph lacks organic connectivity (compare against random graph metrics)
- Cross-linking creates noise rather than meaningful relationships
- sqlite-vec search degrades to O(N) brute force at scale
- System can't maintain quality as knowledge grows

---

## 9. File Structure (Current)

```
cashew/
├── README.md                 # Project overview and usage
├── DESIGN.md                 # This document  
├── core/                     # Core modules
│   ├── config.py            # YAML configuration management
│   ├── context.py           # Context generation and formatting
│   ├── decay.py             # Fitness-based node decay
│   ├── embeddings.py        # sqlite-vec embedding management
│   ├── export.py            # Dashboard export utilities
│   ├── graph_utils.py       # Shared graph utilities
│   ├── retrieval.py         # BFS retrieval engine
│   ├── session.py           # Session lifecycle and think cycles
│   ├── sleep.py             # Cross-linking, decay, dedup
│   ├── stats.py             # Graph metrics and health
│   └── traversal.py         # Graph traversal (why/how/audit)
├── integration/              # External system bridges
│   └── openclaw.py          # OpenClaw agent integration
├── scripts/                  # CLI tools and utilities
│   ├── cashew_context.py    # Main CLI interface
│   ├── declassify.py        # Privacy tag management
│   └── export_dashboard.py  # Dashboard generation
├── tests/                    # Comprehensive test suite
│   ├── test_context.py      # Context generation tests
│   ├── test_retrieval.py    # BFS retrieval tests
│   ├── test_sleep.py        # Sleep cycle tests
│   └── test_traversal.py    # Traversal operation tests
├── dashboard/                # Visualization assets
│   ├── index.html           # Dashboard interface
│   └── data/               # Graph exports
├── data/                    # Database and exports
│   └── graph.db            # SQLite database (3064 nodes, 6122 edges, as of April 2026)
└── docs/                    # Documentation
    └── architecture.md      # Technical architecture details
```

---

## 10. Sleep Protocol Details

### Purpose
Connect isolated thought chains, deduplicate, garbage collect, and consolidate — mirrors neural sleep consolidation.

### Operations (run periodically or daily):

1. **Cross-linking:** Find semantically similar nodes across domains (0.7-0.85 similarity range). Create edges between related but previously disconnected knowledge. Preserves diversity by not over-connecting highly similar nodes.
2. **Fitness-based decay:** Composite scoring (access_count, confidence, age). Below threshold → `decayed=1`. Think cycle outputs face 1.5x higher decay threshold.
3. **Deduplication:** Merge near-identical nodes (>0.82 similarity), redirect edges to canonical version, preserve derivation links.
4. **Core memory promotion:** Frequently accessed, high-confidence nodes get `permanent=1` immunity from decay.
5. **Logging:** Every operation logged for analysis and auditability.

### Self-similarity Across Scales:
| Scale | Consolidation | Pruning | Cross-linking |
|-------|--------------|---------|---------------|
| Neural | Memory replay | Synaptic pruning | New associations |
| Thought graph | Core memory promotion | Fitness-based decay | Cross-domain edges |
| Knowledge systems | Canon formation | Forgotten concepts | Inter-domain synthesis |

---

## 11. Engineering Principles

### Testing Philosophy
"I don't want an unfalsifiable system." — Every behavior gets a test.

**Test categories:**
1. **Unit tests** — Every function, every edge case
   - Graph store: CRUD, dedup, edge creation, integrity
   - Traversal: why() correctness, audit() cycle detection
   - Sleep: fitness scoring, decay behavior, promotion logic

2. **Behavioral tests** — Does the system do what we claim?
   - Derivation: why(A) includes correct parent chain
   - Contradiction detection: audit() flags conflicts
   - Sleep preservation: high-value nodes survive GC
   - Cross-linking: independent clusters connect

3. **Emergence tests** — Can we measure emergent properties?  
   - Power laws: degree distribution vs random graphs
   - Clustering: modularity scores vs random networks
   - Insight generation: think cycle novelty assessment

4. **Regression tests** — Don't break what works
   - Every bug gets test before fix
   - Reproducible experiments (seeded randomness)

**Framework:** pytest, green tests on every commit, no merge without tests

### Code Standards
- Type hints everywhere
- Docstrings on public functions  
- No magic numbers — constants named and documented
- Git hygiene: specific `git add`, one feature per PR

---

## 12. Current State (April 2026)

### Graph Statistics (Author's Personal Graph)
- **3,064 thought nodes** across 9 distinct types (fact, insight, observation, etc.)
- **6,122 derivation edges** with weight and confidence
- **Domain separation** — user/ai domains in single graph
- **288/288 tests passing** — comprehensive coverage
- **sqlite-vec integration** — O(log N) vector search

*Note: These statistics reflect the author's personal knowledge graph as of April 2026. New users start with an empty graph.*

### Proven Capabilities  
1. **BFS retrieval at scale** — 2000+ nodes with sub-second response
2. **Organic graph evolution** — cross-linking creates natural structure
3. **Sleep cycles maintain quality** — decay prevents bloat, dedup prevents redundancy
4. **Zero infrastructure** — single SQLite file, no external servers
5. **Dashboard visualization** — real-time graph rendering

### In Production Use
- Daily context retrieval for agent sessions
- Knowledge extraction from conversations  
- Think cycles for insight generation
- Sleep cycles for graph maintenance

---

## 13. MVP Definition ("Done") ✅ ACHIEVED

**Done = You look at the graph and it surprises you.**

Concrete achievements (April 2026):
1. ✅ Graph scaled — 3,064 nodes, 6,122 edges from organic growth (author's personal graph)
2. ✅ sqlite-vec integration — O(log N) vector search with cosine distance
3. ✅ BFS retrieval — recursive traversal replaces hierarchical hotspots  
4. ✅ Sleep cycle evolution — cross-linking, decay, dedup without clustering
5. ✅ Think cycles via session.py — function-based, not class-based
6. ✅ Test coverage — 288/288 tests passing after major refactor

### Key Learning: Foundation Model AS Reasoning Engine
Don't build Python reasoning modules — the LLM reasoning over structured graph context IS the think cycle. Only tooling needed is graph plumbing (retrieve nodes, insert results).

### Philosophy Confirmed  
- **Orphans are unsolved problems, not bugs** — Don't force connections
- **Honest attempts > curve fitting** — Genuine relationships matter more than graph density
- **Emergent structure validates the architecture** — Power laws and organic connectivity prove self-organization