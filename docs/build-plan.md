# Cashew Brain — Build Plan

## Build Order (each phase is testable before moving to next)

### Phase 1: Migration + Schema
**Build:** Migration script for brain graph → new architecture
**What it does:**
- Copy current graph.db → brain.db (backup original)
- Strip all edge relations → untyped (just source_id, target_id, weight)
- Add new columns: domain, last_accessed, access_count, source, session_id
- Reclassify node_types via LLM pass (seed/belief/environment/derived → observation/belief/decision/insight/fact)
- Assign domain tags via LLM pass (work/personal/fitness/engineering/philosophy/etc.)

**Test:**
```bash
python3 -m pytest tests/test_migration.py
```
- Assert: all nodes preserved, no data loss
- Assert: all edges present, no relation types
- Assert: new columns exist with valid values
- Assert: every node has a domain tag
- Assert: node_type distribution is reasonable (not all "fact")
- Assert: original DB untouched

**Files:** `scripts/migrate_brain.py`, `tests/test_migration.py`

---

### Phase 2: Embedding Layer
**Build:** Embed all nodes, store embeddings, similarity search
**What it does:**
- Embed every node's content using sentence-transformers (local, free)
- Store in `embeddings` table (node_id, vector BLOB)
- `cashew.similar(query, k=10)` → returns top-K nodes by cosine similarity

**Test:**
```bash
python3 -m pytest tests/test_embeddings.py
```
- Assert: every node has an embedding
- Assert: similar("promotion") returns work-related nodes
- Assert: similar("fitness") returns fitness-related nodes
- Assert: similar("silence") returns the silence cluster nodes
- Assert: cosine similarity scores are between 0 and 1
- Assert: k parameter works (returns exactly k results)
- Assert: performance < 100ms for 1000 nodes

**Open question for Raj:**
- sentence-transformers `all-MiniLM-L6-v2` (384-dim, fast, good enough)?
- Or `all-mpnet-base-v2` (768-dim, better quality, 2x slower)?
- Or API embeddings (best quality, costs money per call)?

**Files:** `core/embeddings.py`, `tests/test_embeddings.py`

---

### Phase 3: Hybrid Retrieval
**Build:** Graph-walk expansion on top of embedding search
**What it does:**
- `cashew.retrieve(query, k=10)` =
  1. Embedding search → top-5 entry points
  2. For each entry point, BFS walk 1-2 edges out
  3. Score expanded set: `similarity × recency_decay × log(access_count + 1)`
  4. Return top-K ranked nodes
- Updates `last_accessed` and `access_count` on retrieved nodes

**Test:**
```bash
python3 -m pytest tests/test_retrieval.py
```
- Assert: retrieve("promotion") returns more context than embedding-only search
- Assert: graph walk finds nodes embedding search missed (cross-domain connections)
- Assert: access_count increments on retrieval
- Assert: last_accessed updates
- Assert: recency weighting works (recent nodes rank higher, all else equal)
- Assert: returns ≤ k results
- Assert: no duplicate nodes in results

**Comparison test (the real validation):**
```python
def test_retrieval_vs_memory_search():
    """Compare cashew.retrieve vs current memory_search on same queries"""
    queries = ["promotion tracking", "Raj's communication patterns", "fitness routine"]
    for q in queries:
        cashew_results = cashew.retrieve(q, k=5)
        # Score: are cashew results more relevant? (human eval needed)
        assert len(cashew_results) == 5
```

**Files:** `core/retrieval.py`, `tests/test_retrieval.py`

---

### Phase 4: Knowledge Extraction
**Build:** Extract nodes from conversations as ISOLATED fragments
**What it does:**
- `cashew.extract(conversation_text)` →
  1. LLM call: "Extract key decisions, observations, beliefs, insights from this conversation"
  2. For each: create node with type, domain, content
  3. Embed new node (for later retrieval)
  4. Nodes land ISOLATED — no edges created yet
  5. Flag contradictions for sleep cycle
- Returns list of new node IDs
- **Edge creation happens during sleep, not here** (Raj's design: daytime = fragments, sleep = consolidation)

**Test:**
```bash
python3 -m pytest tests/test_extraction.py
```
- Assert: extraction from a test conversation produces 3-8 nodes (not too few, not too many)
- Assert: each node has valid type and domain
- Assert: new nodes have edges to existing graph (not isolated)
- Assert: contradiction detection flags known contradictions
- Assert: duplicate content doesn't create duplicate nodes (idempotency via content hash)
- Assert: empty/trivial conversations produce 0 nodes

**Decision (from Raj):**
- Extract every N minutes if there's stuff to extract. Lightweight scan. Don't wait for compaction.
- Edge creation happens during SLEEP, not extraction. New nodes land isolated during the day. Sleep rewires — finds nearest neighbors, creates edges, runs think cycles on new clusters, GCs noise. Biologically honest: daytime = fragments, sleep = consolidation.

**Files:** `core/extraction.py`, `tests/test_extraction.py`

---

### Phase 5: Think Cycle Integration
**Build:** Autonomous think cycles on heartbeat
**What it does:**
- `cashew.think()` →
  1. Find underexplored cluster: nodes with high edge_count but low access_count
  2. Pick 3-5 connected nodes from cluster
  3. Feed to LLM: "Here are connected thoughts. What patterns or tensions do you see?"
  4. Parse response → new insight nodes
  5. Connect to source nodes
  6. Optionally run sleep/GC

**Test:**
```bash
python3 -m pytest tests/test_think_cycle.py
```
- Assert: think cycle produces 1-4 new nodes
- Assert: new nodes connect to source nodes
- Assert: cluster selection avoids recently-accessed nodes
- Assert: think cycle doesn't run if graph has < 20 nodes (not enough material)
- Assert: sleep/GC decays low-value nodes (access_count=0, old)
- Assert: sleep/GC promotes high-value nodes (high access_count)

**Files:** `core/think.py`, `tests/test_think_cycle.py`

---

### Phase 6: Integration Hooks (THE HARD PART)
**Build:** Wire cashew into OpenClaw's session lifecycle
**What it does:**
- On session start: retrieve relevant context, inject into system prompt
- On session end: extract knowledge, add to graph
- On heartbeat: optionally run think cycle
- Expose `cashew.retrieve()` as a tool the agent can call explicitly

**Open questions for Raj (need engineering input):**
1. How does OpenClaw inject system context? Can I add to it programmatically?
2. Is there a session lifecycle hook (on_start, on_end) I can tap into?
3. Or is this a SKILL that I invoke explicitly? (simpler, more testable)
4. Should retrieval be automatic (every session) or on-demand (I decide when)?
5. Can I add a custom tool (like `memory_search` but `cashew_retrieve`)?

**Two approaches:**
- **A: Skill-based** — I explicitly call `cashew retrieve "query"` when I need context. Simple, testable, no OpenClaw core changes needed. I control when it's used.
- **B: Hook-based** — Automatically runs on every session start/end. More powerful, but requires OpenClaw integration. Harder to test. Risk of breaking things.

**Decision (from Raj):** Option A. Skill-based. Period. Don't get "comfortable" with A and then creep to B. A is the design. I call cashew explicitly when I need context.

**Files:** `SKILL.md` (for skill-based), or hook scripts

---

## Test Strategy

### Unit Tests (automated, run on every change)
- Migration correctness
- Embedding generation and similarity
- Retrieval ranking
- Extraction parsing
- Think cycle node generation

### Integration Tests (semi-automated)
- End-to-end: message → retrieve → context → response
- Compare cashew.retrieve vs memory_search on same queries
- Graph growth over N conversations (does it stay healthy or explode?)

### Validation Tests (human eval, periodic)
- "Does the retrieved context feel more relevant?" — Raj scores 1-5
- "Did the think cycle produce genuine insights?" — Raj confirms/denies
- "Am I responding better with cashew context?" — qualitative over 2 weeks

### Regression Tests
- Response quality shouldn't degrade during transition
- Latency: retrieval should add < 500ms to session start
- Graph size: should stay < 2000 nodes after 1 month (GC working)

---

## Dependencies
- `sentence-transformers` — local embeddings (pip install)
- `numpy` — vector operations (already installed?)
- `anthropic` — LLM calls for extraction/think cycles (already installed)
- `sqlite3` — graph storage (stdlib)
- `pytest` — testing (pip install)

## Timeline Estimate
- Phase 1 (migration): 1 session
- Phase 2 (embeddings): 1 session
- Phase 3 (retrieval): 1 session
- Phase 4 (extraction): 1 session
- Phase 5 (think cycles): 1 session
- Phase 6 (integration): depends on approach (A: 1 session, B: 2-3 sessions)

Total: ~6 sessions to working prototype
