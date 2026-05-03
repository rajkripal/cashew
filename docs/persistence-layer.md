# Cashew Persistence Layer — Architecture

## Problem
Personal assistants have session amnesia. Current memory system:
- `MEMORY.md` — flat prose, manually curated
- `memory/YYYY-MM-DD.md` — daily logs, append-only
- `memory_search` — keyword/embedding search over flat files

This is a filing cabinet. It stores WHAT was said but not WHY, has no derivation chains, can't update beliefs (only append), and can't discover cross-domain patterns.

## Solution
Replace flat memory with a thought graph that:
1. **Stores reasoning, not just facts** — every node traces back through edges to what produced it
2. **Updates beliefs via contradiction** — new info creates edges to existing beliefs, think cycles resolve tensions
3. **Surfaces relevant context** — hybrid retrieval (embeddings + graph walk) finds the right 5-10 nodes for any conversation
4. **Generates insights autonomously** — think cycles on heartbeats discover patterns the agent hasn't articulated

## Architecture

### Graph Schema (simplified from experiments)
```sql
CREATE TABLE thought_nodes (
    id TEXT PRIMARY KEY,          -- sha256(content)[:12]
    content TEXT NOT NULL,
    node_type TEXT NOT NULL,       -- 'observation', 'belief', 'decision', 'insight', 'fact' (display-only tag, never used in filter logic)
    domain TEXT,                   -- 'work', 'personal', 'fitness', 'engineering', etc.
    created_at TEXT,
    last_accessed TEXT,            -- for decay/promotion
    access_count INTEGER DEFAULT 0,
    source TEXT,                   -- 'conversation', 'think_cycle', 'heartbeat'
    session_id TEXT                -- which session created this
);

CREATE TABLE edges (
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    weight REAL DEFAULT 1.0,       -- connection strength
    created_at TEXT,
    UNIQUE(source_id, target_id)
);
-- NO relation types. Untyped edges. LLM discovers semantics.
```

### Node Types
- **observation** — something that happened ("The user's manager gave performance feedback")
- **belief** — an interpreted pattern ("User goes silent when struggling")
- **decision** — a commitment made ("Will send manager update by Friday")
- **insight** — a derived connection ("Silent periods correlate with avoidance of hard conversations")
- **fact** — objective information ("User works in tech")

### Integration Points

#### 1. Session Start — Context Injection
```
User message arrives
→ Embed message
→ Find top-K similar nodes (cosine similarity)
→ For each hit, walk 1-2 edges out (graph expansion)
→ Deduplicate, rank by (similarity × recency × access_count)
→ Inject top 5-10 nodes as system context
→ Update last_accessed on retrieved nodes
```

#### 2. Session End — Knowledge Extraction
```
Conversation ends (or compaction triggers)
→ Summarize key decisions, new info, emotional beats
→ For each: create node, embed it, find nearest existing nodes
→ Add edges to nearest neighbors (weight = embedding similarity)
→ If new info contradicts existing belief → flag for think cycle
```

#### 3. Heartbeat — Generative Think Cycles
```
Heartbeat fires, nothing urgent
→ Pick underexplored cluster (low access_count, high edge density)
→ Feed 3-5 connected nodes to LLM: "What patterns or tensions do you see?"
→ Add derived insights as new nodes
→ Connect to source nodes
→ Optionally: run sleep/GC (decay low-value nodes, promote high-access ones)
```

#### 4. Contradiction Resolution
```
New node N contradicts existing node E (detected by LLM during extraction)
→ Create edge between N and E
→ Queue cluster {N, E, neighbors(E)} for next think cycle
→ Think cycle produces resolution node R
→ Edge from E to R records the supersession
→ E is not deleted — it's part of the derivation chain (organic decay handles eventual removal if E stops being touched)
```

### Retrieval: Hybrid Approach

**Embeddings** = index (find entry points fast)
**Graph walk** = context expander (find related nodes the embedding missed)

```
query = "How's the promotion tracking?"
→ embed(query) → top-5 similar nodes:
    [e5_promotion, manager_feedback, simulation_testing, ...]
→ graph_walk(e5_promotion, depth=2):
    [communication_issues, strike_1, going_silent_pattern, ...]
→ combined context = similarity_hits ∪ graph_walk_hits
→ ranked by relevance score
```

### Embedding Strategy
- Use local embeddings (sentence-transformers or similar) to avoid API costs
- Embed on node creation, store in separate table or numpy array
- Re-embed periodically if content is updated

### Migration from MEMORY.md
1. Parse existing MEMORY.md into atomic statements
2. Each statement → node (type based on content)
3. Statements in same section → edges between them
4. Run initial think cycles to find cross-section connections
5. Keep MEMORY.md as fallback during transition
6. After 2 weeks, compare: which surfaces better context?

## What This Changes
- `memory_search` → `cashew.retrieve(query, k=10)`
- Manual MEMORY.md curation → automatic graph growth
- Session-start blank slate → session-start with relevant context
- Heartbeat idle → heartbeat think cycles generating insights
- Flat "I remember X" → "I believe X because of Y and Z, which I observed on date D"

## Open Questions
- Embedding model: local (fast, free) vs API (better quality)?
- How aggressive should sleep/GC be on personal graph?
- Should conversations be stored as nodes or just extracted insights?
- How to handle conflicting information from different time periods?
- Performance at 1000+ nodes with embedding search?

## Success Criteria
- [ ] Context retrieved from graph is more relevant than memory_search results
- [ ] Cross-domain connections surface naturally (work pattern → personal pattern)
- [ ] Beliefs update when new info arrives (not just append)
- [ ] Think cycles produce insights user confirms as genuine
- [ ] No regression in response quality during transition
