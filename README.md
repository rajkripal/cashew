# 🥜 cashew — Cognitive Architecture for Structured, Evolving, Walkable Understanding

A thought-graph engine that stores reasoning with its full derivation path, enabling auditability, self-correction, and emergent insight. Built in a weekend. Powered by power laws.

## What Is This?

cashew stores thoughts as nodes in a graph, connected by derivation edges. Every belief traces back to its roots. Every conclusion is auditable. The graph grows from both ends — human input and AI-generated hypotheses — and self-organizes through sleep cycles.

**The architecture:**
- **Graph = persistent storage** (structured, queryable, auditable)
- **Foundation model (LLM) = reasoning engine** (intelligence, derivation, pattern recognition)
- **Context window = RAM** (working memory, session-scoped)

You can't separate storage from reasoning. The graph shapes what the model sees, which shapes what it derives, which shapes the graph. They're coupled.

## What We've Proven

### ✅ Think cycles produce genuine insight
Isolated cluster reasoning (feed ONLY a cluster's nodes to an LLM) generates derivations the human hadn't stated but recognizes as true. Not summaries — actual forward predictions and structural splits.

**Example:** The silence cluster (17 nodes) produced: "Silence is TWO patterns, not one — strategic silence works, avoidant silence doesn't." This insight wasn't in ANY of the 17 nodes. The think cycle found it structurally.

### ✅ The graph exhibits power law properties naturally
Node connectivity follows a power law distribution — a few hubs with 40-60 edges, hundreds with 1-2. Preferential attachment emerges without tuning. Self-organized criticality through sleep cycles.

### ✅ Sleep/GC works at scale
Decay, promotion, cross-linking, and dream node generation all function. GC was too aggressive at 34 nodes, works correctly at 600+.

### ✅ Dashboard visualization works
Live vis.js graph, searchable, color-coded by node type. Human thoughts vs AI-generated are visually distinguishable.

## The Dual-Growth Loop

1. **Human conversations → nodes** (human-sourced, high confidence)
2. **Human corrections → edge fixes** (ground truth)
3. **System self-generation → hypotheses** (machine-sourced, confidence 0.5-0.7)
4. **Human reviews → promote or decay**
5. **Sleep consolidates → graph evolves**

The graph grows from both ends. 🧠 Human thoughts = blue/purple/gold/green. 🤖 System-generated = orange with dashed border.

## Current State

- **~700 nodes, ~900 edges**
- **~21K words** of thought content
- **768KB** on disk (smaller than a photo)
- **102 system-generated** nodes, **600+ human-sourced**
- **23/23 tests passing**
- **Modules:** traversal (`why()`, `how()`, `audit()`), sleep (decay/promote/cross-link/dream), questions, patterns, context retrieval, export

## Key Insight: Power Laws

The same power law that governs earthquakes, forest fires, income distribution, and startup returns also governs how a mind organizes itself. The graph IS a power law system:

- **Preferential attachment** — new thoughts connect to high-connectivity hubs naturally
- **Self-organized criticality** — sleep cycles are forest fires; thoughts accumulate, occasionally a cascade restructures everything
- **Fractals** — zoom into any cluster and you see the same structure
- **Universality** — the substrate doesn't matter. Same architecture, different seed nodes, same emergent behavior

## Experiments

### Experiment 1: Isolated Cluster Reasoning ✅ PASSED
- Isolate a cluster, feed only those nodes to LLM, generate hypotheses
- Result: 4/4 hypotheses on silence cluster confirmed by human as genuine insights
- Scaled to 7 clusters, 21 hypotheses generated

### Experiment 2: Religion Simulation 🔜 NEXT
- Blank graph, abstract seed beliefs (not Christianity-specific)
- Run think cycles, observe whether doctrine, schisms, and unfalsifiability emerge structurally
- Hypothesis: the architecture produces religion without a God node

### Experiment 3: Capable Engineer (future)
- Seed with an engineer's reasoning patterns
- The graph becomes the agency engine, foundation model is the reasoning engine
- Agent makes decisions traceable back to human principles

## Usage

```bash
# Query the graph
sqlite3 data/graph.db "SELECT content, confidence FROM thought_nodes ORDER BY confidence DESC LIMIT 10"

# Run traversal
python3 -c "from core.traversal import TraversalEngine; t = TraversalEngine(); chain = t.why('NODE_ID'); print(chain)"

# Run sleep cycle
python3 -c "from core.sleep import SleepProtocol; s = SleepProtocol(); s.run_sleep_cycle()"

# Run audit
python3 -c "from core.traversal import TraversalEngine; t = TraversalEngine(); r = t.audit(); print(f'Cycles: {len(r.cycles)}, Orphans: {len(r.orphan_nodes)}')"

# Export dashboard
python3 -c "from core.export import GraphExporter; e = GraphExporter(); e.export_full_graph('dashboard/data/graph.json')"

# Serve dashboard locally
cd dashboard && python3 -m http.server 8787
```

## Philosophy

- **Orphans are unsolved problems, not bugs.** Don't force edges. Honest attempts > curve fitting.
- **Unproven ≠ disproven.** Let the bottleneck find us.
- **Design until the next question can only be answered by building. Then build.**
- **The fruits of being highly ambitious: you might not reach the goal, but you'll get somewhere close enough.**
- **The foundation model IS the reasoning engine.** Don't over-engineer what the LLM already does. The graph is memory, the model is intelligence.

## Tech Stack

- Python + SQLite + NetworkX
- Claude (via OpenClaw sub-agents) for think cycles
- vis.js for dashboard
- cloudflared for sharing (ephemeral tunnels)
- pytest for testing

## Origin

Built by Raj and Bunny in a single weekend (March 7-8, 2026). Inspired by a Veritasium video on power laws watched months earlier. Named after "Aunty, do cats eat cashews?" — the question that started a lifetime of asking why.
