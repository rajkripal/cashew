# CLAUDE.md — Cashew Developer Guide

Cashew is a thought-graph memory engine for AI agents. It stores knowledge as nodes in a SQLite graph, connects them with edges, and retrieves relevant context via recursive BFS over organic graph structure. Everything lives in one SQLite file — no external servers, no separate indexes.

**IMPORTANT**: Cashew does NOT call LLMs directly. It is purely the BRAIN (storage + retrieval + structure). The PROCESSOR (LLM) is external and provided by the orchestrator (OpenClaw) via `model_fn` parameters.

## Architecture

```
scripts/cashew_context.py  — Main CLI entry point (context, extract, think, sleep, stats)
cashew_cli.py              — Secondary CLI (init, install-crons, ingest)
core/                      — Engine modules
core/extractors.py         — Extractor plugin interface (BaseExtractor, ExtractorRegistry)
extractors/                — Built-in source extractors (obsidian, sessions, markdown)
extractors/utils.py        — Shared extractor utilities (frontmatter, wikilinks, ignore patterns)
integration/               — OpenClaw bridges
```

### Core Modules

| Module | Purpose |
|--------|---------|
| `config.py` | YAML config loading with env var expansion. Global `config` singleton. |
| `embeddings.py` | Sentence-transformer embeddings (all-MiniLM-L6-v2). sqlite-vec virtual table for O(log N) search. `search()`, `check_novelty()`, `embed_text()`. |
| `retrieval.py` | `retrieve_recursive_bfs()` — seeds via sqlite-vec, BFS graph walk, per-hop scoring. The sole retrieval method. |
| `context.py` | Composes retrieval results into formatted context strings for LLM consumption. |
| `session.py` | Session lifecycle: `start_session()`, `end_session()`, `think_cycle()`, `tension_detection()`. |
| `sleep.py` | Deep consolidation: cross-linking, decay, deduplication, core memory promotion. Runs daily. |
| `decay.py` | Prunes stale nodes (no access, no live edges, old). Sets `decayed=1`. |
| `traversal.py` | Graph walk utilities (DFS, BFS, path finding). |
| `stats.py` | Graph statistics and health metrics. |
| `export.py` | Export graph data to JSON for dashboard visualization. |
| `extractors.py` | Plugin interface: `BaseExtractor` (ABC), `ExtractorRegistry` (register, run, state persistence, dedup). |
| `graph_utils.py` | Shared utilities: `load_embeddings()`, `cosine_similarity()`. |

### Extractors

Built-in source extractors in `extractors/`:

| Module | Purpose |
|--------|---------|
| `obsidian.py` | Obsidian vault extraction. Frontmatter, `[[wikilink]]` edges, `.obsidianignore`, domain from folder structure. |
| `sessions.py` | OpenClaw session JSONL extraction. Incremental line tracking, filters tool calls/system messages. |
| `markdown_dir.py` | General markdown directory extraction. `.cashewignore` support. |
| `utils.py` | Shared: `parse_frontmatter()`, `extract_wikilinks()`, `load_ignore_patterns()`, `should_ignore()`, `split_into_paragraphs()`, `detect_domain_from_path()`. |

All extractors: use `model_fn` for LLM extraction when available, fall back to paragraph splitting when not. Checkpointing via `get_state()`/`set_state()` persisted automatically by the registry.

### Integration

| Module | Purpose |
|--------|---------|
| `integration/openclaw.py` | OpenClaw agent lifecycle hooks. `generate_session_context()`, `extract_from_conversation()`, `run_think_cycle()`. |

## Database Schema

SQLite with 3 tables + 1 virtual table:

- **`thought_nodes`** — Knowledge nodes (id, content, node_type, domain, access_count, decayed, permanent, tags)
- **`derivation_edges`** — Relationships (parent_id, child_id, weight)
- **`embeddings`** — Vector embeddings per node (node_id, vector as BLOB, model name)
- **`vec_embeddings`** — sqlite-vec virtual table for O(log N) nearest neighbor search (node_id, embedding float[384], cosine distance)

### Key Columns

- `domain` — Classifies who the knowledge belongs to. Configurable via config.yaml.
- `node_type` — One of: fact, observation, insight, decision, belief, derived, meta, core_memory, cross_link. Display/informational only — never used in filter or scoring logic.
- `decayed` — 0 or 1. Decayed nodes are excluded from retrieval but kept in DB.
- `permanent` — 0 or 1. Permanent nodes are immune to decay.
- `access_count` — Integer. How many times this node has been retrieved. Deterministic signal for think/tension gates and decay.
- `tags` — Comma-separated tags (e.g., `vault:private` for privacy classification).

## Retrieval: Recursive BFS

The sole retrieval method. No hotspots, no hierarchy, no DFS.

1. **Seed selection** — Embed query, find top-k nearest nodes via `vec_embeddings MATCH` (O(log N) with sqlite-vec, brute-force fallback)
2. **BFS traversal** — From seeds, explore graph neighbors up to `max_depth` hops. At each hop, score neighbors by cosine similarity, keep top `picks_per_hop`.
3. **Final ranking** — All candidates scored and sorted by similarity.

Parameters: `n_seeds=5`, `picks_per_hop=3`, `max_depth=3`.

The graph's organic connectivity (built by sleep cycle cross-linking) provides implicit hierarchy. No synthetic summary nodes needed.

## sqlite-vec Integration

Vector search lives inside the same SQLite file as the graph:

```sql
CREATE VIRTUAL TABLE vec_embeddings USING vec0(
    node_id TEXT PRIMARY KEY,
    embedding float[384] distance_metric=cosine
);
-- Query: O(log N) nearest neighbor
SELECT node_id, distance FROM vec_embeddings
WHERE embedding MATCH ? ORDER BY distance LIMIT 5;
```

- `embed_nodes()` dual-writes to both `embeddings` and `vec_embeddings`
- `search()` uses sqlite-vec when available, falls back to brute force
- `check_novelty()` uses sqlite-vec for O(log N) dedup checking
- `backfill_vec_index()` for one-time migration of existing embeddings

## Novelty Gate

Before any node is inserted, `check_novelty()` checks if content is too similar to existing nodes:
- Threshold: 0.82 cosine similarity (calibrated for MiniLM-L6 where true dupes peak at 0.85-0.90)
- Above threshold → reject (duplicate)
- Below threshold → accept (novel)
- Fails open — if embedding fails, the node gets through

## Sleep Cycle

Periodic background consolidation (`core/sleep.py`):
- **Decay** — Node fitness decays over time. Low-fitness nodes get `decayed=1`.
- **Cross-linking** — Finds semantically similar nodes across domains, creates edges.
- **Deduplication** — Merges near-identical nodes.
- **Core memory promotion** — Frequently accessed nodes get promoted.
- **Think cycle penalty** — Think-cycle-generated nodes face 1.5x higher decay threshold.

No hotspot creation, no clustering, no hierarchy maintenance. The graph self-organizes through cross-linking and decay.

## Configuration

`config.yaml` at project root. Key settings:
- `database.path` — SQLite DB location
- `domains.user` / `domains.ai` — Domain names
- `models.embedding.name` — Embedding model
- `performance.*` — Token budgets, thresholds, top-k

Environment variables override config: `${VAR:-default}` syntax supported in YAML.

## Conventions

- **Node IDs** — 12-char hex strings from content hash (`hashlib.sha256(content)[:12]`)
- **Timestamps** — ISO 8601 strings
- **Edges are untyped** — Topology carries the signal, not labels. In an early ablation (34 seed nodes, religion-debate test set, Claude Sonnet, 12 think cycles per method), structured graphs scored 9/10 on qualitative signatures vs 2/10 for random-edge graphs — topology matters. But an edge-label-swap variant (flipping supports↔contradicts, questions↔derived_from) produced 2.3x more conclusions than the correctly labeled version (84 vs 36), suggesting typed semantics act as constraints that narrow LLM exploration rather than guides that improve it. Edges are stored as pure topology; the LLM infers relationships from node content at read time. Ablation was small and single-domain — design-directional, not a benchmark.
- **Embeddings** — 384-dimensional numpy arrays stored as BLOBs (and in vec_embeddings as float[384])
- **Never delete nodes** — set `decayed=1` instead. The graph is append-mostly.
- **Always embed new nodes** — every node must have rows in both `embeddings` and `vec_embeddings`.
- **Domain assignment** — every node must have a domain. Use `config.get_user_domain()` and `config.get_ai_domain()`.
- **No hardcoded paths** — use `config.get_db_path()` or accept `--db` CLI arg.
- **No direct LLM calls** — Accept `model_fn` parameters, never create LLM clients internally.

## External LLM Pattern

Functions that accept `model_fn`:
- `extract_from_conversation(model_fn=None)` — Uses heuristic extraction if None
- `run_think_cycle(model_fn=None)` — Skips think cycles if None
- `run_sleep_cycle(model_fn=None)` — Uses fallback summaries if None

The CLI auto-discovers a running OpenClaw gateway from `~/.openclaw/openclaw.json` and routes LLM calls through it via the `/v1/chat/completions` endpoint (OpenAI-compatible). No API keys or provider-specific code needed.

## Context Injection Pattern

`generate_session_context(db_path, hints)` returns a pre-formatted string for system prompt injection:

```
=== GRAPH OVERVIEW ===
Graph: 2160 nodes, 3499 edges. Types: 499 fact, 449 insight, 426 observation.

=== RECENT ACTIVITY ===
1. [fact] Some recent fact... (today)

=== RELEVANT CONTEXT ===
1. [FACT] Relevant node content (Domain: user)
2. [DECISION] Another relevant node (Domain: ai)

=== END CONTEXT ===
```

Always use `cashew context --hints "..."` or `generate_session_context()` — never assemble prompts from raw node lists.

## Privacy

- Nodes can be tagged `vault:private` to exclude from group channel queries
- Extraction defaults to private; periodic declassification via `scripts/declassify.py`
- `--exclude-tags vault:private` on context queries filters private nodes

## Engineering Philosophy

These are non-negotiable. Every PR, every fix, every new feature. 

Sourced from the cashew brain (bunny domain). For the full set:
```bash
cashew_context.py context --hints "bunny engineering principles best practices testing rules"
```

### Testing
- **End-to-end tests over unit tests.** Don't write 5 tests for the same helper with varying inputs. Each test should exercise a REAL user flow from start to finish.
- **Tests ship with the code — not optional, not afterthought.** No PR without tests.
- **When fixing a bug, write the test that catches it FIRST.** The test is more important than the fix — it prevents regressions forever.
- **Failures must be LOUD.** Silent failures (returning 0 results, empty arrays, swallowed exceptions) are worse than crashes. If something goes wrong, print why and exit non-zero.
- **Test isolation is mandatory.** Use `tmp_path`, clean up after yourself, never touch production data from tests.

### Code
- **Config path isolation.** If a user specifies `--config-path /somewhere/config.yaml`, ALL paths (DB, logs, backups, models) must resolve relative to that location. Never hardcode `~/.cashew/` anywhere.
- **Environment variables override everything.** `CASHEW_CONFIG_PATH` and `CASHEW_DB_PATH` must be respected by every script, every module, every time.
- **Backward compatibility is sacred.** Existing setups must not break. Ever. Test this explicitly.
- **Never `git add -A`.** Always specific files.
- **Good quality code is a craft.** Human makes all philosophical and design decisions; AI handles syntax.
- **Every question the setup wizard asks must result in something ACTUALLY happening.** No dead config values.

### Architecture
- **Dumb graph, smart reasoning layer.** The graph stores connections. The LLM reasons about meaning. Don't put intelligence in the graph structure.
- **No edge semantics.** Typed edges were ablation-tested (topology scored 9/10 vs 2/10 random; label-swap produced 2.3x more conclusions, suggesting semantics constrain rather than guide) and dropped. No reintroductions under new names — topology stays, labels do not.
- **No temporal scaffolding.** Timestamps are node metadata, not graph structure.
- **Fractal simplicity.** Simple rules, emergent behavior. If a feature adds structural complexity, the burden of proof is on the complexity.
- **Organic decay is the forgetting mechanism.** Don't build structures that fight decay.
- **The philosophy layer is load-bearing architecture, not optional.** Without it cashew is just a memory tool; with it each instance becomes distinct.
- **Discovery tool, not prediction engine.** Surface connections you didn't know existed, then human decides what to do. One hop, human in loop immediately.
- **First-experience philosophy: aim for "holy shit this works" reaction.** Immediate value over technical showcase.

### Process
- **Write the switchboard, don't BE the switchboard.** If you find yourself doing something manually that should be automated, automate it. Scripts for deterministic tasks.
- **An AI should be able to install this itself.** If `cashew init` → import → query doesn't work without human intervention, it's broken.
- **Ship at 80%.** Perfect is the enemy of shipped. Blog 1 shipped at 80% and had massive impact.
- **One feature per PR.** Always use PRs, never push to main.
- **3-exchange rule.** If bot-to-bot hits 3 rounds without an artifact, stop.
- **Pipeline awareness.** Code is 1 month, review is 2 more. Factor this into commitments.
- **Inject principles into CLAUDE.md** so every worker inherits them. Don't hand-hold individual workers with detailed prompts.
- **Shipping means ALL artifacts are consistent.** When a feature lands, scan and update every file that references the affected surface: README, CLAUDE.md, SKILL.md, CLI help text, inline docstrings, config templates. Feedback is about a class of issues, not point fixes. If the README is stale, assume every doc is stale and audit all of them.

## Testing

```bash
python -m pytest tests/          # Full test suite
python scripts/test_e2e.py       # Quick end-to-end test
```

## Common Tasks

**Add a new core module:**
1. Create `core/your_module.py`
2. Import config with `from core.config import config, get_db_path`
3. Add CLI subcommand in `scripts/cashew_context.py` if user-facing
4. Update this file

**Modify the schema:**
1. Add migration script in `scripts/migrate_*.py`
2. Update the init schema in `cashew_cli.py` `cmd_init()`
3. Update this doc's schema section

**Add a new extractor:**
1. Create `extractors/your_source.py` subclassing `BaseExtractor`
2. Implement `name`, `extract()`, `get_state()`, `set_state()`
3. Put shared utilities in `extractors/utils.py`
4. Register in `cmd_ingest()` in `cashew_cli.py`
5. Add tests in `tests/test_extractors.py`
6. Update README (Ingest Sources section + CLI Reference table), this file (Extractors table), and skill files
