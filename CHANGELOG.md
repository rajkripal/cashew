# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.2.1] - 2026-06-29

### Fixed
- `'SentenceTransformer' object has no attribute 'get_embedding_dimension'` crash on every embed and extract call when installed against sentence-transformers < 5.5. v1.2.0 pre-emptively migrated to the new API name (#88) but sentence-transformers wasn't pinned, so existing installs broke. `core/embedding_service.py` now resolves the dim through a `_model_dim` shim that tries `get_embedding_dimension` first and falls back to the deprecated `get_sentence_embedding_dimension`, supporting both API generations.

## [1.2.0] - 2026-06-28

This release pairs a major embedding upgrade with a substantial sleep-cycle reshape. Default embeddings move from MiniLM-384 to gte-large-1024; the sleep cycle is now work-capped and vectorized with batched DB writes; per-model similarity thresholds replace the previously hardcoded MiniLM-calibrated constants. A smooth-upgrade path (`cashew migrate-embeddings` + dim-mismatch warning) lets existing brains migrate cleanly. SQLite concurrency is hardened across every connection site with `PRAGMA busy_timeout`. New `ClaudeArchiveExtractor` ingests claude.ai conversation archives.

### Changed
- **Default embedding model is now `thenlper/gte-large` (1024-dim), replacing `all-MiniLM-L6-v2` (384-dim)** (#46). Benchmarked on a 3,898-node brain, gte-large consumed ~300 MB less peak RAM than MiniLM because fastembed routes it through a quantized ONNX backend while MiniLM loads via the heavier sentence-transformers / PyTorch runtime. Retrieval latency increases ~10ms p50 (still well under 100ms). Better recall on harder queries.
- **Per-model similarity-constant profile layer** (#82). Sleep-cycle thresholds (cross_link, dedup, novelty) are now sourced from a per-model profile rather than hardcoded MiniLM-era constants. Prevents the cross-link edge saturation seen when running the old thresholds against gte-large's tighter cosine distribution.
- **Sleep cycle is now work-capped and vectorized** with batched DB writes (#66). Drops per-cycle wall-clock substantially on large graphs; bounds memory and write amplification with explicit caps rather than implicit loop limits.
- **`node_type` taxonomy is derived from config** instead of duplicated across the codebase (#39). Removes drift between extraction and storage.

### Added
- **`cashew migrate-embeddings` CLI command** (#47) for upgrading an existing brain to the currently configured embedding model. Wipes the `embeddings` table and `vec_embeddings` virtual table, then re-runs the embed pass. Accepts `-y/--yes` for scripted upgrades.
- **Prominent dim-mismatch warning** (#47) on first embed/search against a brain whose stored embedding dim doesn't match the configured model. Tells the user exactly what to run, or how to pin the old model via `CASHEW_EMBEDDING_MODEL`. Fires once per process per db.
- **`ClaudeArchiveExtractor`** (#61) for ingesting claude.ai conversation archives directly into the graph.
- **Shared LLM extraction helper** (#62) consolidates extraction-call plumbing across extractors.
- **CI pytest workflow** runs on every push and PR across Python 3.10 / 3.11 / 3.12 (#81).
- **`decay_audit` table** writes a row on every soft-decay, and `cashew audit` / brain-metrics now surface decay history.
- **`hermes-cashew` integration link** in the README (#48).

### Fixed
- `_embed_orphans` in the sleep cycle hardcoded MiniLM, ignoring the configured embedding model. Now uses `DEFAULT_EMBEDDING_MODEL` so upgraded brains re-embed orphans under the right model (#96).
- `core.permanence` had `EXPECTED_EMBEDDING_DIM=384` hardcoded and would have failed integrity checks on any non-MiniLM brain. Now resolves the dim dynamically.
- `get_default_service` ignored the configured embedding model and always built a service against the install-time default (#54).
- Embedding service now honors `CASHEW_EMBEDDING_MODEL` and resolves the model dim dynamically.
- Extract `--ingest` crashed on bare JSON array inputs (#74); now handled.
- Redundant `skip_llm` kwarg leaked from `cashew_context` into extractors and ignored caller intent (#51).
- Replaced deprecated `get_sentence_embedding_dimension()` calls with `get_embedding_dimension()` to silence sentence-transformers 5.5+ FutureWarnings (#88).
- Extractor parser stored markdown headers (`# Extracted Insights`, `## Decisions`) as no-value nodes when LLMs prepended them to extraction output.
- Think cycle's diversity-sampling source_file filter only matched `%system_generated%`, so `extractor:*`-ingested nodes never populated the random-walk pool on freshly ingested graphs.
- Think cycle's high-activation pool sorted by `last_accessed` even when every node had `access_count=0`, deterministically returning the same oldest-ingested nodes. Now applies a random tiebreaker when access_count is zero.
- Sleep cluster-merge synthesis preserves temporal anchors when collapsing duplicates (#45).
- Extraction prompt's GOOD examples replaced with personal-knowledge templates that don't leak generic-corpus phrasing into the output (#42).
- Dashboard skips `spring_layout` on large graphs and uses random positions instead, preventing multi-minute hangs (#77).

### Performance
- **Lazy-load edges in retrieval BFS** (#80) reduces context-retrieval query latency by avoiding upfront edge materialization for unreachable subtrees.
- **`_graph_walk` IN-subqueries replaced with JOINs** (#79) eliminates per-step query-planner overhead during graph traversal.

### Internal
- `PRAGMA busy_timeout = 5000` is now set on every `sqlite3.connect()` site across the codebase (#56, #60, #83, #89), preventing dead-locks under concurrent access from sleep cycles, extractors, CLI, retrieval, embedding cache, and stats.
- Think-ingest test isolated from the real brain (#57).
- Stale embedding-dimension assertions updated for gte-large default across the test suite (#71).

### Migration

Existing cashew brains built under MiniLM-384 keep working — retrieval falls back to brute-force similarity, and `embed_nodes` prints a clear warning telling you how to upgrade. To migrate:

```
cashew migrate-embeddings -y
```

This wipes existing embeddings and re-embeds every node under the new default. Takes ~17 seconds per 1,000 nodes on Apple Silicon. Nodes, edges, and sleep-cycle structure are preserved; only the vector representations are recomputed.

If you'd rather keep the old model, set:

```
export CASHEW_EMBEDDING_MODEL=all-MiniLM-L6-v2
```

before running cashew.

## [1.1.0] - 2026-05-03

"Dumb graph, smart reasoning" invariant is now enforced end-to-end. No semantic field is read for filter or scoring logic anywhere in the engine. Plus a quality-of-extraction upgrade: LLM-assigned types instead of post-hoc keyword classification.

### Changed
- **Confidence column is gone** (#25). It was an uncalibrated LLM self-report (~70% piled at 0.85+, no signal). Replaced with deterministic graph gates: think + tension candidate selection uses `access_count > 0 OR edge_degree > 0`; decay eligibility uses `access_count == 0 AND age >= 14d AND edge_degree == 0`. Auto-migrates v1 → v3 on next session start.
- **`node_type` is display-only** (#26, #27). All semantic node_type reads in WHERE clauses and scoring removed. Protection of important nodes now flows through a single signal: the `permanent` flag (legacy seed/core_memory nodes backfilled to `permanent=1`).
- **LLM types each statement at extraction time** (#29, closes #12) with one of `[fact|observation|insight|decision|commitment|belief]` instead of post-hoc keyword classification. Falls back to legacy classifier if the LLM omits a tag.

### Added
- **LLM-backed dream synthesis** with template fallback (#22).
- **N-plicate cluster merge** via Bron-Kerbosch replaces pair-wise dedup (#23).
- **Embeddings-integrity health check** catches zero-norm/NaN/wrong-dim vectors and orphan rows (#24).
- **`.cashewignore` support** in `SessionExtractor` (#20).
- Document `commitment` as a distinct core node type in config (#14).

### Fixed
- Permanent nodes protected from GC and dedup decay paths (#21).
- Sleep `dedup_threshold` aligned with DESIGN.md (0.9 → 0.82) (#17).
- Think cycle cold-start determinism + extractor header noise (#18).
- Claude Code plugin inheritance in headless `claude -p` calls (#19).

## [1.0.1] - 2026-04-23

### Fixed
- Packaging: `extractors` package was missing from the built wheel, so `pip install cashew-brain` crashed with `ModuleNotFoundError: No module named 'extractors'` on first run (#7).

### Added
- Public schema API: `core.db.ensure_schema()`, `core.db.get_schema_version()`, and `core.db.schema_version()` for downstream consumers embedding cashew as a library (#4, #6).
- DESIGN.md "Schema Ownership Contract" section documenting cashew-owned tables and columns, the additive-only migration policy within a major version, and the `ext_*` extension namespace for downstream layers (#5).
- `PRAGMA user_version` is now stamped on every database so downstream migrations can branch on applied version.
- Regression test (`tests/test_packaging.py`) asserting every top-level importable package is declared in `pyproject.toml`.

## [1.0.0] - 2026-04-21

First public release on PyPI. `pip install cashew-brain` now works.

### Added
- Persistent thought-graph memory backed by SQLite.
- Local embeddings via `sentence-transformers` (default model: `all-MiniLM-L6-v2`).
- `ContextRetriever` for RAG-style context generation from prior turns.
- Knowledge extraction from completed turns with configurable `model_fn`.
- Autonomous think cycles with organic decay.
- CLI entry point (`cashew`).
- PyPI trusted-publishing release workflow (OIDC, no API tokens).

[Unreleased]: https://github.com/rajkripal/cashew/compare/v1.2.1...HEAD
[1.2.1]: https://github.com/rajkripal/cashew/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/rajkripal/cashew/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/rajkripal/cashew/compare/v1.0.1...v1.1.0
[1.0.1]: https://github.com/rajkripal/cashew/releases/tag/v1.0.1
[1.0.0]: https://github.com/rajkripal/cashew/releases/tag/v1.0.0
