# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- Extractor parser stored markdown headers (`# Extracted Insights`, `## Decisions`) as no-value nodes when LLMs prepended them to extraction output. `extractors/utils.parse_extraction_lines` now drops `#`/`---` lines; sessions, markdown_dir, and obsidian extractors all route through it (#13).
- Think cycle's diversity-sampling source_file filter only matched `%system_generated%`, so `extractor:*`-ingested nodes never populated the random-walk pool on freshly ingested graphs (#15).
- Think cycle's high-activation pool sorted by `last_accessed` even when every node had `access_count=0`, deterministically returning the same oldest-ingested nodes across runs. Now applies a random tiebreaker when access_count is zero, transitioning to recency-based ordering as access history accumulates (#16).

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

[Unreleased]: https://github.com/rajkripal/cashew/compare/v1.0.1...HEAD
[1.0.1]: https://github.com/rajkripal/cashew/releases/tag/v1.0.1
[1.0.0]: https://github.com/rajkripal/cashew/releases/tag/v1.0.0
