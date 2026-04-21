# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/rajkripal/cashew/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/rajkripal/cashew/releases/tag/v1.0.0
