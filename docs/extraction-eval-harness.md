# Extraction-quality eval harness

`scripts/eval_extraction_quality.py` runs cashew's extraction pipeline
(`extract_from_conversation`) against a set of sample transcripts, once per
LLM backend, and reports objective counts: nodes created, edges created,
empty extractions, wall-clock time, and token usage where the backend
reports it.

This exists to eventually answer: "what does running extraction on a local
model cost in extraction quality/throughput vs. the API/subscription
backend" (the "sovereignty tax" question). It does not answer that question
yet — see Limitations below.

## What it does

- Loads every `.txt` file in `eval/sample_transcripts/` (or `--transcripts-dir`)
  as one conversation transcript.
- For each backend named in `--backends` (comma-separated), builds the
  backend via `core.llm.build_backend(name)`, creates a fresh throwaway
  SQLite DB (`scripts.cashew_init.create_database_schema`), and runs
  extraction on every transcript into that DB.
- Prints per-backend totals and, if exactly two backends were run, a raw
  node-count and wall-time delta between them.
- Optionally writes full per-transcript results to a JSON file (`--output`).

## Running it

```bash
# single backend (sanity check / smoke test)
python scripts/eval_extraction_quality.py --backends claude_code

# compare two backends once a second one exists (see Limitations)
python scripts/eval_extraction_quality.py --backends claude_code,ollama_qwen \
  --output results.json
```

Smoke-tested on 2026-07-18 with one sample transcript against `claude_code`
(model `claude-opus-4-7`): 2 nodes, 0 edges, 0 failures, ~8.5s. That run is
in the codebase's git history as evidence the harness executes end-to-end;
it is not a benchmark result.

## Limitations (as of 2026-07-18)

- **No local-model backend exists yet.** `core/llm.py` only implements
  `ClaudeCodeBackend` (shells out to headless `claude -p`, i.e. still the
  API/subscription path, just with the model name configurable via
  `CASHEW_CLAUDE_MODEL`). To actually run the local-vs-API comparison this
  harness is meant for, someone needs to add a new `LLMBackend` subclass
  (e.g. `OllamaBackend`) that talks to a local Qwen or Llama instance
  (via `ollama serve` or similar) and register it in `build_backend()`.
  That's a separate, non-trivial task (get a suitable model running on the
  Mac Mini, tune the extraction prompt format for it, handle its slower/
  less structured JSON output) and is explicitly out of scope for this
  harness.
- **Only two sample transcripts exist** (`eval/sample_transcripts/`). They
  were written by hand for smoke-testing the harness, not sampled from real
  conversation history. A real eval needs a larger, representative set —
  ideally pulled from actual extraction sessions (with private content
  scrubbed).
- **No quality judge.** The harness intentionally only reports countable
  stats (node/edge counts, timing, token usage), not a "was this a good
  extraction" score, because that requires either human review or an
  LLM-judge harness. `bench/free_form_judge.py` on branch
  `feat/membench-free-form-judge` has an AttrScore-style judge pattern for
  a different track (retrieval QA) that could be adapted for extraction
  quality, but that adaptation hasn't been done.
- **No "sovereignty tax" number has been measured.** This harness makes
  that measurement possible once a local backend exists; it does not
  produce the number itself.
