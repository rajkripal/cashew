# RECON — cashew x LoCoMo

Recon brief for the locomo-cashew team. Purpose: give adapter-builder, runner-builder, and smoke-tester everything they need without re-doing the survey.

## TL;DR

A previous subagent already cloned LoCoMo and wrote both the adapter and a checkpointed runner. They are **substantially complete** and the cashew imports they rely on all resolve. The remaining team work is mostly verification, sleep/think wiring for the A/B ablation, smoke testing, and launchd glue.

- Repo cloned at `/Users/bunny/.openclaw/workspace/benchmarks/locomo/` (full snap-research/locomo).
- Adapter: `benchmarks/locomo/cashew_adapter.py` (433 lines, working).
- Runner: `benchmarks/locomo/run_cashew_locomo.py` (395 lines, checkpointed, rate-limit aware).
- Output dir: `/Users/bunny/.openclaw/workspace/cashew/papers/locomo-run/`.
- Prior partial run state present: 1 conv (`conv-26`) had all 9 sessions ingested before the previous subagent was killed. No QA answered yet, no `results.jsonl`.

No blockers. Data is public, not gated. All cashew entry points exist and import cleanly.

## LoCoMo data layout

File: `benchmarks/locomo/data/locomo10.json` — 10 conversations, **1986 QA pairs total**.

Per-sample top-level keys: `sample_id`, `conversation`, `qa`, `event_summary`, `observation`, `session_summary`.

`conversation` shape:
- `speaker_a`, `speaker_b` — names (strings).
- `session_<n>` — list of turn dicts. Turn = `{speaker, dia_id, text, [img_url, blip_caption, query]}`. `dia_id` example: `D1:3` (session 1, turn 3).
- `session_<n>_date_time` — string like `"1:14 pm on 25 May, 2023"` (parser already in adapter, `locomo_date_to_iso`).
- Sessions per conversation range 19–32 (mean ~27). Total 270 sessions across 10 convs.

`qa` shape: list of `{question, answer, evidence: [dia_id...], category: 1..5}`. Some category-5 ("adversarial / not-mentioned") items lack `answer` — runner already handles by substituting the LoCoMo sentinel string.

QA distribution: cat1=282, cat2=321 (date), cat3=96, cat4=841, cat5=446.

## Eval interface

LoCoMo's stock eval (`task_eval/evaluate_qa.py` + `task_eval/evaluation.py::eval_question_answering`) is **closed-book per question**: it stuffs the entire conversation into a long-context model and scores predictions vs gold using:

- `f1_score(prediction, gold)` — token-level F1 over normalized strings (lowercase, strip punctuation, drop a/an/the/and). Implementation lines `evaluation.py:126-138`.
- `exact_match_score` — normalized string equality.
- Cat-2 (date) gets a special grader; cat-5 uses MCQ randomized between gold and "Not mentioned in the conversation".
- Optional bert_score is imported at top of `evaluation.py` — heavy dep we do NOT pull in. Adapter ports the F1 + EM logic locally to avoid bert_score/torch.

**External-memory feedback shape** the LoCoMo evaluator expects in their pipeline: write `out_samples[sample_id]['qa'][i][f"{model_key}_prediction"]` (string) plus optional `..._f1`, `..._recall`. Then call `analyze_aggr_acc(...)` to get per-category breakdown. Our runner sidesteps that and writes its own `results.jsonl` + `summary.json` in the same metric definitions, so we don't depend on bert_score or `task_eval`'s shell scripts.

The runner format per result line:
```
{conv_id, q_idx, category, question, gold, pred, f1, precision, recall, exact_match,
 retrieval_latency_s, answer_latency_s, retrieved_chars,
 judge_input_tokens, judge_output_tokens, ts}
```

## Cashew API surface (verified, all imports resolve)

### Extract path — entry point: `integration.session.extract_from_conversation`

Signature: `extract_from_conversation(db_path, conversation_text, session_id, model_fn, referent_time=None, infer_referent_time=False) -> dict`. Wraps `core.session.end_session`. **Requires a `model_fn(prompt)->str` for LLM extraction**; without it the implementation logs a warning and falls back to heuristics only (very lossy — do NOT run ablation without `model_fn`).

Return dict: `{success, new_nodes (count), new_edges, updated_nodes, node_ids, edges, summary}`. Side effect: nodes inserted into `thought_nodes`, edges into `thought_edges`.

### `scripts/cashew_context.py extract --ingest <json>` standalone behavior

Q: does `extract --ingest` work standalone or does it need a model_fn?

A: **It works standalone — no model_fn required**, but it is NOT a full extraction. Implementation at `scripts/cashew_context.py::_cmd_extract_ingest` (lines 220-273): reads a JSON `{insights: [{content, type, domain}, ...]}`, calls `core.session._create_node` for each, applies tags, done. The LLM step is presumed to have already produced the `insights` array (that's the `--prepare-only`/external-LLM split path). For the LoCoMo ablation we want the full LLM-extract path, so use `extract_from_conversation` with a real `model_fn`, NOT `--ingest`.

The full-LLM CLI form is `python3 scripts/cashew_context.py extract --input <conv.txt> --session-id <sid> --db <path>`. It builds a model_fn internally via `_build_model_fn()` (claude shell-out). The adapter bypasses this CLI and calls `extract_from_conversation` directly with our own `claude -p` model_fn — cleaner, more controllable, what we want.

### Sleep — entry point: `core.sleep.run_sleep_cycle`

Signature: `run_sleep_cycle(db_path=None, model_fn=None) -> dict`. Constructs a `SleepProtocol(db_path)` and runs it. `model_fn` optional — without it, LLM-dependent sleep features (cross-link, novelty re-eval) are skipped silently. For the A/B "full pipeline" arm, pass our claude `model_fn`. Returns a stats dict (keys vary; adapter snapshots node count before/after as ground truth).

Side effects: updates node activations, decays, may add cross-link edges, may collapse near-duplicates.

### Think — entry point: `integration.session.run_think_cycle`

Signature: `run_think_cycle(db_path, focus_domain=None, model_fn=None) -> dict`. Wraps `core.session.think_cycle`. **Requires model_fn** — returns `{success: False, error: "no model"}` immediately if missing. Returns `{success, new_nodes, new_edges, cluster_topic, node_ids, edges, summary}`.

Note: there is **no `core/think.py`** as the task brief assumed — think lives inside `core/session.py::think_cycle`. The integration wrapper is the public surface.

### Context retrieval — entry point: `integration.session.generate_session_context`

Signature: `generate_session_context(db_path, hints=None, tags=None, exclude_tags=None) -> str`. Returns a **formatted string** (header + 3-layer context: tree overview + recent activity + hint-relevant nodes), or empty string on failure. Token estimate is appended to the trailing `*Context layers...*` line.

This is what the adapter's `retrieve()` calls. The string is what we feed back to the answering LLM — no JSON, no structured retrieval interface.

For tighter control there is also `core.context.ContextRetriever` (class with `_extract_keywords`, `_calculate_relevance_score`) returning `RelevantNode` dataclasses, but the wrapped string from `generate_session_context` is the supported public path and what the adapter uses today.

### Schema bootstrap

`core.session._ensure_schema(db_path)` creates a fresh sqlite DB with current schema (v3). The adapter calls this in `init_conv_db` to make per-conversation isolated DBs at `papers/locomo-run/dbs/<conv_id>-<variant>.db`. Variant slot ("A"/"B") is already plumbed for the ablation.

### Daemon caveat (for retrieval)

`generate_session_context` will try to use a shared cashew daemon if one is running. The adapter sets `CASHEW_NO_DAEMON=1` before each `retrieve()` so per-conversation DBs are read in-process. Keep that flag — without it, daemon lookups will hit the wrong DB.

## Salvageable prior work (very high — re-use, don't rebuild)

`benchmarks/locomo/cashew_adapter.py` already provides:
- `claude_p(prompt, model, timeout)` — headless `claude -p` wrapper with rate-limit detection (markers: `rate_limit_error`, `429`, `overloaded_error`, `usage limit reached`, `Too Many Requests`). Raises `RateLimitError` or `RuntimeError`. Uses an empty-MCP config to avoid loading user MCP servers. **Reuse as-is.**
- `make_model_fn(model)` — wraps `claude_p` in a callable with a `.usage` accumulator; cashew-compatible.
- `init_conv_db(out_dir, conv_id, variant)` — per-conv-per-variant DB, schema-bootstrapped.
- `count_nodes(db)`, `run_sleep(db, model_fn)`, `run_think(db, model_fn)` — wrappers that capture before/after node counts and latency. Ready for the ablation runner.
- `locomo_date_to_iso(s)` — parses `"1:14 pm on 25 May, 2023"` → ISO8601 (used as referent_time for cashew temporal reasoning).
- `session_to_text(date, speakers, turns)` — renders a session as `DATE / SPEAKERS / lines` with image captions inlined.
- `ingest_session(db, conv_id, idx, date, speakers, turns, model_fn)` — full LLM extract per session.
- `question_to_hints(q)` — stopword-filtered token extraction for retrieval hints (≤8 hints).
- `retrieve(db, q)` — sets `CASHEW_NO_DAEMON=1`, calls `generate_session_context`, returns `(ctx, latency)`.
- `answer_question(ctx, q, category, gold, judge_model)` — three prompt templates (base, date for cat-2, MCQ for cat-5) with cat-5 randomization mirroring LoCoMo.
- `f1_score`, `exact_match`, `_normalize` — LoCoMo metric port (no bert_score dep).

`benchmarks/locomo/run_cashew_locomo.py` already provides:
- Checkpoint at `papers/locomo-run/checkpoint.json` with `conversations_done/failed`, `current_conv/question_idx`, `ingested_sessions`, `resume_after_iso`, `retry_count`.
- Rate-limit gate: on `RateLimitError`, sets `resume_after_iso = now + 4h` and exits 0 cleanly so launchd doesn't disable the job.
- `--smoke` (first conv, first 3 questions), `--batch N` (default 5), `--reset`.
- Per-question retry up to 2 before skipping.
- `finalize()` writes `summary.json`, `summary.md`, `COMPLETE` marker. Aggregates by category and by conversation. Tracks judge token usage.
- Always exits 0 (launchd-safe) — fatals are logged to `~/bunny-bridge-logs/locomo-runner.log`.

What the existing runner does **not** yet do:
- **No sleep/think hook in the loop.** Adapter has `run_sleep` / `run_think` but the runner never calls them. The A/B ablation needs the runner to call sleep (and optionally think) after ingest in the "full pipeline" arm and skip them in the "ingest-only" arm. Adapter is already plumbed for variant "A"/"B" DB names; runner is hard-coded to variant "A".
- **No A/B variant flag** on the runner CLI (e.g. `--variant ingest_only|full`).
- **No second-pass evaluation** to compare arms — finalize only summarizes one run.
- Models are hardcoded to `claude-sonnet-4-6` via `LOCOMO_JUDGE_MODEL`/`LOCOMO_EXTRACT_MODEL` env (fine, just note it).
- `extract_from_conversation` returns counts under keys `new_nodes`/`new_edges`, but `ingest_session` log line says `nodes+={res.get('new_nodes')}` — works, but adapter silently ignores ingest errors per session. Worth a smoke check.

## Prior run state

`papers/locomo-run/`:
- `checkpoint.json` — `current_conv: conv-26`, all 9 sessions ingested, 0 QA answered, no rate limit gate.
- `dbs/conv-26.db` — populated.
- No `results.jsonl`, `summary.json`, `COMPLETE` yet.

The runner-builder can resume from this state directly, or `--reset` and start fresh. The smoke-tester should `--smoke` against a clean `conv-` to validate end-to-end before any full run.

## Blockers

None hard. Two soft items:

1. **`bert_score` import in `task_eval/evaluation.py`** — if anyone tries to use LoCoMo's stock eval scripts (`evaluate_claude.sh` etc.) they need the heavy stack. Our runner avoids this entirely; only flag it if someone proposes "just use their script."
2. **Daemon contention** — if a long-running cashew daemon points at the main brain DB, retrieval against per-conv DBs must keep `CASHEW_NO_DAEMON=1` set. Adapter handles it; don't strip it.

## Recommended next moves for downstream teammates

- **adapter-builder (#2):** keep `cashew_adapter.py` as-is. Add a `--variant` plumbing helper if any (one-line change to thread variant through `init_conv_db`). Confirm `run_sleep`/`run_think` actually run end-to-end on a small ingested DB. Don't rewrite — ratchet.
- **runner-builder (#3):** add `--variant {ingest_only,full}`; in `full` call `run_sleep(db, model_fn)` after `ensure_all_sessions_ingested` (and optionally `run_think` once per N sessions). Thread variant into `init_conv_db` and `RESULTS` filename. Keep checkpoint, gate, exit-0 behavior.
- **smoke-tester (#4):** `python3 run_cashew_locomo.py --reset --smoke` against both variants. Verify `results.jsonl` lines have non-zero `retrieved_chars`, F1 > 0 on at least 1 of 3 questions, no `RateLimitError` on a fresh run. Tail `~/bunny-bridge-logs/locomo-runner.log`.

## File locations (absolute)

- LoCoMo data: `/Users/bunny/.openclaw/workspace/benchmarks/locomo/data/locomo10.json`
- Adapter: `/Users/bunny/.openclaw/workspace/benchmarks/locomo/cashew_adapter.py`
- Runner: `/Users/bunny/.openclaw/workspace/benchmarks/locomo/run_cashew_locomo.py`
- LoCoMo stock eval (reference only): `/Users/bunny/.openclaw/workspace/benchmarks/locomo/task_eval/evaluation.py`
- Cashew extract entry: `/Users/bunny/.openclaw/workspace/cashew/integration/session.py::extract_from_conversation`
- Cashew sleep entry: `/Users/bunny/.openclaw/workspace/cashew/core/sleep.py::run_sleep_cycle`
- Cashew think entry: `/Users/bunny/.openclaw/workspace/cashew/integration/session.py::run_think_cycle` (real impl in `core/session.py::think_cycle`)
- Cashew context entry: `/Users/bunny/.openclaw/workspace/cashew/integration/session.py::generate_session_context`
- Cashew CLI: `/Users/bunny/.openclaw/workspace/cashew/scripts/cashew_context.py`
- Output dir: `/Users/bunny/.openclaw/workspace/cashew/papers/locomo-run/`
- Runner log: `/Users/bunny/bunny-bridge-logs/locomo-runner.log`
