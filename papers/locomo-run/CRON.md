# LoCoMo full-run cron wiring

The full benchmark runs autonomously via launchd. Each tick claims the
existing checkpoint, processes a batch (up to a wall-clock cap inside
`run_cashew_locomo.py`), writes the checkpoint, and exits. The next tick
resumes. If a rate-limit gate is hit, the runner records `resume_after_iso`
in the checkpoint and short-circuits subsequent ticks until that timestamp
passes.

## Files

- **plist:** `/Users/bunny/Library/LaunchAgents/com.bunny.job.locomo-runner.plist`
  - `Label`: `com.bunny.job.locomo-runner`
  - `StartInterval`: `1800` (every 30 minutes)
  - `RunAtLoad`: false
  - Stdout: `/Users/bunny/bunny-bridge-logs/locomo-runner.launchd.out`
  - Stderr: `/Users/bunny/bunny-bridge-logs/locomo-runner.launchd.err`
- **prompt:** `/Users/bunny/bunny-claude-bridge/cron/prompts/locomo-runner.md`
  (silent worker — runs the runner, prints last log line)
- **runner script:** `/Users/bunny/bunny-claude-bridge/scripts/run-job.sh locomo-runner`
- **runtime log:** `/Users/bunny/bunny-bridge-logs/locomo-runner.log`
- **benchmark code:** `/Users/bunny/.openclaw/workspace/benchmarks/locomo/run_cashew_locomo.py`
- **state/output dir:** `/Users/bunny/.openclaw/workspace/cashew/papers/locomo-run/`
  - `checkpoint.json`, `results.jsonl`, `dbs/`, `summary.{json,md}`

## Manual operations

- Trigger now: `launchctl start com.bunny.job.locomo-runner`
- Verify loaded: `launchctl list | grep locomo-runner`
- Unload: `launchctl unload /Users/bunny/Library/LaunchAgents/com.bunny.job.locomo-runner.plist`
- Tail progress: `tail -f /Users/bunny/bunny-bridge-logs/locomo-runner.log`
- Reset state (DESTRUCTIVE): `python3 run_cashew_locomo.py --reset`

## Expected wall-clock

Smoke timings: ~6s for 3 questions on conv-26 ~= 2s/q ideal once snapshots
are warm. Full corpus is 10 conversations and ~1986 questions per variant
(A and B) for ~3972 question-evals. Pure question-eval time at 2s/q ~= 2.2h.

Add per-conversation overhead:
- Snapshot ingest: ~5min/conv (19 sessions × ~10s) × 10 convs = ~50min
- Variant-B sleep cycle: ~30s-2min/conv × 10 = ~10-20min
- Variant-B think cycle: ~10s/conv × 10 = trivial
- `time.sleep(...)` between ingest and queries: per-conv configurable

Realistic ideal wall-clock without rate limits: **~4-6 hours**.
With rate-limit gates (typical Anthropic 4hr resume windows): **8-15 hours**
spread across many cron ticks.

## How resumption works

1. Tick fires every 30 min via launchd.
2. `run_cashew_locomo.py` loads `checkpoint.json`.
3. If `resume_after_iso` is in the future → log + exit 0.
4. Otherwise process a bounded batch (per-tick wall-clock cap inside the
   runner), updating `checkpoint.json` after each unit of work.
5. On rate-limit error from Anthropic, runner sets `resume_after_iso` and
   exits cleanly. Next tick that fires after that timestamp resumes.
6. When the final conv × variant × question is done, the runner writes
   `summary.json` / `summary.md` and a `complete` marker.

## First-run verification (2026-05-08)

- Reset cleared all state files.
- plist loaded and listed.
- `launchctl start` triggered first batch immediately; runner began ingesting
  conv-26 sessions and answering variant-A questions within seconds.
- Checkpoint and results.jsonl populate as work progresses.
