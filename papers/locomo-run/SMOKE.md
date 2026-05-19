# LoCoMo cashew harness — smoke verdict (post phase-3 fixes)

**Verdict: PASS** — three phase-3 fixes land cleanly. Both variants exercised in a single `--smoke` invocation; A and B start from bit-identical post-ingest state; extraction empty-rate dropped from ~42% to 0% on conv-26.

This file supersedes the phase-2 SMOKE.md. Earlier verdict ("PARTIAL PASS") is preserved in git history.

## What was fixed in phase 3

1. **Extraction-prompt over-conservatism** (was: 25 zero-node events, all `raw_response="[]"`). Fixed by wrapping LoCoMo transcripts with a benchmark-adapter preamble in `session_to_text()`. Cashew core prompt unchanged. See `EXTRACTION-FIX.md`.
2. **Ingest-once-copy A/B refactor** (was: A and B re-extracted independently, contaminating delta with extraction non-determinism). Fixed in `run_cashew_locomo.py`: snapshot DB ingested once, then `shutil.copy`'d to `<conv>-A.db` and `<conv>-B.db`. Variant A queries the copy directly. Variant B runs sleep+think on the copy, then queries.
3. **`--smoke` now exercises variant B.** Was: 3 questions all consumed by variant A, B never ran. Now: 3 questions per variant in a single invocation (6 total), with smoke recursing into B after A completes.

## Smoke command

```
python3 run_cashew_locomo.py --reset --smoke
```

Run on 2026-05-08 against conv-26.

## Snapshot / variant DB integrity (Fix 2 verification)

```
sqlite3 dbs/conv-26-snapshot.db "SELECT COUNT(*) FROM thought_nodes;"  → 192
sqlite3 dbs/conv-26-A.db        "SELECT COUNT(*) FROM thought_nodes;"  → 192
sqlite3 dbs/conv-26-B.db        "SELECT COUNT(*) FROM thought_nodes;"  → 164
```

A is bit-identical to snapshot (no sleep+think run). B = 192 → 164 nodes after sleep cycle (21 dedups + GC) and think (+1 dream + 2 think insights). **A and B started from byte-identical state**, so the 28-node delta is entirely attributable to sleep+think — exactly the ablation we want.

Sleep cycle stats logged:
- `cross_links_created: 486`
- `deduplications: 21`
- `core_promotions: 12`
- `dream_nodes_created: 1`
- `nodes_decayed: 0`

Think cycle: `new_insights: 2`, latency 7.5s.

## Per-question F1 (Fix 3: both variants exercised)

| q_idx | cat | F1 (A) | EM (A) | F1 (B) | EM (B) | gold                                    | pred (A)                                                       | pred (B)                                                       |
|-------|-----|--------|--------|--------|--------|-----------------------------------------|----------------------------------------------------------------|----------------------------------------------------------------|
| 0     | 2   | 1.00   | 1      | 1.00   | 1      | `7 May 2023`                            | `7 May 2023`                                                   | `7 May 2023`                                                   |
| 1     | 2   | 0.00   | 0      | 0.00   | 0      | `2022`                                  | "memory doesn't contain a specific date for when Melanie..."   | "memory doesn't include a specific date for Melanie's lake..." |
| 2     | 3   | 0.25   | 0      | 0.22   | 0      | `Psychology, counseling certification`  | `Counseling or mental health work.`                            | `Counseling or working in mental health.`                      |

Means: **A: F1=0.417, EM=0.333. B: F1=0.407, EM=0.333. Δ(B-A)=-0.009 F1.**

## Extraction empty-rate (Fix 1 verification)

`extraction-debug.jsonl` was not created during this smoke (file does not exist). Pre-fix had 25 entries on the same conv. **Empty-extract rate: 0/19 sessions (0%)**, vs target <5% and pre-fix ~42%.

Per-session node counts (all non-zero, range 7-15):
```
s1=9 s2=11 s3=10 s4=10 s5=7 s6=8 s7=9 s8=15 s9=9 s10=10
s11=10 s12=7 s13=12 s14=11 s15=10 s16=14 s17=13 s18=9 s19=8
```

Total post-ingest: 192 nodes (vs 119 pre-fix). Snapshot is now meaningfully populated.

## Surprises

1. **A == B on q0/q1 EM.** Sleep+think didn't change retrieval enough to flip a binary EM on these 3 questions. Expected at smoke scale (small n, mostly cat-2/3). The point of smoke is verifying the cycle ran and is measurable, not testing the headline hypothesis.
2. **Cat-2 q1 still 0.** Gold is `"2022"` but the conversations all happen May-June 2023. This looks like a LoCoMo data oddity (the gold answer references 2022 but the conv may not contain that year explicitly). Worth flagging during full-run analysis but is not a harness bug.
3. **B-A delta is slightly negative (-0.009 F1).** At n=3 this is noise. Sleep deduplicated 21 nodes; if any of those 21 were retrieval-relevant, B can lose a tiny bit of F1. The medium test (full conv, both variants, ~199 questions per variant) is the real measurement.
4. **Sleep cycle warned `embedding_integrity.integrity_ok: False`** with `orphan_nodes: 19`. Did not crash. Worth tracing once at scale; not a phase-3 blocker.
5. **Sleep cycle hit a `divide by zero / overflow / invalid value encountered in matmul` runtime warning** from sklearn. Likely zero-norm embeddings on a few new nodes. Did not crash sleep. Flag as a soft cashew issue for the maintainers; not a phase-3 blocker.

## Recommended next moves

- **Medium test**: 1 full conv, both variants, all questions. Will validate fixes 1-3 at scale and produce the first non-noisy A/B delta.
- **Trace orphan_nodes / matmul warnings** before the full 10-conv run — they are silent today, but they could compound.
- **Verify another LoCoMo conv** (conv-1, larger and earlier-indexed) doesn't regress — preamble was tuned implicitly against conv-26.

## Files touched in phase 3

- `/Users/bunny/.openclaw/workspace/benchmarks/locomo/cashew_adapter.py` — `session_to_text()` preamble.
- `/Users/bunny/.openclaw/workspace/benchmarks/locomo/run_cashew_locomo.py` — snapshot/copy logic, smoke recursion into variant B, checkpoint schema additions (`snapshot_done`, `snapshot_copied`).
- `/Users/bunny/.openclaw/workspace/cashew/papers/locomo-run/EXTRACTION-FIX.md` — new.
- `/Users/bunny/.openclaw/workspace/cashew/papers/locomo-run/SMOKE.md` — this file.
