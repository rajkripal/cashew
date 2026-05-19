# CRITIQUE — adversarial review of cashew x LoCoMo paper + harness

Reviewer: critic (locomo-cashew team). First pass: 2026-05-08, pre-smoke.
Mandate: pressure-test as if at NeurIPS PC. Push back where it would actually fail review.

Verdict legend: LOOKS GOOD / WEAK / BROKEN.

---

## A. Contribution claim — "untyped edges beats typed edges"

**Verdict: WEAK (as currently planned). Possibly BROKEN as a "headline" claim.**

The outline (Section 2, Claim B) names the typed-vs-untyped ablation as the strongest defensible contribution. Two problems:

1. **The ablation does not exist in this run.** Outline §2 itself hedges: "Cashew docs claim this was ablation-tested. Verify before publication." The current LoCoMo harness runs A=ingest-only vs B=ingest+sleep+think. Neither variant is "typed-edge cashew." So the headline contribution is not being measured by the experiment we're actually running. A reviewer would ask: where is the typed-edge variant? Answer today: it does not exist as code.

2. **Prior work is closer than the outline admits.** A-MEM (2502.12110) explicitly does *dynamic* linking — links are induced, not hand-typed by an ontology. Mem0 (2504.19413) has both Mem0 and Mem0-graph variants and the *paper itself* is the typed-vs-flat ablation, with Mem0-graph adding only ~2% over Mem0 on LoCoMo. That's the field's existing answer to "do edges/types help?": empirically, very little. So cashew's "untyped wins" is not contrarian — it is *consistent with* the published Mem0 result. Framing it as inverting the field is overclaiming. It also doesn't bite on Zep's bet, because Zep's win is bitemporal, not edge-typing-per-se.

**Smallest fix:**
- Rewrite Claim B as "untyped edges + graph-signal decay matches typed-edge graph variants on LoCoMo at lower implementation cost." This is defensible and consistent with Mem0's own published delta. Don't claim inversion of the field.
- If we want the strong claim, build the cashew-with-typed-edges variant as a real ablation arm before the paper. That's at least a week of work the team is not currently scheduled to do.
- Stop calling A=ingest-only vs B=ingest+sleep+think "the contribution ablation." That ablation is about **sleep/think value**, not about edges. Mislabeling will get caught.

---

## B. Evaluation faithfulness — F1/EM port

**Verdict: BROKEN. Numbers as currently computed are NOT directly comparable to the LoCoMo paper.**

Read `task_eval/evaluation.py:75-138` against `cashew_adapter.py:482-506`. Three concrete drifts:

1. **Porter stemming missing.** LoCoMo's `f1_score` (`evaluation.py:127-128`) applies `PorterStemmer().stem()` to every token before counting overlap. Our `f1_score` (`cashew_adapter.py:491-502`) does not. Effect: cashew F1 will be systematically *lower* on morphologically-different-but-equivalent words ("running"/"ran", "decided"/"decide", "talked"/"talking"). Magnitude is non-trivial on conversational free-text answers — easily 2-5 F1 points on category 4 (open-ended).

2. **EM definition diverges.** LoCoMo's `exact_match_score` (`evaluation.py:95-101`) is `set(prediction.split()) == set(ground_truth.split())` after `normalize_answer`. Set-equality, not string-equality, and not order-sensitive. Our `exact_match` (`cashew_adapter.py:505-506`) is `_normalize(pred) == _normalize(gold)`, i.e. ordered string equality. These are different functions. EM as defined by LoCoMo will accept "movie the watched" against "watched the movie"; our implementation won't.

3. **`white_space_fix` collapse.** LoCoMo applies `' '.join(text.split())` (`evaluation.py:82-83`) which collapses all whitespace (incl. tabs/newlines). Our `_normalize` uses `re.sub(r"\s+", " ", s).strip()` which is approximately equivalent — this one is fine. Mentioned for completeness.

There is also a minor difference: our `_normalize` does `unicodedata.normalize("NFD", s)` before stripping punctuation; LoCoMo's `normalize_answer` does not call NFD at all (NFD lives in their `_normalize` helper used for `has_answer`/retrieval, not for `normalize_answer`). NFD will decompose accented chars, then the punctuation strip will remove combining marks. Effect: cashew normalizes "café" → "cafe", LoCoMo keeps "café". On English LoCoMo this is near-zero impact, but it's a real divergence to disclose.

**Why this matters for the paper:** Mem0's headline LoCoMo number is in their reporting using LoCoMo's own evaluator (or an LLM-as-Judge variant). If we cite our F1 against Mem0's F1 in the same table, we are comparing apples to oranges. A reviewer who reads two scoring functions side by side will reject the comparison.

**Smallest fix:**
- Either (a) `pip install nltk`, ship Porter stemming, and use set-equality EM — port faithfully; or (b) keep the current (faster, lighter) scorers but explicitly *also* run LoCoMo's stock F1/EM offline on the saved `results.jsonl` and report **both**: "cashew F1 (no-stem)" and "LoCoMo F1 (stem)". Path (a) is cheaper and removes the asterisk.
- Drop the NFD step or push it into `normalize_answer` only-if-LoCoMo-does. Today it's gratuitous drift.
- Add a unit test: feed `(pred="The cats ran fast", gold="cat run fast")` through both scorers. If they disagree by >0, fail loud.
- bert_score: skipping it is fine for our internal comparisons, but the LoCoMo paper reports bert_score-recall as a primary metric in some categories. If we publish without it, we cannot put our numbers in the same row as LoCoMo paper headline numbers without a footnote. Disclose it.

---

## C. A/B ablation honesty

**Verdict: WEAK. Multiple confounds a reviewer will flag.**

The intent — A=ingest-only, B=ingest+sleep+think on the same ingested sessions — is sound. The implementation introduces several confounds:

1. **Separate DBs, separate ingestions.** `init_conv_db(out_dir, conv_id, variant)` creates `<conv>-A.db` and `<conv>-B.db`. The runner ingests **twice**, once per variant, calling `extract_from_conversation` independently for each (`run_cashew_locomo.py:393-395`). Because `model_fn` is `claude -p` headless and is non-deterministic (no temperature=0 set; no seed; even at temp=0 Claude is not bit-reproducible across calls), A's and B's graphs have different *ingestion* content before sleep/think ever runs. We are not measuring "sleep+think value over the same ingested data." We are measuring "ingest-pass-1 vs ingest-pass-2 + sleep + think." That is not the ablation we want.
   - **Fix:** ingest once into a snapshot DB, then `cp` to A and B copies, then run sleep+think only on B. One-line code change in the runner; large methodological win.

2. **B may add information A doesn't have.** `run_sleep_cycle` and `run_think_cycle` both take `model_fn` and can synthesize new nodes (cross-links, dream nodes, think insights). These nodes are derived from the ingested graph + the LLM's prior. The LLM prior is not in A. So B has more *world knowledge* infused than A, not just more reorganization of the same data. A reviewer will read this as test-time information leakage from the model into the memory store. To be defensible, you must either (i) report sleep/think node deltas explicitly per conv and show that the new info is derivative-only, or (ii) run a third variant `B'` where sleep is mechanical-only (no `model_fn`) to isolate the LLM-injection effect.

3. **Retrieval hyperparams identical?** Both A and B call `generate_session_context(db, hints)` with identical args. Good. But the retrieval *result* depends on graph topology, which differs (intentionally for B, unintentionally for A, see #1). No knob asymmetry, but the structural asymmetry needs to be reported.

4. **No A-only ablation of sleep alone vs think alone.** B bundles both. If B>A, you cannot say which mechanism helps. For a paper, you want at minimum {A, A+sleep, A+sleep+think}. Adding the middle arm is cheap (one variant flag).

**Smallest fix:**
1. Ingest once, copy DB for B. Removes ingestion-noise confound.
2. Add a `B-no-llm-sleep` variant (call `run_sleep_cycle` without `model_fn`) so we can split mechanical reorg from LLM injection.
3. Add `A+sleep` (no think) as a third variant for component attribution.
4. Disclose sleep/think node deltas in the per-conv summary already (`sleep_think_stats`) — good, keep.

---

## D. Hyperparameter discipline

**Verdict: WEAK on disclosure, OK on practice.**

Outline §7.2 explicitly flags this. Current state:

- Cashew thresholds (0.82 dedup, 0.7-0.85 cross-link, sqrt(N) core promotion, gc_threshold) live in cashew config and were tuned on the author's personal graph. No retuning is happening for LoCoMo (good, in spirit).
- But: there is **no pre-registration** of these values in the run output. `summary.json` writes models and aggregates but does not snapshot the cashew config used for the run.
- Smoke results may tempt teammates to nudge thresholds before the full run starts. There is no enforcement against this.

**Smallest fix:**
- At runner startup, read `cashew/config.yaml` (or whatever holds thresholds), hash it, write the hash + full content into `papers/locomo-run/CONFIG_FROZEN.yaml` and refuse to run if it changes mid-experiment. 30 lines of code.
- In the paper, report this hash in the methods section.

---

## E. Baseline integrity

**Verdict: BROKEN for paper-grade comparison.**

Outline §1.1 cites Mem0 at 93.4% on LongMemEval and ~26% gain on LoCoMo over OpenAI Memory. We are not running Mem0 ourselves. Problems:

1. **Mem0's LoCoMo number is not stated anywhere I can see in the outline as a concrete number** — only the LongMemEval headline. The 26% gain is *over OpenAI Memory*, not absolute. We need the absolute Mem0 F1/J on LoCoMo from arXiv:2504.19413 to put in our comparison row. Get it before paper-writing.
2. **A-MEM also reports LoCoMo numbers** (arXiv:2502.12110). Pull those too.
3. **OpenAI Memory baseline alone is not a defensible comparison set.** A NeurIPS/COLM reviewer will demand at least Mem0 + A-MEM + a long-context full-context baseline at the same backbone LLM. Today we have none of these. Citing their numbers from arXiv is acceptable *only if* we also lock our backbone to one they used and disclose that we're cross-referencing. Otherwise you get the "different model, different scoring, different prompt" rejection.
4. **The judge model (claude-sonnet-4-6) is not what Mem0/A-MEM used.** Mem0 reports against GPT-4o-mini and similar. Token-F1 can be model-dependent because the *predictions* are model-dependent. Even if our scorer is identical to LoCoMo's, our predictions came from a different LLM. Cross-paper number borrowing is therefore lossy.

**Smallest fix (bare minimum):**
- Run a "full-context" baseline on at least 1-2 LoCoMo conversations: stuff entire convo into Claude Sonnet 4.6 and answer. Same scoring path. This is the floor that says "our memory is doing something the long-context backbone alone isn't." Cheap to wire.
- Cite Mem0/A-MEM LoCoMo numbers in a separate table with explicit disclaimers about backbone difference.
- If aiming COLM 2027, run Mem0 on LoCoMo ourselves using their public code with our backbone. That is the only way to have a defensible head-to-head.

---

## F. Smoke→full gap

**Verdict: WEAK — pending smoke, but visible risks.**

Smoke is 1 conv × 3 questions (`run_cashew_locomo.py:402-403`). Failure modes that smoke will not catch:

1. **Rate-limit gating math.** With 10 convs × 2 variants × ~199 questions/conv ≈ ~3980 question-LLM-calls + ~270 sessions × 2 variants × 1 ingest call + 10 sleep + 10 think ≈ ~4500+ headless `claude -p` calls minimum. At Claude rate-limit thresholds this guarantees multiple 4-hour pauses. The `RATE_LIMIT_PAUSE_HOURS=4` is a reasonable default but the *total wall-clock* for the full run is not estimated anywhere. Suggest a paper sketch: estimate it.
2. **Per-conv DB grows.** LoCoMo conversations have up to 35 sessions × ~30 turns. After full ingest cashew DBs may reach hundreds-to-low-thousands of nodes per conv. `generate_session_context` retrieval has not been profiled at this scale per-conv. Smoke uses 1 conv (likely smallest) so won't catch this.
3. **Cat 5 (adversarial) behavior under cashew.** Cat 5 is the abstention category — "this isn't in the conversation." cashew's retrieval may *always* return some context (it's BFS-walk over graph, never empty if the graph is non-empty). If retrieval always returns *something*, the answering LLM may pattern-match it as an answer and pick (b) gold instead of (a) "not mentioned." With 446 cat-5 questions (22% of total), this could materially distort headline numbers. Smoke's 3 questions almost certainly miss cat 5.
4. **Question-batch size of 5** at default. With ~199 questions per conv-variant, you need ~40 launchd invocations per conv-variant just to grind through. Multiplied across 20 conv-variants, 800+ invocations. If anything is flaky in the launchd path, you'll see it at scale.
5. **Sleep/think on a 9-session ingestion** (smoke's max) is much smaller graph than full conv. Sleep cycle's behavior can change qualitatively at higher N — cross-link saturation, dedup collisions. Will not be exercised by smoke.
6. **Determinism / reproducibility:** every `claude -p` call is non-deterministic. If we re-run, F1 will jitter. We have no seed control. For a paper, you want at least mean ± std over N=3 seeds for headline numbers. Currently impossible.

**Smallest fix:**
- Before kicking off the full run: run a *medium* test (1 full conv, both variants, all questions). This is ~400 questions, 2-4 hours wall clock with rate limits. Catches all of #2-5 above.
- Add a deterministic-temp=0 flag to `claude -p` (if available) and document that headline numbers are from a single seed. Rerun once with seed perturbation if budget allows.

---

## G. Methodology smells a NeurIPS reviewer would flag

**Verdict: WEAK to BROKEN across multiple axes. Top concerns ranked.**

1. **No statistical significance.** With ~1986 total QA pairs the per-variant n is decent for crude means, but no confidence intervals or significance tests are computed. A paper claim "B beats A by ΔF1=0.03" needs at minimum a paired bootstrap CI or a per-question paired t-test. Currently the runner just reports raw means. **Add paired-by-(conv,q_idx) bootstrap CI to `finalize()`.** ~30 lines.
2. **Cherry-pick risk on conv-26.** Existing checkpoint shows conv-26 already partially run (variant A, 19 sessions ingested, 3 questions answered). If we don't `--reset`, the first 3 questions of conv-26 in the final results came from a previous code version. Mixing harness versions in one results file is unacceptable. **Force `--reset` before the full run** or commit to discarding the partial state.
3. **Selection of LoCoMo categories.** LoCoMo has 5 categories with very different difficulty profiles. Cat-2 (date) and cat-5 (adversarial MCQ) have purpose-built prompts in the adapter — that's appropriate. But cat-3 (only 96 questions, 4.8% of total) and cat-1 (single-hop) probably don't differentiate cashew vs baselines. The interesting category is cat-4 (multi-hop, 841 questions). Make sure the paper *leads* with cat-4 and reports per-category, not just a global mean.
4. **Retrieval budget unfairness when comparing to other systems.** Cashew's `generate_session_context` returns up to a soft 60K-char cap (the adapter passes `ctx[:60000]` to the QA prompt — that's ~15K tokens). Mem0 reports much smaller token budgets in their headline. If we're going to compare token cost to Mem0, we need to enforce comparable retrieval budgets. Today we don't.
5. **LLM-as-judge fragility (outline §7.8).** F1 scoring is not an LLM judge here, but the fact that LoCoMo's headline result in Mem0 is reported using **LLM-as-Judge** ("J") not F1, and we are reporting F1, means our headline number is on a different metric than Mem0's headline. Paper readers will see "cashew F1=0.X, Mem0 J=Y" and either miss or be misled by the difference. **Either run the LLM-as-Judge metric ourselves, or be explicit that our headline is F1 only and Mem0's J number is not directly comparable.**
6. **Test data leakage in extraction.** `extract_from_conversation` uses Claude as the extractor. Claude was trained on data through some cutoff — LoCoMo (Feb 2024) is potentially within Claude's training window. The extractor LLM may have *seen* LoCoMo dialogues during training and have priors about them. This is a less-likely but real concern; a reviewer might raise it. Disclose in limitations.
7. **Orphan: there is no held-out development set.** All hyperparameter freezing aside, if any tuning happens on smoke results and then we run the full set, the full set is effectively the test set for that tuning. Be honest that smoke is dev set and the full 10 convs are the test.

**Smallest fix:** address #1 (paired bootstrap), #2 (reset), and #5 (J vs F1 transparency) before the full run. The rest go in a "Limitations" section.

---

## Top 3 ranked concerns (for team-lead)

1. **F1/EM port drift (B-broken).** Our scoring is not LoCoMo's scoring. Numbers are not directly comparable to anything in the LoCoMo, Mem0, or A-MEM papers. Fix or footnote before any external claim.
2. **A/B confound — separate ingestions (C).** A and B re-extract independently with non-deterministic LLM. We are not measuring "value of sleep+think on the same data." Ingest once, copy DB to B. One-line fix; methodological win.
3. **Headline contribution mislabeled (A).** Outline says "untyped-vs-typed edges" is the contribution; the experiment actually measures "sleep+think value" (B-A). Either build the typed-edge variant or rewrite the contribution to match the experiment.

---

## Pending second-pass items (after smoke + first results)

- Verify smoke run produces non-zero F1 on at least 1/3 questions (RECON suggests this is the smoke success bar).
- Cat-5 abstention behavior — does cashew correctly say "(a) Not mentioned" or default to (b) gold?
- Sleep+think latency at 9-session graph vs predicted full-conv graph.
- Whether B's `nodes_delta` from sleep is meaningfully positive (if zero, sleep is decorative on these conversations).
- Whether `extract_from_conversation` ever silently no-ops (the `ExtractionNoOpError` guard exists — verify it doesn't trigger).

---

## Section H — Real-numbers third pass (post conv-26)

First full A/B conversation landed: 199 questions per variant. Headline: ALL F1 A=0.519, B=0.488 (Δ=-0.031). Cat-2 (temporal) -0.096; cat-1 +0.004; cat-3 -0.022; cat-4 -0.035; cat-5 = 1.000 both.

### H1. Cat-2 -9.6 — does sleep dedup eat dated nodes? **LOOKS GOOD (signal is real and severe).**

**Evidence:** Diffed conv-26-snapshot.db (post-ingest) vs conv-26-B.db (post-sleep+think):
- snapshot: 556 thought_nodes; B: 234 thought_nodes (net -322, -58%).
- 435 nodes present in snapshot are GONE in B. 113 new nodes in B (synthesis/insights from think).
- All 435 removed nodes carry date markers (referent_time set or explicit date strings in content). In ALL of snapshot, 100% had date markers; in B, only 53% do. So the removed set is dated, but baseline is also fully dated — the more telling stat is the survivor profile: B keeps 125 dated of 234 (53%) vs snapshot 100%. Sleep+think is collapsing dated event nodes into denser, less-dated synthesis nodes.
- Sample removed dated nodes: "Caroline attended a council meeting about adoption on Friday 7 July 2023…", "Melanie bought wooden family figurines on 21 October 2023…" — exactly the granular dated facts cat-2 needs.
- B cat-2 retrieved_chars (mean 2026, median 2155) is actually slightly *higher* than A (1884/1878), so this is not a retrieval-volume problem — it's a content problem. The retrieved B chunks contain fewer specific dates.

**Mechanism:** dedup/synthesis during sleep is merging "Melanie bought figurines on 21 Oct 2023" and "Melanie expressed love for family" into a single de-dated insight node. Cat-2 questions ("when did X happen?") then retrieve the synthesis without the date.

**Smallest fix:** in dedup/merge logic, require preservation of `referent_time` and any ISO-8601 / explicit-date substring from source nodes when emerging a merged node. Or: tag merged nodes with the union of source `referent_time`s and surface them in retrieval context. Cheapest test — re-run conv-26 B with dedup gated to "only merge if both sources lack referent_time."

### H2. Cat-1 flat (+0.004) — significant or noise? **WEAK (underpowered).**

**Evidence:** n=34 (A), n=32 (B). Per-question F1 sd ~0.26 each side. SE of delta = 0.064. 95% CI on Δ = [-0.116, +0.136]. The observed +0.010 is well inside noise; cannot distinguish from zero, and CI easily covers ±10 F1. Single-conv cat-1 is too small to claim "sleep doesn't move single-hop."

**Smallest fix:** wait for more conversations; pool cat-1 across all 10 convs before any claim. Report CI in the paper, not point estimate.

### H3. Are we sub-SOTA on cat-2 anyway? **LOOKS GOOD — we are at/above Mem0-graph.**

**Evidence:** Mem0 paper (arXiv:2504.19413) Table 1 — Temporal F1: Mem0=51.55 (Mem0_g graph variant achieves the 51.55; LoCoMo baseline 18.41). Our cashew A=0.613 (61.3), B=0.516 (51.6). Variant A is +10 F1 over Mem0_g; B is at parity. We are not "below SOTA before sleep runs" — we are above it on cat-2. The story "sleep+think currently degrades a SOTA-class temporal baseline" stands; this is a real finding, not a measurement artifact of being a weak system.

**Caveat:** our F1 implementation is locally-ported (see Section B "F1/EM port drift"), so absolute comparison to Mem0's number is not airtight. Even granting ±5 F1 of port drift, A clears Mem0_g.

### H4. Cat-5 perfection — broken metric? **BROKEN.**

**Evidence:** 94 cat-5 predictions on conv-26 (47 per variant). 90 of 94 gold answers are literally the string "Not mentioned in the conversation" (the other 4 are "No"). Of 94 predictions, 90 contain "not mentioned." System emits a near-constant abstention; gold is near-constant abstention; F1 = 1.0 by tautology. This is not measuring abstention skill — it's measuring whether the system ever speaks. A system that always emits "Not mentioned" would also score 1.0.

**Smallest fix:** either (a) drop cat-5 from F1 reporting and replace with "abstention precision/recall" computed against an injected positive-control set where gold is *not* abstention, or (b) report cat-5 as "always abstains; not a discriminative signal." Currently the cat-5 row is propping up the headline ALL F1 by ~12 points (47/199 = 24% of questions at 1.0). A more honest ALL F1 excluding cat-5: A ≈ 0.367, B ≈ 0.330 (Δ=-0.037, similar story but different absolute level).

### H5. Methodology smell sweep on real data. **MOSTLY CLEAN, two notes.**

- **Schema:** all 17 fields present, no nulls in critical fields, 199 rows per variant matches expectation.
- **Retrieved_chars:** A mean 1881 / median 1868; B mean 2054 / median 2137. No zeros, no extreme outliers. B retrieving slightly more is consistent with sleep producing larger synthesis nodes.
- **Latencies:** answer_latency median 2.77s (A) / 2.73s (B), p95 7.25 / 6.73 — clean and roughly equal. Retrieval latency median 0.097s (A) vs 0.025s (B) — B is ~4x faster. Plausible: B has 234 nodes vs A's much larger graph, so vector search is faster. Worth confirming this doesn't reflect a different retrieval path.
- **Retrieval consistency A vs B same-q:** 100/199 questions produce identical pred strings across A and B. |Δretrieved_chars| median = 312. So half the time, the two variants land on the same answer despite different graphs — which means our A/B signal is being computed on the other ~100 questions. That's still enough power for the 199-question ALL F1 delta but worth noting: the per-category n is effectively halved on the "differential" subset. **WEAK** (not broken, but reduces effective power).
- **A vs B re-extraction confound:** Section C concern — was this fixed? Task #8 says "ingest-once-copy" was completed; the snapshot.db→B.db diff above is consistent with B being seeded from a single ingestion (snapshot is the shared starting point). Confirmed by structure. Good.

### Top 3 H-section findings (ranked)

1. **Cat-2 degradation has a clear mechanism (H1):** sleep+think dedup is collapsing dated event nodes into de-dated synthesis nodes. 435 dated nodes removed, 322 net node loss. Fix: preserve `referent_time` on merge. This is the most actionable finding from the real run.
2. **Cat-5 metric is tautological (H4):** 90/94 gold = "Not mentioned"; system always emits "Not mentioned"; F1=1.0 means nothing. Inflates headline ALL F1 by ~12 points. Stop reporting cat-5 in the headline number, or replace with abstention P/R on positive controls.
3. **Cat-1 delta is noise, not signal (H2):** 95% CI [-0.116, +0.136]. Don't claim "sleep is neutral on single-hop" from one conversation. Pool across convs.
