# MemBench Extension — Scope

**Date:** 2026-05-07
**Status:** Scope only. No implementation in this doc.
**Anchors:** MEMBENCH-RECON.md (option (c) greenlit), NOVELTY-CHECK.md (positioning).
**Plan in one line:** run cashew + Mem0 on MemBench PS-RM/OS-RM (MCQ anchor), author ~50 free-form synthesis questions on the same corpora, score with AttrScore-style citation-faithfulness LLM judging, report both numbers honestly.

---

## 1. MCQ track plan (anchor numbers)

### 1.1 Adapter shape

MemBench has no `Memory` ABC. We write one ourselves and adapt both systems to it. Minimal interface:

```python
class MemorySystem(Protocol):
    def reset(self, user_id: str) -> None: ...
    def ingest_session(self, user_id: str, session_msgs: list[Msg]) -> None: ...
    def answer_mcq(self, user_id: str, question: str, options: list[str]) -> str:  # returns "A".."D"
        ...
```

For each user in MemBench:
1. `reset(user_id)`
2. For each session in chronological order: `ingest_session(...)`
3. For each MCQ question targeting that user: call `answer_mcq(...)`, log returned letter

Scoring is exact-match against MemBench gold letter; we replicate their accuracy aggregation per subtype (preference vs emotion, single/multi-hop, etc.).

### 1.2 Cashew wiring

- Ingest: replay session messages through cashew's standard ingest pipeline (chunker → extractor → graph writer). Embedding model: **gte-large** (matches our LoCoMo numbers and Mem0 head-to-head).
- Sleep cycles: trigger one sleep pass per user after all sessions ingested. (MemBench is offline; we don't need streaming sleep.)
- Think cycles: one think pass per user post-sleep. This is the cashew differentiator — present on the full pipeline cell.
- Retrieval: query expansion (QE) on the MCQ stem + uniform retrieval (k=10–15, tune on a 50-user dev slice). Concat retrieved nodes + question + options into the answer prompt.
- Answer LLM: claude-sonnet-4-6 with a strict "respond with one letter A/B/C/D" suffix and a regex extractor.

### 1.3 Mem0 wiring

We already have the LoCoMo Mem0 adapter. Translation needed:
- Same gte-large embeddings (already configured).
- `add()` per session, `search()` on the MCQ stem, k matched to cashew.
- Same sonnet-4-6 answer prompt for fairness.

### 1.4 Models — single model across the board

`claude-sonnet-4-6` for: ingest LLM (cashew extraction), retrieval QE, sleep, think, answer. Justification: matches our LoCoMo run, removes "model swap" as a confound, comparable internally. **Caveat for paper:** MemBench's published baselines used Qwen2.5-7B; cross-paper accuracy comparisons get an explicit asterisk.

### 1.5 Format mapping

MemBench MCQ JSONL row → adapter input:
- `user_id` → `user_id`
- `session_history: list[list[Msg]]` → ingest loop
- `question: str`, `options: {A,B,C,D}` → `answer_mcq`
- `answer: str` → gold for scoring (held out at adapter layer)

Output: results JSONL with `{user_id, qid, subtype, system, predicted_letter, gold_letter, correct, latency_ms, retrieved_node_ids}`.

### 1.6 Effort + cost

| Item | Estimate |
|---|---|
| Cashew adapter (MemorySystem shim + MCQ prompt + extractor) | 8–12 h |
| Mem0 adapter (port from LoCoMo harness) | 2–4 h |
| Eval harness (rate-limit-aware cron, checkpointing — reuse from LoCoMo) | 3–4 h |
| Smoke run on 20 users | 2 h |
| Full PS-RM + OS-RM run (5.5k Q × 2 systems) | ~6 h wall-clock under existing cron |
| **Total engineering** | **~2 days** |
| **Total compute (Sonnet)** | ~5.5k Q × 2 × ~$0.005 + ingest/think tokens ≈ **$70–$100** |

---

## 2. Free-form question authoring (the contribution)

50 hand-authored questions, ~17 per corpus (MovieLens, Food, Goodreads). Each question must be (a) answerable from the user's MemBench history, (b) require synthesis across multiple history items (no single-item lookups), (c) have at least one defensible answer that a thoughtful reader can argue for from the data.

### 2.1 Question style + examples

**MovieLens reviews (~17 Qs)**
1. What is this user's evolving relationship with violence as entertainment? Does their tolerance shift over time?
2. Where do this user's stated genre preferences (in dialogue) diverge from their actual high-rating clusters?
3. What latent dimension (auteur loyalty? era nostalgia? runtime tolerance?) best explains their rating variance?
4. Does this user rate higher when watching alone vs with others, based on what they describe?
5. When does this user rate against critical consensus, and is there a pattern in those contrarian picks?
6. What does this user dislike that they refuse to admit they dislike?
7. Which director or actor functions as a "blind spot" — consistently rated above the user's own quality bar?

**Food reviews (~17 Qs)**
1. What is this user's evolving relationship with spice tolerance / unfamiliar cuisine?
2. Where does this user's stated health-consciousness diverge from their actual indulgence patterns?
3. Is "value for money" or "novelty" the stronger driver of high ratings here?
4. What food category does this user explore most when stressed vs celebrating, based on context?
5. Which restaurant types does this user keep returning to despite middling ratings?
6. What does this user describe as "comfort food" implicitly through repeat behavior?
7. Where do this user's reviews shift in tone (terse vs effusive) and what triggers the shift?

**Goodreads (~17 Qs)**
1. What is this user's evolving relationship with literary difficulty? Are they trading down or up?
2. Where do this user's stated "favorite genres" diverge from where their 5-stars actually cluster?
3. What latent theme (grief? ambition? identity?) recurs across their highest-rated books?
4. Does this user finish books they hate or DNF them — and what does that say about commitment patterns?
5. Which authors does this user follow loyally vs sample once, and what predicts which bucket?
6. What gap exists between this user's "to-read" aspirations and what they actually read?
7. When does this user rate against the popular consensus, and what kind of book triggers it?

### 2.2 Authoring protocol

- **Author:** Raj writes the first 50, working from a fresh MemBench user's full history each time. Estimated 15 min/question = ~12 h authoring.
- **Anti-bias rules** (binding):
  - Do not author a question while looking at cashew's output for that user. The corpus comes first; the system never seeds the question.
  - For each candidate question, write a one-paragraph "defensible answer sketch" *before* running any system. Lock it in a file, timestamped.
  - Do not author questions whose form is conspicuously cashew-shaped (e.g. "what is this user's emotional architecture across X" — too on-the-nose for think cycles). Favor questions where flat retrieval *could* succeed.
- **Answerability check:** for each question, the author must cite ≥3 specific user-history items (review IDs / rating IDs) that ground the defensible answer. Questions that cannot meet this bar are dropped.

### 2.3 Citation format

System answers must produce JSON of shape:

```json
{
  "answer": "free-form prose, ~150-300 words",
  "citations": [
    {"item_id": "movielens_user42_rating_007", "claim": "Quoted/paraphrased claim this item supports"},
    ...
  ]
}
```

`item_id` must be a key present in MemBench's source history for the user. We resolve to original row in the rec-system corpus.

### 2.4 Validation panel

- **Panel:** Raj + 2 readers (recruit from teammates not on this paper, e.g. team-lead and one external).
- **Protocol per question:** each reader independently writes their own one-paragraph defensible answer, then sees the author's. Question is **accepted** if ≥2/3 readers' answers agree on the dominant theme. Otherwise drop or rewrite.
- **Target post-validation:** ≥40 surviving questions out of 50 authored. If fewer survive, author more.

---

## 3. Scoring methodology

### 3.1 Citation precision (primary metric, AttrScore port)

For each (claim, citation) pair in a system's answer:
1. Resolve `item_id` → actual user-history row from MemBench source.
2. LLM judge prompt (Sonnet-4-6): given the cited row and the claim, classify as `attributable` / `extrapolatory` / `contradictory`.
3. Per-answer precision = #attributable / #cited pairs.
4. Per-system precision = mean across 50 questions.

### 3.2 Recall (v1: skip; v1.5: optional)

Recall is hard because there is no ground-truth set of "items a good answer must cite." Two options:

- **(a) v1 — skip recall.** Headline metric is precision-only. Justified for v1 because precision asymmetrically punishes hallucination, which is the failure mode we care about. Acceptable.
- **(b) v1.5 — author reference citation set.** During panel validation, each reader lists "items I would expect a good answer to cite." Union → reference set. Recall = |system_cites ∩ reference| / |reference|. Adds ~30 min/question of panel work. Defer unless v1 results converge.

### 3.3 Inter-judge reliability (must-do)

- Sample 20% of judge calls (≈ 50 q × ~5 cites × 0.2 ≈ 50 calls per system).
- Raj rates each manually with the same 3-class scheme.
- Compute agreement (Cohen's κ + raw % agreement).
- **Pass bar:** ≥90% raw agreement. If lower, the methodology is unsafe and we redesign the judge prompt or escalate to a human-only subsample.

### 3.4 Aggregate reporting

Per system: `{precision_mean, precision_std, n_questions, kappa_with_human}`. Plus per-corpus breakdown so we can see if cashew wins on Goodreads-style narrative depth but ties on MovieLens factual taste.

---

## 4. Comparison cells

| # | System | Track | Purpose |
|---|---|---|---|
| 1 | Cashew full pipeline (gte-large + QE + sleep + think + uniform retrieval) | MCQ (PS-RM + OS-RM) | Main MCQ number |
| 2 | Mem0 (gte-large) | MCQ (PS-RM + OS-RM) | Head-to-head baseline |
| 3 | Cashew full pipeline | Free-form synthesis | Main contribution number |
| 4 | Mem0 | Free-form synthesis | Head-to-head baseline |
| 5 | Cashew without think cycles (ingest+sleep only) | Free-form synthesis | **Optional ablation** — isolates think contribution |
| 6 | LLM-only (entire user history stuffed in context, no memory layer) | Free-form synthesis | **Optional baseline** — sanity check that memory layer earns its keep |

If budget pressed, cells 5 and 6 are deferrable. Cells 1–4 are mandatory.

---

## 5. Honest reporting plan

- **MCQ first.** Headline: cashew vs Mem0 on PS-RM and OS-RM accuracy. Caveat box: "MemBench's published numbers use Qwen2.5-7B; we use Sonnet-4-6 for parity with our LoCoMo run. Numbers not directly comparable across papers." Acknowledge weakened cross-paper comparability up front.
- **Free-form second.** Headline: citation precision, cashew vs Mem0 (vs ablations if run). This is our actual contribution.
- **If cashew loses on MCQ:** report it. Frame as "MCQ accuracy is dominated by retrieval quality on synthetic preference dialogues; our architecture targets a different axis." Don't bury.
- **If cashew wins on free-form citation precision:** the headline. Frame as "the contribution is showing the metric matters, not just that we win it" — pair with Mem0's MCQ parity to argue both systems are competent.
- **If cashew loses both:** report and adjust the paper's central claim. Falsification is fine; spinning is not.

---

## 6. Risks and decisions

1. **MemBench's harness assumes Qwen2.5-7B.** *Mitigation:* we don't use their harness. We re-implement evaluation from the dataset JSONL; harness is ours. Document the substitution in §5 caveat box.
2. **Free-form questions too easy or too hard.** *Mitigation:* validation panel + the "author's defensible answer ≥3 citations" rule. Pilot 5 questions and run both systems before authoring all 50; recalibrate difficulty if both systems score >90% or <30%.
3. **Citation precision converges (both ~80%).** *Mitigation:* fall back metrics ranked by readiness:
   - Citation *density* (cites per claim) — measures whether system bothers to ground.
   - Citation *diversity* (distinct items cited / answer length) — penalizes shallow citing of the same hot item.
   - Recall against panel reference set (§3.2 option b).
   - Pairwise human preference between cashew vs Mem0 answers, blinded — last-resort, expensive.
4. **Author bias (Raj favoring cashew's strengths).** *Mitigation:* blind validation panel + rule against on-the-nose question shapes (§2.2) + a post-hoc audit where one teammate flags any question that seems cashew-favored, and we drop or rewrite.
5. **Synthetic-corpus distribution mismatch.** MemBench dialogues are GPT-generated from rating pairs, not real conversation. Cashew's extraction was tuned on real logs. *Mitigation:* note as a limitation; do not over-claim "real-world synthesis." Plan a follow-up on real corpus (LoCoMo or our own dialogue captures) if time allows.
6. **Judge model = answerer model.** Both are Sonnet-4-6. Risk of self-preference. *Mitigation:* spot-check 20% with human (§3.3) + consider a cross-judge ablation with Opus-4-7 on a 10% subsample.

---

## 7. Implementation sequence

| Phase | Days | Work | Decision gate |
|---|---|---|---|
| **A** | 3 | MCQ adapter (cashew + Mem0), smoke on 20 users, first PS-RM numbers | If cashew within ±5% of Mem0 on MCQ, proceed. If cashew is >10% behind, debug retrieval before continuing. |
| **B** | 3 | Free-form question authoring (50 Qs) + validation panel (target ≥40 survive) | Need ≥40 validated questions before scoring run. |
| **C** | 3 | Free-form scoring run (cells 3–4, optional 5–6) + AttrScore judge wire + 20% human-judge audit | If judge κ < 0.8 or raw agreement < 90%, redesign judge before reporting. |
| **D** | 2 | Writeup, plots, paper section draft | — |
| **Total** | **~11 working days (~2 calendar weeks with slack)** | | |

---

## 8. What NOT to do

- **Don't bundle with the LoCoMo extension story.** That's a separate paper or a clearly separate section. This benchmark stands or falls on its own.
- **Don't claim novelty on the methodology.** AttrScore (Yue 2023), Liu 2023, Bohnet 2022, Rashkin 2023 own LLM-as-judge citation faithfulness. We are *porting* it to personal-memory synthesis. Frame as recombination + first port to this benchmark line.
- **Don't author cashew-shaped softballs.** Questions like "what's user X's evolving emotional architecture" are too on-the-nose for think cycles. Reject during authoring, not after.
- **Don't run free-form before MCQ closes Phase A.** If the MCQ adapter is broken we'll waste authoring time.
- **Don't skip the human-judge audit.** Without it, the precision number is unfalsifiable and a reviewer will eat us.
- **Don't pre-commit to a "cashew wins" story.** §5 must be honored regardless of outcome.
