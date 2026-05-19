# QUAL-DIFF — cashew/A vs Mem0 on conv-26 (199 paired questions)

Reviewer: critic-pass4 (locomo-cashew). Method: paired by (conv_id, q_idx), all numbers same scoring path (Porter+set-EM F1).

Headline: cashew/A 0.519 vs Mem0 0.566 ALL; ex-cat-5 0.371 vs 0.431 (Δ = -0.060). Wins narrowly cat-2/cat-3, loses cat-1 (-0.095), cat-4 (-0.105).

---

## 1. Failure-mode classification

Bucket: paired questions where Mem0 beats cashew by F1 ≥ 0.40 (n = 17 of 199, 8.5%).

| | cat-1 | cat-2 | cat-3 | cat-4 | total |
|---|---|---|---|---|---|
| big-loss count | 5 | 2 | 0 | 10 | 17 |
| (A) cashew abstained ("no information") | 3 | 1 | 0 | 5 | 9 |
| (B) cashew confidently wrong | 2 | 1 | 0 | 5 | 8 |

Then I cross-checked the cashew-A DB (`dbs/conv-26-A.db`, 556 thought_nodes) for the literal facts cashew failed to recall. Of 12 probed losing cases, **10 of 12 facts are present in the graph as nodes** with the right keywords (Sweden, guinea pig, sunflower-warmth-happiness, council/adoption, sunset, adoption agencies, Perseid, Becoming Nicole, all three pet names Oliver/Luna/Bailey, de-stress). Only 2 looked genuinely absent: "rainbow sidewalk" and "palm tree" (and "palm tree" is likely a paraphrase issue — sunset paintings are extracted).

**This collapses Class C (never-extracted) to ~17% of the failure set. ~83% of cashew's big losses are Class A: the fact is in the graph, retrieval did not surface it.**

Class D (Mem0 extracted something cashew didn't) is essentially empty here — the data flatly says cashew's extractor caught the same facts. Class B (retrieved-right-but-answered-wrong) is hard to separate from A without re-running the QA prompt with full graph dump, but the abstention rate (9/17) plus ~1900-char retrieved context says retrieval was the bottleneck more often than not.

---

## 2. Representative examples (side-by-side)

| q | cat | gold | cashew/A | Mem0 | in DB? | verdict |
|---|---|---|---|---|---|---|
| 11 | 1 | Sweden | "No information available." | "Sweden (her home country)" | YES (2 nodes) | Class A — retrieval miss |
| 3 | 1 | Adoption agencies | "No information available." | "Adoption agencies." | YES (12 nodes) | Class A — retrieval miss |
| 123 | 4 | guinea pig | "No information available." | "A guinea pig named Oscar." | YES ("Oscar, parsley") | Class A — retrieval miss |
| 118 | 4 | Perseid meteor shower | "Nature, hiking, marshmallows" (off-topic) | "The Perseid meteor shower…" | YES (3 nodes) | Class A — retrieval pulled wrong nodes |
| 114 | 4 | warmth and happiness | "No information available." | "Warmth and happiness." | YES (sunflower+warmth+happiness all present) | Class A — retrieval miss |
| 55 | 1 | Sunsets | "No information available." | "Sunsets." | YES (12 sunset nodes) | Class A — retrieval miss |
| 52 | 1 | Oliver, Luna, Bailey | "Inconsistent. Multiple conflicting facts: dog Oliver / cat Luna; cat Bailey…" | "Luna, Oliver, Bailey" | YES (all 3 names) | Class B — retrieval surfaced contradictions, no resolver |
| 116 | 4 | LGBTQ center, unity and strength | "Trans journey, rejection of binary gender, red/blue blending" | "Visit to LGBTQ center, unity and strength" | mixed | Class B — wrong but plausible alt-fact picked |
| 135 | 4 | Got hurt, took pottery break (Oct '23) | "Son was in a car accident on a road trip" | "Got injured Sep '23, pottery break" | both extracted | Class B — picked the wrong "setback" node |
| 128 | 4 | rainbow sidewalk | "No information available." | "A rainbow sidewalk design painted for Pride" | NO | Class C — extraction miss |

Pattern: when cashew answers it is often right or close (cat-2 dates: "August 2023" vs "week of 23 Aug 2023"; running reason 0.55 F1; library books 0.57 F1). When it loses it is usually because a fact that **exists as a graph node** never made it into the 60K-char retrieved context.

## 3. cashew's wins (n=2 strict, both cat-2)

- q=80, "When did Melanie buy figurines?" — cashew "21 October 2023" (1.0); Mem0 "no information about figurines" (0.0). cashew's referent_time tagging worked; Mem0 didn't surface the dated event.
- q=72, "When did Melanie's friend adopt a child?" — cashew "2022" (1.0); Mem0 "2022–2023" (0.0). Same story: cashew's date metadata wins on a precise temporal query.

This is the only place cashew has a clean architectural advantage: **dated event nodes with `referent_time`**. It shows up as the cat-2 win (+2.8 F1 vs Mem0).

## 4. Identical-prediction analysis (n=74 of 199, 37%)

| cat | identical | total | %identical |
|---|---|---|---|
| 1 | 0 | 33 | 0% |
| 2 | 13 | 64 | 20% |
| 3 | 3 | 6 | 50% |
| 4 | 11 | 49 | 22% |
| 5 | 47 | 47 | 100% |

cat-5 is fully tautological (both systems emit "Not mentioned", which is gold). On cat-1 they never agree word-for-word, but both are usually right just phrased differently. The "easy" overlap is concentrated in cat-2 dates where both systems converge on the right ISO string.

---

## 5. Regime-fit verdict — does LoCoMo test what cashew is built for?

**Mostly NO.** Three lines of evidence:

1. **Scale.** conv-26 has 556 thought_nodes after ingest, 234 after sleep+think. cashew's design points (organic decay, dream consolidation, cross-domain synthesis, sqrt(N) core promotion at 10K+ nodes) are inert at this scale. Decay does nothing in 10 days of bench data; dreams have nothing to consolidate across domains because LoCoMo is a single conversational domain per conv.

2. **Question shape.** cat-1 (single-hop "What is X's pet?") and cat-4 (open "What did Caroline take from book Y?") are bounded retrieval over a fixed corpus. The information is one or two extracted nodes away. This rewards a flat, well-indexed memory store (Mem0), not a graph that bets on multi-hop walks. cashew's BFS+vector retrieval has no special advantage here, and as Section 1 shows, it actually under-retrieves vs Mem0's flatter store.

3. **No cross-domain synthesis is tested.** The questions where cashew's design *should* shine — "given everything you know about Caroline across her career, family, and creative work, what would she likely do?" — do not exist in LoCoMo. Even cat-4 "open" questions are factual recall within one conversation, not cross-corpus inference.

What LoCoMo *does* legitimately test that cashew handles well: **temporal grounding**. cashew's `referent_time` metadata produced both the cat-2 wins above. That's a real and defensible win, but it is one feature, not the architecture.

The author's hypothesis stands: this is the wrong benchmark for the headline claims. The right benchmark is a 10K+ node personal corpus with cross-domain "synthesis" probes — which doesn't exist publicly today.

---

## 6. Recommendation

**Fix order, in priority:**

1. **Retrieval, not extraction, is the bug.** ~83% of big losses are retrieval misses on facts that *are* in the graph. Before any architecture-level work, profile `generate_session_context` on conv-26: for each abstention, dump the candidate node set, the retrieved set, and the rank cutoff. The hypothesis to falsify: vector-similarity for short factoid questions ("Where did Caroline move from 4 years ago?") doesn't match the way the fact is phrased in the node ("Caroline is originally from Sweden"). Likely fixes: hybrid keyword+vector, query expansion with extracted entities, or raise the recall budget. This is the single highest-leverage fix.

2. **Contradiction resolution.** q=52 (pet names) shows cashew correctly stored all three pets but the QA prompt got contradictory snippets and refused to commit. A merge/conflict resolution pass during retrieval — or a "consolidate before answer" step — would turn a 0.09 into a 1.0.

3. **Don't retune the architecture for LoCoMo.** Sleep+think is already net-negative on this benchmark (cashew B < cashew A). The honest paper move is to scope the LoCoMo result as "untyped graph + temporal metadata holds parity with Mem0_g on dates while losing 6 F1 elsewhere on a benchmark that doesn't exercise our scale-regime claims." Then build (or partner on) a 10K-node personal-corpus eval as the headline contribution.

**Different benchmark, AND a different cashew on this benchmark.** Both. The retrieval fix would close most of the 6-F1 gap on LoCoMo without claiming new architecture. The architecture claims need their own evaluation.
