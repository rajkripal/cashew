# Ablation A: BM25 hybrid vs dense retrieval (seed step)

Run 2026-07-05 on a copy of the live graph (6,384 nodes / 3,507 live, gte-large 1024d).
16 hand-authored queries balanced across semantic-paraphrase, exact-term lexical, and mixed.
Ground truth = nodes whose content matches a distinctive identifier (needle = <=5 expected).

## Numbers

SEED step (which nodes each method selects, top-10):
| variant | hit@10 | MRR   | rec@10 | needle hit@10 | needle MRR |
|---------|--------|-------|--------|---------------|------------|
| dense   | 1.000  | 0.730 | 0.619  | 1.000         | 0.875      |
| bm25    | 0.688  | 0.505 | 0.469  | 0.625         | 0.542      |
| hybrid  | 1.000  | 0.719 | 0.656  | 1.000         | 0.760      |

Full pipeline (retrieve_recursive_bfs, seeds re-scored to TRUE cosine so ranking is honest):
| variant | hit@10 | MRR   | rec@10 |
|---------|--------|-------|--------|
| dense   | 0.938  | 0.723 | 0.625  |
| bm25    | 0.938  | 0.673 | 0.594  |
| hybrid  | 1.000  | 0.746 | 0.650  |

## Findings

1. **Dense already saturates coverage.** hit@10 = 1.0 at the seed step — dense finds a
   relevant node for EVERY query, including the rare-term lexical ones (Trupanion, HippoRAG,
   RRF, gte-large, queue-sweeper all at rank 1). gte-large is strong enough that lexical
   queries are not a dense weakness at this corpus size, so BM25 adds no coverage.
2. **BM25 alone is strictly worse** (hit@10 0.69): it whiffs the pure-semantic paraphrase
   queries entirely (goes-silent, imminent-success, dumb-graph, Chiki, rabbit-silhouette)
   because the query shares no keywords with the node.
3. **Hybrid's only value is ranking, and it's marginal.** RRF pulls up nodes both signals
   agree on (sycophancy 7->2, organic-decay 5->2, Meta 5->1), nudging pipeline MRR from
   0.723 to 0.746 and hit@10 from 0.938 to 1.0 — but that's ONE query on a 16-query set,
   within noise, and hybrid also demotes some (Chiki 1->4) where BM25 disagrees.
4. **Integration gotcha:** retrieve_recursive_bfs uses the seed's SELECTION score as the
   final cosine rank (cosine_sim returns seed_scores[nid] for seeds). A naive BM25/hybrid
   seed swap corrupts ranking — hybrid collapsed to hit@10 0.375 until seeds were re-scored
   by cosine. Any real hybrid impl must decouple selection score from ranking score.

## Recommendation

Do NOT add BM25/hybrid at cashew's current scale. gte-large saturates recall; hybrid's edge
is a marginal, noisy MRR nudge that doesn't justify the FTS5 table + RRF + ranking-decouple
work. Revisit only if (a) the graph grows large (10k-100k+) where brute-force dense recall
degrades, or (b) exact-match retrieval of specific IDs/rare tokens becomes critical AND the
embedding model starts missing them. Ablation B (hybrid walk/ranking) is unlikely to change
this given dense already hits ceiling on coverage.

## Follow-up: parallel BM25 append (Raj's variant, better architecture)

Run BM25 as a SEPARATE parallel search and append its top hits directly to the dense
result set, instead of fusing at the seed and letting cosine re-rank bury them. This is
strictly better at preserving BM25's contribution (no cosine neutralization).

Result (16 queries, dense pipeline top-10 vs bm25 parallel top-10):
- BM25-parallel rescued **12 relevant nodes** dense's top-10 missed — vs ~0 net from the
  RRF-into-seed approach. So the architecture matters: append > fuse-then-rerank.
- BUT 11 of 12 rescues are extra nodes on topics dense ALREADY covered (depth, not coverage).
  Only **1** is a genuine coverage fix (the "sycophancy" query — dense missed the topic
  entirely, BM25 caught it; hit@10 0.938 -> 1.0).
- Fixed result budget (7 dense + 3 bm25): rec@10 0.625 -> 0.656 (+0.03, marginal).
- The larger gain only materializes if you GROW the result set to include the appends, which
  spends more context — and bounded context is cashew's core value prop.

Updated take: Raj's parallel-append is the correct architecture and captures value the
seed-fusion approach discarded. But at current scale the payoff is still modest (one
coverage fix + more depth per already-covered topic), because dense saturates topic coverage.
Clearly worth it at larger scale, or if the goal shifts from "hit the topic" to "surface more
relevant nodes per topic." Reasonable next step: wire it as an optional flag and A/B live.
