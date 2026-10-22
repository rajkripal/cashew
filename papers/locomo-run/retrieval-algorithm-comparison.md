# Retrieval Algorithm Comparison: LoCoMo Results

Evaluated four retrieval configurations on the LoCoMo benchmark to determine whether uniform scoring or query expansion (QE) improves over the vector+recency baseline.

## Configurations

| Config | Scoring | QE |
|---|---|---|
| cashew-gte-baseline | vector + recency blend | no |
| cashew-uniform-gte | uniform (no recency weight) | no |
| cashew-qe-gte | vector + recency blend | yes |
| cashew-uniform-gte+qe | uniform | yes |

## conv-26 Results (n=199)

| Config | F1 | ex@5 |
|---|---|---|
| cashew-gte-baseline | 0.537 | 0.394 |
| cashew-uniform-gte | 0.544 | 0.402 |
| cashew-qe-gte | 0.547 | 0.407 |
| cashew-uniform-gte+qe | 0.551 | 0.413 |

## Cross-Validation

| Conv | n | cashew-qe-gte F1 | cashew-uniform-gte+qe F1 | Delta |
|---|---|---|---|---|
| conv-30 | 105 | 0.535 | 0.537 | +0.002 |
| conv-41 | 193 | 0.464 | 0.283 | -0.181 |

## Recommendation

Keep vector+recency blend as default. Do not ship uniform scoring.

Uniform shows a small gain on conv-26 (+0.014 F1 over baseline) but collapses on conv-41 (-0.181 F1 vs qe-gte). The gain is not robust across conversations. The risk of a large regression outweighs the marginal upside.

QE provides a consistent small improvement (+0.010 F1 on conv-26, roughly flat on conv-30, and stable on conv-41 at 0.464). If QE is ever shipped, it should use the vector+recency blend, not uniform scoring.
