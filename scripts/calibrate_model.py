#!/usr/bin/env python3
"""Measure an embedding model's cosine distribution and recommend thresholds.

Similarity thresholds (cross-link, dedup, novelty) are model-specific and must
be measured, not guessed. Run this against a representative graph for a model,
then add/update the resulting ModelProfile in core/model_profiles.py.

Usage:
    python scripts/calibrate_model.py --db data/graph.db [--pairs 40000]

It reports:
- the unrelated-pair cosine distribution (random pairs)
- the per-node nearest-neighbor distribution (the floor below which distinct
  nodes live, so novelty must sit above it)
- recommended cross-link / dedup / novelty thresholds derived from those
"""

import argparse
import sqlite3
import sys

import numpy as np


def _load_vectors(db_path: str) -> np.ndarray:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT vector FROM embeddings WHERE vector IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        print(f"No embeddings found in {db_path}", file=sys.stderr)
        sys.exit(1)
    M = np.stack([np.frombuffer(r[0], dtype=np.float32) for r in rows]).astype(np.float32)
    M /= np.clip(np.linalg.norm(M, axis=1, keepdims=True), 1e-8, None)
    return M


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--pairs", type=int, default=40000, help="random pairs to sample")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    M = _load_vectors(args.db)
    n, dim = M.shape
    print(f"model space: {n} vectors, dim {dim}")

    rng = np.random.default_rng(args.seed)
    i = rng.integers(0, n, size=args.pairs)
    j = rng.integers(0, n, size=args.pairs)
    keep = i != j
    rand_sims = np.einsum("ij,ij->i", M[i[keep]], M[j[keep]])

    # Per-node nearest neighbor (full matrix is fine for graph-sized n).
    S = M @ M.T
    np.fill_diagonal(S, -1.0)
    nn = S.max(axis=1)

    def pct(a, p):
        return round(float(np.percentile(a, p)), 3)

    print("\nunrelated-pair cosine (random pairs):")
    print(f"  mean {round(float(rand_sims.mean()), 3)}  "
          f"P50 {pct(rand_sims, 50)}  P95 {pct(rand_sims, 95)}  P99 {pct(rand_sims, 99)}")
    print("per-node nearest-neighbor cosine:")
    print(f"  P25 {pct(nn, 25)}  P50 {pct(nn, 50)}  P90 {pct(nn, 90)}  P99 {pct(nn, 99)}")

    # Recommendations: cross-link well into the tail of unrelated pairs (P99-ish,
    # rounded up), dedup above that, novelty above the distinct-node NN median.
    cross = max(round(pct(rand_sims, 99) + 0.03, 2), round(pct(rand_sims, 95) + 0.05, 2))
    dedup = round(cross + 0.04, 2)
    novelty = round(max(dedup + 0.01, pct(nn, 50) + 0.03), 2)
    print("\nrecommended ModelProfile thresholds (verify before committing):")
    print(f"  cross_link_threshold = {cross}")
    print(f"  dedup_threshold      = {dedup}")
    print(f"  novelty_threshold    = {novelty}")


if __name__ == "__main__":
    main()
