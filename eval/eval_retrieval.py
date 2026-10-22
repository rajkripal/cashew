#!/usr/bin/env python3
"""
Retrieval evaluation harness — first cut.

Runs a fixed "needle set" of (query, expected_node_ids) pairs against the
production retrieval function (core.retrieval.retrieve_recursive_bfs) and
reports Recall@k (k=1,5,10), mean rank, and MRR.

This exists so future retrieval-quality decisions (hybrid BM25, reranking,
HNSW vs exact scan, etc.) can be measured against a baseline instead of
judged on vibes. The needle set at eval/retrieval_needles.json is a first
draft seeded from real node content in data/graph.db — expected node picks
need human review, not ground truth yet.

Usage:
    python3 eval/eval_retrieval.py [--db data/graph.db] [--needles eval/retrieval_needles.json] [--top-k 10]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.retrieval import retrieve_recursive_bfs


def load_needles(path: str):
    with open(path) as f:
        data = json.load(f)
    return data["needles"]


def rank_of_expected(ranked_ids, expected_ids):
    """Return 1-indexed rank of the first expected id found in ranked_ids, or None."""
    expected_set = set(expected_ids)
    for i, nid in enumerate(ranked_ids, start=1):
        if nid in expected_set:
            return i
    return None


def evaluate(db_path: str, needles, top_k: int):
    ks = [1, 5, 10]
    hits_at_k = {k: 0 for k in ks}
    ranks = []
    reciprocal_ranks = []
    per_query = []

    for needle in needles:
        query = needle["query"]
        expected = needle["expected"]

        results = retrieve_recursive_bfs(db_path, query, top_k=max(top_k, max(ks)))
        ranked_ids = [r.node_id for r in results]

        rank = rank_of_expected(ranked_ids, expected)
        per_query.append({"query": query, "expected": expected, "rank": rank, "n_results": len(ranked_ids)})

        if rank is not None:
            ranks.append(rank)
            reciprocal_ranks.append(1.0 / rank)
            for k in ks:
                if rank <= k:
                    hits_at_k[k] += 1
        else:
            reciprocal_ranks.append(0.0)

    n = len(needles)
    recall_at_k = {k: hits_at_k[k] / n for k in ks}
    mean_rank = sum(ranks) / len(ranks) if ranks else float("nan")
    mrr = sum(reciprocal_ranks) / n if n else float("nan")

    return {
        "n_queries": n,
        "recall_at_k": recall_at_k,
        "mean_rank": mean_rank,
        "mrr": mrr,
        "n_found": len(ranks),
        "per_query": per_query,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality against a needle set.")
    parser.add_argument("--db", default="data/graph.db", help="Path to graph.db")
    parser.add_argument("--needles", default="eval/retrieval_needles.json", help="Path to needle set JSON")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k results to request from retrieval")
    parser.add_argument("--verbose", action="store_true", help="Print per-query rank detail")
    args = parser.parse_args()

    needles = load_needles(args.needles)
    metrics = evaluate(args.db, needles, args.top_k)

    print(f"Retrieval eval — {metrics['n_queries']} queries against {args.db}")
    print(f"Found expected node in results for {metrics['n_found']}/{metrics['n_queries']} queries")
    for k, v in metrics["recall_at_k"].items():
        print(f"Recall@{k}: {v:.2%}")
    print(f"Mean rank (when found): {metrics['mean_rank']:.2f}")
    print(f"MRR: {metrics['mrr']:.3f}")

    if args.verbose:
        print("\nPer-query detail:")
        for pq in metrics["per_query"]:
            status = f"rank {pq['rank']}" if pq["rank"] is not None else "NOT FOUND"
            print(f"  [{status}] {pq['query']!r} -> expected {pq['expected']}")


if __name__ == "__main__":
    main()
