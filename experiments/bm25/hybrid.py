"""BM25 (FTS5) seed + RRF hybrid seed for cashew retrieval ablation.

Same signature as core.embeddings.search: (db_path, query, top_k) -> [(node_id, score)].
So any of these can be monkeypatched onto core.retrieval.embedding_search to vary
ONLY the seed step (ablation A: walk held at cosine).
"""
import re
import sqlite3
from core.embeddings import search as dense_search


def ensure_fts5(db_path: str) -> int:
    """Build an FTS5 index over live thought_nodes.content. Returns rows indexed."""
    conn = sqlite3.connect(db_path)
    conn.execute("DROP TABLE IF EXISTS nodes_fts")
    conn.execute("CREATE VIRTUAL TABLE nodes_fts USING fts5(node_id UNINDEXED, content)")
    rows = conn.execute(
        "SELECT id, content FROM thought_nodes WHERE (decayed IS NULL OR decayed=0) AND content IS NOT NULL"
    ).fetchall()
    conn.executemany("INSERT INTO nodes_fts(node_id, content) VALUES (?,?)", rows)
    conn.commit()
    n = conn.execute("SELECT count(*) FROM nodes_fts").fetchone()[0]
    conn.close()
    return n


def _fts_query(query: str) -> str:
    # Extract word tokens, OR them (wide net for recall), quote to dodge FTS5 syntax.
    terms = re.findall(r"[A-Za-z0-9]+", query.lower())
    terms = [t for t in terms if len(t) > 2][:20]
    return " OR ".join(f'"{t}"' for t in terms) if terms else '""'


def bm25_search(db_path, query, top_k=10):
    """Lexical seed via FTS5 bm25(). bm25() is more-negative=better; negate to a score."""
    match = _fts_query(query)
    if match == '""':
        return []
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT node_id, bm25(nodes_fts) AS s FROM nodes_fts WHERE nodes_fts MATCH ? "
            "ORDER BY s LIMIT ?", (match, top_k)
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    conn.close()
    # bm25() returns negative (better = more negative); flip to positive score.
    return [(nid, -s) for nid, s in rows]


def hybrid_search(db_path, query, top_k=10, k_rrf=60, fetch=30):
    """RRF fusion of dense + bm25 seed lists. score = sum 1/(k_rrf + rank)."""
    dense = dense_search(db_path, query, top_k=fetch)
    bm = bm25_search(db_path, query, top_k=fetch)
    scores = {}
    for lst in (dense, bm):
        for rank, (nid, _) in enumerate(lst):
            scores[nid] = scores.get(nid, 0.0) + 1.0 / (k_rrf + rank + 1)
    ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
    return ranked
