"""Ablation A: dense vs BM25 vs hybrid SEED. Walk/final-rank held at cosine.
Measures the seed step directly (the real comparison) AND the full pipeline
(retrieve_recursive_bfs), since the pipeline re-ranks by cosine and may neutralize
seed gains — that gap is itself the finding that motivates ablation B."""
import json, sqlite3, sys
sys.path.insert(0, '.')
import core.retrieval as R
from core.embeddings import search as dense_search
from experiments.bm25.hybrid import ensure_fts5, bm25_search, hybrid_search

DB = 'data/graph.db'
TOPK = 10
ensure_fts5(DB)
qs = json.load(open('experiments/bm25/eval_queries.json'))
c = sqlite3.connect(DB)

def expected_ids(identifier):
    return set(r[0] for r in c.execute(
        "select id from thought_nodes where (decayed is null or decayed=0) and lower(content) like ?",
        ('%' + identifier.lower() + '%',)).fetchall())

def metrics(returned_ids, exp):
    hit = 1.0 if any(n in exp for n in returned_ids[:TOPK]) else 0.0
    rr = 0.0
    for i, n in enumerate(returned_ids[:TOPK]):
        if n in exp: rr = 1.0 / (i + 1); break
    rec = len(set(returned_ids[:TOPK]) & exp) / min(len(exp), TOPK)
    return hit, rr, rec

SEEDS = {'dense': dense_search, 'bm25': bm25_search, 'hybrid': hybrid_search}

# Precompute expected + needle flag
for q in qs:
    q['_exp'] = expected_ids(q['id']); q['_needle'] = len(q['_exp']) <= 5

def run(level):
    out = {}
    for name, fn in SEEDS.items():
        agg = {'all': [0,0,0,0], 'needle': [0,0,0,0]}  # hit,rr,rec,count
        for q in qs:
            exp = q['_exp']
            if not exp: continue
            if level == 'seed':
                ids = [nid for nid, _ in fn(DB, q['query'], TOPK)]
            else:  # pipeline: swap the seed, keep cosine walk/rank
                R.embedding_search = fn
                ids = [r.node_id for r in R.retrieve_recursive_bfs(DB, q['query'], top_k=TOPK, n_seeds=5)]
                R.embedding_search = dense_search
            h, rr, rec = metrics(ids, exp)
            for bucket in (['all'] + (['needle'] if q['_needle'] else [])):
                agg[bucket][0]+=h; agg[bucket][1]+=rr; agg[bucket][2]+=rec; agg[bucket][3]+=1
        out[name] = agg
    return out

for level in ('seed', 'pipeline'):
    res = run(level)
    print(f"\n===== {level.upper()} level (n={len(qs)} queries) =====")
    print(f"{'variant':8} | {'hit@10':>7} {'MRR':>6} {'rec@10':>7} || needle: {'hit@10':>7} {'MRR':>6} {'rec@10':>7}")
    for name, agg in res.items():
        a = agg['all']; nd = agg['needle']
        an = a[3] or 1; ndn = nd[3] or 1
        print(f"{name:8} | {a[0]/an:7.3f} {a[1]/an:6.3f} {a[2]/an:7.3f} || {'':8} {nd[0]/ndn:7.3f} {nd[1]/ndn:6.3f} {nd[2]/ndn:7.3f}  (n_needle={nd[3]})")
