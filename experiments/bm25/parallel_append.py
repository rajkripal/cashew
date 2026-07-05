"""Raj's variant: BM25 as a SEPARATE parallel search, its hits APPENDED to the
dense result set (no cosine re-rank). Measures whether BM25 rescues any relevant
node the dense pipeline missed — the only way append adds value."""
import json, sqlite3, sys
sys.path.insert(0, '.')
import core.retrieval as R
from experiments.bm25.hybrid import ensure_fts5, bm25_search
DB='data/graph.db'; K=10; ensure_fts5(DB)
qs=json.load(open('experiments/bm25/eval_queries.json')); c=sqlite3.connect(DB)
def exp(idr): return set(r[0] for r in c.execute("select id from thought_nodes where (decayed is null or decayed=0) and lower(content) like ?",('%'+idr.lower()+'%',)).fetchall())
for q in qs: q['_exp']=exp(q['id'])

def dense_pipeline(query): return [r.node_id for r in R.retrieve_recursive_bfs(DB,query,top_k=K,n_seeds=5)]

tot_rescue=0; dh=0; drec=0; brec=0; n=0
print(f"{'type':8} {'query id':22} | dense_hit  bm25_rescues (relevant nodes dense's top10 missed that bm25 top10 caught)")
for q in qs:
    e=q['_exp']
    if not e: continue
    n+=1
    d=dense_pipeline(q['query'])
    b=[nid for nid,_ in bm25_search(DB,q['query'],K)]
    rescued=(e - set(d)) & set(b)
    tot_rescue+=len(rescued)
    dh += 1 if set(d)&e else 0
    drec += len(set(d)&e)/min(len(e),K)
    blend=d[:7]+[x for x in b if x not in d[:7]][:3]      # fixed-budget 7 dense + 3 bm25
    brec += len(set(blend)&e)/min(len(e),K)
    if len(rescued) or not (set(d)&e):
        print(f"{q['type']:8} {q['id'][:22]:22} | {'Y' if set(d)&e else 'N':>8}  {len(rescued)} rescued")
print(f"\nDense pipeline: hit@10={dh/n:.3f}  rec@10={drec/n:.3f}")
print(f"Fixed-budget blend (7 dense + 3 bm25): rec@10={brec/n:.3f}")
print(f"TOTAL relevant nodes BM25-parallel rescued that dense missed: {tot_rescue}  across {n} queries")
