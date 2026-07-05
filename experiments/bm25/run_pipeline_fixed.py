"""Clean ablation A pipeline: seed SELECTION varies (dense/bm25/hybrid), but seeds
are re-scored with TRUE cosine so retrieve_recursive_bfs ranks honestly (holding
walk+rank at cosine). Isolates 'does hybrid seed selection help end-to-end'."""
import json, sqlite3, sys
sys.path.insert(0, '.')
import numpy as np
import core.retrieval as R
from core.embeddings import search as dense_search, embed_text
from experiments.bm25.hybrid import ensure_fts5, bm25_search, hybrid_search

DB='data/graph.db'; TOPK=10
ensure_fts5(DB)
qs=json.load(open('experiments/bm25/eval_queries.json'))
c=sqlite3.connect(DB)

_veccache={}
def node_vec(nid):
    if nid in _veccache: return _veccache[nid]
    row=c.execute("SELECT vector FROM embeddings WHERE node_id=?",(nid,)).fetchone()
    v=np.frombuffer(row[0],dtype=np.float32) if row and row[0] else None
    _veccache[nid]=v; return v

def cosine_scored(fn):
    def wrapped(db,query,top_k=10):
        qv=np.array(embed_text(query),dtype=np.float32); qn=np.linalg.norm(qv)
        ids=[nid for nid,_ in fn(db,query,top_k)]
        out=[]
        for nid in ids:
            v=node_vec(nid)
            cos=float(np.dot(qv,v)/(qn*np.linalg.norm(v))) if v is not None and np.linalg.norm(v)>0 else 0.0
            out.append((nid,cos))
        return out
    return wrapped

def expected(idr):
    return set(r[0] for r in c.execute("select id from thought_nodes where (decayed is null or decayed=0) and lower(content) like ?",('%'+idr.lower()+'%',)).fetchall())
for q in qs: q['_exp']=expected(q['id']); q['_needle']=len(q['_exp'])<=5

def metrics(ids,exp):
    hit=1.0 if any(n in exp for n in ids[:TOPK]) else 0.0
    rr=next((1.0/(i+1) for i,n in enumerate(ids[:TOPK]) if n in exp),0.0)
    rec=len(set(ids[:TOPK])&exp)/min(len(exp),TOPK)
    return hit,rr,rec

print(f"{'variant':8} | {'hit@10':>7} {'MRR':>6} {'rec@10':>7} || needle {'hit@10':>7} {'MRR':>6} {'rec@10':>7}")
for name,fn in [('dense',dense_search),('bm25',bm25_search),('hybrid',hybrid_search)]:
    R.embedding_search=cosine_scored(fn)
    A=[0,0,0,0]; N=[0,0,0,0]
    for q in qs:
        if not q['_exp']: continue
        ids=[r.node_id for r in R.retrieve_recursive_bfs(DB,q['query'],top_k=TOPK,n_seeds=5)]
        h,rr,rec=metrics(ids,q['_exp'])
        A[0]+=h;A[1]+=rr;A[2]+=rec;A[3]+=1
        if q['_needle']: N[0]+=h;N[1]+=rr;N[2]+=rec;N[3]+=1
    an=A[3] or 1; nn=N[3] or 1
    print(f"{name:8} | {A[0]/an:7.3f} {A[1]/an:6.3f} {A[2]/an:7.3f} || {'':6} {N[0]/nn:7.3f} {N[1]/nn:6.3f} {N[2]/nn:7.3f}")
R.embedding_search=dense_search
