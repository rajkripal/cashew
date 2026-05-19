#!/usr/bin/env python3
"""embed-swap runner: re-run conv-26 variant A with thenlper/gte-large embeddings.

Constraints:
- Does NOT touch the main brain at /Users/bunny/.openclaw/workspace/cashew/data/graph.db.
- Does NOT touch the launchd MiniLM benchmark output (papers/locomo-run/results.jsonl + dbs/).
- Uses a side-snapshot of cashew (papers/locomo-run/cashew-gte/) with patched dim constants.
- Writes to dbs-gte/ and results-gte-large.jsonl.

Set CASHEW_ROOT, CASHEW_EMBEDDING_MODEL, CASHEW_EMBEDDING_DIM, CASHEW_SOCKET before
importing the adapter so the patched copy is loaded and the live daemon is bypassed.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Force the patched cashew copy.
GTE_ROOT = "/Users/bunny/.openclaw/workspace/cashew/papers/locomo-run/cashew-gte"
os.environ["CASHEW_ROOT"] = GTE_ROOT
os.environ["CASHEW_EMBEDDING_MODEL"] = "thenlper/gte-large"
os.environ["CASHEW_EMBEDDING_DIM"] = "1024"
os.environ["CASHEW_SOCKET"] = "/tmp/__embed_swap_no_daemon.sock"  # force socket miss
os.environ["CASHEW_NO_DAEMON"] = "1"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Make sure cashew-gte is FIRST on sys.path so `import core.embeddings` resolves to it.
sys.path.insert(0, GTE_ROOT)

# Now import the adapter (which inserts CASHEW_ROOT, defaulting to gte env override).
ADAPTER_DIR = "/Users/bunny/.openclaw/workspace/benchmarks/locomo"
sys.path.insert(0, ADAPTER_DIR)

from cashew_adapter import (  # noqa: E402
    answer_question,
    conv_db_path,
    exact_match,
    f1_score,
    init_conv_db,
    ingest_session,
    make_model_fn,
    retrieve,
)

# Sanity check: the loaded core.embedding_service must be the patched one.
from core import embedding_service as _esvc  # noqa: E402
assert _esvc.EMBEDDING_DIM == 1024, f"wrong embedding dim loaded: {_esvc.EMBEDDING_DIM}"
assert _esvc.DEFAULT_MODEL == "thenlper/gte-large", _esvc.DEFAULT_MODEL
print(f"[embed-swap] confirmed model={_esvc.DEFAULT_MODEL} dim={_esvc.EMBEDDING_DIM}")
print(f"[embed-swap] core module file: {_esvc.__file__}")

DATA_FILE = Path(ADAPTER_DIR) / "data" / "locomo10.json"
OUT_DIR = Path("/Users/bunny/.openclaw/workspace/cashew/papers/locomo-run")
DBS_DIR = OUT_DIR / "dbs-gte"
RESULTS = OUT_DIR / "results-gte-large.jsonl"
LOG_FILE = OUT_DIR / "embed-swap.log"
JUDGE_MODEL = os.environ.get("LOCOMO_JUDGE_MODEL", "claude-sonnet-4-6")
EXTRACT_MODEL = os.environ.get("LOCOMO_EXTRACT_MODEL", "claude-sonnet-4-6")
TARGET_CONV = "conv-26"


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a") as f:
        f.write(line + "\n")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def session_keys(conv: dict) -> list:
    return sorted(
        [k for k in conv.keys() if k.startswith("session_") and "date_time" not in k],
        key=lambda k: int(k.split("_")[1]),
    )


def conv_db_path_gte(conv_id: str, variant: str = "A") -> str:
    DBS_DIR.mkdir(parents=True, exist_ok=True)
    return str(DBS_DIR / f"{conv_id}-{variant}.db")


def init_db_gte(conv_id: str, variant: str = "A") -> str:
    db = conv_db_path_gte(conv_id, variant)
    if not Path(db).exists():
        # Re-use cashew's _ensure_schema by calling the adapter helper, but we
        # need to swap the path. Adapter's init_conv_db uses out_dir / 'dbs' —
        # we want out_dir / 'dbs-gte'. Simplest: temp-symlink-style by calling
        # the underlying function directly.
        from core.session import _ensure_schema  # noqa: F401
        _ensure_schema(db)
    return db


def ingest_all_sessions(sample: dict, db: str, model_fn) -> None:
    keys = session_keys(sample["conversation"])
    speakers = [sample["conversation"]["speaker_a"], sample["conversation"]["speaker_b"]]
    conv_id = sample["sample_id"]
    for key in keys:
        idx = int(key.split("_")[1])
        date = sample["conversation"].get(f"session_{idx}_date_time", "unknown date")
        turns = sample["conversation"][key]
        try:
            res = ingest_session(db, conv_id, idx, date, speakers, turns, model_fn)
            log(f"ingest s{idx}: new_nodes={res.get('new_nodes', '?')} "
                f"new_edges={res.get('new_edges', '?')} delta={res.get('nodes_delta', '?')}")
        except Exception as e:
            log(f"ingest ERROR s{idx}: {e!r}")
            traceback.print_exc()


def append_result(rec: dict) -> None:
    with RESULTS.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def main() -> int:
    log(f"START embed-swap conv={TARGET_CONV} model={os.environ['CASHEW_EMBEDDING_MODEL']} "
        f"dim={os.environ['CASHEW_EMBEDDING_DIM']}")
    log(f"results -> {RESULTS}")
    log(f"dbs -> {DBS_DIR}")

    # Load locomo data, find conv-26.
    samples = json.loads(DATA_FILE.read_text())
    sample = next(s for s in samples if s["sample_id"] == TARGET_CONV)
    qa = sample["qa"]
    log(f"qa count = {len(qa)}")

    db = init_db_gte(TARGET_CONV, variant="A")
    log(f"db path = {db}")

    # If RESULTS already has entries for this conv variant A, resume from after.
    done_idx = set()
    if RESULTS.exists():
        for line in RESULTS.read_text().splitlines():
            try:
                r = json.loads(line)
                if r.get("conv_id") == TARGET_CONV and r.get("variant") == "A":
                    done_idx.add(r.get("q_idx"))
            except Exception:
                pass
    log(f"resume: {len(done_idx)} questions already in results")

    # Ingest if DB looks empty (resume-safe).
    import sqlite3
    conn = sqlite3.connect(db)
    n_nodes = conn.execute("SELECT count(*) FROM thought_nodes").fetchone()[0]
    conn.close()
    log(f"existing nodes in db: {n_nodes}")
    model_fn = make_model_fn(EXTRACT_MODEL)
    if n_nodes == 0:
        t0 = time.time()
        ingest_all_sessions(sample, db, model_fn)
        log(f"ingest complete in {time.time()-t0:.1f}s")
        conn = sqlite3.connect(db)
        n_nodes = conn.execute("SELECT count(*) FROM thought_nodes").fetchone()[0]
        n_emb = conn.execute("SELECT count(*) FROM embeddings").fetchone()[0]
        conn.close()
        log(f"after ingest: nodes={n_nodes} embeddings={n_emb}")
    else:
        log("ingest SKIPPED (db has nodes)")

    # Run QA on all 199 questions, variant A semantics (no sleep+think).
    for q_idx, item in enumerate(qa):
        if q_idx in done_idx:
            continue
        question = item.get("question", "")
        gold = str(item.get("answer", ""))
        category = int(item.get("category", 1))
        if category == 5 and not gold:
            gold = "Not mentioned in the conversation"

        try:
            ctx, retr_latency = retrieve(db, question)
        except Exception as e:
            log(f"retrieve ERROR q{q_idx}: {e!r}")
            ctx, retr_latency = "", 0.0

        try:
            t0 = time.time()
            pred, meta = answer_question(ctx, question, category, gold, judge_model=JUDGE_MODEL)
            ans_latency = time.time() - t0
        except Exception as e:
            log(f"answer ERROR q{q_idx}: {e!r}")
            pred, meta, ans_latency = "", {"usage": {}}, 0.0

        f1, prec, rec = f1_score(pred, gold)
        em = exact_match(pred, gold)
        out = {
            "conv_id": TARGET_CONV,
            "variant": "A",
            "q_idx": q_idx,
            "category": category,
            "question": question,
            "gold": gold,
            "pred": pred,
            "f1": f1,
            "precision": prec,
            "recall": rec,
            "exact_match": em,
            "retrieval_latency_s": retr_latency,
            "answer_latency_s": ans_latency,
            "retrieved_chars": len(ctx),
            "judge_input_tokens": meta.get("usage", {}).get("input_tokens", 0),
            "judge_output_tokens": meta.get("usage", {}).get("output_tokens", 0),
            "ts": now_iso(),
            "embedding_model": "thenlper/gte-large",
        }
        append_result(out)
        log(f"q{q_idx} cat{category} F1={f1:.2f} EM={em} pred={pred[:60]!r} gold={gold[:50]!r}")

    log("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
