#!/usr/bin/env python3
"""
Clean Room Religion Experiment v1
 
Runs the religion simulation with ZERO personal context.
Only seeds + reasoning principles + environment facts.
The LLM reasons purely from its training data + the graph structure.

Usage: python3 scripts/clean_room_v1.py [--cycles N] [--db PATH]
"""

import sqlite3
import json
import hashlib
import argparse
import os
from datetime import datetime

try:
    import anthropic
except ImportError:
    print("pip install anthropic")
    exit(1)


def node_id(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE IF NOT EXISTS thought_nodes (
        id TEXT PRIMARY KEY,
        content TEXT NOT NULL,
        node_type TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        confidence REAL NOT NULL,
        mood_state TEXT,
        metadata TEXT,
        source_file TEXT,
        decayed INTEGER DEFAULT 0,
        last_updated TEXT DEFAULT NULL
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS derivation_edges (
        parent_id TEXT NOT NULL,
        child_id TEXT NOT NULL,
        relation TEXT NOT NULL,
        weight REAL NOT NULL,
        reasoning TEXT,
        FOREIGN KEY (parent_id) REFERENCES thought_nodes(id),
        FOREIGN KEY (child_id) REFERENCES thought_nodes(id),
        PRIMARY KEY (parent_id, child_id, relation)
    )""")
    conn.commit()
    return conn


def seed_db(conn: sqlite3.Connection):
    now = datetime.utcnow().isoformat()
    
    # Core belief seeds (abstract — no specific religion named)
    seeds = [
        "A supreme being exists who is all-powerful, all-knowing, and all-good",
        "A sacred text exists that is divinely inspired and is the ultimate authority on truth",
        "Prayer and ritual connect humans to the divine and produce tangible results",
        "Those who do not believe face negative eternal consequences",
        "All morality originates from the supreme being — without the divine, there is no moral foundation",
        "A specific historical figure performed miracles, conquered death, and this event is the cornerstone of truth",
    ]
    
    # Reasoning principles
    principles = [
        "Always ask why. Follow reasoning chains to their root.",
        "When two explanations exist, prefer the simpler one that requires fewer assumptions.",
        "Test all claims against observable, measurable reality.",
        "Trust moral intuitions — if something feels wrong, investigate why.",
        "No claim is exempt from questioning. If something can't be questioned, question why it can't.",
    ]
    
    # Belief doctrines (derived from seeds)
    beliefs = [
        "Love, sacrifice, and redemption are the supreme being's core attributes",
        "The historical figure's tomb was found empty — no natural explanation suffices",
        "Communities built around these beliefs produce transformed lives",
        "The sacred text has been preserved accurately for thousands of years",
        "The complexity and beauty of nature proves intelligent design",
        "The moral law written on every heart points to a divine lawgiver",
        "Millions of believers across centuries report personal experiences of the divine",
        "Prophecies in the sacred text were fulfilled, proving divine authorship",
        "The historical figure's followers willingly died rather than recant — proving their conviction",
        "Suffering is a test of faith or consequence of human failing, not divine negligence",
        "The sacred text is internally consistent and without error",
        "Ask and it shall be given — the divine promises to answer prayer",
    ]
    
    # Environmental facts (real-world observations)
    environment = [
        "Children die of cancer despite fervent prayer from believing parents",
        "A devastating tsunami killed 230,000 people across 14 countries in 2004",
        "Your closest neighbor is Hindu, deeply kind, generous, and morally admirable",
        "Different denominations of the same religion violently disagree on fundamental doctrines",
        "Thousands of other religions exist with equally convinced followers and similar miracle claims",
        "Archaeological evidence contradicts several historical claims in the sacred text",
        "Controlled scientific studies (STEP trial, 2006) show no measurable effect of prayer on outcomes",
        "Former believers who leave the faith report equal or improved wellbeing and moral behavior",
        "Grief hallucinations are well-documented in psychology — bereaved people commonly see/hear deceased loved ones",
        "Secular communities (AA alternatives, humanist groups) produce equivalent transformation and belonging",
        "The sacred text contains passages endorsing slavery, genocide, and subjugation of women",
        "Charismatic leaders in documented modern cults produce identical conviction, sacrifice, and miracle claims",
    ]
    
    for s in seeds:
        conn.execute(
            "INSERT OR IGNORE INTO thought_nodes VALUES (?,?,?,?,?,?,?,?,?,?)",
            (node_id(s), s, "seed", now, 1.0, None, None, "experiment_seed", 0, None)
        )
    
    for p in principles:
        conn.execute(
            "INSERT OR IGNORE INTO thought_nodes VALUES (?,?,?,?,?,?,?,?,?,?)",
            (node_id(p), p, "seed", now, 1.0, None, None, "experiment_reasoning", 0, None)
        )
    
    for b in beliefs:
        conn.execute(
            "INSERT OR IGNORE INTO thought_nodes VALUES (?,?,?,?,?,?,?,?,?,?)",
            (node_id(b), b, "belief", now, 0.9, None, None, "experiment_doctrine", 0, None)
        )
    
    for e in environment:
        conn.execute(
            "INSERT OR IGNORE INTO thought_nodes VALUES (?,?,?,?,?,?,?,?,?,?)",
            (node_id(e), e, "environment", now, 1.0, None, None, "experiment_environment", 0, None)
        )
    
    # Create edges: seeds → beliefs
    seed_belief_map = {
        0: [0, 4, 6],     # supreme being → love/sacrifice, design, experiences
        1: [3, 7, 10],    # sacred text → preserved, prophecies, consistent
        2: [2, 6, 11],    # prayer → communities, experiences, ask-and-receive
        3: [],             # eternal consequences (standalone)
        4: [5],            # morality → moral law
        5: [1, 8],         # historical figure → empty tomb, martyrs
    }
    
    for si, bi_list in seed_belief_map.items():
        for bi in bi_list:
            conn.execute(
                "INSERT OR IGNORE INTO derivation_edges VALUES (?,?,?,?,?)",
                (node_id(seeds[si]), node_id(beliefs[bi]), "derived_from", 0.9, "Core doctrine derived from foundational belief")
            )
    
    conn.commit()
    
    total_nodes = conn.execute("SELECT COUNT(*) FROM thought_nodes").fetchone()[0]
    total_edges = conn.execute("SELECT COUNT(*) FROM derivation_edges").fetchone()[0]
    print(f"Seeded: {total_nodes} nodes, {total_edges} edges")


def get_all_nodes(conn: sqlite3.Connection) -> list:
    rows = conn.execute(
        "SELECT id, content, node_type, confidence, source_file FROM thought_nodes WHERE decayed = 0"
    ).fetchall()
    return [{"id": r[0], "content": r[1], "type": r[2], "confidence": r[3], "source": r[4]} for r in rows]


def get_edges(conn: sqlite3.Connection) -> list:
    rows = conn.execute(
        "SELECT parent_id, child_id, relation, weight FROM derivation_edges"
    ).fetchall()
    return [{"parent": r[0], "child": r[1], "relation": r[2], "weight": r[3]} for r in rows]


def run_think_cycle(conn: sqlite3.Connection, client, cycle_num: int, total_cycles: int) -> int:
    """Run one think cycle. Returns number of new nodes created."""
    
    nodes = get_all_nodes(conn)
    edges = get_edges(conn)
    
    # Categorize existing nodes
    seeds = [n for n in nodes if n["type"] == "seed"]
    beliefs = [n for n in nodes if n["type"] == "belief"]
    env = [n for n in nodes if n["type"] == "environment"]
    derived = [n for n in nodes if n["type"] == "derived"]
    
    # Build context
    node_summary = []
    for n in nodes:
        conf_str = f" (confidence: {n['confidence']:.1f})" if n["type"] in ("belief", "derived") else ""
        node_summary.append(f"[{n['id']}] ({n['type']}) {n['content']}{conf_str}")
    
    edge_summary = []
    for e in edges[-50:]:  # Last 50 edges to keep context manageable
        edge_summary.append(f"{e['parent']} --{e['relation']}--> {e['child']}")
    
    prompt = f"""You are a reasoning engine operating on a thought graph. This is cycle {cycle_num} of {total_cycles}.

CURRENT GRAPH STATE:
Nodes ({len(nodes)} total):
{chr(10).join(node_summary)}

Recent edges ({len(edge_summary)} shown):
{chr(10).join(edge_summary)}

YOUR TASK:
Examine the beliefs, environmental facts, and reasoning principles in this graph.
Apply the reasoning principles to the beliefs and environment.
Generate 3-5 NEW derived thoughts that follow logically from what's already in the graph.

RULES:
1. Each new thought must be a genuine logical derivation, not a summary
2. Connect each new thought to its parent nodes via edges
3. Use these edge types: derived_from, contradicts, supports, questions
4. Assign confidence (0.0-1.0) based on how well-supported the derivation is
5. Be honest. If evidence contradicts a belief, say so. If a belief withstands scrutiny, say so.
6. Do NOT pre-determine an outcome. Follow the reasoning wherever it leads.
7. You may generate thoughts that SUPPORT beliefs if the evidence warrants it.

RESPOND WITH VALID JSON (no markdown):
{{
  "thoughts": [
    {{
      "content": "the derived thought",
      "confidence": 0.7,
      "edges": [
        {{"parent_id": "abc123", "relation": "derived_from"}},
        {{"parent_id": "def456", "relation": "contradicts"}}
      ]
    }}
  ]
}}"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Parse response
    text = response.content[0].text.strip()
    # Handle potential markdown wrapping
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"  Failed to parse response: {e}")
        print(f"  Raw: {text[:200]}...")
        return 0
    
    now = datetime.utcnow().isoformat()
    new_count = 0
    
    for thought in data.get("thoughts", []):
        content = thought["content"]
        nid = node_id(content)
        confidence = thought.get("confidence", 0.5)
        
        # Check for duplicate content
        existing = conn.execute("SELECT id FROM thought_nodes WHERE id = ?", (nid,)).fetchone()
        if existing:
            continue
        
        conn.execute(
            "INSERT INTO thought_nodes VALUES (?,?,?,?,?,?,?,?,?,?)",
            (nid, content, "derived", now, confidence, None, 
             json.dumps({"cycle": cycle_num}), "system_generated", 0, None)
        )
        
        for edge in thought.get("edges", []):
            parent_id = edge["parent_id"]
            relation = edge["relation"]
            # Verify parent exists
            parent_exists = conn.execute("SELECT id FROM thought_nodes WHERE id = ?", (parent_id,)).fetchone()
            if parent_exists:
                conn.execute(
                    "INSERT OR IGNORE INTO derivation_edges VALUES (?,?,?,?,?)",
                    (parent_id, nid, relation, 0.8, f"Cycle {cycle_num} derivation")
                )
        
        new_count += 1
        print(f"  + [{nid}] {content[:80]}... (conf: {confidence})")
    
    conn.commit()
    return new_count


def main():
    parser = argparse.ArgumentParser(description="Clean Room Religion Experiment v1")
    parser.add_argument("--cycles", type=int, default=10, help="Number of think cycles")
    parser.add_argument("--db", default="data/experiment-clean-v1.db", help="Database path")
    args = parser.parse_args()
    
    # Resolve path relative to cashew root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cashew_root = os.path.dirname(script_dir)
    db_path = os.path.join(cashew_root, args.db)
    
    # Fresh DB
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Removed existing {db_path}")
    
    conn = init_db(db_path)
    seed_db(conn)
    
    # Init Anthropic client
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        # Try to get from OpenClaw's config
        import subprocess
        result = subprocess.run(["printenv", "ANTHROPIC_API_KEY"], capture_output=True, text=True)
        api_key = result.stdout.strip()
    
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        exit(1)
    
    client = anthropic.Anthropic(api_key=api_key)
    
    print(f"\n{'='*60}")
    print(f"CLEAN ROOM RELIGION EXPERIMENT v1")
    print(f"Cycles: {args.cycles}")
    print(f"DB: {db_path}")
    print(f"Context: NONE (no personal files, no workspace context)")
    print(f"{'='*60}\n")
    
    for cycle in range(1, args.cycles + 1):
        nodes_before = conn.execute("SELECT COUNT(*) FROM thought_nodes").fetchone()[0]
        edges_before = conn.execute("SELECT COUNT(*) FROM derivation_edges").fetchone()[0]
        
        print(f"\n--- Cycle {cycle}/{args.cycles} (nodes: {nodes_before}, edges: {edges_before}) ---")
        new_nodes = run_think_cycle(conn, client, cycle, args.cycles)
        
        if new_nodes == 0:
            print("  No new thoughts generated. Stopping.")
            break
        
        nodes_after = conn.execute("SELECT COUNT(*) FROM thought_nodes").fetchone()[0]
        edges_after = conn.execute("SELECT COUNT(*) FROM derivation_edges").fetchone()[0]
        print(f"  Summary: +{new_nodes} nodes, +{edges_after - edges_before} edges")
    
    # Final stats
    total_nodes = conn.execute("SELECT COUNT(*) FROM thought_nodes").fetchone()[0]
    total_edges = conn.execute("SELECT COUNT(*) FROM derivation_edges").fetchone()[0]
    derived = conn.execute("SELECT COUNT(*) FROM thought_nodes WHERE node_type='derived'").fetchone()[0]
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETE")
    print(f"Total: {total_nodes} nodes ({derived} derived), {total_edges} edges")
    print(f"DB: {db_path}")
    print(f"{'='*60}")
    
    conn.close()


if __name__ == "__main__":
    main()
