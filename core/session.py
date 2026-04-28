#!/usr/bin/env python3
"""
Cashew Session Integration Layer
Glue between thought graph and AI assistant session lifecycle
"""

import sqlite3
import json
import hashlib
import re
from typing import List, Dict, Optional, Tuple, Callable
from datetime import datetime, timezone
from dataclasses import dataclass
import logging
import sys
import argparse

from .config import config, get_token_budget, get_top_k, get_walk_depth, get_user_domain, get_ai_domain
from .retrieval import retrieve, retrieve_recursive_bfs, RetrievalResult
from .embeddings import embed_text, embed_nodes
from .stats import get_active_node_count

@dataclass
class SessionContext:
    """Result from start_session"""
    context_str: str
    nodes_used: List[str]
    token_estimate: int
    
    def to_dict(self) -> dict:
        return {
            "context_str": self.context_str,
            "nodes_used": self.nodes_used,
            "token_estimate": self.token_estimate
        }

@dataclass
class ExtractionResult:
    """Result from end_session"""
    new_nodes: List[str]
    new_edges: List[Tuple[str, str, str]]  # (parent_id, child_id, reasoning)
    updated_nodes: List[str]
    
    def to_dict(self) -> dict:
        return {
            "new_nodes": self.new_nodes,
            "new_edges": self.new_edges,
            "updated_nodes": self.updated_nodes
        }

@dataclass
class ThinkResult:
    """Result from think_cycle"""
    new_nodes: List[str]
    new_edges: List[Tuple[str, str, str]]
    cluster_topic: str
    
    def to_dict(self) -> dict:
        return {
            "new_nodes": self.new_nodes,
            "new_edges": self.new_edges,
            "cluster_topic": self.cluster_topic
        }

def _get_connection(db_path: str) -> sqlite3.Connection:
    """Get database connection"""
    return sqlite3.connect(db_path)

# ----- Schema contract ----------------------------------------------------
# Cashew owns these tables and the columns listed below. See DESIGN.md
# ("Schema ownership contract") for the full policy. Downstream consumers
# (e.g. hermes-cashew) may add their own tables and columns — the contract
# is that cashew will never rename or drop its own columns within a major
# version, and will only add columns in minor versions.
#
# SCHEMA_VERSION is stored in `PRAGMA user_version`. Bump it whenever a new
# migration is appended to _MIGRATIONS.
SCHEMA_VERSION = 1


def _apply_v1(cursor: sqlite3.Cursor) -> None:
    """Canonical from-scratch schema. Idempotent."""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS thought_nodes (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            node_type TEXT NOT NULL,
            domain TEXT,
            timestamp TEXT,
            access_count INTEGER DEFAULT 0,
            last_accessed TEXT,
            confidence REAL,
            source_file TEXT,
            decayed INTEGER DEFAULT 0,
            metadata TEXT DEFAULT '{}',
            last_updated TEXT,
            mood_state TEXT,
            permanent INTEGER DEFAULT 0,
            tags TEXT,
            referent_time TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS derivation_edges (
            parent_id TEXT,
            child_id TEXT,
            weight REAL,
            reasoning TEXT,
            confidence REAL,
            timestamp TEXT,
            PRIMARY KEY (parent_id, child_id),
            FOREIGN KEY (parent_id) REFERENCES thought_nodes(id),
            FOREIGN KEY (child_id) REFERENCES thought_nodes(id)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            node_id TEXT PRIMARY KEY,
            vector BLOB NOT NULL,
            model TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (node_id) REFERENCES thought_nodes(id)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS hotspots (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            status TEXT,
            domain TEXT,
            file_pointers TEXT,
            cluster_node_ids TEXT,
            tags TEXT,
            created TEXT,
            last_updated TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            tags TEXT
        )
    """)
    for stmt in (
        "CREATE INDEX IF NOT EXISTS idx_nodes_domain ON thought_nodes(domain)",
        "CREATE INDEX IF NOT EXISTS idx_nodes_timestamp ON thought_nodes(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_nodes_confidence ON thought_nodes(confidence)",
        "CREATE INDEX IF NOT EXISTS idx_nodes_referent_time ON thought_nodes(referent_time)",
        "CREATE INDEX IF NOT EXISTS idx_edges_parent ON derivation_edges(parent_id)",
        "CREATE INDEX IF NOT EXISTS idx_edges_child ON derivation_edges(child_id)",
    ):
        try:
            cursor.execute(stmt)
        except sqlite3.OperationalError:
            pass


# Legacy column patch-ups for databases created before the canonical schema
# was centralised. Additive only; cashew will not drop or rename columns
# within a major version.
_LEGACY_NODE_COLUMNS = (
    ("domain",        "ALTER TABLE thought_nodes ADD COLUMN domain TEXT"),
    ("access_count",  "ALTER TABLE thought_nodes ADD COLUMN access_count INTEGER DEFAULT 0"),
    ("last_accessed", "ALTER TABLE thought_nodes ADD COLUMN last_accessed TEXT"),
    ("confidence",    "ALTER TABLE thought_nodes ADD COLUMN confidence REAL"),
    ("source_file",   "ALTER TABLE thought_nodes ADD COLUMN source_file TEXT"),
    ("decayed",       "ALTER TABLE thought_nodes ADD COLUMN decayed INTEGER DEFAULT 0"),
    ("metadata",      "ALTER TABLE thought_nodes ADD COLUMN metadata TEXT DEFAULT '{}'"),
    ("last_updated",  "ALTER TABLE thought_nodes ADD COLUMN last_updated TEXT"),
    ("mood_state",    "ALTER TABLE thought_nodes ADD COLUMN mood_state TEXT"),
    ("permanent",     "ALTER TABLE thought_nodes ADD COLUMN permanent INTEGER DEFAULT 0"),
    ("tags",          "ALTER TABLE thought_nodes ADD COLUMN tags TEXT"),
    ("referent_time", "ALTER TABLE thought_nodes ADD COLUMN referent_time TEXT"),
)


def _ensure_schema(db_path: str):
    """Create or upgrade the cashew schema in place.

    Idempotent. Safe to call on an empty file, a legacy database missing
    some columns, or an already-current database. This is the blessed
    entry point for both CLI init and library consumers.
    """
    conn = _get_connection(db_path)
    cursor = conn.cursor()
    try:
        _apply_v1(cursor)

        cursor.execute("PRAGMA table_info(thought_nodes)")
        existing = {row[1] for row in cursor.fetchall()}
        for col, sql in _LEGACY_NODE_COLUMNS:
            if col not in existing:
                try:
                    cursor.execute(sql)
                except sqlite3.OperationalError:
                    pass

        cursor.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()
    finally:
        conn.close()


def get_schema_version(db_path: str) -> int:
    """Return the applied schema version (PRAGMA user_version). 0 means unmanaged."""
    conn = _get_connection(db_path)
    try:
        row = conn.execute("PRAGMA user_version").fetchone()
        return int(row[0]) if row else 0
    finally:
        conn.close()

def _normalize_referent_time(value: Optional[str]) -> Optional[str]:
    """Normalize a caller-supplied event time to UTC ISO8601.

    Rules:
    - None/empty → None (no event clock recorded).
    - Accepts ISO8601 with explicit tz offset ('Z' or +HH:MM). Converted to UTC.
    - Naive datetimes (no tz) are REJECTED — we refuse to guess local tz.
    - Any parse failure raises ValueError. Fail loud over silent drift.
    """
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
    else:
        raise ValueError(f"referent_time must be a string, got {type(value).__name__}")

    # Python's fromisoformat handles 'Z' only from 3.11+; normalize manually.
    candidate = s.replace('Z', '+00:00') if s.endswith('Z') else s
    try:
        dt = datetime.fromisoformat(candidate)
    except ValueError as e:
        raise ValueError(f"referent_time is not valid ISO8601: {value!r} ({e})")
    if dt.tzinfo is None:
        raise ValueError(
            f"referent_time is timezone-ambiguous (no offset): {value!r}. "
            "Supply explicit UTC offset or 'Z' — we will not guess local tz."
        )
    return dt.astimezone(timezone.utc).isoformat()


def _estimate_tokens(text: str) -> int:
    """Rough token estimation (conservative: ~3.5 chars per token)"""
    return len(text) // 3

def _get_tree_overview(db_path: str) -> str:
    """Get Layer 1 - Tree Overview: total nodes and edges"""
    try:
        conn = _get_connection(db_path)
        cursor = conn.cursor()
        
        # Total nodes (excluding decayed)
        total_nodes = get_active_node_count(cursor)

        # Total edges
        cursor.execute("SELECT COUNT(*) FROM derivation_edges")
        total_edges = cursor.fetchone()[0]
        
        # Count distinct node types for a quick shape overview
        cursor.execute("""
            SELECT node_type, COUNT(*) FROM thought_nodes
            WHERE decayed IS NULL OR decayed = 0
            GROUP BY node_type ORDER BY COUNT(*) DESC LIMIT 5
        """)
        type_counts = cursor.fetchall()
        
        conn.close()
        
        lines = [f"Graph: {total_nodes} nodes, {total_edges} edges."]
        
        if type_counts:
            type_str = ", ".join(f"{count} {ntype}" for ntype, count in type_counts[:3])
            lines.append(f"Types: {type_str}.")
        
        return " ".join(lines)
        
    except Exception as e:
        logging.error(f"Error getting tree overview: {e}")
        return f"Graph overview unavailable: {e}"

def _get_recent_activity(db_path: str, domain: str = None) -> str:
    """Get Layer 2 - Recent Activity: recently updated nodes/summaries"""
    try:
        conn = _get_connection(db_path)
        cursor = conn.cursor()
        
        # Get 3 most recently updated nodes
        if domain:
            cursor.execute("""
                SELECT substr(content, 1, 80), node_type, 
                       COALESCE(last_updated, timestamp) as update_time
                FROM thought_nodes 
                WHERE (decayed IS NULL OR decayed = 0) 
                
                AND COALESCE(last_updated, timestamp) IS NOT NULL
                AND domain = ?
                ORDER BY update_time DESC
                LIMIT 3
            """, (domain,))
        else:
            cursor.execute("""
                SELECT substr(content, 1, 80), node_type, 
                       COALESCE(last_updated, timestamp) as update_time
                FROM thought_nodes 
                WHERE (decayed IS NULL OR decayed = 0) 
                
                AND COALESCE(last_updated, timestamp) IS NOT NULL
                ORDER BY update_time DESC
                LIMIT 3
            """)
        
        recent_items = []
        for row in cursor.fetchall():
            content, node_type, update_time = row
            # Format timestamp to be readable (just show date if today, otherwise date)
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(update_time.replace('Z', '+00:00'))
                if dt.strftime('%Y-%m-%d') == datetime.now().strftime('%Y-%m-%d'):
                    time_str = "today"
                else:
                    time_str = dt.strftime('%m-%d')
                recent_items.append(f"[{node_type}] {content} ({time_str})")
            except:
                recent_items.append(f"[{node_type}] {content}")
        
        conn.close()
        
        if recent_items:
            return "\n".join([f"{i+1}. {item}" for i, item in enumerate(recent_items)])
        else:
            return "No recent activity found."
            
    except Exception as e:
        logging.error(f"Error getting recent activity: {e}")
        return f"Recent activity unavailable: {e}"

def _format_context_string(results: List[RetrievalResult]) -> str:
    """Format retrieval results into context string"""
    if not results:
        return ""
    
    lines = []
    
    for i, result in enumerate(results, 1):
        # Format: [TYPE] Content (Domain: domain_name)
        domain_str = f" (Domain: {result.domain})" if result.domain != "unknown" else ""
        lines.append(f"{i}. [{result.node_type.upper()}] {result.content}{domain_str}")
    
    return "\n".join(lines)

def _update_access_tracking(db_path: str, node_ids: List[str]):
    """Update last_accessed and access_count for retrieved nodes"""
    if not node_ids:
        return
    
    conn = _get_connection(db_path)
    cursor = conn.cursor()
    
    now = datetime.now(timezone.utc).isoformat()
    
    placeholders = ','.join(['?'] * len(node_ids))
    cursor.execute(f"""
        UPDATE thought_nodes 
        SET last_accessed = ?, 
            access_count = COALESCE(access_count, 0) + 1
        WHERE id IN ({placeholders})
    """, [now] + node_ids)
    
    conn.commit()
    conn.close()

def start_session(db_path: str, session_id: str, hints: Optional[List[str]] = None, domain: Optional[str] = None, tags: Optional[List[str]] = None, exclude_tags: Optional[List[str]] = None) -> SessionContext:
    """
    Start a session and inject relevant context with three layers:
    Layer 1 - Tree Overview (always)
    Layer 2 - Recent Activity (always) 
    Layer 3 - Hint-driven depth (when hints provided)
    
    Args:
        db_path: Path to SQLite database
        session_id: Unique session identifier
        hints: Optional topics/keywords for this session
        domain: Optional domain filter for context retrieval
        
    Returns:
        SessionContext with three-layer context string, node IDs, and token estimate
    """
    _ensure_schema(db_path)
    
    # Layer 1: Tree Overview (always shown)
    tree_overview = _get_tree_overview(db_path)
    
    # Layer 2: Recent Activity (always shown)
    recent_activity = _get_recent_activity(db_path, domain=domain)
    
    # Layer 3: Hint-driven depth (only if hints provided)
    hint_context = ""
    nodes_used = []
    hint_tokens = 0
    
    if hints:
        # Build query from hints
        query = " ".join(hints)
        
        # Retrieve relevant nodes using DFS hierarchical approach
        top_k = get_top_k()
        
        results = retrieve_recursive_bfs(db_path, query, top_k, domain=domain, tags=tags, exclude_tags=exclude_tags)
        
        if results:
            # Apply token budget constraint for hint-driven content
            token_budget = get_token_budget() // 2  # Reserve space for layers 1+2
            filtered_results = []
            current_tokens = 0
            
            for result in results:
                # Estimate tokens for this result (including formatting)
                result_text = f"[{result.node_type.upper()}] {result.content}"
                if result.domain != "unknown":
                    result_text += f" (Domain: {result.domain})"
                
                result_tokens = _estimate_tokens(result_text)
                
                if current_tokens + result_tokens > token_budget:
                    logging.info(f"Hint context token budget reached: {current_tokens}/{token_budget} tokens")
                    break
                
                filtered_results.append(result)
                current_tokens += result_tokens
            
            # Format hint-driven context
            hint_context = _format_context_string(filtered_results)
            nodes_used = [r.node_id for r in filtered_results]
            hint_tokens = current_tokens
            
            # Update access tracking
            _update_access_tracking(db_path, nodes_used)
    
    # Build three-layer context string
    context_parts = []
    context_parts.append("=== GRAPH OVERVIEW ===")
    context_parts.append(tree_overview)
    context_parts.append("")
    context_parts.append("=== RECENT ACTIVITY ===")
    context_parts.append(recent_activity)
    
    if hints and hint_context:
        context_parts.append("")
        context_parts.append("=== RELEVANT CONTEXT ===")
        context_parts.append(hint_context)
    
    context_parts.append("")
    context_parts.append("=== END CONTEXT ===")
    
    context_str = "\n".join(context_parts)
    
    # Estimate total tokens (overview + recent + hints)
    overview_tokens = _estimate_tokens(tree_overview)
    recent_tokens = _estimate_tokens(recent_activity) 
    total_tokens = overview_tokens + recent_tokens + hint_tokens + 50  # +50 for structure
    
    logging.info(f"Session {session_id} started with 3-layer context: "
                f"overview({overview_tokens}t) + recent({recent_tokens}t) + hints({hint_tokens}t) = {total_tokens}t")
    
    return SessionContext(
        context_str=context_str,
        nodes_used=nodes_used,
        token_estimate=total_tokens
    )

def _extract_with_heuristics(conversation_text: str) -> List[Dict[str, str]]:
    """Fallback extraction using heuristics when no model_fn provided"""
    extractions = []
    
    # Decision markers
    decision_patterns = [
        r'will\s+(\w+(?:\s+\w+){1,8})',
        r'decided\s+to\s+(\w+(?:\s+\w+){1,8})',
        r'plan\s+to\s+(\w+(?:\s+\w+){1,8})',
        r'going\s+to\s+(\w+(?:\s+\w+){1,8})'
    ]
    
    for pattern in decision_patterns:
        matches = re.finditer(pattern, conversation_text, re.IGNORECASE)
        for match in matches:
            extractions.append({
                "content": f"Decision: {match.group(1).strip()}",
                "node_type": "decision",
                "confidence": 0.6
            })
    
    # Belief markers
    belief_patterns = [
        r'(think|believe|seems\s+like|probably|likely)\s+(\w+(?:\s+\w+){1,10})',
        r'my\s+opinion\s+is\s+(\w+(?:\s+\w+){1,10})',
    ]
    
    for pattern in belief_patterns:
        matches = re.finditer(pattern, conversation_text, re.IGNORECASE)
        for match in matches:
            # Get the last capture group that contains the content
            groups = match.groups()
            content = groups[-1] if len(groups) > 1 and groups[-1] else groups[0]
            extractions.append({
                "content": f"Belief: {content.strip()}",
                "node_type": "belief",
                "confidence": 0.5
            })
    
    # Fact markers (simple heuristic)
    sentences = conversation_text.split('.')
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 20 and len(sentence) < 100:
            # Look for factual statements (contains proper nouns, numbers, etc.)
            if re.search(r'[A-Z][a-z]+|\\d+', sentence):
                extractions.append({
                    "content": f"Observation: {sentence}",
                    "node_type": "observation",
                    "confidence": 0.4
                })
    
    # Limit to most promising extractions
    return sorted(extractions, key=lambda x: x["confidence"], reverse=True)[:5]

def _set_node_tags(db_path: str, node_id: str, tags: list):
    """Store tags as comma-separated string on a node."""
    import json as _json
    conn = _get_connection(db_path)
    cursor = conn.cursor()
    tags_str = ",".join(t.strip().lower() for t in tags if t.strip())
    cursor.execute("UPDATE thought_nodes SET tags = ? WHERE id = ?", (tags_str, node_id))
    conn.commit()
    conn.close()


def _create_node(db_path: str, content: str, node_type: str,
                session_id: str, confidence: float = 0.7,
                domain: str = 'default',
                referent_time: Optional[str] = None) -> str:
    """Create a new thought node and return its ID.

    `referent_time` (optional) is the event clock — when the fact/event
    actually happened. Distinct from `timestamp` which is the ingestion
    clock. Must be a UTC ISO8601 string or None. Callers are responsible
    for normalizing to UTC; see `_normalize_referent_time` helper.
    """
    # Generate deterministic ID based on content
    node_id = hashlib.sha256(content.encode()).hexdigest()[:12]

    conn = _get_connection(db_path)
    cursor = conn.cursor()

    # Check if node already exists
    cursor.execute("SELECT id FROM thought_nodes WHERE id = ?", (node_id,))
    if cursor.fetchone():
        conn.close()
        return node_id  # Node already exists

    # Insert new node
    now = datetime.now(timezone.utc).isoformat()

    cursor.execute("""
        INSERT INTO thought_nodes
        (id, content, node_type, timestamp, confidence, source_file,
         last_accessed, access_count, domain, referent_time)
        VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
    """, (node_id, content, node_type, now, confidence, session_id, now,
          domain, referent_time))
    
    conn.commit()
    conn.close()
    
    return node_id

def _find_similar_nodes(db_path: str, node_id: str, threshold: float = 0.3) -> List[Tuple[str, float]]:
    """Find nodes similar to the given node using embeddings"""
    conn = _get_connection(db_path)
    cursor = conn.cursor()
    
    # Get the node content
    cursor.execute("SELECT content FROM thought_nodes WHERE id = ?", (node_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return []
    
    content = row[0]
    conn.close()
    
    # Use existing embedding search
    from .embeddings import search
    results = search(db_path, content, top_k=10)
    
    # Filter by threshold and exclude self
    similar = [(node_id_res, score) for node_id_res, score in results 
               if score >= threshold and node_id_res != node_id]
    
    return similar

def _create_edge(db_path: str, parent_id: str, child_id: str, reasoning: str):
    """Create an edge between two nodes"""
    conn = _get_connection(db_path)
    cursor = conn.cursor()
    
    # Check if edge already exists
    cursor.execute("""
        SELECT COUNT(*) FROM derivation_edges 
        WHERE parent_id = ? AND child_id = ?
    """, (parent_id, child_id))
    
    if cursor.fetchone()[0] > 0:
        conn.close()
        return  # Edge already exists
    
    # Insert new edge
    cursor.execute("""
        INSERT INTO derivation_edges (parent_id, child_id, weight, reasoning)
        VALUES (?, ?, 0.8, ?)
    """, (parent_id, child_id, f"extracted_from - {reasoning}"))
    
    conn.commit()
    conn.close()

def _llm_infer_referent_time(content: str, model_fn: Callable[[str], str]) -> Optional[str]:
    """Best-effort LLM inference of event time from prose.

    Opt-in only (see --infer-referent-time). LLM-inferred times are
    untrustworthy — if the prose has no clear temporal anchor the model
    MUST return NONE. We also reject any output that doesn't parse as
    tz-aware ISO8601.
    """
    prompt = (
        "Extract the event time (when the fact/event actually happened) from "
        "this statement as strict ISO8601 with a UTC offset (ending in 'Z' "
        "or '+00:00'). If no clear temporal anchor is present, reply with "
        "exactly the word NONE. Do not guess. Do not add prose.\n\n"
        f"Statement: {content}\n\nAnswer:"
    )
    try:
        resp = (model_fn(prompt) or "").strip()
    except Exception as e:
        logging.debug(f"referent_time inference failed: {e}")
        return None
    if not resp or resp.upper().startswith("NONE"):
        return None
    # Keep only the first whitespace-delimited token
    token = resp.split()[0].strip().rstrip('.,;')
    try:
        _normalize_referent_time(token)
    except ValueError:
        return None
    return token


def end_session(db_path: str, session_id: str, conversation_text: str,
               model_fn: Optional[Callable[[str], str]] = None,
               default_referent_time: Optional[str] = None,
               infer_referent_time: bool = False) -> ExtractionResult:
    """
    End a session and extract new knowledge from conversation
    
    Args:
        db_path: Path to SQLite database  
        session_id: Session identifier
        conversation_text: Full conversation text to extract from
        model_fn: Optional function to call LLM for structured extraction
        
    Returns:
        ExtractionResult with new nodes, edges, and updated nodes
    """
    _ensure_schema(db_path)

    # Normalize caller-supplied default referent_time once up front — fails
    # loud on ambiguous/naive input rather than silently assuming local tz.
    default_referent_time = _normalize_referent_time(default_referent_time)

    if not conversation_text or len(conversation_text.strip()) < 20:
        logging.info(f"Session {session_id} ended with minimal content, skipping extraction")
        return ExtractionResult(new_nodes=[], new_edges=[], updated_nodes=[])
    
    extractions = []
    
    if model_fn:
        # Use LLM for structured extraction
        extraction_prompt = f"""You are extracting knowledge from a conversation into a personal thought graph. Extract ONLY genuinely new, specific, substantive knowledge — not summaries or meta-comments.

For each item, classify as:
{config.node_type_prompt_fragment}

Each content field must be a specific, standalone statement that makes sense without the conversation context.

For each item, assign:
- "domain": who this thought belongs to. Use "{get_user_domain()}" for the human's knowledge, experiences, decisions, beliefs, facts about their life, work, relationships. Use "{get_ai_domain()}" for the AI assistant's operational knowledge, engineering decisions about the system, behavioral rules, or self-reflective observations about the AI's own processes.
- "tags": short descriptive labels (e.g. "career", "family", "engineering", "philosophy", "health", "finance", "project:cashew"). Lowercase, specific, reusable. Multiple tags encouraged.

BAD: "They discussed embeddings" (meta-comment)
BAD: "The conversation covered several topics" (summary)
GOOD: "Local embedding models (all-MiniLM-L6-v2) are sufficient for graphs under 100K nodes — brute force cosine similarity stays under 50ms"
GOOD: "Extraction should be triggered by context fullness monitoring, not left to manual memory — don't let compaction happen to you"

Respond with ONLY a JSON array. No markdown, no explanation, no code fences.

[{{"content": "specific knowledge here", "type": "{config.node_type_pipe_list}", "confidence": 0.7, "domain": "{get_user_domain()}", "tags": ["engineering", "embeddings"]}}]

Conversation to extract from:
{conversation_text}
"""
        
        try:
            response = model_fn(extraction_prompt)
            # Try to parse JSON response — handle markdown code fences
            import json as _json
            cleaned = response.strip()
            # Strip markdown code fences if present
            if cleaned.startswith('```'):
                lines = cleaned.split('\n')
                lines = [l for l in lines if not l.strip().startswith('```')]
                cleaned = '\n'.join(lines).strip()
            # Find JSON array in response
            start = cleaned.find('[')
            end = cleaned.rfind(']')
            if start != -1 and end != -1:
                json_str = cleaned[start:end+1]
                extractions = _json.loads(json_str)
                logging.info(f"Extracted {len(extractions)} items via LLM")
            else:
                logging.warning("No JSON array found in LLM response, using heuristics")
                extractions = _extract_with_heuristics(conversation_text)
        
        except Exception as e:
            logging.warning(f"LLM extraction failed: {e}, using heuristics")
            extractions = _extract_with_heuristics(conversation_text)
    
    else:
        # Use heuristic extraction
        extractions = _extract_with_heuristics(conversation_text)
    
    # Create nodes and edges
    new_nodes = []
    new_edges = []
    updated_nodes = []
    
    for extraction in extractions:
        if not extraction.get("content"):
            continue
        
        content = extraction["content"]
        node_type = extraction.get("type", "observation")
        node_type = config.validate_node_type(node_type)
        confidence = extraction.get("confidence", 0.5)
        tags = extraction.get("tags", [])
        domain = extraction.get("domain", get_user_domain())
        # Validate domain — only allow configured domains
        valid_domains = {get_user_domain(), get_ai_domain()}
        if domain not in valid_domains:
            domain = config.get_user_domain()
        
        # Resolve event clock: explicit > default > (optional) LLM inference > None.
        # LLM-inferred times are untrustworthy, hence opt-in via infer_referent_time.
        ref_time_raw = extraction.get("referent_time") or default_referent_time
        if ref_time_raw is None and infer_referent_time and model_fn is not None:
            ref_time_raw = _llm_infer_referent_time(content, model_fn)
        try:
            node_referent_time = _normalize_referent_time(ref_time_raw)
        except ValueError as e:
            logging.warning(
                f"Dropping unparseable referent_time {ref_time_raw!r}: {e}"
            )
            node_referent_time = None

        # Create the new node
        node_id = _create_node(db_path, content, node_type, session_id, confidence,
                              domain=domain, referent_time=node_referent_time)
        
        # Store tags if provided
        if tags and isinstance(tags, list):
            _set_node_tags(db_path, node_id, tags)
        
        new_nodes.append(node_id)
    
    # Embed new nodes
    embed_nodes(db_path)
    
    # Find similar existing nodes and create edges
    for node_id in new_nodes:
        similar_nodes = _find_similar_nodes(db_path, node_id)
        
        for similar_id, similarity in similar_nodes[:3]:  # Link to top 3 similar
            reasoning = f"Session extraction similarity: {similarity:.3f}"
            _create_edge(db_path, similar_id, node_id, reasoning)
            new_edges.append((similar_id, node_id, reasoning))
    
    logging.info(f"Session {session_id} ended: extracted {len(new_nodes)} nodes, "
                f"created {len(new_edges)} edges")
    
    return ExtractionResult(
        new_nodes=new_nodes,
        new_edges=new_edges,
        updated_nodes=updated_nodes
    )

def _find_cluster_for_thinking(db_path: str, focus_domain: Optional[str] = None) -> List[str]:
    """Find a coherent cluster of nodes for think cycle with random walk diversity"""
    conn = _get_connection(db_path)
    cursor = conn.cursor()
    
    if focus_domain:
        # Look for nodes in specific domain
        cursor.execute("""
            SELECT id, last_accessed, access_count, node_type, COALESCE(domain, 'unknown') as domain
            FROM thought_nodes 
            WHERE (decayed IS NULL OR decayed = 0)
            AND COALESCE(domain, 'unknown') = ?
            ORDER BY COALESCE(access_count, 0) ASC,
                     CASE WHEN COALESCE(access_count, 0) = 0
                          THEN RANDOM() ELSE last_accessed END ASC
            LIMIT 20
        """, (focus_domain,))
        high_activation_candidates = cursor.fetchall()
        
        # Also get some random walk candidates from other domains
        cursor.execute("""
            SELECT id, last_accessed, access_count, node_type, COALESCE(domain, 'unknown') as domain
            FROM thought_nodes
            WHERE (decayed IS NULL OR decayed = 0)
            AND COALESCE(domain, 'unknown') != ?
            AND node_type != 'seed'
            ORDER BY RANDOM()
            LIMIT 10
        """, (focus_domain,))
        random_walk_candidates = cursor.fetchall()
    else:
        # Find high-activation candidates (least recently accessed)
        cursor.execute("""
            SELECT id, last_accessed, access_count, node_type, COALESCE(domain, 'unknown') as domain
            FROM thought_nodes
            WHERE (decayed IS NULL OR decayed = 0)
            AND node_type != 'seed'
            ORDER BY COALESCE(access_count, 0) ASC,
                     CASE WHEN COALESCE(access_count, 0) = 0
                          THEN RANDOM() ELSE last_accessed END ASC
            LIMIT 20
        """)
        high_activation_candidates = cursor.fetchall()
        
        # Find underrepresented domains/types for random walk
        cursor.execute("""
            SELECT node_type, COALESCE(domain, 'unknown') as domain, COUNT(*) as cnt
            FROM thought_nodes 
            WHERE (decayed IS NULL OR decayed = 0)
            AND timestamp > datetime('now', '-30 days')
            AND (source_file LIKE '%system_generated%'
                 OR source_file LIKE 'extractor:%')
            GROUP BY node_type, domain
            ORDER BY cnt ASC
        """)
        underrep_stats = cursor.fetchall()
        
        # Get random walk candidates from underrepresented areas
        random_walk_candidates = []
        for node_type, domain, _ in underrep_stats[:3]:  # Top 3 underrepresented
            cursor.execute("""
                SELECT id, last_accessed, access_count, node_type, COALESCE(domain, 'unknown') as domain
                FROM thought_nodes
                WHERE (decayed IS NULL OR decayed = 0)
                AND node_type = ? AND COALESCE(domain, 'unknown') = ?
                AND source_file NOT LIKE '%system_generated%'  -- Prefer human-authored content (extractor:* counts as human-authored)
                ORDER BY RANDOM()
                LIMIT 2
            """, (node_type, domain))
            random_walk_candidates.extend(cursor.fetchall())
    
    conn.close()
    
    if not high_activation_candidates:
        return []
    
    # Combine high-activation and random walk candidates
    # Target: ~70% high-activation, ~30% random walk for diversity
    import random
    total_size = min(config.think_cycle_nodes, 8)  # Cap at 8 for manageability
    
    if random_walk_candidates:
        high_activation_size = max(1, int(total_size * 0.7))
        random_walk_size = total_size - high_activation_size
        
        selected_high = random.sample(
            high_activation_candidates, 
            min(high_activation_size, len(high_activation_candidates))
        )
        selected_random = random.sample(
            random_walk_candidates,
            min(random_walk_size, len(random_walk_candidates))
        )
    else:
        # No random walk candidates (e.g., fresh migration with no system_generated nodes)
        # Fill entirely from high-activation pool
        selected_high = random.sample(
            high_activation_candidates, 
            min(total_size, len(high_activation_candidates))
        )
        selected_random = []
    
    # Combine and extract node IDs
    all_selected = selected_high + selected_random
    return [node_id for node_id, _, _, _, _ in all_selected]

def think_cycle(db_path: str, model_fn: Callable[[str], str], 
               focus_domain: Optional[str] = None) -> ThinkResult:
    """
    Run a think cycle on the graph
    
    Args:
        db_path: Path to SQLite database
        model_fn: Function to call LLM for thinking
        focus_domain: Optional domain to focus thinking on
        
    Returns:
        ThinkResult with new nodes, edges, and cluster topic
    """
    _ensure_schema(db_path)
    
    # Find a cluster to think about
    cluster_nodes = _find_cluster_for_thinking(db_path, focus_domain)
    
    if not cluster_nodes:
        logging.info("No nodes found for think cycle")
        return ThinkResult(
            new_nodes=[],
            new_edges=[],
            cluster_topic="No cluster found"
        )
    
    # Load node details
    conn = _get_connection(db_path)
    cursor = conn.cursor()
    
    placeholders = ','.join(['?'] * len(cluster_nodes))
    cursor.execute(f"""
        SELECT id, content, node_type, COALESCE(metadata, '{{}}') as metadata
        FROM thought_nodes 
        WHERE id IN ({placeholders})
    """, cluster_nodes)
    
    nodes_info = []
    for row in cursor.fetchall():
        node_id, content, node_type, metadata = row
        try:
            metadata_dict = json.loads(metadata) if metadata else {}
        except:
            metadata_dict = {}
        
        nodes_info.append({
            "id": node_id,
            "content": content,
            "type": node_type,
            "domain": metadata_dict.get("domain", "unknown")
        })
    
    conn.close()
    
    # Build thinking prompt
    cluster_description = "\\n".join([
        f"[{node['type'].upper()}] {node['content']} (Domain: {node['domain']})"
        for node in nodes_info
    ])
    
    cluster_topic = f"cluster of {len(nodes_info)} nodes"
    if focus_domain:
        cluster_topic += f" in {focus_domain} domain"
    
    thinking_prompt = f"""You are analyzing a cluster of connected thoughts from a personal knowledge graph. Your job is to find non-obvious connections and generate genuine insights.

THOUGHTS:
{cluster_description}

Look for:
- Patterns that connect these ideas in ways the author might not have noticed
- Tensions or contradictions worth naming
- A higher-level principle that unifies seemingly unrelated thoughts

Respond with ONLY a JSON array (no markdown, no explanation). Each insight must be a specific, substantive statement — not a meta-comment about the thoughts.

BAD: "These thoughts share common themes about growth"
GOOD: "The silence pattern at work and the silence during faith transition are the same defense mechanism — withdrawal when the gap between performance and identity becomes visible"

JSON format:
[{{"content": "your specific insight here", "type": "insight", "confidence": 0.7}}]
"""
    
    try:
        response = model_fn(thinking_prompt)
        
        # Parse response
        new_thoughts = []
        # Strip markdown code fences if present (LLMs often wrap JSON in ```json ... ```)
        cleaned = response.strip()
        if cleaned.startswith('```'):
            # Remove opening fence (```json or ```)
            first_newline = cleaned.index('\n') if '\n' in cleaned else len(cleaned)
            cleaned = cleaned[first_newline + 1:]
            # Remove closing fence
            if cleaned.rstrip().endswith('```'):
                cleaned = cleaned.rstrip()[:-3].rstrip()
        
        if cleaned.startswith('['):
            try:
                import json
                new_thoughts = json.loads(cleaned)
            except:
                pass
        
        if not new_thoughts:
            # No valid thoughts generated — don't insert placeholder noise
            logging.info("Think cycle produced no parseable insights, skipping insertion")
            return ThinkResult(
                new_nodes=[],
                new_edges=[],
                cluster_topic=cluster_topic
            )
        
        # Filter by confidence threshold — not all thoughts are insights
        THINK_CONFIDENCE_THRESHOLD = 0.75
        filtered = [t for t in new_thoughts if t.get("confidence", 0.5) >= THINK_CONFIDENCE_THRESHOLD]
        skipped = len(new_thoughts) - len(filtered)
        if skipped > 0:
            logging.info(f"Think cycle: filtered out {skipped}/{len(new_thoughts)} thoughts below confidence {THINK_CONFIDENCE_THRESHOLD}")
        
        if not filtered:
            logging.info("Think cycle: all thoughts below confidence threshold, nothing to insert")
            return ThinkResult(
                new_nodes=[],
                new_edges=[],
                cluster_topic=cluster_topic
            )
        
        # Diversity check: filter out thoughts too similar to existing nodes
        DIVERSITY_THRESHOLD = 0.85
        diversity_filtered = []
        
        for thought in filtered:
            content = thought.get("content", "")
            if not content:
                continue
            
            # Check similarity to existing nodes using embeddings search
            try:
                from .embeddings import search
                
                # Search for similar existing nodes using the proposed content as query
                similar_nodes = search(db_path, content, top_k=5)
                
                # Check if any existing node is too similar
                max_similarity = 0.0
                if similar_nodes:
                    max_similarity = max(score for _, score in similar_nodes)
                
                if max_similarity > DIVERSITY_THRESHOLD:
                    logging.info(f"Think cycle: skipping thought due to high similarity ({max_similarity:.3f}) to existing node")
                    continue
                
                diversity_filtered.append(thought)
                
            except Exception as e:
                # If similarity check fails, be permissive and include the thought
                logging.warning(f"Think cycle: diversity check failed for thought, including anyway: {e}")
                diversity_filtered.append(thought)
        
        diversity_skipped = len(filtered) - len(diversity_filtered)
        if diversity_skipped > 0:
            logging.info(f"Think cycle: filtered out {diversity_skipped}/{len(filtered)} thoughts due to similarity")
        
        if not diversity_filtered:
            logging.info("Think cycle: all thoughts too similar to existing nodes")
            return ThinkResult(
                new_nodes=[],
                new_edges=[],
                cluster_topic=cluster_topic
            )
        
        filtered = diversity_filtered
        
        # Create new nodes (only high-confidence thoughts)
        new_nodes = []
        new_edges = []
        
        for thought in filtered:
            if not thought.get("content"):
                continue
            
            # Create derived node
            derived_id = _create_node(
                db_path, 
                thought["content"],
                thought.get("type", "insight"),
                "system_generated",  # Use consistent tag for dashboard styling
                thought.get("confidence", 0.7)
            )
            new_nodes.append(derived_id)
            
            # Connect to source cluster nodes
            reasoning = "Think cycle derivation"
            for node_info in nodes_info:
                _create_edge(db_path, node_info["id"], derived_id, reasoning)
                new_edges.append((node_info["id"], derived_id, reasoning))
        
        # Embed new nodes
        embed_nodes(db_path)
        
        logging.info(f"Think cycle completed: {len(new_nodes)} new insights "
                    f"from {cluster_topic}")
        
        return ThinkResult(
            new_nodes=new_nodes,
            new_edges=new_edges,
            cluster_topic=cluster_topic
        )
    
    except Exception as e:
        logging.error(f"Think cycle failed: {e}")
        return ThinkResult(
            new_nodes=[],
            new_edges=[],
            cluster_topic=f"Failed: {cluster_topic}"
        )


def tension_detection(db_path: str, model_fn: Callable[[str], str],
                     focus_domain: Optional[str] = None) -> ThinkResult:
    """
    Find and articulate tensions/contradictions in the thought graph.
    Uses embeddings to find semantically close but potentially contradictory nodes.
    """
    _ensure_schema(db_path)
    
    conn = _get_connection(db_path)
    cursor = conn.cursor()
    
    # Get nodes with embeddings
    domain_filter = ""
    params = []
    if focus_domain:
        domain_filter = "AND json_extract(tn.metadata, '$.domain') = ?"
        params.append(focus_domain)
    
    cursor.execute(f"""
        SELECT tn.id, tn.content, tn.node_type, 
               COALESCE(json_extract(tn.metadata, '$.domain'), 'unknown') as domain,
               e.vector
        FROM thought_nodes tn
        JOIN embeddings e ON tn.id = e.node_id
        WHERE (tn.decayed IS NULL OR tn.decayed = 0)
        {domain_filter}
    """, params)
    
    rows = cursor.fetchall()
    conn.close()
    
    if len(rows) < 4:
        return ThinkResult(new_nodes=[], new_edges=[], cluster_topic="Not enough nodes for tension detection")
    
    # Find pairs with moderate similarity (0.4-0.8 range = same topic, different stance)
    import numpy as np
    import struct
    
    nodes = []
    embeddings = []
    for row in rows:
        nodes.append({"id": row[0], "content": row[1], "type": row[2], "domain": row[3]})
        vec_blob = row[4]
        if isinstance(vec_blob, bytes):
            emb = list(struct.unpack(f'{len(vec_blob)//4}f', vec_blob))
        elif isinstance(vec_blob, str):
            emb = json.loads(vec_blob)
        else:
            emb = list(vec_blob)
        embeddings.append(np.array(emb, dtype=np.float32))
    
    embeddings = np.array(embeddings)
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms
    
    # Compute similarity matrix
    sim_matrix = embeddings @ embeddings.T
    
    # Find tension candidates: moderate similarity (same topic area, potentially different views)
    # Exclude near-duplicates (>0.85) and unrelated (<0.30)
    tension_pairs = []
    n = len(nodes)
    for i in range(n):
        for j in range(i+1, n):
            sim = float(sim_matrix[i][j])
            if np.isnan(sim) or np.isinf(sim):
                continue
            if 0.30 <= sim <= 0.70:
                # Prefer different node types (more likely to be genuine tensions)
                type_bonus = 0.05 if nodes[i]["type"] != nodes[j]["type"] else 0
                # Prefer different domains (cross-domain tensions are more interesting)
                domain_bonus = 0.05 if nodes[i]["domain"] != nodes[j]["domain"] else 0
                score = abs(sim - 0.50) * -1 + type_bonus + domain_bonus  # Lower = better
                tension_pairs.append((i, j, sim, score))
    
    # Sort by score (lower = better tension candidate)
    tension_pairs.sort(key=lambda x: x[3])
    
    # Take top 10 candidate pairs
    candidates = [(i, j, sim) for i, j, sim, _ in tension_pairs[:10]]
    
    if not candidates:
        return ThinkResult(new_nodes=[], new_edges=[], cluster_topic="No tension candidates found")
    
    # Build prompt with candidate pairs
    pairs_text = ""
    for idx, (i, j, sim) in enumerate(candidates):
        pairs_text += f"\nPair {idx+1} (similarity: {sim:.2f}):\n"
        pairs_text += f"  A [{nodes[i]['type']}]: {nodes[i]['content']}\n"
        pairs_text += f"  B [{nodes[j]['type']}]: {nodes[j]['content']}\n"
    
    tension_prompt = f"""You are a tension detector for a personal knowledge graph. You're looking at pairs of thoughts from the same person that might be in tension with each other — contradictions, unresolved conflicts, competing values, or beliefs that pull in opposite directions.

CANDIDATE PAIRS:
{pairs_text}

For each pair that has a GENUINE tension (not all will), articulate what the tension is. Skip pairs that are simply different topics or complementary ideas.

A tension is:
- A contradiction the person hasn't resolved
- Two values or goals that compete for the same resources (time, energy, identity)
- A stated belief vs observed behavior pattern
- An aspiration that conflicts with a comfort zone

Respond with ONLY a JSON array. Each item:
{{"pair": <pair_number>, "tension": "specific articulation of the tension", "type": "contradiction|competing_values|belief_behavior_gap|aspiration_comfort", "confidence": 0.7, "resolution_hint": "optional brief suggestion"}}

If no genuine tensions exist, return an empty array [].
Only include tensions you're confident about (>= 0.75).
"""
    
    try:
        response = model_fn(tension_prompt)
        
        # Parse response - strip markdown fences if present
        clean = response.strip()
        if clean.startswith("```"):
            clean = re.sub(r'^```\w*\n?', '', clean)
            clean = re.sub(r'\n?```$', '', clean)
        
        tensions = json.loads(clean) if clean.strip().startswith('[') else []
        
        # Filter by confidence
        tensions = [t for t in tensions if t.get("confidence", 0) >= 0.75]
        
        if not tensions:
            return ThinkResult(new_nodes=[], new_edges=[], cluster_topic="tension detection (no high-confidence tensions found)")
        
        # Create tension nodes
        new_nodes = []
        new_edges = []
        
        for t in tensions:
            pair_idx = t.get("pair", 1) - 1
            if pair_idx < 0 or pair_idx >= len(candidates):
                continue
                
            i, j, sim = candidates[pair_idx]
            
            content = f"TENSION ({t.get('type', 'unknown')}): {t['tension']}"
            if t.get("resolution_hint"):
                content += f" [Resolution hint: {t['resolution_hint']}]"
            
            node_id = _create_node(
                db_path, content, "tension",
                "system_generated", t.get("confidence", 0.75)
            )
            new_nodes.append(node_id)
            
            # Connect to both source nodes
            _create_edge(db_path, nodes[i]["id"], node_id, "Tension detection")
            _create_edge(db_path, nodes[j]["id"], node_id, "Tension detection")
            new_edges.append((nodes[i]["id"], node_id, "Tension detection"))
            new_edges.append((nodes[j]["id"], node_id, "Tension detection"))
        
        # Embed new nodes
        embed_nodes(db_path)
        
        return ThinkResult(
            new_nodes=new_nodes,
            new_edges=new_edges,
            cluster_topic=f"tension detection ({len(new_nodes)} tensions found)"
        )
    
    except Exception as e:
        logging.error(f"Tension detection failed: {e}")
        return ThinkResult(
            new_nodes=[],
            new_edges=[],
            cluster_topic=f"Failed: tension detection ({e})"
        )


def main():
    """CLI interface for session management"""
    parser = argparse.ArgumentParser(description="Cashew Session Integration")
    parser.add_argument("command", choices=["start", "end", "think"], help="Command to run")
    parser.add_argument("--db", default="./data/test_session.db", 
                       help="Database path")
    parser.add_argument("--session-id", default="test_session", help="Session ID")
    parser.add_argument("--hints", nargs="*", help="Session hints (for start command)")
    parser.add_argument("--conversation", help="Conversation text (for end command)")
    parser.add_argument("--domain", help="Focus domain (for think command)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    # Mock model function for testing
    def mock_model_fn(prompt: str) -> str:
        """Simple mock that returns basic extraction"""
        if "conversation" in prompt.lower():
            return '[{"content": "Mock extracted insight", "type": "insight", "confidence": 0.8}]'
        else:
            return '[{"content": "Mock think cycle result", "type": "insight", "confidence": 0.7}]'
    
    if args.command == "start":
        result = start_session(args.db, args.session_id, args.hints)
        print(f"Session started: {args.session_id}")
        print(f"Context nodes: {len(result.nodes_used)}")
        print(f"Token estimate: {result.token_estimate}")
        if result.context_str:
            print("\\nContext:")
            print(result.context_str)
    
    elif args.command == "end":
        if not args.conversation:
            print("Error: --conversation required for end command")
            return 1
        
        result = end_session(args.db, args.session_id, args.conversation, mock_model_fn)
        print(f"Session ended: {args.session_id}")
        print(f"New nodes: {len(result.new_nodes)}")
        print(f"New edges: {len(result.new_edges)}")
        print(f"Nodes: {result.new_nodes}")
    
    elif args.command == "think":
        result = think_cycle(args.db, mock_model_fn, args.domain)
        print(f"Think cycle completed")
        print(f"Cluster: {result.cluster_topic}")
        print(f"New nodes: {len(result.new_nodes)}")
        print(f"New edges: {len(result.new_edges)}")
        print(f"Nodes: {result.new_nodes}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

def _get_saturated_themes(db_path: str, days: int = 14, min_count: int = 3) -> List[str]:
    """Get themes that are saturated (frequently generated) in recent think cycles.
    
    Returns content snippets from recent system_generated nodes to help
    the think cycle avoid producing redundant insights.
    
    Args:
        db_path: Path to SQLite database
        days: Look back window in days
        min_count: Minimum occurrences to consider saturated
        
    Returns:
        List of content strings from frequently generated themes
    """
    conn = _get_connection(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT content FROM thought_nodes
        WHERE source_file = 'system_generated'
        AND timestamp > datetime('now', ? || ' days')
        AND (decayed IS NULL OR decayed = 0)
        ORDER BY timestamp DESC
    """, (f"-{days}",))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [row[0] for row in rows]
