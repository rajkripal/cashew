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

from .config import config, get_token_budget, get_top_k, get_walk_depth
from .retrieval import retrieve, RetrievalResult
from .embeddings import embed_text, embed_nodes

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

def _ensure_schema(db_path: str):
    """Ensure required tables and columns exist"""
    conn = _get_connection(db_path)
    cursor = conn.cursor()
    
    # Check if last_accessed column exists, add if missing
    cursor.execute("PRAGMA table_info(thought_nodes)")
    columns = [row[1] for row in cursor.fetchall()]
    
    if 'last_accessed' not in columns:
        cursor.execute("ALTER TABLE thought_nodes ADD COLUMN last_accessed TEXT")
    
    if 'access_count' not in columns:
        cursor.execute("ALTER TABLE thought_nodes ADD COLUMN access_count INTEGER DEFAULT 0")
    
    conn.commit()
    conn.close()

def _estimate_tokens(text: str) -> int:
    """Rough token estimation (conservative: ~3.5 chars per token)"""
    return len(text) // 3

def _format_context_string(results: List[RetrievalResult]) -> str:
    """Format retrieval results into context string"""
    if not results:
        return ""
    
    lines = ["=== RELEVANT CONTEXT ==="]
    
    for i, result in enumerate(results, 1):
        # Format: [TYPE] Content (Domain: domain_name)
        domain_str = f" (Domain: {result.domain})" if result.domain != "unknown" else ""
        lines.append(f"{i}. [{result.node_type.upper()}] {result.content}{domain_str}")
    
    lines.append("=== END CONTEXT ===")
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

def start_session(db_path: str, session_id: str, hints: Optional[List[str]] = None, domain: Optional[str] = None) -> SessionContext:
    """
    Start a session and inject relevant context
    
    Args:
        db_path: Path to SQLite database
        session_id: Unique session identifier
        hints: Optional topics/keywords for this session
        domain: Optional domain filter for context retrieval
        
    Returns:
        SessionContext with context string, node IDs, and token estimate
    """
    _ensure_schema(db_path)
    
    # Build query from hints or use default
    query = " ".join(hints) if hints else "recent relevant context"
    
    # Retrieve relevant nodes using hybrid approach
    top_k = get_top_k()
    walk_depth = get_walk_depth()
    
    results = retrieve(db_path, query, top_k, walk_depth, domain)
    
    if not results:
        logging.info(f"No relevant context found for session {session_id}")
        return SessionContext(
            context_str="",
            nodes_used=[],
            token_estimate=0
        )
    
    # Apply token budget constraint
    token_budget = get_token_budget()
    filtered_results = []
    current_tokens = 0
    
    for result in results:
        # Estimate tokens for this result (including formatting)
        result_text = f"[{result.node_type.upper()}] {result.content}"
        if result.domain != "unknown":
            result_text += f" (Domain: {result.domain})"
        
        result_tokens = _estimate_tokens(result_text)
        
        if current_tokens + result_tokens > token_budget:
            logging.info(f"Token budget reached: {current_tokens}/{token_budget} tokens")
            break
        
        filtered_results.append(result)
        current_tokens += result_tokens
    
    # Format context string
    context_str = _format_context_string(filtered_results)
    nodes_used = [r.node_id for r in filtered_results]
    
    # Update access tracking
    _update_access_tracking(db_path, nodes_used)
    
    logging.info(f"Session {session_id} started with {len(nodes_used)} context nodes "
                f"({current_tokens} tokens)")
    
    return SessionContext(
        context_str=context_str,
        nodes_used=nodes_used,
        token_estimate=current_tokens
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

def _create_node(db_path: str, content: str, node_type: str, 
                session_id: str, confidence: float = 0.7) -> str:
    """Create a new thought node and return its ID"""
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
         last_accessed, access_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, 0)
    """, (node_id, content, node_type, now, confidence, f"session_{session_id}", now))
    
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
        INSERT INTO derivation_edges (parent_id, child_id, relation, weight, reasoning)
        VALUES (?, ?, 'extracted_from', 0.8, ?)
    """, (parent_id, child_id, reasoning))
    
    conn.commit()
    conn.close()

def end_session(db_path: str, session_id: str, conversation_text: str, 
               model_fn: Optional[Callable[[str], str]] = None) -> ExtractionResult:
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
    
    if not conversation_text or len(conversation_text.strip()) < 20:
        logging.info(f"Session {session_id} ended with minimal content, skipping extraction")
        return ExtractionResult(new_nodes=[], new_edges=[], updated_nodes=[])
    
    extractions = []
    
    if model_fn:
        # Use LLM for structured extraction
        extraction_prompt = f"""
Analyze this conversation and extract key information. Identify:

1. NEW FACTUAL INFORMATION (who, what, when, where)
2. DECISIONS OR COMMITMENTS made  
3. BELIEFS OR OPINIONS expressed
4. INSIGHTS OR PATTERNS discovered
5. EMOTIONAL CONTEXT or confidence levels

For each item, provide:
- content: The actual information
- type: one of 'observation', 'belief', 'decision', 'insight', 'fact'
- confidence: 0.0-1.0 confidence score

Format as JSON array:
[{{"content": "...", "type": "...", "confidence": 0.8}}]

Conversation:
{conversation_text}
"""
        
        try:
            response = model_fn(extraction_prompt)
            # Try to parse JSON response
            if response.strip().startswith('['):
                import json
                extractions = json.loads(response)
            else:
                # Fallback if LLM didn't return JSON
                logging.warning("LLM response was not JSON, using heuristics")
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
        confidence = extraction.get("confidence", 0.5)
        
        # Create the new node
        node_id = _create_node(db_path, content, node_type, session_id, confidence)
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
    """Find a coherent cluster of nodes for think cycle"""
    conn = _get_connection(db_path)
    cursor = conn.cursor()
    
    if focus_domain:
        # Look for nodes in specific domain
        cursor.execute("""
            SELECT id, last_accessed, access_count
            FROM thought_nodes 
            WHERE (decayed IS NULL OR decayed = 0)
            AND json_extract(metadata, '$.domain') = ?
            ORDER BY COALESCE(access_count, 0) ASC, last_accessed ASC
            LIMIT 10
        """, (focus_domain,))
    else:
        # Find least recently accessed nodes
        cursor.execute("""
            SELECT id, last_accessed, access_count
            FROM thought_nodes
            WHERE (decayed IS NULL OR decayed = 0)
            AND node_type != 'seed'
            ORDER BY COALESCE(access_count, 0) ASC, last_accessed ASC
            LIMIT 10
        """)
    
    candidates = cursor.fetchall()
    
    if not candidates:
        conn.close()
        return []
    
    # Pick a random subset for thinking
    import random
    cluster_size = min(config.think_cycle_nodes, len(candidates))
    selected = random.sample(candidates, cluster_size)
    
    conn.close()
    return [node_id for node_id, _, _ in selected]

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
    
    thinking_prompt = f"""
Analyze these connected thoughts and look for new patterns, connections, or insights:

{cluster_description}

What new connections, patterns, tensions, or derived insights do you see? 
Consider:
1. Cross-domain connections between these ideas
2. Missing links or gaps in reasoning
3. Potential contradictions that need resolution  
4. Higher-level patterns or principles

Generate 1-3 new derived thoughts that synthesize or extend these ideas.
Format as JSON array:
[{{"content": "...", "type": "insight|derived|connection", "confidence": 0.0-1.0}}]
"""
    
    try:
        response = model_fn(thinking_prompt)
        
        # Parse response
        new_thoughts = []
        if response.strip().startswith('['):
            try:
                import json
                new_thoughts = json.loads(response)
            except:
                pass
        
        if not new_thoughts:
            # Fallback: create a simple synthesis
            new_thoughts = [{
                "content": f"Think cycle synthesis of {len(nodes_info)} related concepts",
                "type": "insight", 
                "confidence": 0.6
            }]
        
        # Create new nodes
        new_nodes = []
        new_edges = []
        
        for thought in new_thoughts:
            if not thought.get("content"):
                continue
            
            # Create derived node
            derived_id = _create_node(
                db_path, 
                thought["content"],
                thought.get("type", "insight"),
                "think_cycle",
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

def main():
    """CLI interface for session management"""
    parser = argparse.ArgumentParser(description="Cashew Session Integration")
    parser.add_argument("command", choices=["start", "end", "think"], help="Command to run")
    parser.add_argument("--db", default="/Users/bunny/.openclaw/workspace/cashew/data/test_session.db", 
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