#!/usr/bin/env python3
"""
Cashew Placement-Aware Extraction Module
Immediately assigns new nodes to best-matching hotspots upon creation.
No node ever exists without cluster membership.
"""

import sqlite3
import json
import hashlib
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple, Callable
from datetime import datetime, timezone

from core.embeddings import embed_text, embed_nodes, search as embedding_search
from core.hotspots import create_hotspot, HOTSPOT_TYPE
from core.complete_clustering import infer_emergent_domains

logger = logging.getLogger("cashew.placement_aware_extraction")

# Database path is now configurable via environment variable or CLI
from .config import get_db_path

# Quality gate parameters
# MiniLM-L6 cosine similarity distribution: mean ~0.13, P99 ~0.49, true dupes peak ~0.85-0.90
NOVELTY_THRESHOLD = 0.82  # reject if nearest neighbor similarity > this
BORDERLINE_THRESHOLD = 0.72  # confidence tiebreaker zone: 0.72-0.82


def check_novelty(db_path: str, content: str, threshold: float = NOVELTY_THRESHOLD) -> Tuple[bool, float, Optional[str]]:
    """
    Check if a candidate node is sufficiently novel compared to existing graph.
    
    Returns:
        (is_novel, max_similarity, nearest_node_id)
        is_novel=True means the node should be kept.
    """
    try:
        candidate_embedding = np.array(embed_text(content))
    except Exception as e:
        logger.warning(f"Failed to embed candidate for novelty check: {e}")
        return True, 0.0, None  # fail open — allow the node
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT e.node_id, e.vector 
        FROM embeddings e
        JOIN thought_nodes tn ON e.node_id = tn.id
        WHERE tn.decayed IS NULL OR tn.decayed = 0
    """)
    
    max_sim = 0.0
    nearest_id = None
    
    for node_id, vector_bytes in cursor.fetchall():
        try:
            stored = np.frombuffer(vector_bytes, dtype=np.float32)
            dot = np.dot(candidate_embedding, stored)
            cn = np.linalg.norm(candidate_embedding)
            sn = np.linalg.norm(stored)
            if cn > 0 and sn > 0:
                sim = float(dot / (cn * sn))
                if sim > max_sim:
                    max_sim = sim
                    nearest_id = node_id
        except Exception:
            continue
    
    conn.close()
    
    is_novel = max_sim < threshold
    return is_novel, max_sim, nearest_id


# Placement parameters
HOTSPOT_MATCH_THRESHOLD = 0.3  # Min similarity to assign to existing hotspot
UNCATEGORIZED_DOMAIN = "inbox"  # Domain for uncategorized hotspot


def _get_connection(db_path: str) -> sqlite3.Connection:
    """Get database connection"""
    return sqlite3.connect(db_path)


def _find_best_matching_hotspot(db_path: str, node_content: str, 
                               domain_filter: Optional[str] = None) -> Tuple[Optional[str], float]:
    """
    Find the best-matching hotspot for a new node using embedding similarity.
    
    Args:
        db_path: Path to database
        node_content: Content of the new node
        domain_filter: Optional domain to restrict hotspot search
        
    Returns:
        Tuple of (best_hotspot_id, similarity_score) or (None, 0.0)
    """
    try:
        # Get embedding for the new node content
        query_embedding = np.array(embed_text(node_content), dtype=np.float32)
        
        conn = _get_connection(db_path)
        cursor = conn.cursor()
        
        # Get all hotspot embeddings (optionally filtered by domain)
        if domain_filter:
            cursor.execute("""
                SELECT tn.id, e.vector, tn.content
                FROM thought_nodes tn
                JOIN embeddings e ON tn.id = e.node_id
                WHERE tn.node_type = ? 
                AND (tn.decayed IS NULL OR tn.decayed = 0)
                AND COALESCE(tn.domain, 'unknown') = ?
            """, (HOTSPOT_TYPE, domain_filter))
        else:
            cursor.execute("""
                SELECT tn.id, e.vector, tn.content
                FROM thought_nodes tn
                JOIN embeddings e ON tn.id = e.node_id
                WHERE tn.node_type = ? 
                AND (tn.decayed IS NULL OR tn.decayed = 0)
            """, (HOTSPOT_TYPE,))
        
        hotspot_data = cursor.fetchall()
        conn.close()
        
        if not hotspot_data:
            return None, 0.0
        
        best_hotspot = None
        best_similarity = 0.0
        
        for hotspot_id, vector_blob, hotspot_content in hotspot_data:
            try:
                hotspot_vector = np.frombuffer(vector_blob, dtype=np.float32)
                
                # Calculate cosine similarity
                dot_product = np.dot(query_embedding, hotspot_vector)
                norm_query = np.linalg.norm(query_embedding)
                norm_hotspot = np.linalg.norm(hotspot_vector)
                
                if norm_query > 0 and norm_hotspot > 0:
                    similarity = float(dot_product / (norm_query * norm_hotspot))
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_hotspot = hotspot_id
            
            except Exception as e:
                logger.warning(f"Error computing similarity for hotspot {hotspot_id}: {e}")
                continue
        
        return best_hotspot, best_similarity
        
    except Exception as e:
        logger.warning(f"Error finding best matching hotspot: {e}")
        return None, 0.0


def _ensure_uncategorized_hotspot(db_path: str) -> str:
    """
    Ensure an 'uncategorized' inbox hotspot exists for new nodes that don't match existing hotspots.
    
    Returns:
        ID of the uncategorized hotspot
    """
    conn = _get_connection(db_path)
    cursor = conn.cursor()
    
    # Check if uncategorized hotspot already exists
    cursor.execute("""
        SELECT id FROM thought_nodes 
        WHERE node_type = ? AND domain = ? 
        AND content LIKE '%uncategorized%' OR content LIKE '%inbox%'
        AND (decayed IS NULL OR decayed = 0)
        LIMIT 1
    """, (HOTSPOT_TYPE, UNCATEGORIZED_DOMAIN))
    
    existing = cursor.fetchone()
    if existing:
        conn.close()
        return existing[0]
    
    conn.close()
    
    # Create new uncategorized hotspot
    hotspot_id = create_hotspot(
        db_path=db_path,
        content="[INBOX] Uncategorized thoughts awaiting classification during sleep cycles",
        status="auto_generated_inbox",
        file_pointers={},
        cluster_node_ids=[],  # Will be populated as nodes are assigned
        domain=UNCATEGORIZED_DOMAIN,
        tags=["inbox", "auto_cluster", "uncategorized"]
    )
    
    logger.info(f"Created uncategorized inbox hotspot: {hotspot_id}")
    return hotspot_id


def _assign_node_to_hotspot(db_path: str, node_id: str, hotspot_id: str, 
                           reasoning: str = "Placement-aware extraction assignment"):
    """
    Create a 'summarizes' edge from hotspot to node, placing the node in the cluster.
    
    Args:
        db_path: Path to database
        node_id: ID of node to assign
        hotspot_id: ID of hotspot to assign to
        reasoning: Reason for the assignment
    """
    conn = _get_connection(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT OR IGNORE INTO derivation_edges 
            (parent_id, child_id, relation, weight, reasoning)
            VALUES (?, ?, 'summarizes', 0.8, ?)
        """, (hotspot_id, node_id, reasoning))
        
        conn.commit()
        logger.info(f"Assigned node {node_id} to hotspot {hotspot_id}")
        
    except sqlite3.IntegrityError:
        logger.warning(f"Edge already exists: {hotspot_id} -> {node_id}")
    
    conn.close()


def create_node_with_placement(db_path: str, content: str, node_type: str,
                              session_id: str, confidence: float = 0.7,
                              domain_hint: Optional[str] = None,
                              model_fn: Optional[Callable[[str], str]] = None) -> Tuple[str, str]:
    """
    Create a new thought node with immediate hotspot placement.
    
    Args:
        db_path: Path to database
        content: Node content
        node_type: Type of node
        session_id: Session identifier (used as source_file)
        confidence: Node confidence score
        domain_hint: Optional hint about which domain this belongs to
        model_fn: Optional LLM function for creating new hotspots
        
    Returns:
        Tuple of (node_id, assigned_hotspot_id)
    """
    # Generate deterministic ID based on content
    node_id = hashlib.sha256(content.encode()).hexdigest()[:12]
    
    conn = _get_connection(db_path)
    cursor = conn.cursor()
    
    # Check if node already exists
    cursor.execute("SELECT id FROM thought_nodes WHERE id = ?", (node_id,))
    if cursor.fetchone():
        conn.close()
        # Node exists - find its current hotspot assignment
        existing_hotspot = _find_nodes_current_hotspot(db_path, node_id)
        return node_id, existing_hotspot or "unknown"
    
    # Infer domain if not provided
    if not domain_hint:
        node_meta = {node_id: {"content": content, "node_type": node_type, "source_file": session_id}}
        domain_mapping = infer_emergent_domains(node_meta)
        inferred_domain = domain_mapping.get(node_id, "general")
    else:
        inferred_domain = domain_hint
    
    # Insert new node
    now = datetime.now(timezone.utc).isoformat()
    
    cursor.execute("""
        INSERT INTO thought_nodes 
        (id, content, node_type, timestamp, confidence, source_file, 
         last_accessed, access_count, domain)
        VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)
    """, (node_id, content, node_type, now, confidence, session_id, now, inferred_domain))
    
    conn.commit()
    conn.close()
    
    # Embed the new node immediately
    try:
        embed_nodes(db_path)  # This will pick up the new node
        logger.info(f"Embedded new node {node_id}")
    except Exception as e:
        logger.warning(f"Failed to embed node {node_id}: {e}")
    
    # Find best-matching hotspot
    best_hotspot, similarity = _find_best_matching_hotspot(db_path, content, inferred_domain)
    
    if best_hotspot and similarity >= HOTSPOT_MATCH_THRESHOLD:
        # Assign to existing hotspot
        _assign_node_to_hotspot(
            db_path, node_id, best_hotspot, 
            f"Placement-aware extraction (similarity: {similarity:.3f})"
        )
        assigned_hotspot = best_hotspot
        logger.info(f"Assigned node {node_id} to existing hotspot {best_hotspot} (sim: {similarity:.3f})")
        
    else:
        # Check if we should create a new hotspot for this domain area
        if similarity < HOTSPOT_MATCH_THRESHOLD and _should_create_new_hotspot(db_path, content, inferred_domain, model_fn):
            # Create new hotspot for this topic area
            new_hotspot_id = _create_new_hotspot_for_node(db_path, node_id, content, inferred_domain, model_fn)
            assigned_hotspot = new_hotspot_id
            logger.info(f"Created new hotspot {new_hotspot_id} for node {node_id}")
        else:
            # Assign to uncategorized inbox
            uncategorized_hotspot = _ensure_uncategorized_hotspot(db_path)
            _assign_node_to_hotspot(
                db_path, node_id, uncategorized_hotspot,
                f"Placement-aware extraction - uncategorized (best similarity: {similarity:.3f})"
            )
            assigned_hotspot = uncategorized_hotspot
            logger.info(f"Assigned node {node_id} to uncategorized inbox")
    
    return node_id, assigned_hotspot


def _find_nodes_current_hotspot(db_path: str, node_id: str) -> Optional[str]:
    """Find which hotspot currently contains this node"""
    conn = _get_connection(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT de.parent_id
        FROM derivation_edges de
        JOIN thought_nodes tn ON de.parent_id = tn.id
        WHERE de.child_id = ? AND de.relation = 'summarizes'
        AND tn.node_type = ?
    """, (node_id, HOTSPOT_TYPE))
    
    result = cursor.fetchone()
    conn.close()
    
    return result[0] if result else None


MIN_CLUSTER_SIZE = 5  # Minimum nodes to justify creating a new hotspot

def _should_create_new_hotspot(db_path: str, content: str, domain: str, model_fn) -> bool:
    """
    Decide if we should create a new hotspot for this content.
    
    Philosophy: orphan nodes are a feature, not a bug. Only create a new hotspot
    when there are MIN_CLUSTER_SIZE+ inbox nodes that cluster around a similar topic.
    Otherwise, let the node sit in the inbox until a natural cluster emerges.
    """
    try:
        candidate_embedding = np.array(embed_text(content))
    except Exception:
        return False
    
    # Count inbox nodes similar to this content
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Find the inbox hotspot
    cursor.execute("""
        SELECT id FROM thought_nodes 
        WHERE node_type = 'hotspot' AND content LIKE '%INBOX%Uncategorized%'
        AND (decayed IS NULL OR decayed = 0)
        LIMIT 1
    """)
    inbox_row = cursor.fetchone()
    if not inbox_row:
        conn.close()
        return False
    
    inbox_id = inbox_row[0]
    
    # Get inbox children with embeddings
    cursor.execute("""
        SELECT e.node_id, emb.vector 
        FROM derivation_edges e
        JOIN embeddings emb ON e.child_id = emb.node_id
        WHERE e.parent_id = ?
    """, (inbox_id,))
    
    similar_count = 0
    for node_id, vec_bytes in cursor.fetchall():
        try:
            stored = np.frombuffer(vec_bytes, dtype=np.float32)
            sim = float(np.dot(candidate_embedding, stored) / 
                       (np.linalg.norm(candidate_embedding) * np.linalg.norm(stored)))
            if sim > 0.6:  # Reasonably similar topic
                similar_count += 1
        except Exception:
            continue
    
    conn.close()
    
    # Only create hotspot if enough similar nodes are waiting in inbox
    return similar_count >= (MIN_CLUSTER_SIZE - 1)  # -1 because the current node would join too


def _create_new_hotspot_for_node(db_path: str, node_id: str, content: str, 
                                domain: str, model_fn) -> str:
    """
    Create a new hotspot to contain this node and similar future nodes.
    
    Args:
        db_path: Path to database
        node_id: ID of the node to be contained
        content: Content of the node
        domain: Inferred domain
        model_fn: LLM function for generating hotspot summary
        
    Returns:
        ID of the created hotspot
    """
    # Generate hotspot summary
    if model_fn:
        prompt = f"""Create a concise hotspot title/summary (1-2 sentences max) for this new topic area in the {domain} domain.

The hotspot should capture the main theme and be broad enough to contain related future thoughts.

Founding thought: {content}

Hotspot summary:"""
        try:
            summary = model_fn(prompt).strip()
            if not summary:
                raise ValueError("Empty summary from model")
        except Exception as e:
            logger.warning(f"LLM hotspot summary failed: {e}")
            summary = f"[{domain.upper()}] {content[:60]}..."
    else:
        summary = f"[{domain.upper()}] {content[:60]}..."
    
    # Create the hotspot
    hotspot_id = create_hotspot(
        db_path=db_path,
        content=summary,
        status="auto_generated_placement",
        file_pointers={},
        cluster_node_ids=[node_id],
        domain=domain,
        tags=["auto_cluster", "placement_aware", "new_topic"]
    )
    
    # The create_hotspot function will automatically create the summarizes edge
    logger.info(f"Created new hotspot {hotspot_id} for node {node_id} in domain {domain}")
    
    return hotspot_id


def batch_assign_orphaned_nodes(db_path: str, model_fn: Optional[Callable[[str], str]] = None,
                               dry_run: bool = False) -> Dict:
    """
    Find all orphaned nodes (not assigned to any hotspot) and assign them.
    Useful for migrating from old clustering to placement-aware system.
    
    Args:
        db_path: Path to database  
        model_fn: Optional LLM function for creating new hotspots
        dry_run: If True, don't modify database
        
    Returns:
        Dict with assignment statistics
    """
    conn = _get_connection(db_path)
    cursor = conn.cursor()
    
    # Find all nodes that aren't assigned to any hotspot
    cursor.execute("""
        SELECT tn.id, tn.content, tn.node_type, COALESCE(tn.domain, 'general') as domain
        FROM thought_nodes tn
        WHERE (tn.decayed IS NULL OR tn.decayed = 0)
        AND tn.node_type != ?
        AND tn.id NOT IN (
            SELECT DISTINCT de.child_id 
            FROM derivation_edges de
            JOIN thought_nodes hotspot ON de.parent_id = hotspot.id
            WHERE de.relation = 'summarizes' AND hotspot.node_type = ?
        )
    """, (HOTSPOT_TYPE, HOTSPOT_TYPE))
    
    orphaned_nodes = cursor.fetchall()
    conn.close()
    
    results = {
        "orphaned_nodes_found": len(orphaned_nodes),
        "assigned_to_existing": 0,
        "assigned_to_new": 0,
        "assigned_to_inbox": 0,
        "assignments": []
    }
    
    logger.info(f"Found {len(orphaned_nodes)} orphaned nodes for assignment")
    
    for node_id, content, node_type, domain in orphaned_nodes:
        if dry_run:
            best_hotspot, similarity = _find_best_matching_hotspot(db_path, content, domain)
            assignment_type = "existing" if (best_hotspot and similarity >= HOTSPOT_MATCH_THRESHOLD) else "inbox"
            results["assignments"].append({
                "node_id": node_id,
                "content": content[:80],
                "domain": domain,
                "assignment_type": assignment_type,
                "similarity": similarity
            })
            continue
        
        # Actually assign the node
        best_hotspot, similarity = _find_best_matching_hotspot(db_path, content, domain)
        
        if best_hotspot and similarity >= HOTSPOT_MATCH_THRESHOLD:
            # Assign to existing hotspot
            _assign_node_to_hotspot(
                db_path, node_id, best_hotspot,
                f"Batch assignment migration (similarity: {similarity:.3f})"
            )
            results["assigned_to_existing"] += 1
            assignment_type = "existing"
            
        elif _should_create_new_hotspot(db_path, content, domain, model_fn):
            # Create new hotspot
            new_hotspot_id = _create_new_hotspot_for_node(db_path, node_id, content, domain, model_fn)
            results["assigned_to_new"] += 1
            assignment_type = "new"
            
        else:
            # Assign to inbox
            uncategorized_hotspot = _ensure_uncategorized_hotspot(db_path)
            _assign_node_to_hotspot(
                db_path, node_id, uncategorized_hotspot,
                f"Batch assignment migration - uncategorized (best similarity: {similarity:.3f})"
            )
            results["assigned_to_inbox"] += 1
            assignment_type = "inbox"
        
        results["assignments"].append({
            "node_id": node_id,
            "content": content[:80],
            "domain": domain,
            "assignment_type": assignment_type,
            "similarity": similarity
        })
    
    return results


def extract_with_placement(db_path: str, conversation_text: str, session_id: str,
                          model_fn: Optional[Callable[[str], str]] = None) -> Dict:
    """
    Extract knowledge from conversation with immediate placement-aware assignment.
    
    This replaces the extraction logic in session.py with placement-aware behavior.
    
    Args:
        db_path: Path to database
        conversation_text: Full conversation to extract from
        session_id: Session identifier
        model_fn: Optional LLM function for extraction and hotspot creation
        
    Returns:
        Dict with extraction and placement results
    """
    if not conversation_text or len(conversation_text.strip()) < 20:
        logger.info(f"Session {session_id} has minimal content, skipping extraction")
        return {
            "success": False,
            "reason": "Insufficient content",
            "new_nodes": [],
            "placements": []
        }
    
    extractions = []
    
    if model_fn:
        # Use LLM for structured extraction
        extraction_prompt = f"""You are extracting knowledge from a conversation into a personal thought graph with immediate cluster placement. Extract ONLY genuinely new, specific, substantive knowledge — not summaries or meta-comments.

For each item, classify as:
- "belief": a held opinion or conviction
- "insight": a non-obvious connection or pattern discovered  
- "decision": a commitment or choice made
- "observation": a factual pattern noticed
- "fact": a concrete verifiable fact

Each content field must be a specific, standalone statement that makes sense without the conversation context.

BAD: "They discussed embeddings" (meta-comment)
BAD: "The conversation covered several topics" (summary)  
GOOD: "Local embedding models (all-MiniLM-L6-v2) are sufficient for graphs under 100K nodes — brute force cosine similarity stays under 50ms"
GOOD: "Placement-aware extraction should assign nodes to hotspots immediately during creation, not leave them orphaned"

IMPORTANT: Only nodes with confidence >= 0.8 will be saved to the database. Be selective and only extract high-quality, substantive knowledge.

Respond with ONLY a JSON array. No markdown, no explanation, no code fences.

[{{"content": "specific knowledge here", "type": "belief|insight|decision|observation|fact", "confidence": 0.7}}]

Conversation to extract from:
{conversation_text}
"""
        
        try:
            response = model_fn(extraction_prompt)
            # Try to parse JSON response
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
                extractions = json.loads(json_str)
                logger.info(f"Extracted {len(extractions)} items via LLM")
            else:
                logger.warning("No JSON array found in LLM response, using heuristics")
                extractions = _extract_with_heuristics(conversation_text)
        
        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}, using heuristics")
            extractions = _extract_with_heuristics(conversation_text)
    
    else:
        # Use heuristic extraction
        extractions = _extract_with_heuristics(conversation_text)
    
    # Create nodes with immediate placement
    new_nodes = []
    placements = []
    
    for extraction in extractions:
        if not extraction.get("content"):
            continue
        
        content = extraction["content"]
        node_type = extraction.get("type", "observation") 
        confidence = extraction.get("confidence", 0.5)
        
        # Primary gate: semantic novelty check
        is_novel, max_sim, nearest_id = check_novelty(db_path, content)
        if not is_novel:
            logger.info(f"Rejecting duplicate (sim={max_sim:.3f} to {nearest_id}): {content[:60]}")
            continue
        
        # Secondary gate: use confidence as tiebreaker for borderline novelty
        # If similarity is in borderline zone (0.72-0.82) AND confidence is low, skip
        if max_sim > BORDERLINE_THRESHOLD and confidence < 0.7:
            logger.info(f"Rejecting borderline node (sim={max_sim:.3f}, conf={confidence}): {content[:60]}")
            continue
            
        # Create node with placement
        try:
            node_id, assigned_hotspot = create_node_with_placement(
                db_path, content, node_type, session_id, confidence, model_fn=model_fn
            )
            new_nodes.append(node_id)
            placements.append({
                "node_id": node_id,
                "content": content[:80],
                "type": node_type,
                "assigned_hotspot": assigned_hotspot,
                "confidence": confidence
            })
        except Exception as e:
            logger.error(f"Failed to create node with placement: {e}")
            continue
    
    logger.info(f"Placement-aware extraction: {len(new_nodes)} nodes created and placed")
    
    return {
        "success": True,
        "new_nodes": new_nodes,
        "placements": placements,
        "extraction_count": len(extractions)
    }


def _extract_with_heuristics(conversation_text: str) -> List[Dict[str, str]]:
    """Fallback extraction using heuristics when no model_fn provided"""
    import re
    
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
                "type": "decision",
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
                "type": "belief",
                "confidence": 0.5
            })
    
    # Fact markers (simple heuristic)
    sentences = conversation_text.split('.')
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 20 and len(sentence) < 100:
            # Look for factual statements (contains proper nouns, numbers, etc.)
            if re.search(r'[A-Z][a-z]+|\d+', sentence):
                extractions.append({
                    "content": f"Observation: {sentence}",
                    "type": "observation",
                    "confidence": 0.4
                })
    
    # Limit to most promising extractions
    return sorted(extractions, key=lambda x: x["confidence"], reverse=True)[:5]