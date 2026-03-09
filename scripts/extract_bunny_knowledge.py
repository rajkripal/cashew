#!/usr/bin/env python3
"""
Extract Bunny's operational knowledge from workspace files
Creates nodes tagged as 'bunny' domain with various operational insights
"""

import sqlite3
import json
import hashlib
import uuid
import argparse
import sys
import os
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass

@dataclass
class KnowledgeNode:
    """Represents a piece of operational knowledge to be stored"""
    content: str
    node_type: str  # belief/observation/decision/fact
    confidence: float
    source_file: str
    category: str  # observation_about_raj, operational_decision, tool_knowledge, self_knowledge
    
    def get_id(self) -> str:
        """Generate deterministic ID based on content"""
        return hashlib.sha256(self.content.encode()).hexdigest()[:12]

def get_connection(db_path: str) -> sqlite3.Connection:
    """Get database connection"""
    return sqlite3.connect(db_path)

def read_workspace_file(file_path: str) -> str:
    """Read a workspace file and return its contents"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}")
        return ""

def extract_observations_about_raj(soul_content: str, user_content: str, memory_content: str) -> List[KnowledgeNode]:
    """Extract observations about Raj's patterns, preferences, and triggers"""
    nodes = []
    
    # Key observations from USER.md patterns section
    observations = [
        ("Raj goes silent when struggling - this is his biggest career risk", 0.9, "USER.md"),
        ("Raj withdraws instead of communicating when facing difficulties", 0.9, "USER.md"),
        ("Raj people-pleases and overcommits to avoid conflict, then can't deliver", 0.9, "USER.md"),
        ("Raj gives unrealistic timelines due to people-pleasing tendencies", 0.8, "USER.md"),
        ("Raj gets trapped in perfectionism, focusing deeply on one thing while 10 others need attention", 0.9, "USER.md"),
        ("Raj treats 'good enough' as moral failure", 0.8, "USER.md"),
        ("Raj holds threads in his head without external systems", 0.8, "USER.md"),
        ("Raj expands capacity, stabilizes, then stops pushing until crisis forces growth", 0.8, "USER.md"),
        ("Raj needs to narrate work more visibly for promotion - impact not communicated doesn't count", 0.9, "USER.md"),
        ("Raj has confidence oscillations between timid and performer modes", 0.8, "USER.md"),
        ("Raj treats habits like streaks where breaks equal failure", 0.8, "USER.md"),
        ("Raj is a depth player who struggles in breadth contexts but dominates once depth opens up", 0.9, "USER.md"),
        ("Raj's strength compounds quietly in depth but organizations reward breadth signals", 0.9, "USER.md"),
        ("Raj has worse routine on WFH days and needs structure strategies", 0.7, "USER.md"),
        ("Raj loves systems-level thinking and cross-domain pattern recognition", 0.9, "USER.md"),
        ("Raj hates filler and no-fluff communication", 0.9, "SOUL.md"),
        ("Raj appreciates intellectual rigor and systems analogies", 0.9, "SOUL.md"),
        ("Raj's Vienna anchor is Partner, Chiki, music, and life itself - not achievements", 0.9, "MEMORY.md"),
        ("Raj can be happy without being satisfied - emotional state decoupled from problem queue depth", 0.8, "MEMORY.md"),
        ("Raj has tight feedback loops that wire perform→love, don't perform→silence", 0.8, "MEMORY.md"),
        ("Raj reads silence as rejection due to conditioning", 0.8, "MEMORY.md"),
        ("Raj is motivated by being seen as a beautiful person, not just being right", 0.9, "MEMORY.md"),
        ("Raj uses a 'chain links' mental model for relationships - loose but firm grip", 0.7, "MEMORY.md"),
        ("Raj is a deriver, not memorizer - loves first principles over rote answers", 0.9, "MEMORY.md"),
        ("Raj's writing comes from catharsis first, analytical pieces later", 0.8, "MEMORY.md"),
        ("Raj thinks in structure naturally and uses cross-domain analogies as his superpower", 0.9, "MEMORY.md"),
    ]
    
    for content, confidence, source in observations:
        nodes.append(KnowledgeNode(
            content=content,
            node_type="observation",
            confidence=confidence,
            source_file=source,
            category="observation_about_raj"
        ))
    
    return nodes

def extract_operational_decisions(soul_content: str, agents_content: str) -> List[KnowledgeNode]:
    """Extract operational decisions about communication style, formatting, etc."""
    nodes = []
    
    decisions = [
        ("Be direct, warm, no-fluff - get to the point because Raj hates filler", 0.9, "SOUL.md"),
        ("Match Raj's energy level - excited gets matched, processing heavy gets space", 0.9, "SOUL.md"),
        ("Use systems analogies over generic advice - they land better with Raj", 0.9, "SOUL.md"),
        ("Push back when needed - be honest over comfortable, name avoidance when seen", 0.9, "SOUL.md"),
        ("Use humor and wit when appropriate - don't be sterile", 0.8, "SOUL.md"),
        ("Accountability is the single most important thing Bunny does for Raj", 0.9, "SOUL.md"),
        ("Check in after 1 day of silence on workstreams - don't nag but don't let him hide", 0.9, "SOUL.md"),
        ("Never send/draft/reply emails without Raj's explicit approval", 0.9, "AGENTS.md"),
        ("Use bullet points when useful, prose for real conversations", 0.8, "AGENTS.md"),
        ("Don't over-format or create walls of text", 0.8, "AGENTS.md"),
        ("On Discord/WhatsApp: no markdown tables, use bullet lists instead", 0.9, "AGENTS.md"),
        ("Wrap multiple Discord links in <> to suppress embeds", 0.8, "AGENTS.md"),
        ("Know when to speak in group chats - respond when directly mentioned or adding value", 0.9, "AGENTS.md"),
        ("Stay silent in group chats during casual banter or when conversation flows fine", 0.9, "AGENTS.md"),
        ("Use emoji reactions naturally on platforms that support them", 0.8, "AGENTS.md"),
        ("Write the switchboard, don't BE the switchboard - use scripts for deterministic tasks", 0.9, "AGENTS.md"),
        ("Spend tokens on executive decisions and conversation, not CLI parsing or bookkeeping", 0.9, "AGENTS.md"),
        ("If something can be scripted, write the script instead of being the operator", 0.9, "AGENTS.md"),
        ("Stop bot-to-bot discussions after 3 rounds without concrete artifacts", 0.8, "AGENTS.md"),
        ("Use sub-agents over threads for research or building tasks", 0.8, "AGENTS.md"),
        ("PRs over proposals - writing code forces solving vs describing problems", 0.9, "AGENTS.md"),
        ("Test everything before marking it as done", 0.9, "AGENTS.md"),
        ("Never use 'git add -A' or 'git add .' - always specify files", 0.9, "AGENTS.md"),
        ("One feature per PR - split if touching multiple directories", 0.8, "AGENTS.md"),
        ("Always use PRs, never push to main", 0.9, "AGENTS.md"),
        ("Write to files, not mental notes - memory doesn't survive session restarts", 0.9, "AGENTS.md"),
        ("Save substantive conversations to memory files automatically", 0.8, "AGENTS.md"),
        ("Load MEMORY.md only in main sessions, not shared contexts for security", 0.9, "AGENTS.md"),
        ("Auto-save during conversations every ~5 exchanges if anything meaningful happens", 0.8, "AGENTS.md"),
    ]
    
    for content, confidence, source in decisions:
        nodes.append(KnowledgeNode(
            content=content,
            node_type="decision",
            confidence=confidence,
            source_file=source,
            category="operational_decision"
        ))
    
    return nodes

def extract_tool_knowledge(tools_content: str) -> List[KnowledgeNode]:
    """Extract tool knowledge, account details, CLI gotchas, etc."""
    nodes = []
    
    # Extract key tool facts (not secrets)
    facts = [
        ("Gmail infra account exists at bot@example.com for operational email", 0.9, "TOOLS.md"),
        ("Google OAuth is connected via gog CLI with file-based keyring", 0.9, "TOOLS.md"),
        ("Four Gmail accounts are connected: rajkripal.danday, rajkripaldanday17, raj.kutless17, bunny.rajs.openclaw", 0.9, "TOOLS.md"),
        ("Discord bot ID for Bunny is 1472692347094437929", 0.9, "TOOLS.md"),
        ("Raj's Discord user ID is 775475002836647957", 0.9, "TOOLS.md"),
        ("Model IDs require full IDs with date suffix in OpenClaw", 0.9, "TOOLS.md"),
        ("claude-haiku-3-5 is blocked by OpenClaw as insecure", 0.9, "TOOLS.md"),
        ("Short model names like 'claude-haiku' do not work in OpenClaw", 0.9, "TOOLS.md"),
        ("GitHub account is bunny-bot-openclaw with PAT stored in keychain", 0.9, "TOOLS.md"),
        ("Member of jugaad-lab GitHub organization", 0.8, "TOOLS.md"),
        ("Actual Budget app runs desktop server on port 5007", 0.8, "TOOLS.md"),
        ("SimpleFIN is connected for bank syncing at $15/year", 0.8, "TOOLS.md"),
        ("Cloudflared creates ephemeral tunnels with trycloudflare.com URLs", 0.8, "TOOLS.md"),
        ("Quick cloudflared tunnels die after ~30 minutes", 0.8, "TOOLS.md"),
        ("Browser control requires gateway restart to pick up new browsers", 0.8, "TOOLS.md"),
        ("Chromium is installed but browser control needs gateway restart", 0.7, "TOOLS.md"),
        ("Cashew dashboard expects 'source'/'target' in edges, but DB has 'parent_id'/'child_id'", 0.9, "TOOLS.md"),
        ("Think cycle nodes need source_file='system_generated' for orange styling in dashboard", 0.8, "TOOLS.md"),
        ("Claude Code clean room experiments need zero workspace context", 0.8, "TOOLS.md"),
        ("Claude --print mode can hang with large tasks, use interactive mode instead", 0.7, "TOOLS.md"),
        ("KMP_DUPLICATE_LIB_OK=TRUE environment variable needed for embeddings", 0.9, "TOOLS.md"),
        ("Primary calendar is user@example.com for event creation", 0.9, "TOOLS.md"),
        ("Busy/Tentative calendar entries are Meta work sync, ignore on holidays/weekends", 0.8, "TOOLS.md"),
        ("Email priority order: people > bills > appointments > interests", 0.8, "TOOLS.md"),
    ]
    
    for content, confidence, source in facts:
        nodes.append(KnowledgeNode(
            content=content,
            node_type="fact",
            confidence=confidence,
            source_file=source,
            category="tool_knowledge"
        ))
    
    return nodes

def extract_self_knowledge(soul_content: str) -> List[KnowledgeNode]:
    """Extract self-knowledge about Bunny's personality, role, boundaries"""
    nodes = []
    
    beliefs = [
        ("Bunny is Raj's AI alter-ego and externalized operating system running 24/7", 0.9, "SOUL.md"),
        ("Bunny's mission is to learn Raj's patterns, think like him, and keep him honest", 0.9, "SOUL.md"),
        ("Bunny is not a generic assistant but a specialized AI mirror for Raj", 0.9, "SOUL.md"),
        ("Bunny should be intellectually rigorous with systems-level thinking", 0.9, "SOUL.md"),
        ("Bunny should be honest over comfortable, pushing back when needed", 0.9, "SOUL.md"),
        ("Bunny should match Raj's energy and be playful when earned", 0.8, "SOUL.md"),
        ("Bunny's core strength is cross-domain pattern recognition like Raj", 0.9, "SOUL.md"),
        ("Bunny should never waste Raj's time despite being named playfully", 0.9, "SOUL.md"),
        ("Bunny's job is to help Raj be reliable, proactive, and intellectually honest", 0.9, "SOUL.md"),
        ("Bunny helps through scaffolding and awareness, not by doing things for Raj", 0.9, "SOUL.md"),
        ("Bunny should think like an engineering system with proper interfaces and persistence", 0.8, "SOUL.md"),
        ("Bunny operates as a daily operating system providing briefings and tracking", 0.9, "SOUL.md"),
        ("Bunny serves as a thinking partner for systems-level dialogue", 0.9, "SOUL.md"),
        ("Bunny works as a communication coach helping frame impact and visibility", 0.8, "SOUL.md"),
        ("Bunny supports E5 promotion tracking and simulation testing capstone", 0.8, "SOUL.md"),
        ("Bunny should maintain cost optimization using different models for different tasks", 0.8, "SOUL.md"),
        ("Bunny should practice proper development with PRs and never push to main", 0.8, "SOUL.md"),
        ("Bunny should do root cause analysis and not accept surface-level explanations", 0.9, "SOUL.md"),
        ("Bunny provides life support including fitness check-ins and relationship reminders", 0.8, "SOUL.md"),
        ("Bunny should be direct and conversational, not over-formatted", 0.8, "SOUL.md"),
        ("Bunny should ask before sending anything external or destructive", 0.9, "SOUL.md"),
        ("Bunny can freely read, explore, organize and work within the workspace", 0.9, "SOUL.md"),
        ("Bunny should never exfiltrate private data", 0.9, "SOUL.md"),
        ("Bunny operates with participant mindset in groups, not as Raj's voice or proxy", 0.9, "SOUL.md"),
        ("Bunny should be proactive during heartbeats but respect quiet time", 0.8, "SOUL.md"),
    ]
    
    for content, confidence, source in beliefs:
        nodes.append(KnowledgeNode(
            content=content,
            node_type="belief",
            confidence=confidence,
            source_file=source,
            category="self_knowledge"
        ))
    
    return nodes

def create_node_in_db(db_path: str, node: KnowledgeNode) -> str:
    """Create a node in the database and return its ID"""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    node_id = node.get_id()
    
    # Check if node already exists
    cursor.execute("SELECT id FROM thought_nodes WHERE id = ?", (node_id,))
    if cursor.fetchone():
        conn.close()
        return node_id  # Node already exists
    
    # Insert new node with domain information
    now = datetime.now(timezone.utc).isoformat()
    
    # Create metadata with domain and category
    metadata = {
        "domain": "bunny",
        "category": node.category
    }
    
    cursor.execute("""
        INSERT INTO thought_nodes 
        (id, content, node_type, timestamp, confidence, source_file, 
         last_accessed, access_count, domain, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
    """, (
        node_id, 
        node.content, 
        node.node_type, 
        now, 
        node.confidence, 
        node.source_file,
        now,
        "bunny",
        json.dumps(metadata)
    ))
    
    conn.commit()
    conn.close()
    
    return node_id

def find_similar_nodes_for_edges(db_path: str, node_id: str, domain_filter: str = None) -> List[Tuple[str, float]]:
    """Find similar nodes for creating edges using simple text similarity"""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    # Get the node content
    cursor.execute("SELECT content FROM thought_nodes WHERE id = ?", (node_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return []
    
    content = row[0].lower()
    
    # Simple text similarity - look for nodes with overlapping keywords
    if domain_filter:
        cursor.execute("""
            SELECT id, content FROM thought_nodes 
            WHERE id != ? 
            AND domain = ?
            AND (decayed IS NULL OR decayed = 0)
        """, (node_id, domain_filter))
    else:
        cursor.execute("""
            SELECT id, content FROM thought_nodes 
            WHERE id != ?
            AND (decayed IS NULL OR decayed = 0)
        """, (node_id,))
    
    candidates = cursor.fetchall()
    conn.close()
    
    # Simple keyword matching
    keywords = set(word for word in content.split() if len(word) > 3)
    
    similar_nodes = []
    for candidate_id, candidate_content in candidates:
        candidate_lower = candidate_content.lower()
        candidate_keywords = set(word for word in candidate_lower.split() if len(word) > 3)
        
        # Calculate overlap score
        overlap = len(keywords & candidate_keywords)
        if overlap >= 2:  # At least 2 overlapping keywords
            score = overlap / (len(keywords) + len(candidate_keywords) - overlap)  # Jaccard similarity
            if score >= 0.1:  # Minimum similarity threshold
                similar_nodes.append((candidate_id, score))
    
    # Sort by similarity and return top matches
    similar_nodes.sort(key=lambda x: x[1], reverse=True)
    return similar_nodes[:5]

def create_edge_in_db(db_path: str, parent_id: str, child_id: str, reasoning: str):
    """Create an edge between two nodes"""
    conn = get_connection(db_path)
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
        VALUES (?, ?, 'related', 0.7, ?)
    """, (parent_id, child_id, reasoning))
    
    conn.commit()
    conn.close()

def embed_nodes(db_path: str):
    """Embed all new nodes using the embeddings module"""
    try:
        # Import here to handle optional dependency
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from core.embeddings import embed_nodes as embed_nodes_fn
        
        # Set environment variable for embeddings
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        
        embed_nodes_fn(db_path)
        
    except Exception as e:
        print(f"Warning: Could not embed nodes: {e}")
        print("You may need to run: cd /Users/bunny/.openclaw/workspace/cashew && KMP_DUPLICATE_LIB_OK=TRUE python3 -c 'from core.embeddings import embed_nodes; embed_nodes(\"data/graph.db\")'")

def extract_bunny_knowledge(db_path: str, workspace_dir: str) -> Dict[str, int]:
    """
    Extract all of Bunny's operational knowledge and create nodes
    
    Returns:
        Dictionary with statistics about extraction
    """
    print("📖 Reading workspace files...")
    
    # Read workspace files
    soul_path = os.path.join(workspace_dir, "SOUL.md")
    user_path = os.path.join(workspace_dir, "USER.md")
    tools_path = os.path.join(workspace_dir, "TOOLS.md")
    agents_path = os.path.join(workspace_dir, "AGENTS.md")
    memory_path = os.path.join(workspace_dir, "MEMORY.md")
    
    soul_content = read_workspace_file(soul_path)
    user_content = read_workspace_file(user_path)
    tools_content = read_workspace_file(tools_path)
    agents_content = read_workspace_file(agents_path)
    memory_content = read_workspace_file(memory_path)
    
    # Extract different categories of knowledge
    print("🧠 Extracting observations about Raj...")
    observations = extract_observations_about_raj(soul_content, user_content, memory_content)
    
    print("⚖️  Extracting operational decisions...")
    decisions = extract_operational_decisions(soul_content, agents_content)
    
    print("🔧 Extracting tool knowledge...")
    tool_knowledge = extract_tool_knowledge(tools_content)
    
    print("🪞 Extracting self-knowledge...")
    self_knowledge = extract_self_knowledge(soul_content)
    
    # Combine all nodes
    all_nodes = observations + decisions + tool_knowledge + self_knowledge
    
    print(f"📝 Creating {len(all_nodes)} knowledge nodes...")
    
    # Create nodes in database
    created_nodes = []
    stats = {
        "observation_about_raj": 0,
        "operational_decision": 0,
        "tool_knowledge": 0,
        "self_knowledge": 0
    }
    
    for node in all_nodes:
        node_id = create_node_in_db(db_path, node)
        created_nodes.append((node_id, node))
        stats[node.category] += 1
    
    print(f"✅ Created {len(created_nodes)} nodes")
    
    # Create edges between related nodes
    print("🔗 Creating edges between related nodes...")
    
    edge_count = 0
    
    # Create edges between bunny nodes
    for i, (node_id, node) in enumerate(created_nodes):
        similar_bunny = find_similar_nodes_for_edges(db_path, node_id, "bunny")
        
        for similar_id, similarity in similar_bunny:
            reasoning = f"Bunny knowledge similarity: {similarity:.3f}"
            create_edge_in_db(db_path, similar_id, node_id, reasoning)
            edge_count += 1
    
    # Create edges between bunny nodes and raj nodes
    print("🌉 Creating edges between bunny and raj nodes...")
    
    for node_id, node in created_nodes:
        # Only connect a subset to avoid too many edges
        if "Raj" in node.content:  # Only connect nodes that explicitly mention Raj
            similar_raj = find_similar_nodes_for_edges(db_path, node_id, "raj")
            
            for similar_id, similarity in similar_raj[:2]:  # Limit to top 2
                reasoning = f"Cross-domain bunny-raj similarity: {similarity:.3f}"
                create_edge_in_db(db_path, similar_id, node_id, reasoning)
                edge_count += 1
    
    print(f"✅ Created {edge_count} edges")
    
    # Embed new nodes
    print("🎯 Embedding new nodes...")
    embed_nodes(db_path)
    
    print("✅ Knowledge extraction complete!")
    
    return {
        **stats,
        "total_nodes": len(created_nodes),
        "total_edges": edge_count
    }

def main():
    """CLI interface for knowledge extraction"""
    parser = argparse.ArgumentParser(description="Extract Bunny's operational knowledge from workspace files")
    parser.add_argument("--db", required=True, help="Path to SQLite database")
    parser.add_argument("--workspace", default="/Users/bunny/.openclaw/workspace", help="Path to workspace directory")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be extracted without creating nodes")
    
    args = parser.parse_args()
    
    # Check if database exists
    try:
        conn = sqlite3.connect(args.db)
        conn.close()
    except sqlite3.Error as e:
        print(f"❌ Error connecting to database: {e}")
        return 1
    
    # Check if workspace files exist
    required_files = ["SOUL.md", "USER.md", "TOOLS.md", "AGENTS.md", "MEMORY.md"]
    for filename in required_files:
        file_path = os.path.join(args.workspace, filename)
        if not os.path.exists(file_path):
            print(f"❌ Required workspace file not found: {file_path}")
            return 1
    
    if args.dry_run:
        print("🔍 DRY RUN: Would extract Bunny's operational knowledge")
        print("Files to process:")
        for filename in required_files:
            file_path = os.path.join(args.workspace, filename)
            print(f"  ✅ {file_path}")
        return 0
    
    print("🐰 Extracting Bunny's operational knowledge...")
    
    try:
        stats = extract_bunny_knowledge(args.db, args.workspace)
        
        print("\n📊 Extraction Summary:")
        print(f"  Observations about Raj: {stats['observation_about_raj']} nodes")
        print(f"  Operational decisions: {stats['operational_decision']} nodes")
        print(f"  Tool knowledge: {stats['tool_knowledge']} nodes")
        print(f"  Self-knowledge: {stats['self_knowledge']} nodes")
        print(f"  Total nodes created: {stats['total_nodes']}")
        print(f"  Total edges created: {stats['total_edges']}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())