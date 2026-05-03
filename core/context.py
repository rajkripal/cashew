#!/usr/bin/env python3
"""
Cashew Context Retrieval Module
Given a topic/query string, return the most relevant thought nodes and their derivation chains.
This is used to inject the user's existing reasoning into LLM prompts before responding.
"""

import sqlite3
import json
import re
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import argparse
import sys
from dataclasses import dataclass

# Database path is now configurable via environment variable or CLI
from .config import get_db_path

@dataclass
class RelevantNode:
    id: str
    content: str
    node_type: str
    relevance_score: float
    parent_chain: List[Dict]

class ContextRetriever:
    """Retrieve relevant thought nodes for context injection"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = get_db_path()
        self.db_path = db_path
        # Common stopwords to exclude from keyword matching
        self.stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'were', 'will', 'with', 'i', 'you', 'they', 'we',
            'this', 'but', 'not', 'or', 'can', 'had', 'would', 'could',
            'what', 'when', 'where', 'why', 'how', 'all', 'also', 'just',
            'so', 'do', 'does', 'did', 'have', 'been', 'being', 'if', 'up'
        }
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from query"""
        # Normalize and tokenize
        words = re.findall(r'\b\w+\b', query.lower())
        
        # Filter out stopwords and short words
        keywords = [word for word in words 
                   if word not in self.stopwords and len(word) > 2]
        
        return keywords
    
    def _calculate_relevance_score(self, node_content: str, keywords: List[str]) -> float:
        """Calculate relevance score based on keyword overlap"""
        if not keywords:
            return 0.0
        
        content_words = set(re.findall(r'\b\w+\b', node_content.lower()))
        
        # Count keyword matches
        matches = sum(1 for keyword in keywords if keyword in content_words)
        
        # Base relevance is proportion of keywords found
        base_score = matches / len(keywords)
        
        # Boost for exact phrase matches
        exact_matches = 0
        for keyword in keywords:
            if keyword in node_content.lower():
                exact_matches += 1
        
        phrase_boost = exact_matches * 0.1
        
        return min(base_score + phrase_boost, 1.0)
    
    def _get_parent_chain(self, node_id: str, max_depth: int = 3) -> List[Dict]:
        """Get derivation chain using traversal engine"""
        try:
            from .traversal import TraversalEngine
            engine = TraversalEngine(self.db_path)
            chain = engine.why(node_id, max_depth=max_depth)
            
            if not chain or any("error" in step or "cycle_detected" in step for step in chain):
                return []
            
            return chain
        except Exception as e:
            # Fallback: manual parent retrieval
            return self._get_parent_chain_fallback(node_id, max_depth)
    
    def _get_parent_chain_fallback(self, node_id: str, max_depth: int = 3) -> List[Dict]:
        """Fallback parent chain retrieval"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        chain = []
        current_id = node_id
        depth = 0
        visited = set()
        
        while depth < max_depth and current_id not in visited:
            visited.add(current_id)
            
            # Get current node
            cursor.execute("""
                SELECT content, node_type FROM thought_nodes WHERE id = ?
            """, (current_id,))

            node_row = cursor.fetchone()
            if not node_row:
                break

            content, node_type = node_row
            
            # Get parents
            cursor.execute("""
                SELECT de.parent_id, de.weight, de.reasoning, tn.content
                FROM derivation_edges de
                JOIN thought_nodes tn ON de.parent_id = tn.id
                WHERE de.child_id = ?
                ORDER BY de.weight DESC
                LIMIT 1
            """, (current_id,))
            
            parent_row = cursor.fetchone()
            
            chain.append({
                "node": {
                    "id": current_id,
                    "content": content,
                    "type": node_type,
                },
                "depth": depth,
                "derived_from": parent_row[0] if parent_row else None,  # parent_id
                "reasoning": parent_row[2] if parent_row else None  # reasoning (now index 2)
            })
            
            if not parent_row:
                break
            
            current_id = parent_row[0]
            depth += 1
        
        conn.close()
        return chain
    
    def retrieve(self, query: str, max_nodes: int = 5) -> List[RelevantNode]:
        """
        Retrieve most relevant thought nodes for a query
        
        Args:
            query: Search query string
            max_nodes: Maximum number of nodes to return
            
        Returns:
            List of RelevantNode objects ranked by relevance
        """
        keywords = self._extract_keywords(query)
        
        if not keywords:
            return []
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get all non-decayed nodes. node_type is display-only metadata
        # and never participates in retrieval logic — let the LLM decide
        # which shapes are useful context.
        cursor.execute("""
            SELECT id, content, node_type
            FROM thought_nodes
            WHERE (decayed = 0 OR decayed IS NULL)
        """)

        candidates = []

        for row in cursor.fetchall():
            node_id, content, node_type = row

            # Calculate relevance score
            relevance = self._calculate_relevance_score(content, keywords)

            if relevance > 0.1:  # Only include somewhat relevant nodes
                # Get parent chain for context
                parent_chain = self._get_parent_chain(node_id)

                candidates.append(RelevantNode(
                    id=node_id,
                    content=content,
                    node_type=node_type,
                    relevance_score=relevance,
                    parent_chain=parent_chain
                ))

        conn.close()

        # Sort by relevance score
        candidates.sort(key=lambda n: n.relevance_score, reverse=True)
        
        return candidates[:max_nodes]
    
    def format_context(self, nodes: List[RelevantNode]) -> str:
        """
        Format retrieved nodes into clean text block for LLM injection
        
        Args:
            nodes: List of relevant nodes
            
        Returns:
            Formatted context string
        """
        if not nodes:
            return ""
        
        context_lines = ["Existing reasoning on this topic:"]
        context_lines.append("")
        
        for i, node in enumerate(nodes, 1):
            # Main node content
            context_lines.append(f"{i}. {node.content}")
            
            # Show derivation if available
            if node.parent_chain:
                derivation = self._format_derivation_chain(node.parent_chain)
                if derivation:
                    context_lines.append(f"   (derived from: {derivation})")
            
            context_lines.append("")
        
        return "\n".join(context_lines)
    
    def _format_derivation_chain(self, chain: List[Dict]) -> str:
        """Format derivation chain for display"""
        if not chain or len(chain) < 1:
            return ""
        
        try:
            # Extract derivation from traversal engine format
            current = chain[0]
            
            if "derived_from" not in current:
                return ""
            
            derivations = []
            for derivation in current["derived_from"]:
                if "parent_chain" in derivation and derivation["parent_chain"]:
                    parent = derivation["parent_chain"][0]
                    parent_content = parent.get("node", {}).get("content", "")
                    if parent_content:
                        derivations.append(parent_content[:60] + "...")
            
            return "; ".join(derivations) if derivations else ""
            
        except (KeyError, IndexError, TypeError):
            # Fallback: use simple format
            if "reasoning" in chain[0]:
                return chain[0]["reasoning"] or ""
            return ""
    
    def search_by_content(self, content_fragment: str, max_nodes: int = 3) -> List[RelevantNode]:
        """Search for nodes containing specific content"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Use SQLite FTS if available, otherwise LIKE
        cursor.execute("""
            SELECT id, content, node_type
            FROM thought_nodes
            WHERE content LIKE ?
            AND (decayed = 0 OR decayed IS NULL)
            LIMIT ?
        """, (f"%{content_fragment}%", max_nodes))

        results = []
        for row in cursor.fetchall():
            node_id, content, node_type = row
            parent_chain = self._get_parent_chain(node_id)

            results.append(RelevantNode(
                id=node_id,
                content=content,
                node_type=node_type,
                relevance_score=1.0,  # Exact content match
                parent_chain=parent_chain
            ))
        
        conn.close()
        return results
    
    def get_related_nodes(self, node_id: str, max_nodes: int = 3) -> List[RelevantNode]:
        """Get nodes related to a specific node via edges"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get related nodes through edges (both directions)
        cursor.execute("""
            SELECT tn.id, tn.content, tn.node_type, de.weight
            FROM derivation_edges de
            JOIN thought_nodes tn ON (
                (de.parent_id = ? AND tn.id = de.child_id) OR
                (de.child_id = ? AND tn.id = de.parent_id)
            )
            WHERE (tn.decayed = 0 OR tn.decayed IS NULL)
            ORDER BY de.weight DESC
            LIMIT ?
        """, (node_id, node_id, max_nodes))

        results = []
        for row in cursor.fetchall():
            rel_id, content, node_type, weight = row
            parent_chain = self._get_parent_chain(rel_id)

            results.append(RelevantNode(
                id=rel_id,
                content=content,
                node_type=node_type,
                relevance_score=weight,
                parent_chain=parent_chain
            ))
        
        conn.close()
        return results


def main():
    """CLI interface for context retrieval"""
    parser = argparse.ArgumentParser(description="Cashew Context Retrieval")
    parser.add_argument("command", choices=["query", "content", "related"], help="Command to run")
    parser.add_argument("search_term", help="Query string, content fragment, or node ID")
    parser.add_argument("--max-nodes", type=int, default=5, help="Maximum nodes to return")
    parser.add_argument("--format", choices=["context", "json"], default="context", help="Output format")
    
    args = parser.parse_args()
    
    retriever = ContextRetriever()
    
    if args.command == "query":
        nodes = retriever.retrieve(args.search_term, args.max_nodes)
        
        if args.format == "json":
            # Output as JSON for programmatic use
            result = []
            for node in nodes:
                result.append({
                    "id": node.id,
                    "content": node.content,
                    "type": node.node_type,
                    "relevance": node.relevance_score,
                    "parent_chain": node.parent_chain
                })
            print(json.dumps(result, indent=2))
        else:
            # Output formatted context
            context = retriever.format_context(nodes)
            print(f"\n🔍 Context for '{args.search_term}':")
            print("=" * 50)
            print(context if context else "No relevant context found.")
    
    elif args.command == "content":
        nodes = retriever.search_by_content(args.search_term, args.max_nodes)
        context = retriever.format_context(nodes)
        print(f"\n🔎 Nodes containing '{args.search_term}':")
        print("=" * 50)
        print(context if context else "No matching content found.")
    
    elif args.command == "related":
        nodes = retriever.get_related_nodes(args.search_term, args.max_nodes)
        context = retriever.format_context(nodes)
        print(f"\n🔗 Nodes related to {args.search_term[:12]}...:")
        print("=" * 50)
        print(context if context else "No related nodes found.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())