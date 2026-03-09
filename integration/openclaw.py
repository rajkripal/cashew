#!/usr/bin/env python3
"""
OpenClaw Integration Module for Cashew
Bridge between cashew's session layer and OpenClaw's lifecycle
"""

import os
import json
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path

from core.session import start_session, end_session, think_cycle, tension_detection, SessionContext, ExtractionResult, ThinkResult


def _load_anthropic_api_key() -> Optional[str]:
    """Load Anthropic API key from OpenClaw auth profiles"""
    try:
        auth_profiles_path = "/Users/bunny/.openclaw/agents/main/agent/auth-profiles.json"
        with open(auth_profiles_path, 'r') as f:
            auth_data = json.load(f)
        
        api_key = auth_data.get("profiles", {}).get("anthropic:manual", {}).get("token")
        if not api_key:
            logging.warning("No Anthropic API key found in auth profiles")
            return None
        
        return api_key
    except Exception as e:
        logging.error(f"Failed to load Anthropic API key: {e}")
        return None


def _create_anthropic_model_fn():
    """Create a model function that uses the Anthropic API"""
    api_key = _load_anthropic_api_key()
    if not api_key:
        return None
    
    try:
        import anthropic
    except ImportError:
        logging.error("anthropic package not installed")
        return None
    
    client = anthropic.Anthropic(api_key=api_key)
    
    def model_fn(prompt: str) -> str:
        """Call Anthropic API with the given prompt"""
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            return response.content[0].text
        except Exception as e:
            logging.error(f"Anthropic API call failed: {e}")
            raise
    
    return model_fn


def generate_session_context(db_path: str, hints: Optional[List[str]] = None) -> str:
    """
    Generate session context from the thought graph
    
    Args:
        db_path: Path to the SQLite database
        hints: Optional list of topic hints for context retrieval
        
    Returns:
        Formatted context string ready for injection into AGENTS.md or system context
        Returns empty string on failure (never crashes)
    """
    try:
        # Check if database exists
        if not os.path.exists(db_path):
            logging.warning(f"Database not found at {db_path}")
            return ""
        
        # Use a dummy session ID for context generation
        session_id = "openclaw_context_generation"
        
        # Get session context
        context = start_session(db_path, session_id, hints)
        
        if not context.context_str:
            return ""
        
        # Format with proper header for AGENTS.md injection
        formatted_context = f"""## Relevant Context from Thought Graph

{context.context_str}

*Retrieved {len(context.nodes_used)} nodes, ~{context.token_estimate} tokens*"""
        
        return formatted_context
        
    except Exception as e:
        logging.error(f"Error generating session context: {e}")
        return ""


def extract_from_conversation(db_path: str, conversation_text: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract insights and knowledge from a conversation
    
    Args:
        db_path: Path to the SQLite database
        conversation_text: Full conversation text to extract from
        session_id: Optional session identifier
        
    Returns:
        Dictionary with extraction results and summary
    """
    try:
        # Check if database exists
        if not os.path.exists(db_path):
            raise ValueError(f"Database not found at {db_path}")
        
        session_id = session_id or "openclaw_extraction"
        
        # Create model function for extraction
        model_fn = _create_anthropic_model_fn()
        if not model_fn:
            raise ValueError("Failed to create Anthropic model function")
        
        # Extract from conversation
        result = end_session(db_path, session_id, conversation_text, model_fn)
        
        # Format response
        response = {
            "success": True,
            "new_nodes": len(result.new_nodes),
            "new_edges": len(result.new_edges),
            "updated_nodes": len(result.updated_nodes),
            "node_ids": result.new_nodes,
            "edges": result.new_edges,
            "summary": f"Extracted {len(result.new_nodes)} new thoughts and created {len(result.new_edges)} connections"
        }
        
        return response
        
    except Exception as e:
        logging.error(f"Error extracting from conversation: {e}")
        return {
            "success": False,
            "error": str(e),
            "new_nodes": 0,
            "new_edges": 0,
            "summary": f"Extraction failed: {e}"
        }


def run_think_cycle(db_path: str, focus_domain: Optional[str] = None) -> Dict[str, Any]:
    """
    Run a think cycle on the thought graph
    
    Args:
        db_path: Path to the SQLite database
        focus_domain: Optional domain to focus thinking on
        
    Returns:
        Dictionary with think cycle results and summary
    """
    try:
        # Check if database exists
        if not os.path.exists(db_path):
            raise ValueError(f"Database not found at {db_path}")
        
        # Create model function for thinking
        model_fn = _create_anthropic_model_fn()
        if not model_fn:
            raise ValueError("Failed to create Anthropic model function")
        
        # Run think cycle
        result = think_cycle(db_path, model_fn, focus_domain)
        
        # Format response
        response = {
            "success": True,
            "new_nodes": len(result.new_nodes),
            "new_edges": len(result.new_edges),
            "cluster_topic": result.cluster_topic,
            "node_ids": result.new_nodes,
            "edges": result.new_edges,
            "summary": f"Think cycle on '{result.cluster_topic}': {len(result.new_nodes)} new insights, {len(result.new_edges)} connections"
        }
        
        return response
        
    except Exception as e:
        logging.error(f"Error running think cycle: {e}")
        return {
            "success": False,
            "error": str(e),
            "new_nodes": 0,
            "new_edges": 0,
            "cluster_topic": "Failed",
            "summary": f"Think cycle failed: {e}"
        }


def run_tension_detection(db_path: str, focus_domain: Optional[str] = None) -> Dict[str, Any]:
    """Run tension detection on the thought graph."""
    try:
        if not os.path.exists(db_path):
            raise ValueError(f"Database not found at {db_path}")
        
        model_fn = _create_anthropic_model_fn()
        if not model_fn:
            raise ValueError("Failed to create Anthropic model function")
        
        result = tension_detection(db_path, model_fn, focus_domain)
        
        return {
            "success": True,
            "new_nodes": len(result.new_nodes),
            "new_edges": len(result.new_edges),
            "cluster_topic": result.cluster_topic,
            "node_ids": result.new_nodes,
            "edges": result.new_edges,
            "summary": f"Tension detection: {len(result.new_nodes)} tensions found"
        }
    except Exception as e:
        logging.error(f"Error in tension detection: {e}")
        return {
            "success": False,
            "error": str(e),
            "new_nodes": 0,
            "new_edges": 0,
            "cluster_topic": "Failed",
            "summary": f"Tension detection failed: {e}"
        }


# Convenience functions for common use cases

def get_work_context(db_path: str) -> str:
    """Get context relevant to work/career topics"""
    return generate_session_context(db_path, ["work", "career", "promotion", "manager", "performance"])


def get_personal_context(db_path: str) -> str:
    """Get context relevant to personal topics"""
    return generate_session_context(db_path, ["personal", "fitness", "health", "relationships", "family"])


def get_technical_context(db_path: str) -> str:
    """Get context relevant to technical/engineering topics"""
    return generate_session_context(db_path, ["engineering", "technical", "programming", "software", "architecture"])


def get_bunny_context(db_path: str, hints: Optional[List[str]] = None) -> str:
    """
    Get context from Bunny's operational knowledge domain only
    
    Args:
        db_path: Path to the SQLite database
        hints: Optional list of topic hints for context retrieval
        
    Returns:
        Formatted context string from bunny domain only
    """
    try:
        # Check if database exists
        if not os.path.exists(db_path):
            logging.warning(f"Database not found at {db_path}")
            return ""
        
        from core.retrieval import retrieve_dfs
        
        # Build query from hints or use default
        query = " ".join(hints) if hints else "operational knowledge patterns decisions"
        
        # Retrieve only bunny domain nodes using DFS
        results = retrieve_dfs(db_path, query, top_k=10, domain="bunny")
        
        if not results:
            return ""
        
        # Format with proper header
        from core.retrieval import format_context
        context_str = format_context(results, include_paths=False)
        
        formatted_context = f"""## Bunny's Operational Knowledge

{context_str}

*Retrieved {len(results)} bunny domain nodes*"""
        
        return formatted_context
        
    except Exception as e:
        logging.error(f"Error generating bunny context: {e}")
        return ""


def get_raj_context(db_path: str, hints: Optional[List[str]] = None) -> str:
    """
    Get context from Raj's thought domain only
    
    Args:
        db_path: Path to the SQLite database
        hints: Optional list of topic hints for context retrieval
        
    Returns:
        Formatted context string from raj domain only
    """
    try:
        # Check if database exists
        if not os.path.exists(db_path):
            logging.warning(f"Database not found at {db_path}")
            return ""
        
        from core.retrieval import retrieve_dfs
        
        # Build query from hints or use default
        query = " ".join(hints) if hints else "thoughts insights patterns decisions"
        
        # Retrieve only raj domain nodes using DFS
        results = retrieve_dfs(db_path, query, top_k=10, domain="raj")
        
        if not results:
            return ""
        
        # Format with proper header
        from core.retrieval import format_context
        context_str = format_context(results, include_paths=False)
        
        formatted_context = f"""## Raj's Thoughts and Insights

{context_str}

*Retrieved {len(results)} raj domain nodes*"""
        
        return formatted_context
        
    except Exception as e:
        logging.error(f"Error generating raj context: {e}")
        return ""


def run_work_think_cycle(db_path: str) -> Dict[str, Any]:
    """Run a think cycle focused on work-related insights"""
    return run_think_cycle(db_path, "work")


def run_personal_think_cycle(db_path: str) -> Dict[str, Any]:
    """Run a think cycle focused on personal insights"""
    return run_think_cycle(db_path, "personal")


# Main integration point for OpenClaw
def integrate_with_openclaw(db_path: str, operation: str, **kwargs) -> Dict[str, Any]:
    """
    Main integration function for OpenClaw to call cashew operations
    
    Args:
        db_path: Path to the SQLite database
        operation: Operation to perform ('context', 'extract', 'think')
        **kwargs: Operation-specific arguments
        
    Returns:
        Dictionary with operation results
    """
    try:
        if operation == "context":
            hints = kwargs.get("hints")
            context = generate_session_context(db_path, hints)
            return {
                "success": True,
                "operation": "context",
                "result": context,
                "has_content": bool(context.strip())
            }
        
        elif operation == "extract":
            conversation = kwargs.get("conversation_text", "")
            session_id = kwargs.get("session_id")
            result = extract_from_conversation(db_path, conversation, session_id)
            result["operation"] = "extract"
            return result
        
        elif operation == "think":
            focus_domain = kwargs.get("focus_domain")
            result = run_think_cycle(db_path, focus_domain)
            result["operation"] = "think"
            return result
        
        else:
            return {
                "success": False,
                "operation": operation,
                "error": f"Unknown operation: {operation}",
                "result": ""
            }
    
    except Exception as e:
        logging.error(f"Integration operation '{operation}' failed: {e}")
        return {
            "success": False,
            "operation": operation,
            "error": str(e),
            "result": ""
        }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Cashew OpenClaw Integration")
    parser.add_argument("operation", choices=["context", "extract", "think"], help="Operation to perform")
    parser.add_argument("--db", default="/Users/bunny/.openclaw/workspace/cashew/data/graph.db", help="Database path")
    parser.add_argument("--hints", nargs="*", help="Hints for context generation")
    parser.add_argument("--conversation", help="Conversation text for extraction")
    parser.add_argument("--session-id", help="Session ID")
    parser.add_argument("--domain", help="Focus domain for think cycle")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    if args.operation == "context":
        result = generate_session_context(args.db, args.hints)
        if result:
            print(result)
        else:
            print("No context generated")
    
    elif args.operation == "extract":
        if not args.conversation:
            print("Error: --conversation required for extract operation")
        else:
            result = extract_from_conversation(args.db, args.conversation, args.session_id)
            print(json.dumps(result, indent=2))
    
    elif args.operation == "think":
        result = run_think_cycle(args.db, args.domain)
        print(json.dumps(result, indent=2))