#!/usr/bin/env python3
"""
OpenClaw Integration Module for Cashew
Bridge between cashew's session layer and OpenClaw's lifecycle
"""

import os
import json
import logging
from typing import List, Dict, Optional, Any, Callable
from pathlib import Path

from core.session import start_session, end_session, think_cycle, tension_detection, SessionContext, ExtractionResult, ThinkResult
from core.config import config, get_user_domain, get_ai_domain


def generate_session_context(db_path: str, hints: Optional[List[str]] = None) -> str:
    """
    Generate three-layer session context from the thought graph
    
    Args:
        db_path: Path to the SQLite database
        hints: Optional list of topic hints for context retrieval
        
    Returns:
        Formatted three-layer context string ready for injection
        Returns empty string on failure (never crashes)
    """
    try:
        # Check if database exists
        if not os.path.exists(db_path):
            logging.warning(f"Database not found at {db_path}")
            return ""
        
        # Use a dummy session ID for context generation
        session_id = "openclaw_context_generation"
        
        # Get three-layer session context
        context = start_session(db_path, session_id, hints)
        
        if not context.context_str:
            return ""
        
        # The context_str now already has the three-layer format, just add header
        formatted_context = f"""## Context from Thought Graph

{context.context_str}

*Context layers: overview + recent activity{' + relevant hints' if hints else ''} (~{context.token_estimate} tokens)*"""
        
        return formatted_context
        
    except Exception as e:
        logging.error(f"Error generating session context: {e}")
        return ""


def extract_from_conversation(db_path: str, conversation_text: str, session_id: Optional[str] = None, model_fn: Optional[Callable[[str], str]] = None) -> Dict[str, Any]:
    """
    Extract insights and knowledge from a conversation
    
    Args:
        db_path: Path to the SQLite database
        conversation_text: Full conversation text to extract from
        session_id: Optional session identifier
        model_fn: Optional model function for LLM-powered extraction. If None, uses heuristic extraction only.
        
    Returns:
        Dictionary with extraction results and summary
    """
    try:
        # Check if database exists
        if not os.path.exists(db_path):
            raise ValueError(f"Database not found at {db_path}")
        
        session_id = session_id or "openclaw_extraction"
        
        if not model_fn:
            logging.warning("No model function provided - LLM-dependent extraction features disabled")
        
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


def run_think_cycle(db_path: str, focus_domain: Optional[str] = None, model_fn: Optional[Callable[[str], str]] = None) -> Dict[str, Any]:
    """
    Run a think cycle on the thought graph
    
    Args:
        db_path: Path to the SQLite database
        focus_domain: Optional domain to focus thinking on
        model_fn: Optional model function for LLM-powered think cycles. If None, skips LLM-dependent operations.
        
    Returns:
        Dictionary with think cycle results and summary
    """
    try:
        # Check if database exists
        if not os.path.exists(db_path):
            raise ValueError(f"Database not found at {db_path}")
        
        if not model_fn:
            return {
                "success": False,
                "error": "No model function provided - think cycles require LLM access",
                "new_nodes": 0,
                "new_edges": 0,
                "cluster_topic": "Skipped",
                "summary": "Think cycle skipped - no LLM access"
            }
        
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


def run_tension_detection(db_path: str, focus_domain: Optional[str] = None, model_fn: Optional[Callable[[str], str]] = None) -> Dict[str, Any]:
    """Run tension detection on the thought graph."""
    try:
        if not os.path.exists(db_path):
            raise ValueError(f"Database not found at {db_path}")
        
        if not model_fn:
            return {
                "success": False,
                "error": "No model function provided - tension detection requires LLM access",
                "new_nodes": 0,
                "new_edges": 0,
                "cluster_topic": "Skipped",
                "summary": "Tension detection skipped - no LLM access"
            }
        
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


def get_ai_context(db_path: str, hints: Optional[List[str]] = None) -> str:
    """
    Get context from AI's operational knowledge domain only
    
    Args:
        db_path: Path to the SQLite database
        hints: Optional list of topic hints for context retrieval
        
    Returns:
        Formatted context string from AI domain only
    """
    try:
        # Check if database exists
        if not os.path.exists(db_path):
            logging.warning(f"Database not found at {db_path}")
            return ""
        
        from core.retrieval import retrieve_dfs
        
        # Build query from hints or use default
        query = " ".join(hints) if hints else "operational knowledge patterns decisions"
        
        # Retrieve only AI domain nodes using DFS
        ai_domain = get_ai_domain()
        results = retrieve_dfs(db_path, query, top_k=10, domain=ai_domain)
        
        if not results:
            return ""
        
        # Format with proper header
        from core.retrieval import format_context
        context_str = format_context(results, include_paths=False)
        
        formatted_context = f"""## AI Operational Knowledge

{context_str}

*Retrieved {len(results)} {ai_domain} domain nodes*"""
        
        return formatted_context
        
    except Exception as e:
        logging.error(f"Error generating AI context: {e}")
        return ""


def get_user_context(db_path: str, hints: Optional[List[str]] = None) -> str:
    """
    Get context from user's thought domain only
    
    Args:
        db_path: Path to the SQLite database
        hints: Optional list of topic hints for context retrieval
        
    Returns:
        Formatted context string from user domain only
    """
    try:
        # Check if database exists
        if not os.path.exists(db_path):
            logging.warning(f"Database not found at {db_path}")
            return ""
        
        from core.retrieval import retrieve_dfs
        
        # Build query from hints or use default
        query = " ".join(hints) if hints else "thoughts insights patterns decisions"
        
        # Retrieve only user domain nodes using DFS
        user_domain = get_user_domain()
        results = retrieve_dfs(db_path, query, top_k=10, domain=user_domain)
        
        if not results:
            return ""
        
        # Format with proper header
        from core.retrieval import format_context
        context_str = format_context(results, include_paths=False)
        
        formatted_context = f"""## User Thoughts and Insights

{context_str}

*Retrieved {len(results)} {user_domain} domain nodes*"""
        
        return formatted_context
        
    except Exception as e:
        logging.error(f"Error generating user context: {e}")
        return ""


def run_work_think_cycle(db_path: str, model_fn: Optional[Callable[[str], str]] = None) -> Dict[str, Any]:
    """Run a think cycle focused on work-related insights"""
    return run_think_cycle(db_path, "work", model_fn)


def run_personal_think_cycle(db_path: str, model_fn: Optional[Callable[[str], str]] = None) -> Dict[str, Any]:
    """Run a think cycle focused on personal insights"""
    return run_think_cycle(db_path, "personal", model_fn)


# Main integration point for OpenClaw
def integrate_with_openclaw(db_path: str, operation: str, model_fn: Optional[Callable[[str], str]] = None, **kwargs) -> Dict[str, Any]:
    """
    Main integration function for OpenClaw to call cashew operations
    
    Args:
        db_path: Path to the SQLite database
        operation: Operation to perform ('context', 'extract', 'think')
        model_fn: Optional model function for LLM-powered operations. If None, some operations will be skipped or use fallbacks.
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
            result = extract_from_conversation(db_path, conversation, session_id, model_fn)
            result["operation"] = "extract"
            return result
        
        elif operation == "think":
            focus_domain = kwargs.get("focus_domain")
            result = run_think_cycle(db_path, focus_domain, model_fn)
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
    from core.config import get_db_path
    parser.add_argument("--db", default=get_db_path(), help="Database path")
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
            # CLI usage has no model function - will use heuristic extraction only
            result = extract_from_conversation(args.db, args.conversation, args.session_id, model_fn=None)
            print(json.dumps(result, indent=2))
    
    elif args.operation == "think":
        # CLI usage has no model function - will be skipped
        print("Think cycles require LLM access. Use through OpenClaw cron jobs instead.")
        result = {"success": False, "error": "No LLM access from CLI", "summary": "CLI think cycles not supported"}
        print(json.dumps(result, indent=2))