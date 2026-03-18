#!/usr/bin/env python3
"""
Cashew Complete Integration Module
Integrates the new 100% coverage clustering, placement-aware extraction, 
hierarchy evolution, and complete retrieval systems.
"""

import logging
import json
from typing import Optional, List, Dict, Callable

logger = logging.getLogger("cashew.complete_integration")

# Import the complete coverage modules
from core.complete_clustering import run_complete_clustering_cycle, check_complete_coverage
from core.placement_aware_extraction import (
    extract_with_placement, batch_assign_orphaned_nodes, create_node_with_placement
)
from core.hierarchy_evolution import run_hierarchy_evolution_cycle
from core.complete_retrieval import (
    retrieve_complete_dfs, retrieve_complete_hierarchical, retrieve_complete_breadth_first,
    format_complete_context, explain_complete_retrieval
)
from core.embeddings import embed_nodes

# Database path is now configurable via environment variable or CLI  
from core.config import get_db_path, get_user_domain, get_ai_domain

# Removed direct Anthropic API client creation - model_fn should be passed from external orchestrator

def generate_complete_session_context(db_path: str = None, 
                                     hints: Optional[List[str]] = None,
                                     domain: Optional[str] = None,
                                     method: str = "dfs",
                                     top_k: int = 5) -> str:
    """
    Generate session context using complete retrieval (no fallback pools).
    
    Args:
        db_path: Path to graph database
        hints: Optional topic hints for context retrieval
        domain: Optional domain filter
        method: Retrieval method ("dfs", "hierarchical", "breadth_first")
        top_k: Number of context items to retrieve
        
    Returns:
        Formatted context string ready for LLM prompt
    """
    if db_path is None:
        db_path = get_db_path()
        
    if not hints:
        # Default to broad query
        query = "recent important insights decisions projects status"
    else:
        query = " ".join(hints)
    
    try:
        if method == "dfs":
            results = retrieve_complete_dfs(db_path, query, top_k, domain)
        elif method == "hierarchical":
            results = retrieve_complete_hierarchical(db_path, query, top_k, domain=domain)
        elif method == "breadth_first":
            results = retrieve_complete_breadth_first(db_path, query, top_k, domain)
        else:
            logger.warning(f"Unknown retrieval method: {method}, falling back to DFS")
            results = retrieve_complete_dfs(db_path, query, top_k, domain)
        
        if not results:
            return "No relevant context found in the complete graph."
        
        context = format_complete_context(results, include_paths=False)
        logger.info(f"Generated complete context with {len(results)} items using {method} method")
        return context
        
    except Exception as e:
        logger.error(f"Failed to generate complete session context: {e}")
        return f"Error generating context: {e}"

def extract_from_conversation_complete(db_path: str = None,
                                      conversation_text: str = "",
                                      session_id: str = "complete_session",
                                      model_fn: Optional[Callable[[str], str]] = None) -> Dict:
    """
    Extract knowledge from conversation with placement-aware assignment.
    
    Args:
        db_path: Path to graph database
        conversation_text: Full conversation to extract from
        session_id: Session identifier
        model_fn: Optional model function for LLM-powered extraction. If None, uses heuristic extraction only.
        
    Returns:
        Dict with extraction and placement results
    """
    if db_path is None:
        db_path = get_db_path()
        
    try:
        # Ensure schema is up to date before any writes
        from core.embeddings import ensure_schema
        ensure_schema(db_path)
        
        if not model_fn:
            logger.warning("No model function provided - LLM-dependent extraction features disabled")
        
        result = extract_with_placement(
            db_path, conversation_text, session_id, model_fn
        )
        
        # Ensure embeddings are up to date
        try:
            embed_nodes(db_path)
        except Exception as e:
            logger.warning(f"Failed to embed new nodes: {e}")
        
        return result
        
    except Exception as e:
        logger.error(f"Complete extraction failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "new_nodes": [],
            "placements": []
        }

def run_complete_think_cycle(db_path: str = None,
                            focus_domain: Optional[str] = None,
                            model_fn: Optional[Callable[[str], str]] = None) -> Dict:
    """
    Run a think cycle that creates nodes with immediate placement.
    
    Args:
        db_path: Path to graph database
        focus_domain: Optional domain to focus thinking on
        model_fn: Optional model function for LLM-powered think cycles. If None, returns error.
        
    Returns:
        Dict with think cycle results and placements
    """
    if db_path is None:
        db_path = get_db_path()
        
    try:
        from core.session import think_cycle
        
        if not model_fn:
            return {
                "success": False,
                "error": "No model function provided - think cycles require LLM access",
                "new_nodes": []
            }
        
        # Run the think cycle (this creates nodes but doesn't place them)
        think_result = think_cycle(db_path, model_fn, focus_domain)
        
        # Ensure all new nodes are placed in clusters
        if think_result.new_nodes:
            placement_result = batch_assign_orphaned_nodes(db_path, model_fn, dry_run=False)
            
            return {
                "success": True,
                "think_result": think_result.to_dict(),
                "placement_result": placement_result,
                "new_nodes": think_result.new_nodes
            }
        else:
            return {
                "success": True,
                "think_result": think_result.to_dict(),
                "placement_result": {"orphaned_nodes_found": 0},
                "new_nodes": []
            }
            
    except Exception as e:
        logger.error(f"Complete think cycle failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "new_nodes": []
        }

def run_complete_sleep_cycle(db_path: str = None,
                            enable_hierarchy_evolution: bool = True,
                            model_fn: Optional[Callable[[str], str]] = None) -> Dict:
    """
    Run a complete sleep cycle with clustering, hierarchy evolution, and 100% coverage.
    
    Args:
        db_path: Path to graph database
        enable_hierarchy_evolution: Whether to run hierarchy evolution operations
        model_fn: Optional model function for LLM-powered sleep operations. If None, some features will be disabled.
        
    Returns:
        Dict with comprehensive sleep cycle results
    """
    if db_path is None:
        db_path = get_db_path()
        
    try:
        # Ensure schema is up to date before any writes
        from core.embeddings import ensure_schema
        ensure_schema(db_path)
        
        if not model_fn:
            logger.warning("No model function provided - LLM-dependent sleep features disabled")
        
        logger.info("Starting complete sleep cycle with 100% coverage")
        
        # Phase 1: Complete clustering (eliminates noise points)
        logger.info("Phase 1: Running complete clustering cycle")
        clustering_results = run_complete_clustering_cycle(db_path, model_fn, dry_run=False)
        
        # Phase 2: Hierarchy evolution (merge, split, promote, reclassify)
        evolution_results = {}
        if enable_hierarchy_evolution and model_fn:
            logger.info("Phase 2: Running hierarchy evolution cycle")
            evolution_results = run_hierarchy_evolution_cycle(db_path, model_fn, dry_run=False)
        elif not model_fn:
            logger.info("Phase 2: Skipping hierarchy evolution - no LLM access")
            evolution_results = {"skipped": "no_llm_access"}
        
        # Phase 3: Verify 100% coverage
        logger.info("Phase 3: Verifying complete coverage")
        coverage_stats = check_complete_coverage(db_path)
        
        # Phase 4: Assign any remaining orphans (shouldn't be any, but safety check)
        orphan_assignment = {"orphaned_nodes_found": 0}
        if coverage_stats.get("orphaned_nodes", 0) > 0:
            if model_fn:
                logger.warning(f"Found {coverage_stats['orphaned_nodes']} orphans - running emergency assignment")
                orphan_assignment = batch_assign_orphaned_nodes(db_path, model_fn, dry_run=False)
            else:
                logger.warning(f"Found {coverage_stats['orphaned_nodes']} orphans but no LLM access for assignment")
                orphan_assignment = {"skipped": "no_llm_access", "orphaned_nodes_found": coverage_stats['orphaned_nodes']}
        
        # Combine results
        complete_sleep_results = {
            "clustering_phase": clustering_results,
            "evolution_phase": evolution_results,
            "coverage_verification": coverage_stats,
            "orphan_assignment": orphan_assignment,
            "total_actions": (
                clustering_results.get("new_hotspots_created", 0) +
                evolution_results.get("merges_performed", 0) +
                evolution_results.get("splits_performed", 0) +
                evolution_results.get("promotions_performed", 0) +
                evolution_results.get("reclassifications_performed", 0)
            )
        }
        
        logger.info(f"Complete sleep cycle finished: {complete_sleep_results['total_actions']} total actions")
        return complete_sleep_results
        
    except Exception as e:
        import traceback
        error_detail = f"{type(e).__name__}: {e}" if str(e) else f"{type(e).__name__} (no message)"
        logger.error(f"Complete sleep cycle failed: {error_detail}\n{traceback.format_exc()}")
        return {
            "error": error_detail,
            "clustering_phase": {},
            "evolution_phase": {},
            "coverage_verification": {},
            "orphan_assignment": {},
            "total_actions": 0
        }

def migrate_to_complete_coverage(db_path: str = None,
                                dry_run: bool = False,
                                model_fn: Optional[Callable[[str], str]] = None) -> Dict:
    """
    Migrate an existing cashew graph to the complete coverage system.
    
    This assigns all orphaned nodes to clusters and ensures 100% coverage.
    
    Args:
        db_path: Path to graph database  
        dry_run: If True, don't modify database
        model_fn: Optional model function for LLM-powered migration. If None, migration will be limited.
        
    Returns:
        Dict with migration results
    """
    if db_path is None:
        db_path = get_db_path()
    
    try:
        logger.info(f"Starting migration to complete coverage (dry_run={dry_run})")
        
        # Check current coverage
        initial_coverage = check_complete_coverage(db_path)
        logger.info(f"Initial coverage: {initial_coverage['coverage_percentage']:.1f}% "
                   f"({initial_coverage['orphaned_nodes']} orphans)")
        
        if initial_coverage["coverage_percentage"] >= 99.9:
            logger.info("Graph already has complete coverage")
            return {
                "migration_needed": False,
                "initial_coverage": initial_coverage,
                "final_coverage": initial_coverage,
                "assignments": []
            }
        
        if not model_fn:
            return {
                "migration_needed": True,
                "error": "No model function provided - migration requires LLM access for node assignment",
                "initial_coverage": initial_coverage,
                "final_coverage": initial_coverage,
                "assignment_results": {"skipped": "no_llm_access"}
            }
        
        # Assign all orphaned nodes
        assignment_result = batch_assign_orphaned_nodes(db_path, model_fn, dry_run)
        
        if not dry_run:
            # Re-check coverage after assignment
            final_coverage = check_complete_coverage(db_path)
            logger.info(f"Final coverage: {final_coverage['coverage_percentage']:.1f}% "
                       f"({final_coverage['orphaned_nodes']} orphans)")
        else:
            final_coverage = {"coverage_percentage": "unknown (dry run)"}
        
        migration_results = {
            "migration_needed": True,
            "initial_coverage": initial_coverage,
            "final_coverage": final_coverage,
            "assignment_results": assignment_result
        }
        
        logger.info("Migration to complete coverage completed")
        return migration_results
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return {
            "migration_needed": False,
            "error": str(e),
            "initial_coverage": {},
            "final_coverage": {},
            "assignment_results": {}
        }

def explain_complete_system(db_path: str = None, query: str = "test query") -> Dict:
    """
    Generate a comprehensive explanation of the complete coverage system.
    
    Args:
        db_path: Path to graph database
        query: Test query for demonstration
        
    Returns:
        Dict with system explanation and examples
    """
    if db_path is None:
        db_path = get_db_path()
    
    try:
        # Get coverage stats
        coverage_stats = check_complete_coverage(db_path)
        
        # Test all retrieval methods
        dfs_explanation = explain_complete_retrieval(db_path, query, method="dfs")
        hierarchical_explanation = explain_complete_retrieval(db_path, query, method="hierarchical")
        breadth_explanation = explain_complete_retrieval(db_path, query, method="breadth_first")
        
        system_explanation = {
            "system_overview": {
                "description": "Complete coverage clustering system eliminates DBSCAN noise",
                "key_features": [
                    "100% node coverage - zero orphans",
                    "Placement-aware extraction - immediate cluster assignment",
                    "Hierarchy evolution - merge, split, promote, reclassify",
                    "Pure tree traversal retrieval - no fallback pools needed",
                    "Emergent domains - no hardcoded categories"
                ]
            },
            "current_coverage": coverage_stats,
            "retrieval_methods": {
                "dfs": dfs_explanation,
                "hierarchical": hierarchical_explanation, 
                "breadth_first": breadth_explanation
            },
            "architecture_comparison": {
                "old_system": {
                    "dbscan_noise": "~82% nodes orphaned",
                    "extraction": "Nodes created without cluster assignment",
                    "retrieval": "Requires fallback pools for orphaned nodes",
                    "hierarchy": "Static - minimal evolution",
                    "domains": "Hardcoded (user, ai)"
                },
                "new_system": {
                    "dbscan_noise": "0% nodes orphaned - complete coverage",
                    "extraction": "Immediate placement-aware assignment",
                    "retrieval": "Pure tree traversal - no fallbacks needed",
                    "hierarchy": "Dynamic evolution with merge/split/promote/reclassify",
                    "domains": "Fully emergent from content analysis"
                }
            }
        }
        
        return system_explanation
        
    except Exception as e:
        logger.error(f"System explanation failed: {e}")
        return {"error": str(e)}

def get_complete_system_stats(db_path: str = None) -> Dict:
    """
    Get comprehensive statistics about the complete coverage system.
    
    Args:
        db_path: Path to graph database
        
    Returns:
        Dict with system statistics
    """
    if db_path is None:
        db_path = get_db_path()
    
    try:
        coverage_stats = check_complete_coverage(db_path)
        
        from core.complete_clustering import load_embeddings_with_metadata
        from core.complete_retrieval import _build_hotspot_hierarchy
        
        # Load basic node info
        node_ids, vectors, node_meta = load_embeddings_with_metadata(db_path)
        
        # Build hierarchy info
        children_map, parent_map, cluster_members_map = _build_hotspot_hierarchy(db_path)
        
        # Calculate statistics
        domains = {}
        node_types = {}
        for node_id, meta in node_meta.items():
            domain = meta.get("domain", "unknown")
            domains[domain] = domains.get(domain, 0) + 1
            node_type = meta.get("node_type", "unknown")
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        stats = {
            "coverage": coverage_stats,
            "node_statistics": {
                "total_nodes": len(node_ids),
                "by_domain": domains,
                "by_type": node_types
            },
            "hierarchy_statistics": {
                "total_hotspots": len(cluster_members_map),
                "root_hotspots": len([h for h in cluster_members_map.keys() if h not in parent_map]),
                "parent_hotspots": len(children_map),
                "leaf_hotspots": len([h for h in cluster_members_map.keys() if h not in children_map]),
                "avg_cluster_size": sum(len(members) for members in cluster_members_map.values()) / len(cluster_members_map) if cluster_members_map else 0,
                "hierarchy_depth": max([_calculate_depth(h, parent_map) for h in cluster_members_map.keys()]) if cluster_members_map else 0
            },
            "system_health": {
                "has_complete_coverage": coverage_stats.get("coverage_percentage", 0) >= 99.9,
                "emergent_domains": len([d for d in domains.keys() if d not in [get_user_domain(), get_ai_domain(), "unknown"]]),
                "hierarchical_structure": len(parent_map) > 0
            }
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get system stats: {e}")
        return {"error": str(e)}

def _calculate_depth(hotspot_id: str, parent_map: Dict[str, str], visited: Optional[set] = None) -> int:
    """Calculate depth of hotspot in hierarchy (with cycle detection)"""
    if visited is None:
        visited = set()
    
    if hotspot_id in visited:
        return 0  # Cycle detected
    
    visited.add(hotspot_id)
    
    if hotspot_id not in parent_map:
        return 0  # Root node
    
    parent = parent_map[hotspot_id]
    return 1 + _calculate_depth(parent, parent_map, visited)