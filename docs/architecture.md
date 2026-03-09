# Cashew Hierarchical Retrieval Architecture

## Overview

Cashew implements a hierarchical retrieval system that uses recursive clustering to build a tree of hotspot nodes. Instead of flat search over 900+ nodes, the system performs a DFS (Depth-First Search) through a tree of hotspots where search is O(log N) comparisons.

## Core Principles

1. **Brain is source of truth** - The graph database holds the authoritative state of all knowledge
2. **Files are blob storage** - Markdown files contain raw content but no status information
3. **Hotspots are the index** - Hierarchical summary nodes that act as routing points for efficient retrieval

## Hierarchical Retrieval System

### Hotspot Tree Structure

Hotspots form a tree where:
- **Root hotspots**: High-level summaries not children of other hotspots
- **Parent hotspots**: Summarize large clusters and point to sub-hotspots
- **Leaf hotspots**: Summarize final clusters and point to detail nodes
- **Detail nodes**: Original knowledge nodes (thoughts, facts, decisions)

```
Root Hotspot: "Work & Career Progress"
├── Parent Hotspot: "Meta Engineering Role" 
│   ├── Leaf Hotspot: "E5 Promotion Timeline"
│   │   ├── Detail: "Received E5 promotion in Q2"
│   │   ├── Detail: "Manager feedback on technical leadership"
│   │   └── Detail: "Cross-team collaboration examples"
│   └── Leaf Hotspot: "Technical Skills"
│       ├── Detail: "Python expertise development"
│       └── Detail: "System design experience"
└── Parent Hotspot: "Career Development"
    └── Leaf Hotspot: "Performance Reviews"
        ├── Detail: "2023 review outcomes"
        └── Detail: "Goal setting for 2024"
```

### Recursive Cluster Detection

The clustering system (`core/clustering.py`) uses recursive DBSCAN:

1. **Initial clustering**: Run DBSCAN on all nodes with `eps=0.35`, `min_samples=3`
2. **Size check**: If cluster > `max_cluster_size=15`, recursively split
3. **Tighter clustering**: Re-run DBSCAN on large clusters with `eps = eps * 0.7`
4. **Hotspot creation**: Create parent hotspot for original cluster, child hotspots for sub-clusters
5. **Hierarchy building**: Connect parent→child with `summarizes` edges
6. **Recursion**: Repeat until all clusters ≤ max_cluster_size

### DFS Search Algorithm

The retrieval system (`core/retrieval.py`) implements DFS traversal:

```python
def dfs_search(current_hotspots: Set[str]) -> str:
    # 1. Compute embedding similarity for current level hotspots
    hotspot_sims = [(id, similarity(query, id)) for id in current_hotspots]
    
    # 2. Sort by similarity, take top 2-3 for exploration  
    hotspot_sims.sort(reverse=True)
    
    # 3. For each promising hotspot:
    for hotspot_id, sim in hotspot_sims[:3]:
        children = get_child_hotspots(hotspot_id)
        
        if children:
            # Has children - recurse deeper
            return dfs_search(children)
        else:
            # Leaf hotspot - return this as final selection
            return hotspot_id
```

### Search Flow

1. **Root selection**: Get all root-level hotspots (not children of other hotspots)
2. **Embedding comparison**: Compare query embedding against root hotspots
3. **Best match selection**: Pick top 2-3 most similar root hotspots
4. **Recursive descent**: For each selected hotspot:
   - Check if it has child hotspots
   - If yes: compare query against children, pick best, recurse
   - If no (leaf): use this hotspot's cluster members
5. **Result assembly**: Return 1 leaf hotspot (context) + N detail nodes (ranked by similarity)

## Sleep Cycle Maintenance

The sleep cycle (`core/sleep.py`) maintains the hierarchical structure:

### Clustering Phase

- Runs recursive clustering with `max_cluster_size=15`
- Creates new parent/child hotspots as needed
- Preserves existing hotspot hierarchy
- Updates stale hotspots that drift from their clusters

### Staleness Detection

- Computes similarity between hotspot embedding and cluster centroid
- Marks hotspots as stale if similarity < 0.65
- Regenerates stale hotspot summaries using LLM

## Scalability Analysis

| Node Count | Flat Search | DFS Search | Improvement |
|------------|-------------|------------|-------------|
| 100        | 100 comparisons | ~10 comparisons | 10x |
| 1,000      | 1,000 comparisons | ~15 comparisons | 67x |
| 10,000     | 10,000 comparisons | ~20 comparisons | 500x |
| 100,000    | 100,000 comparisons | ~25 comparisons | 4,000x |

DFS search achieves O(log N) complexity by:
- Only comparing against 5-10 hotspots at each level
- Tree depth typically 3-4 levels for reasonable cluster sizes
- Avoiding brute-force comparison against all detail nodes

## Implementation Details

### Files Modified

1. `core/clustering.py`: Added `detect_clusters_recursive()` and hierarchical edge creation
2. `core/retrieval.py`: Added `retrieve_dfs()` with tree traversal
3. `core/session.py`: Updated to use `retrieve_dfs()` by default
4. `core/sleep.py`: Updated to use recursive clustering
5. `integration/openclaw.py`: Updated to use DFS retrieval

### Database Schema

Uses existing schema with `relation='summarizes'`:
- Parent hotspot → child hotspot: `summarizes` relation
- Leaf hotspot → detail nodes: `summarizes` relation
- `node_type='hotspot'` for all hotspot nodes

### Backward Compatibility

- Existing CLI commands unchanged
- `retrieve()` function preserved for legacy use
- `retrieve_hierarchical()` still available as intermediate option
- All existing test queries should work

## Performance Characteristics

### Search Performance
- **Cold start**: ~200ms (embedding computation + DFS traversal)
- **Warm cache**: ~50ms (DFS traversal only)  
- **Memory usage**: O(hotspots) not O(all_nodes)

### Tree Maintenance
- **Sleep cycle frequency**: Every 10-20 new nodes
- **Clustering cost**: O(N²) for distance matrix, amortized over many searches
- **Staleness detection**: O(hotspots) per sleep cycle

## Testing & Validation

### Test Queries
Run these to verify recall@5 ≥ 78% (current baseline):

```bash
cd cashew && KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py context --hints "is Raj still Christian"
cd cashew && KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py context --hints "what does Raj do for work" 
cd cashew && KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py context --hints "Raj family religion conflict"
cd cashew && KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py context --hints "Raj promotion timeline"
cd cashew && KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py context --hints "blog 2 progress status"
cd cashew && KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py context --hints "who is Vinny"
cd cashew && KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py context --hints "what is Bunny working on"
```

### Success Criteria
- Queries should find relevant information in first 5 results
- Response time should be <100ms for most queries  
- Tree structure should be visible in hotspot nodes
- New clusters should automatically get hierarchical organization

## Future Enhancements

1. **Dynamic rebalancing**: Automatically split/merge hotspots based on usage patterns
2. **Query routing hints**: Use query analysis to skip irrelevant tree branches
3. **Incremental updates**: Update tree structure without full re-clustering
4. **Cross-domain bridging**: Special handling for queries spanning multiple domains
5. **Temporal decay**: Factor in recency when building/traversing hierarchy