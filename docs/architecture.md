# Cashew Recursive BFS Architecture

## Overview

Cashew implements a flat-graph retrieval system that uses recursive BFS (Breadth-First Search) combined with sqlite-vec for efficient vector search. Instead of hierarchical hotspots, the system performs O(log N) seed selection followed by graph traversal through organic connectivity patterns built by sleep cycles.

## Core Principles

1. **Graph is source of truth** - The SQLite database holds the authoritative state of all knowledge
2. **Files are blob storage** - Markdown files contain raw content but no status information  
3. **Organic connectivity is the index** - Cross-linked relationships provide implicit hierarchy
4. **Emergent structure** - No synthetic categories; organization forms through cross-linking and decay

## Current System State (April 2026, Author's Personal Graph)

- **3,064 thought nodes** across 9 distinct node types
- **6,122 derivation edges** with weight scores

- **Flat graph structure** with organic cross-linking (0 hotspots)
- **O(log N) retrieval** via sqlite-vec seeding + recursive BFS traversal
- **Domain separation** with cross-domain insight generation

*Note: These statistics reflect the author's personal knowledge graph as of April 2026. New users start with an empty graph.*

## Recursive BFS Retrieval System

### sqlite-vec Integration

Vector search lives inside the same SQLite file as the graph using sqlite-vec virtual tables:

```sql
CREATE VIRTUAL TABLE vec_embeddings USING vec0(
    node_id TEXT PRIMARY KEY,
    embedding float[384] distance_metric=cosine
);

-- Query: O(log N) nearest neighbor search
SELECT node_id, distance FROM vec_embeddings
WHERE embedding MATCH ? ORDER BY distance LIMIT 5;
```

### BFS Search Algorithm

The complete retrieval system (`core/retrieval.py`) implements recursive BFS:

```python
def retrieve_recursive_bfs(hints: List[str], n_seeds=5, picks_per_hop=3, max_depth=3) -> List[str]:
    """
    Recursive BFS through organic graph structure
    Returns: List of relevant node IDs ranked by relevance
    """
    # 1. Embed query hints
    query_embedding = embed_text(hints)
    
    # 2. O(log N) seed selection via sqlite-vec
    seeds = search_similar_nodes(query_embedding, limit=n_seeds)
    
    # 3. BFS traversal from seeds
    candidates = set(seeds)
    current_level = seeds
    
    for depth in range(max_depth):
        next_level = []
        
        # Get all neighbors of current level
        neighbors = get_graph_neighbors(current_level)
        
        # Score neighbors by cosine similarity to query
        scored_neighbors = [
            (node_id, cosine_similarity(query_embedding, get_embedding(node_id)))
            for node_id in neighbors if node_id not in candidates
        ]
        
        # Pick top candidates for next level
        scored_neighbors.sort(key=lambda x: x[1], reverse=True)
        best_neighbors = [node_id for node_id, _ in scored_neighbors[:picks_per_hop]]
        
        next_level.extend(best_neighbors)
        candidates.update(best_neighbors)
        
        if not next_level:
            break
        current_level = next_level
    
    # 4. Final ranking by similarity to original query
    final_results = [
        (node_id, cosine_similarity(query_embedding, get_embedding(node_id)))
        for node_id in candidates
    ]
    final_results.sort(key=lambda x: x[1], reverse=True)
    
    return [node_id for node_id, _ in final_results]
```

### Search Flow

1. **Query embedding**: Convert search hints to 384-dimensional vector using sentence-transformers
2. **Seed selection**: Find top-k most similar nodes via sqlite-vec O(log N) search  
3. **BFS traversal**: For each hop (up to max_depth=3):
   - Get neighbors of current level nodes
   - Score neighbors by cosine similarity to original query
   - Select top picks_per_hop candidates for next level
   - Add to candidate set
4. **Final ranking**: Rank all candidates by similarity to original query

**Parameters:**
- `n_seeds=5` - Initial seed nodes from vector search
- `picks_per_hop=3` - Nodes selected per BFS hop
- `max_depth=3` - Maximum traversal depth

## Sleep Cycle Maintenance  

The sleep protocol (`core/sleep.py`) maintains graph structure through organic processes:

### Cross-linking Phase
- Finds semantically similar nodes across domains (0.7-0.85 similarity range)
- Creates edges between related but previously disconnected knowledge
- Preserves diversity by not over-connecting highly similar nodes
- Builds the organic connectivity that BFS traversal exploits

### Decay and Fitness Scoring
- Computes composite fitness scores based on access count, edge degree, and age
- Marks low-fitness nodes as `decayed=1` (excluded from retrieval)
- Think-cycle-generated nodes face 1.5x higher decay threshold
- Preserves high-value knowledge while pruning noise

### Deduplication
- Identifies near-duplicate nodes (>0.82 similarity)
- Merges redundant content, preserving derivation links
- Redirects edges to canonical versions
- Prevents graph bloat from repeated information

### Core Memory Promotion
- Promotes frequently accessed nodes to `permanent=1`
- Immune to decay cycles
- Represents stable, foundational knowledge

**No clustering, no hotspot creation, no hierarchy maintenance** - structure emerges through cross-linking and natural decay.

## Scalability Analysis

| Node Count | Flat Search (O(N)) | sqlite-vec + BFS (O(log N + K)) | Speedup |
|------------|-------------------|---------------------------|---------|
| 100        | 100 comparisons   | ~10 comparisons           | 10x     |
| 1,000      | 1,000 comparisons | ~15 comparisons          | 67x    |
| 10,000     | 10,000 comparisons| ~20 comparisons          | 500x    |
| 100,000    | 100,000 comparisons| ~25 comparisons         | 4,000x  |

BFS search achieves sub-linear complexity through:
- sqlite-vec O(log N) seed selection (not O(N) brute force)
- Limited BFS traversal depth (3 hops maximum)
- Picks-per-hop constraint prevents combinatorial explosion
- Organic connectivity provides efficient pathways to relevant content

## Implementation Details

### Core Modules

1. **`core/embeddings.py`**: sqlite-vec integration, embedding generation, similarity search
2. **`core/retrieval.py`**: BFS traversal implementation  
3. **`core/context.py`**: Context generation orchestration
4. **`core/session.py`**: Session lifecycle with BFS retrieval
5. **`core/sleep.py`**: Sleep cycles with cross-linking and decay
6. **`core/traversal.py`**: Graph traversal operations (why/how/audit)

### Database Schema Utilization

Uses flat node/edge schema with vector acceleration:
- **`thought_nodes`**: Content, metadata, decay status
- **`derivation_edges`**: Parent-child relationships with weights
- **`embeddings`**: BLOB storage for backward compatibility 
- **`vec_embeddings`**: sqlite-vec virtual table for O(log N) search

### Command-Line Interface

```bash
# Context retrieval via recursive BFS
python3 scripts/cashew_context.py context --hints "keywords"

# System statistics
python3 scripts/cashew_context.py stats

# Sleep cycle with cross-linking
python3 scripts/cashew_context.py sleep

# Think cycle for insight generation
python3 scripts/cashew_context.py think
```

## Performance Characteristics

### Search Performance
- **Cold start**: ~150-300ms (embedding computation + BFS traversal)
- **Warm retrieval**: ~50-100ms (BFS traversal with cached embeddings)
- **Memory usage**: O(active_set) not O(all_nodes), typically visits <5% of graph

### Graph Maintenance  
- **Sleep cycle frequency**: Daily or triggered by growth thresholds
- **Cross-linking computation**: O(N²) for similarity matrix, amortized across many sessions
- **Decay operations**: O(N) for fitness scoring, infrequent execution
- **Deduplication**: O(N log N) for similarity clustering

### Storage Efficiency
- **Database size**: ~16MB for 3,064 nodes (as of April 2026, author's personal graph)
- **Embedding vectors**: Local sentence-transformers, no API dependencies
- **sqlite-vec overhead**: Minimal additional storage for accelerated search
- **No synthetic nodes**: 100% of storage used for real knowledge

## Testing & Validation

### Automated Test Coverage
The test suite (`tests/`) covers:
- **`test_retrieval.py`**: BFS traversal correctness and performance
- **`test_embeddings.py`**: sqlite-vec integration and fallback behavior
- **`test_sleep.py`**: Sleep cycle maintenance operations
- **`test_traversal.py`**: Graph traversal correctness (why/how/audit)

### Quality Metrics
- **Recall@5**: Percentage of relevant results in top 5 hits
- **Query latency**: Response time for context generation  
- **Graph connectivity**: Distribution of node degrees and path lengths
- **Storage efficiency**: Ratio of real knowledge to synthetic overhead

### Integration Testing
```bash
# Run complete system test
cd cashew
KMP_DUPLICATE_LIB_OK=TRUE python3 -m pytest tests/ -v

# Test retrieval quality
python3 scripts/cashew_context.py context --hints "engineering work"
python3 scripts/cashew_context.py context --hints "project decisions"  
python3 scripts/cashew_context.py context --hints "system architecture"
```

## Think Cycle Integration

### Cross-Domain Synthesis
The think cycle system (`core/session.py`) uses BFS retrieval for pattern detection:

1. **Diverse context assembly**: BFS retrieval finds connections across knowledge domains
2. **Pattern recognition**: LLM analyzes retrieved context for insights and tensions
3. **Novel hypothesis generation**: Creates new understanding from existing knowledge
4. **Derivation linking**: New insights link to source nodes as parents
5. **Validation**: Human feedback validates genuinely novel vs. derivative insights

### Think Cycle Results
- **Cross-domain insights**: Patterns that span technical, personal, and strategic domains
- **Validated novelty**: Human confirmation that insights are genuinely new, not mere summaries
- **Organic knowledge growth**: Graph evolves through autonomous reasoning cycles

## Domain Separation

### Multi-Domain Architecture  
The graph supports multiple knowledge domains within a single database:

- **Domain field**: Each node tagged with domain (user, ai, project-specific, etc.)
- **Cross-domain edges**: Derivation links can connect across domain boundaries
- **Unified retrieval**: BFS traversal finds relevant knowledge regardless of domain
- **Domain-filtered queries**: Optional domain scoping for privacy or focus

### Cross-Domain Bridge Building
Sleep cycles create connections spanning domains:
- **Cross-linking**: Semantic similarity connects related concepts across boundaries
- **Think cycle synthesis**: Autonomous generation of insights bridging domains
- **Emergent interdisciplinarity**: No hardcoded cross-domain rules

## Privacy and Access Control

### Tag-Based Filtering
- Nodes can be tagged `vault:private` to exclude from group contexts
- `--exclude-tags vault:private` filters private content during retrieval
- Default extraction marks content private; declassification via `scripts/declassify.py`

### Domain Boundaries
- User vs AI domain separation for attribution and trust
- Configurable domain names in `config.yaml`
- Retrieval can be scoped to specific domains when needed

## Future Enhancements

### Short-term
1. **Query optimization**: Cache embeddings, batch similarity computations
2. **Adaptive parameters**: Dynamic picks_per_hop based on graph density  
3. **Performance benchmarking**: Systematic comparison against baseline retrieval methods
4. **Advanced filtering**: Time-based, access-weighted retrieval

### Medium-term  
1. **Incremental indexing**: Update sqlite-vec index without full rebuild
2. **Multi-modal embeddings**: Support for images, audio in thought nodes
3. **Federation**: Multiple agents sharing graph knowledge
4. **Explainable retrieval**: Human-readable explanations for result selection

### Long-term
1. **Distributed graphs**: Multiple databases with synchronized cross-linking
2. **Real-time collaboration**: Live multi-agent graph updates
3. **Transfer learning**: Retrieval patterns optimized for specific use cases
4. **Adaptive architecture**: System automatically tunes parameters for workload

## Key Architectural Insights

### Organic Over Synthetic
Unlike systems with predetermined hierarchies, cashew's connectivity emerges from actual knowledge relationships discovered through cross-linking. This creates more semantically meaningful organization.

### BFS Over DFS
Breadth-first search explores diverse neighborhoods before going deep, finding connections across knowledge domains rather than drilling down narrow paths.

### sqlite-vec Scaling
Vector search acceleration within SQLite eliminates external dependencies while providing sub-linear query performance as the graph grows.

### Cross-linking as Infrastructure
Sleep cycle cross-linking builds the pathways that BFS traversal exploits, creating a positive feedback loop between structure-building and retrieval efficiency.

### Simplicity Through Emergence
Complex retrieval behavior emerges from simple rules (similarity-based cross-linking + BFS traversal) without requiring complex hierarchical maintenance algorithms.