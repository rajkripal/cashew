# Migration Guide for Cashew

This guide covers how to migrate from various data states to a fully-organized Cashew thought graph, and how to onboard new users to the system.

## Overview

Cashew organizes knowledge into a flat graph with:
- **Content nodes**: Individual thoughts, decisions, insights with organic connectivity
- **Cross-linking**: Sleep cycles build semantic relationships between related nodes
- **sqlite-vec acceleration**: O(log N) vector search within the same SQLite file
- **BFS retrieval**: Recursive traversal through organic graph structure

## Migration Scenarios

### 1. Existing User Migration: From Flat Files to Organized Graph

**Scenario**: You have markdown files, notes, or an unorganized collection of thoughts that need to be structured into the cashew graph.

#### Step 1: Extract Initial Content
```bash
# Extract from a conversation or content file
cd /path/to/cashew
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py extract --input /path/to/content.md
```

This creates nodes in the graph with basic connectivity.

#### Step 2: Run Sleep Cycles to Build Connectivity
```bash
# Run sleep cycle for cross-linking and organization
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py sleep
```

The sleep cycle will:
- Create cross-links between semantically related nodes
- Decay low-quality or stale content
- Deduplicate near-identical nodes
- Build the organic connectivity that BFS retrieval exploits

#### Step 3: Verify Coverage
```bash
# Check system stats
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py stats

# Run test queries to validate recall
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py context --hints "your important topics here"
```

#### Step 4: Iterative Refinement
- Review the node structure: check graph statistics for connectivity metrics
- Extract additional content and repeat sleep cycles
- Use think cycles to generate cross-domain insights

### 2. New User Onboarding: Building from Scratch

**Scenario**: Brand new user with no existing data who wants to start using cashew.

#### Step 1: Initial Seed Content
Start with a few representative thoughts or decisions to seed the system:

```bash
# Create initial extraction file
cat > /tmp/seed-content.md << 'EOF'
I'm starting to use cashew to organize my thoughts. Key areas I want to track:

**Work/Career**: I'm pursuing a promotion to Senior Engineer. Need to demonstrate technical leadership and project ownership.

**Personal Projects**: Working on a blog about technology and philosophy. Want to publish 2-3 posts per month.

**Learning**: Studying distributed systems and machine learning. Goal is to become more well-rounded.

**Health**: Trying to establish a consistent exercise routine. Running 3x per week currently.
EOF

# Extract the seed content
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py extract --input /tmp/seed-content.md
```

#### Step 2: Let the Graph Grow Organically
As you use the system, cross-links will naturally form through sleep cycles:

```bash
# Add thoughts over time through conversations
echo "Had a great 1:1 with my manager today. She mentioned I should take ownership of the API redesign project." > /tmp/work-update.md
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py extract --input /tmp/work-update.md

# Run think cycles to generate insights
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py think
```

#### Step 3: Periodic Maintenance
Run weekly/monthly maintenance to keep the graph organized:

```bash
# Weekly sleep cycle for cross-linking
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py sleep
```

### 3. Bulk Import Migration

**Scenario**: Importing a large amount of existing content (journals, notes, documents).

#### Preparation
1. **Chunk your content**: Break large files into topic-focused chunks
2. **Clean the text**: Remove metadata, formatting that doesn't add meaning
3. **Create domain boundaries**: Separate personal vs work vs technical content

#### Batch Processing
```bash
# Process multiple files
for file in /path/to/content/*.md; do
    echo "Processing $file..."
    KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py extract --input "$file" --session-id "bulk_import_$(basename $file .md)"
done

# Run multiple sleep cycles to build connectivity
for i in {1..3}; do
    echo "Sleep cycle $i..."
    KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py sleep
done
```

## sqlite-vec Migration

### Migrating from Legacy Embedding Storage

If you have an existing cashew database without sqlite-vec support, migrate with:

```bash
# Backup existing database
cp data/graph.db data/graph_backup_$(date +%Y%m%d).db

# Run sqlite-vec migration
KMP_DUPLICATE_LIB_OK=TRUE python3 -c "
from core.embeddings import backfill_vec_index
print('Starting sqlite-vec migration...')
backfill_vec_index('data/graph.db')
print('Migration complete!')
"

# Verify migration
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py stats
```

This will:
- Create the `vec_embeddings` virtual table
- Copy all existing embeddings from the `embeddings` table
- Set up O(log N) vector search acceleration
- Maintain backward compatibility with the BLOB-based embeddings table

### Post-Migration Verification

```bash
# Test vector search performance
time KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py context --hints "test query"

# Check database integrity
sqlite3 data/graph.db "PRAGMA integrity_check;"

# Verify both embedding tables are populated
sqlite3 data/graph.db "SELECT COUNT(*) FROM embeddings; SELECT COUNT(*) FROM vec_embeddings;"
```

## Graph Connectivity Patterns

### What Good Connectivity Looks Like
A healthy cashew graph should exhibit:
- **Power law degree distribution**: Few high-connectivity nodes, many low-connectivity nodes
- **Small world properties**: Short paths between any two nodes
- **Domain boundaries**: Clear separation between different knowledge areas
- **Cross-domain bridges**: Occasional connections spanning domains

### Measuring Graph Health
```bash
# Check basic connectivity stats
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py stats

# Look for key metrics:
# - Average degree: 2-5 edges per node is healthy
# - Connected components: Should be 1 (fully connected graph)
# - Diameter: Should be small (3-6 hops max between any nodes)
```

## Post-Migration Verification

### 1. Retrieval Quality Check
Verify that your important thoughts are findable:

```bash
# Test key topic retrieval with BFS
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py context --hints "career promotion goals"
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py context --hints "blog writing ideas"
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py context --hints "health fitness routine"

# Check overall system health
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py stats
```

### 2. Cross-Domain Connectivity
Test whether BFS can find connections across knowledge domains:

```bash
# Query for concepts that should span domains
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py context --hints "decisions made this month"
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py context --hints "patterns across work and personal"
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py context --hints "lessons learned from projects"
```

### 3. Performance Validation
Ensure sqlite-vec is providing the expected performance benefits:

```bash
# Test query performance (should be sub-second for 1000s of nodes)
time KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py context --hints "complex query with multiple terms"

# Check vector index status
sqlite3 data/graph.db "SELECT COUNT(*) FROM vec_embeddings;"
```

### 4. Quality Metrics
Good migration results should show:
- **High connectivity**: Average degree 2-5 per node
- **Efficient retrieval**: Query response under 500ms for most graphs
- **Cross-domain synthesis**: BFS finds relevant nodes across domains
- **No isolated nodes**: Every node reachable within 3-6 hops
- **sqlite-vec acceleration**: O(log N) seed selection working

## Common Migration Issues

### Issue: Poor Query Recall
**Cause**: Insufficient cross-linking between related concepts
**Solution**: Run additional sleep cycles to build connectivity

```bash
# Run multiple sleep cycles
for i in {1..5}; do
    echo "Sleep cycle $i..."
    KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py sleep
    sleep 1
done
```

### Issue: Slow Query Performance
**Cause**: sqlite-vec not properly installed or fallback to brute force
**Solution**: Verify sqlite-vec installation and re-run migration

```bash
# Check if sqlite-vec is available
python3 -c "
import sqlite3
db = sqlite3.connect(':memory:')
try:
    db.execute('CREATE VIRTUAL TABLE test USING vec0(embedding float[384])')
    print('sqlite-vec is working!')
except Exception as e:
    print(f'sqlite-vec error: {e}')
"

# Re-run backfill if needed
KMP_DUPLICATE_LIB_OK=TRUE python3 -c "from core.embeddings import backfill_vec_index; backfill_vec_index('data/graph.db')"
```

### Issue: Fragmented Graph
**Cause**: Content extracted without sufficient semantic overlap
**Solution**: Add bridging content and run cross-linking cycles

```bash
# Add some general bridging thoughts
echo "Looking at my overall priorities and how different areas of life connect..." > /tmp/bridge.md
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py extract --input /tmp/bridge.md

# Run sleep cycle to create cross-domain links
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py sleep
```

## Maintenance Schedule

### Daily (if actively adding content)
- Quick context queries to verify new thoughts are being captured
- Check query performance with `time` command

### Weekly
```bash
# Sleep cycle for cross-linking
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py sleep

# Think cycle for insights
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py think
```

### Monthly
```bash
# Full system health check
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py stats

# Dashboard export for visualization
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/export_dashboard.py data/graph.db dashboard/data/graph.json --html "My Thought Graph"

# Database maintenance
sqlite3 data/graph.db "VACUUM; ANALYZE;"
```

## Advanced Migration Techniques

### Cross-System Import
If migrating from other PKM systems (Obsidian, Roam, etc.):

1. **Extract atomic thoughts**: Break complex notes into single-concept nodes
2. **Preserve connections**: Important relationships become derivation edges
3. **Use domain tags**: Leverage existing tag structure for domain assignment
4. **Gradual switchover**: Run both systems in parallel during transition

### Multi-User Migration
For teams or families sharing knowledge:

1. **Domain separation**: Use domain field to separate personal vs shared knowledge
2. **Staged rollout**: Start with one person's content, then add others
3. **Privacy boundaries**: Use vault:private tags to control access
4. **Merge conflicts**: Use timestamp and access_count to resolve overlaps

## Troubleshooting

### Database Issues
```bash
# Check database integrity
sqlite3 data/graph.db "PRAGMA integrity_check;"

# Check sqlite-vec table status
sqlite3 data/graph.db "SELECT name FROM sqlite_master WHERE type='table' AND name='vec_embeddings';"

# Rebuild embeddings if corrupted
KMP_DUPLICATE_LIB_OK=TRUE python3 -c "from core.embeddings import embed_nodes; embed_nodes('data/graph.db')"
```

### Performance Issues
```bash
# Check database size and vacuum
ls -lh data/graph.db
sqlite3 data/graph.db "VACUUM; ANALYZE;"

# Verify sqlite-vec is being used (not fallback)
grep -i "fallback\|brute" logs/cashew.log
```

### Migration Rollback
If migration goes wrong, restore from backup:
```bash
# Always backup before major migrations
cp data/graph.db data/graph_backup_$(date +%Y%m%d).db

# Restore if needed
cp data/graph_backup_YYYYMMDD.db data/graph.db
```

## Success Metrics

A successful migration should achieve:
- **Retrieval accuracy**: 80%+ of queries return relevant results
- **Graph connectivity**: Average degree 2-5 edges per node
- **Query performance**: <500ms response time for most queries
- **Cross-domain synthesis**: BFS finds connections across knowledge areas
- **Growth sustainability**: System handles new content without degradation
- **sqlite-vec acceleration**: O(log N) seed selection working correctly

## Next Steps

After successful migration:
1. **Integrate with workflow**: Set up automatic extraction from conversations
2. **Dashboard monitoring**: Regular visualization of growth and connectivity patterns
3. **Think cycle automation**: Schedule regular insight generation cycles
4. **Performance monitoring**: Track query latency and graph health metrics

For questions or issues, check the troubleshooting section or review the system design documentation in `DESIGN.md`.