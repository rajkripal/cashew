# Migration Guide for Cashew

This guide covers how to migrate from various data states to a fully-organized Cashew thought graph, and how to onboard new users to the system.

## Overview

Cashew organizes knowledge into a hierarchical graph with:
- **Hotspots**: Summary nodes that act as cluster centers (like "Blog 2", "Cashew Project", "Faith Journey")
- **Content nodes**: Individual thoughts, decisions, insights that belong to hotspots
- **The Inbox Pattern**: Uncategorized nodes that await classification during sleep cycles
- **Complete Coverage**: Every node belongs somewhere in the hierarchy

## Migration Scenarios

### 1. Existing User Migration: From Flat Files to Organized Graph

**Scenario**: You have markdown files, notes, or an unorganized collection of thoughts that need to be structured into the cashew graph.

#### Step 1: Extract Initial Content
```bash
# Extract from a conversation or content file
cd /path/to/cashew
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py extract --input /path/to/content.md
```

This creates nodes in the graph but they'll likely end up in the inbox initially.

#### Step 2: Run Sleep Cycles to Organize
```bash
# Run inbox triage to move nodes to proper clusters
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/triage_inbox.py --threshold 0.20

# Run full hierarchy evolution for deeper organization
KMP_DUPLICATE_LIB_OK=TRUE python3 -c "
from core.hierarchy_evolution import run_hierarchy_evolution_cycle
from integration.openclaw import _create_anthropic_model_fn

model_fn = _create_anthropic_model_fn()
result = run_hierarchy_evolution_cycle('data/graph.db', model_fn)
print('Evolution result:', result)
"
```

#### Step 3: Verify Coverage
```bash
# Check system stats
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py system-stats

# Run test queries to validate recall
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py context --hints "your important topics here"
```

#### Step 4: Iterative Refinement
- Review the hotspots created: `python3 scripts/cashew_context.py hotspot list`
- Manually adjust clusters if needed: `python3 scripts/cashew_context.py hotspot update --id <hotspot_id> --content "better summary"`
- Extract additional content and repeat sleep cycles

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

#### Step 2: Let the Tree Grow Organically
As you use the system, nodes will naturally cluster:

```bash
# Add thoughts over time through conversations
echo "Had a great 1:1 with my manager today. She mentioned I should take ownership of the API redesign project." > /tmp/work-update.md
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py extract --input /tmp/work-update.md

# Run think cycles to generate insights
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py think --domain bunny
```

#### Step 3: Periodic Maintenance
Run weekly/monthly maintenance to keep the tree organized:

```bash
# Weekly inbox cleanup
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/triage_inbox.py --threshold 0.25

# Monthly full evolution
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py complete-sleep
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

# Run aggressive triage after bulk import
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/triage_inbox.py --threshold 0.15

# Run multiple evolution cycles to settle the hierarchy
for i in {1..3}; do
    echo "Evolution cycle $i..."
    KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py complete-sleep
done
```

## The Inbox Pattern

### What is the Inbox?
The inbox is a special hotspot (usually ID `07621e89a08e`) that catches:
- Newly extracted thoughts that don't have an obvious cluster
- Thoughts that are too general or cross multiple domains
- Content that needs human review before classification

### Why it Exists
1. **Graceful degradation**: System never fails to store new thoughts
2. **Incremental organization**: Allows processing large amounts of content without perfect upfront categorization
3. **Discovery mechanism**: Inbox review reveals patterns that suggest new hotspots

### How Sleep Cycles Triage It
Sleep cycles automatically move nodes from inbox to appropriate clusters when:
- **Similarity match**: Node has >0.2-0.35 similarity to existing cluster content
- **Domain alignment**: Node fits clearly in a domain (work, personal, technical)
- **Confidence threshold**: System is confident about the placement

```bash
# Manual inbox review
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py hotspot show --id 07621e89a08e

# Targeted triage with different thresholds
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/triage_inbox.py --threshold 0.30 --dry-run  # Conservative
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/triage_inbox.py --threshold 0.20          # Moderate
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/triage_inbox.py --threshold 0.15          # Aggressive
```

### When to Manually Intervene
- **Inbox stays above 100 nodes**: May need new hotspots for emerging themes
- **Same nodes keep getting misclassified**: Similarity model needs adjustment
- **Important thoughts stay in inbox**: May need manual hotspot creation

## Post-Migration Verification

### 1. Coverage Check
Verify that your important thoughts are findable:

```bash
# Test key topic retrieval
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py context --hints "career promotion goals"
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py context --hints "blog writing ideas"
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py context --hints "health fitness routine"

# Check overall system health
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py system-stats
```

### 2. Recall Validation
Test whether important decisions and insights surface correctly:

```bash
# Query for specific types of content
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py context --hints "decisions made this month"
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py context --hints "insights about work culture"
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py context --hints "lessons learned from projects"
```

### 3. Hierarchy Review
Examine the cluster structure:

```bash
# List all hotspots by domain
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py hotspot list

# Review inbox size
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py hotspot show --id 07621e89a08e

# Check for orphaned or duplicate clusters
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py stats
```

### 4. Quality Metrics
Good migration results should show:
- **Inbox under 100 nodes** (typically 5-15% of total)
- **10-30 hotspots** for 500-1000 nodes
- **Average cluster size 15-50 nodes**
- **Query recall >80%** for important topics
- **No orphaned nodes** (every node belongs to a hotspot)

## Common Migration Issues

### Issue: Everything Goes to Inbox
**Cause**: No existing hotspots to anchor new content
**Solution**: Create seed hotspots manually before bulk import

```bash
# Create anchor hotspots for your main domains
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py hotspot create \
    --content "Work and career development activities" \
    --status "active" --domain bunny --tags "career,work"

KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py hotspot create \
    --content "Personal projects and creative pursuits" \
    --status "active" --domain bunny --tags "personal,projects"
```

### Issue: Clusters Too Large or Small
**Cause**: Similarity thresholds not calibrated for your content
**Solution**: Adjust triage thresholds and run split/merge cycles

```bash
# Split large clusters
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py complete-sleep

# Merge small clusters (handled automatically in evolution)
```

### Issue: Poor Query Recall
**Cause**: Important content stuck in wrong clusters or inbox
**Solution**: Manual hotspot curation and re-triaging

```bash
# Find and move misplaced content
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py context --hints "missing content keywords"
# Then manually update hotspot assignments
```

## Maintenance Schedule

### Daily (if actively adding content)
- Quick context queries to verify new thoughts are being captured
- Manual inbox review if it grows >20% of total nodes

### Weekly
```bash
# Inbox triage
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/triage_inbox.py --threshold 0.25

# Think cycle for insights
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py think
```

### Monthly
```bash
# Full system evolution
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py complete-sleep

# System health check
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/cashew_context.py system-stats

# Dashboard export for visualization
KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/export_dashboard.py data/graph.db dashboard/data/graph.json --html "My Thought Graph"
```

## Advanced Migration Techniques

### Cross-System Import
If migrating from other PKM systems (Obsidian, Roam, etc.):

1. **Extract atomic thoughts**: Break complex notes into single-concept nodes
2. **Preserve connections**: Important relationships become derivation edges
3. **Migrate tags to domains**: Use your existing tag structure to inform domain assignment
4. **Gradual switchover**: Run both systems in parallel during transition

### Multi-User Migration
For teams or families sharing knowledge:

1. **Domain separation**: Use domain field to separate personal vs shared knowledge
2. **Staged rollout**: Start with one person's content, then add others
3. **Access patterns**: Configure retrieval to respect privacy boundaries
4. **Merge conflicts**: Use timestamp and confidence to resolve overlaps

## Troubleshooting

### Database Issues
```bash
# Check database integrity
sqlite3 data/graph.db "PRAGMA integrity_check;"

# Rebuild embeddings if corrupted
KMP_DUPLICATE_LIB_OK=TRUE python3 -c "from core.embeddings import embed_nodes; embed_nodes('data/graph.db')"
```

### Performance Issues
```bash
# Check database size and vacuum
ls -lh data/graph.db
sqlite3 data/graph.db "VACUUM;"

# Check for missing indexes
sqlite3 data/graph.db ".indices"
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
- **Organization efficiency**: <20% of nodes remain in inbox long-term  
- **Growth sustainability**: System handles new content without degradation
- **Insight generation**: Think cycles produce valuable connections
- **Maintenance burden**: <30min/week to keep system healthy

## Next Steps

After successful migration:
1. **Integrate with workflow**: Set up automatic extraction from conversations
2. **Dashboard monitoring**: Regular visualization of growth and patterns
3. **Sharing**: Export contexts for collaboration or backup
4. **Extension**: Add domain-specific schemas or custom node types

For questions or issues, check the troubleshooting section or review the system design documentation in `DESIGN.md`.