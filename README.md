# Cashew 🥜

**Persistent thought-graph memory for AI agents.** Cashew provides context generation, knowledge extraction, and autonomous think cycles to maintain coherent memory across sessions and compactions.

Never lose context again. Cashew builds a hierarchical graph of interconnected knowledge nodes that grows smarter over time through autonomous reasoning cycles.

## Quick Start

```bash
# Clone and install
git clone <your-repo>
cd cashew
pip install -e .

# Initialize your brain
cashew init

# Verify setup
cashew context --hints "test"

# Start using in conversations
echo "I prefer TypeScript over JavaScript for complex projects" | cashew extract --input -
```

## What is Cashew?

Cashew is a **pure infrastructure** thought-graph memory engine that gives AI agents persistent, hierarchical memory across sessions. This is engine-only: graph database, retrieval, extraction, think cycles, and sleep cycles. No opinionated identity layer or philosophical content.

Unlike simple RAG systems, Cashew:

- **Builds knowledge graphs**: Facts, insights, and decisions become interconnected nodes
- **Learns autonomously**: Think cycles consolidate and extend knowledge without human input (when LLM access provided)
- **Provides smart context**: Retrieves relevant information using semantic similarity and graph traversal
- **Handles scale**: Efficient clustering and indexing keeps performance high as knowledge grows
- **Integrates seamlessly**: Drop-in enhancement for OpenClaw agents and other AI systems

## LLM Architecture

**Cashew does NOT call LLMs directly.** It follows a strict separation:

- **Cashew (Brain)**: Storage, retrieval, clustering, structure
- **Orchestrator (Processor)**: Provides LLM access via `model_fn` parameters

### Feature Availability:
- **CLI usage**: Heuristic extraction, graph operations, context retrieval (no LLM needed)
- **OpenClaw crons**: Full LLM features (hotspot summaries, think cycles, smart extraction) via orchestrator
- **Custom integrations**: Any system can provide LLM access through the `model_fn` parameter pattern

## Installation

### Requirements
- Python 3.10+
- 2GB RAM (for embedding model)
- 100MB disk space (grows with your knowledge graph)

### Install

```bash
pip install cashew
```

Or for development:
```bash
git clone <repo>
cd cashew
pip install -e .
```

## Setup

### 1. Initialize
```bash
cashew init
```

This creates:
- `config.yaml` - Configuration (edit this!)
- `data/graph.db` - Your knowledge graph database
- `logs/` - Application logs
- `models/` - Downloaded embedding models

### 2. Configure

Edit `config.yaml` to customize:

```yaml
# Database and storage
database:
  path: "./data/graph.db"
  backup_dir: "./data/backups"

# Domain names (replaces hardcoded 'raj'/'bunny')
domains:
  user: "user"    # Things the human said/decided
  ai: "ai"        # AI analysis and suggestions

# Performance tuning
performance:
  token_budget: 2000      # Context size limit
  top_k_results: 10       # Max results to retrieve
  similarity_threshold: 0.3  # Minimum relevance
  
# Model configuration  
models:
  embedding:
    name: "all-MiniLM-L6-v2"
  # LLM config removed - cashew doesn't call LLMs directly
  # LLM access provided by orchestrator via model_fn parameters
```

### 3. Verify

```bash
cashew context --hints "test"
# Should show: "No relevant context found" (expected for empty brain)
```

## Core Usage

### Query for Context
Before answering questions, query your brain:
```bash
cashew context --hints "project status work priorities"
```

This returns relevant knowledge to inform your response.

### Extract Knowledge
After important conversations, extract insights:
```bash
# From file (uses heuristic extraction - no LLM)
cashew extract --input conversation.txt

# From stdin  
echo "User prefers TypeScript for type safety" | cashew extract --input -
```

**Note**: CLI extraction uses heuristic methods. For smart LLM-powered extraction, use OpenClaw cron jobs which provide the necessary model functions.

### Think Cycles
**Note**: Think cycles require LLM access. Use through OpenClaw cron jobs for full functionality.
```bash
cashew think  # Limited functionality without LLM
```

### Sleep Cycles  
Deep reorganization and clustering (structural operations work without LLM):
```bash
cashew sleep  # Clustering works, hotspot summaries use fallbacks
```

### Statistics
Check your brain's health:
```bash
cashew stats
```

## Cron Automation

Install automated maintenance:

```bash
cashew install-crons
```

This generates `cashew-crons.yaml` with OpenClaw cron job configurations:

- **Brain extraction** (every 2hrs) - Reads session history, extracts to brain
- **Think cycle** (2x daily) - Consolidation and insight generation  
- **Sleep cycle** (daily) - Deep reorganization and clustering
- **Backup** (daily) - Database backup and health checks

Copy the generated jobs to your OpenClaw config file (`~/.openclaw/config/config.yaml`).

## Configuration Reference

### Database Configuration
```yaml
database:
  path: "./data/graph.db"           # Main database location
  backup_dir: "./data/backups"      # Backup storage
  auto_backup: true                 # Auto-backup before major operations
```

### Domain Configuration
```yaml
domains:
  default: "general"                # Fallback domain
  user: "user"                      # Human user's knowledge
  ai: "ai"                         # AI-generated insights
  classifications:                  # Additional categories
    - personal
    - work  
    - projects
    - learning
```

### Performance Tuning
```yaml
performance:
  token_budget: 2000               # Max tokens for context generation
  top_k_results: 10                # Max results per query
  walk_depth: 2                    # Graph traversal depth
  similarity_threshold: 0.3         # Minimum relevance score
  novelty_threshold: 0.82          # Prevent near-duplicates
  clustering_eps: 0.35             # DBSCAN clustering sensitivity
  think_cycle_nodes: 5             # Max nodes per think cycle
```

### Model Configuration
```yaml
models:
  embedding:
    name: "all-MiniLM-L6-v2"       # Sentence transformer model
    provider: "sentence-transformers"
    cache_dir: "./models"           # Local model cache
    
  llm:
    provider: "anthropic"           # or "openai"
    model: "claude-sonnet-4-20250514"
    api_key_env: "ANTHROPIC_API_KEY"  # Environment variable name
```

### Integration Settings
```yaml
integration:
  openclaw:
    # Path to OpenClaw auth profiles
    auth_profile_path: "${HOME}/.openclaw/agents/${OPENCLAW_AGENT:-main}/agent/auth-profiles.json"
    # OpenClaw workspace
    workspace_path: "${HOME}/.openclaw/workspace"
```

## API Integration

### Python API

```python
from cashew.core.context import ContextRetriever
from cashew.integration.openclaw import extract_from_conversation

# Query context
retriever = ContextRetriever("./data/graph.db")
context = retriever.generate_context_from_hints(["work", "projects"])

# Extract knowledge
result = extract_from_conversation(
    db_path="./data/graph.db",
    conversation="User decided to use React for the frontend",
    session_id="session_123"
)
```

### CLI Integration

Perfect for shell scripts and automation:
```bash
# Context for current task
CONTEXT=$(cashew context --hints "$(echo $USER_INPUT | head -c 100)")

# Extract after completion  
echo "$CONVERSATION_LOG" | cashew extract --input -
```

## Advanced Usage

### Custom Domains

Add your own domain classifications in `config.yaml`:

```yaml
domains:
  classifications:
    - work
    - personal  
    - learning
    - projects
    - research
```

### Performance Optimization

For large graphs (10k+ nodes):

```yaml
performance:
  clustering_eps: 0.3        # Tighter clustering
  similarity_threshold: 0.4  # Higher relevance bar
  token_budget: 1500         # Smaller context window
```

For real-time applications:
```yaml
performance:
  top_k_results: 5          # Fewer results
  walk_depth: 1             # Shallow traversal
  novelty_threshold: 0.9    # Aggressive deduplication
```

### Database Migration

Moving to a new system:
```bash
# Backup current database
cashew backup

# Copy to new system
cp data/graph.db /new/location/

# Update config
vim config.yaml  # Update database.path

# Verify
cashew stats
```

## Troubleshooting

### Empty Context Results
- Check database exists: `ls -la data/graph.db`
- Verify embeddings: `cashew stats`
- Try broader hints: `cashew context --hints "general"`

### Extraction Failing
- Check API key: `echo $ANTHROPIC_API_KEY`
- Test model access: Try simple extraction
- Review logs: `tail -f logs/cashew.log`

### Performance Issues
- Reduce token budget in config.yaml
- Lower top_k_results
- Run `cashew sleep` to reorganize clusters

### Database Corruption
- Restore from backup: `cp data/backups/latest.db data/graph.db`
- Reinitialize: `rm data/graph.db && cashew init`

## Development

### Running Tests
```bash
pip install -e .[dev]
pytest
```

### Adding Features
1. Core logic goes in `core/`
2. Integration code in `integration/`
3. CLI commands in `cashew_cli.py`
4. Update tests in `tests/`

### Contributing
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/jugaad-lab/cashew/issues)
- **Documentation**: See `docs/` directory
- **Examples**: See `examples/` directory

---

**Built for the OpenClaw ecosystem** - Cashew integrates seamlessly with OpenClaw agents to provide persistent memory across sessions, compactions, and system restarts.