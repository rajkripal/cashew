---
name: cashew
description: Persistent thought-graph memory for OpenClaw agents. Query context before answering substantive questions, extract knowledge during conversations, run think/sleep cycles for consolidation. Triggers on session start, knowledge-worthy conversations, heartbeats, and any question about prior context.
---

# cashew — Persistent Thought-Graph Memory

## What This Does
Installs cashew as your brain. Every conversation gets extracted into a knowledge graph. Every new session starts with relevant context retrieved from the graph. Think cycles run autonomously to find cross-domain connections.

## Installation

### Step 1: Install cashew
```bash
pip install cashew-brain  # or: pip install -e /path/to/cashew
```

### Step 2: Initialize your brain
```bash
cashew --db ~/.cashew/graph.db init
```

### Step 3: Verify
```bash
cashew --db ~/.cashew/graph.db stats
```

### Step 4: Enable pre-compaction extraction
Apply the memoryFlush config so knowledge is extracted to cashew before every compaction:
```
gateway config.patch with:
{
  "agents": {
    "defaults": {
      "compaction": {
        "memoryFlush": {
          "enabled": true,
          "softThresholdTokens": 4000,
          "systemPrompt": "Session nearing compaction. Extract all important knowledge to cashew before context is lost.",
          "prompt": "Compaction is imminent. Write key facts, decisions, insights, and commitments from this conversation to /tmp/cashew-extract-input.md, then run: cashew --db ~/.cashew/graph.db extract --input /tmp/cashew-extract-input.md -- Reply NO_REPLY when done."
        }
      }
    }
  }
}
```

## CLI Reference

**Note:** `--db` is a global flag and must go BEFORE the subcommand.

### Retrieve context
```bash
cashew --db <path> context --hints "<space-separated keywords>"
cashew --db <path> context --hints "<hints>" --tags "finance,career"        # scope to specific domains
cashew --db <path> context --hints "<hints>" --exclude-tags "vault:private"  # exclude private nodes
```
Returns three layers: graph overview, recent activity, and hint-matched nodes.

### Extract from conversation text
```bash
# Write conversation highlights to a file, then:
cashew --db <path> extract --input /tmp/cashew-extract-input.md
cashew --db <path> extract --input /tmp/file.md --tags vault:private  # apply tags to all extracted nodes
```

### Extract via prepare→ingest pattern (for cron/automation)
```bash
# Step 1: Get extraction prompt
cashew --db <path> extract --prepare-only --input /tmp/conversation.md

# Step 2: LLM generates JSON (you do this part)

# Step 3: Ingest the JSON
cashew --db <path> extract --ingest /tmp/results.json
cashew --db <path> extract --ingest /tmp/results.json --tags vault:private  # with tags
```

**Ingest JSON format:**
```json
{
  "insights": [
    {
      "content": "specific knowledge statement",
      "type": "fact|observation|insight|decision|belief",
      "confidence": 0.7,
      "domain": "raj|bunny"
    }
  ]
}
```

### Think cycle
```bash
cashew --db <path> think                          # random cluster
cashew --db <path> think --domain "career"         # focused domain
cashew --db <path> think --prepare-only            # get cluster for external LLM
cashew --db <path> think --ingest /tmp/results.json # ingest think results
```

### Sleep cycle
```bash
cashew --db <path> sleep         # full consolidation: cross-linking, decay, dedup
cashew --db <path> sleep --debug # with diagnostics
```

### Ingest from sources
```bash
cashew --db <path> ingest obsidian /path/to/vault     # Obsidian vault (frontmatter, wikilink edges, .obsidianignore)
cashew --db <path> ingest sessions /path/to/sessions/  # OpenClaw session JSONL (incremental)
cashew --db <path> ingest markdown /path/to/notes/     # Markdown directory (.cashewignore)
cashew --db <path> ingest --list                       # Show available extractors
cashew --db <path> ingest obsidian /path --no-llm      # Skip LLM, use paragraph splitting fallback
```
All extractors checkpoint automatically — re-running only processes new/modified files.

### Other commands
```bash
cashew --db <path> stats         # graph statistics
cashew --db <path> init          # initialize new database
```

## Extraction Quality Rules

### What to extract
- Pattern-level insights, not raw events ("tends to X" not "did X today")
- Decisions and their reasoning ("decided X because Y")
- Corrections to existing knowledge ("actually X, not Y as previously thought")
- Operating principles and lessons learned
- Cross-domain connections and meta-observations
- Commitments/TODOs from anyone in the conversation

### What NOT to extract
- Ephemeral activity logs ("posted in Discord", "ran a command", "deployed X")
- Transient status ("204 tests passing" — changes constantly)
- Information only relevant for a few hours
- Restating things already in the brain with different words

### Domain assignment
Each node has a domain. Get this right:
- **User domain** (`raj` or configured user name): user's knowledge, beliefs, preferences, decisions, personal info, relationships, finances, career, creative work
- **AI domain** (`bunny` or configured AI name): operational knowledge, lessons learned, behavioral rules, workflow patterns, tool quirks, infrastructure facts
- Project decisions the user makes → user domain (their decision)
- Operational lessons the AI learns → AI domain (your learning)
- Blog content, project strategy, architecture decisions → user domain
- CLI patterns, deployment facts, cron configs → AI domain

### Privacy tagging
- Tag `vault:private` for: personal finances, health, relationships, credentials, pre-launch IP, private conversations, unpublished drafts
- Do NOT tag as private: general engineering principles, anything already shared publicly
- When in doubt, tag private (declassify later)
- In group channels, always query with `--exclude-tags vault:private`

## Cron Jobs

Frequencies below are sensible defaults. Adjust to your usage patterns — heavy daily use warrants more frequent extraction; light use can dial everything back. Sleep schedule is configurable in `config.yaml` under `sleep.schedule` and `sleep.frequency`.

### Session-to-brain extraction (default: every 2 hours)
Reads recent session history, extracts knowledge using prepare→ingest pattern. Split private/public nodes into separate ingest calls.

### Think cycle (default: twice daily)
Picks a random cluster, finds cross-domain patterns, generates insights. Use `--prepare-only` + external LLM + `--ingest` pattern.

### Sleep cycle (default: every 6 hours, configurable in config.yaml)
Full consolidation: cross-linking, decay, deduplication, hierarchy evolution.

### DB backup (default: every 6 hours)
Copy graph.db to timestamped backup. Never lose the brain.

### Declassify (default: daily)
Review aged `vault:private` nodes for safe declassification.

## Key Principles

- **Brain is the sole source of truth.** No memory files. No fallback to flat files.
- **Brain can be extended and corrected** — new nodes supersede old ones naturally.
- **Query before replying** to any substantive message. The cost of an unnecessary query is 5 seconds. The cost of a context-free response is missing the A+ answer.
- **Extract during conversation**, not after. Don't wait for cron — you are the primary extraction path.
- **Quality over quantity** — 3 good nodes beats 15 mediocre ones.
- **Don't dump raw brain output to user** — it informs YOU, you inform the user naturally.

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `CASHEW_DB` | Path to graph database | `./data/graph.db` |
| `KMP_DUPLICATE_LIB_OK` | Set to `TRUE` for MKL/OpenMP conflicts | unset |
| `CASHEW_METRICS` | Set to `1` to enable metrics recording | `0` |

## Architecture
- **Graph**: SQLite file with nodes, edges, embeddings in one file
- **Retrieval**: O(log N) seed selection via sqlite-vec + recursive BFS graph walk
- **Extraction**: LLM extracts facts/insights/observations/decisions/beliefs from conversations
- **Think cycles**: Autonomous cross-domain pattern detection
- **Decay**: Unused nodes naturally fade — forgetting is the quality filter
- **Sleep**: Deep consolidation — cross-linking, dedup, hierarchy evolution

## Requirements
- Python 3.10+
- ~500MB disk for embedding model (downloaded on first use)
- LLM access via OpenClaw's model_fn (no separate API key needed)
