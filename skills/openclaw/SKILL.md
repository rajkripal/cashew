---
name: cashew
description: Persistent thought-graph memory for OpenClaw agents. Gives your agent a brain that survives across sessions -- query context before answering questions, extract knowledge during conversations, run think cycles for autonomous insight generation. Triggers on session start, knowledge-worthy conversations, and idle heartbeats.
---

# cashew -- Persistent Thought-Graph Memory

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
This is critical — without it, knowledge is lost on every compaction.

## How To Use

**Note:** `--db` is a global flag and must go BEFORE the subcommand.

### Retrieve context (do this at session start)
```bash
cashew --db ~/.cashew/graph.db context --hints "keywords from the conversation"
```
This returns relevant knowledge from your graph. Use it to inform your responses.

### Extract knowledge (do this during/after conversations)
Write key insights to a temp file, then extract:
```bash
cat > /tmp/cashew-extract-input.md << 'EOF'
<paste conversation highlights, decisions, insights>
EOF
cashew --db ~/.cashew/graph.db extract --input /tmp/cashew-extract-input.md
```

### Think cycle (autonomous insight generation)
```bash
cashew --db ~/.cashew/graph.db think
```
Picks a random cluster of nodes, finds cross-domain patterns, generates new insights.

## Cron Jobs to Set Up

Use OpenClaw's cron system to automate:

### 1. Session-to-brain extraction (every 2 hours)
Reads recent session history and extracts to brain.
```
Schedule: every 2 hours
Payload: agentTurn
Message: "Read the most recent session history. Extract key facts, insights, decisions, and observations to cashew: write them to /tmp/cashew-extract-input.md then run: cashew --db ~/.cashew/graph.db extract --input /tmp/cashew-extract-input.md"
```

### 2. Think cycle (twice daily)
```
Schedule: cron "0 5,17 * * *" (5am and 5pm)
Payload: agentTurn  
Message: "Run a cashew think cycle: cashew --db ~/.cashew/graph.db think"
```

### 3. Dashboard deploy (daily, optional)
```
Schedule: cron "30 7 * * *" (7:30am daily)
Payload: agentTurn
Message: "Export and deploy the cashew dashboard: python3 scripts/export_dashboard.py ~/.cashew/graph.db dashboard/data/graph.json && deploy"
```

## Integration with OpenClaw

### Pre-compaction extraction (memoryFlush hook)
Add this to your OpenClaw config (`openclaw.json` or via `openclaw config`) to automatically extract knowledge to cashew before compaction:
```json5
{
  agents: {
    defaults: {
      compaction: {
        memoryFlush: {
          enabled: true,
          softThresholdTokens: 4000,
          systemPrompt: "Session nearing compaction. Extract all important knowledge to cashew before context is lost.",
          prompt: "Compaction is imminent. Write key facts, decisions, insights, and commitments from this conversation to /tmp/cashew-extract-input.md, then run: cashew --db ~/.cashew/graph.db extract --input /tmp/cashew-extract-input.md -- Reply NO_REPLY when done."
        }
      }
    }
  }
}
```
This ensures no knowledge is lost during compaction — the agent extracts to the brain automatically before context is compressed.

### Session start context
In AGENTS.md or SOUL.md, add:
```
Before replying to any substantive message, query the brain:
cashew --db ~/.cashew/graph.db context --hints "<keywords from the message>"
```

## Architecture
- **Graph**: SQLite file (`graph.db`) with nodes, edges, embeddings
- **Retrieval**: O(log N) seed selection via sqlite-vec + recursive BFS graph walk
- **Extraction**: LLM extracts facts/insights/observations/decisions/beliefs from conversations
- **Think cycles**: Autonomous cross-domain pattern detection with serendipity seeds
- **Decay**: Unused nodes naturally decay -- forgetting is the quality filter

## Requirements
- Python 3.10+
- ~500MB disk for embedding model (downloaded on first use)
- LLM access via OpenClaw's model_fn (no separate API key needed)
