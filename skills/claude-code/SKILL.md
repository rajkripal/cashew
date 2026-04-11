---
name: cashew
description: Persistent thought-graph memory across Claude Code sessions. Query context before answering questions about prior work, extract knowledge during conversations, run think cycles during idle time. Use when starting a session, when asked about prior context, or when important decisions/insights emerge.
allowed-tools: Bash Read Write Edit
---

# Cashew — Persistent Thought-Graph Memory

Cashew gives you a persistent brain across sessions. Without it, you forget everything when context resets. With it, you carry forward decisions, patterns, relationships, and project knowledge.

## Installation

```bash
pip install cashew-brain
# Or from source:
pip install git+https://github.com/rajkripal/cashew.git

# Initialize your database
cashew --db ~/.cashew/graph.db init
```

Set the `CASHEW_DB` environment variable to avoid passing `--db` every time:
```bash
export CASHEW_DB=~/.cashew/graph.db
```

## Core Protocol

### 1. Session Start — Query First (MANDATORY)

Before answering any substantive question in a new session, query the brain:

```bash
KMP_DUPLICATE_LIB_OK=TRUE cashew context --hints "<keywords from the conversation>"
```

This returns:
- **Graph overview** — shape of what you know (domains, clusters)
- **Recent activity** — what happened in the last few sessions
- **Relevant nodes** — knowledge matching your hints

Do NOT answer questions about prior work, decisions, people, or preferences without querying first. The cost of a query is seconds. The cost of a context-free answer is missing the A+ response sitting in the brain.

### 2. During Conversation — Extract Proactively

When any of these happen, extract immediately:
- A project status changes or a decision is made
- A new insight, pattern, or correction surfaces
- A new person, relationship, or commitment is established

```bash
cat > /tmp/cashew-extract.md << 'EOF'
- Decided to use Rust for the new service because of memory safety requirements
- User prefers async-first architecture patterns
- Project deadline moved to Q3
EOF
KMP_DUPLICATE_LIB_OK=TRUE cashew extract --input /tmp/cashew-extract.md
```

**Extract:** Pattern-level insights, decisions with reasoning, corrections, cross-domain connections, commitments/TODOs, operating principles and lessons learned.

**Skip:** Transient chat ("ok thanks"), info already in the graph, raw data without interpretation, ephemeral activity logs ("deployed X", "ran a command"), transient status ("204 tests passing"), code (it belongs in files).

**Domain assignment:** Each node gets a domain. The user's knowledge, beliefs, preferences, decisions, creative work → user domain. Your operational knowledge, lessons learned, workflow patterns, tool quirks → AI domain. Project decisions the user makes → user domain. Operational lessons you learn → AI domain.

**Privacy tagging:** Use `--tags vault:private` for personal finances, health, relationships, credentials, pre-launch IP, unpublished drafts. Don't tag as private: general engineering principles, anything already public. When in doubt, tag private.

### 3. Think Cycles — Autonomous Consolidation

During idle time or at end of session:

```bash
KMP_DUPLICATE_LIB_OK=TRUE cashew think
```

This finds cross-domain connections, detects tensions (contradictory knowledge), and consolidates clusters. Run 1-2x per day.

## Scheduling with Claude Code Cron

If Claude Code's `Cron` tool is available, set up automated maintenance:

```
# Extract from recent work every 2 hours
/loop 2h "Summarize key decisions and insights from this session into /tmp/cashew-extract.md, then run: KMP_DUPLICATE_LIB_OK=TRUE cashew extract --input /tmp/cashew-extract.md"

# Think cycle once per day
/loop 24h "KMP_DUPLICATE_LIB_OK=TRUE cashew think"
```

Or use system cron for persistence across sessions:
```bash
(crontab -l 2>/dev/null; echo '0 */6 * * * KMP_DUPLICATE_LIB_OK=TRUE cashew think') | crontab -
```

## Bootstrapping from Claude Archives

If you have exported Claude conversations (`conversations.json`):

```bash
# Export conversations to individual files
python3 -c "
import json, os
os.makedirs('/tmp/cashew-bootstrap', exist_ok=True)
convos = json.load(open('conversations.json'))
for i, c in enumerate(convos):
    with open(f'/tmp/cashew-bootstrap/convo_{i}.md', 'w') as f:
        for msg in c.get('chat_messages', []):
            f.write(f\"{msg.get('sender','?')}: {msg.get('text','')}\n\n\")
"

# Extract each file (divide-and-conquer is more reliable than bulk)
for f in /tmp/cashew-bootstrap/*.md; do
    KMP_DUPLICATE_LIB_OK=TRUE cashew extract --input "$f"
done
```

## CLI Reference

| Command | Purpose |
|---------|---------|
| `cashew context --hints "..."` | Retrieve relevant context from brain |
| `cashew context --hints "..." --exclude-tags "vault:private"` | Exclude private nodes (use in shared contexts) |
| `cashew extract --input file.md` | Extract knowledge from text |
| `cashew extract --input file.md --tags vault:private` | Extract with privacy tags |
| `cashew think` | Run think cycle (cross-domain connections) |
| `cashew sleep` | Full sleep cycle (clustering + hierarchy) |
| `cashew stats` | Graph statistics (node/edge counts) |
| `cashew [--db path] init` | Initialize new database |

## Key Principles

- **Brain is source of truth** for decisions, patterns, relationships
- **One context query per session** is usually enough — don't over-query
- **Extract sparingly** — only genuinely new knowledge, quality over quantity
- **Don't dump raw brain output to the user** — it informs YOU, you inform the user naturally
- **If results are thin**, re-query with different/broader hints before giving up

## Architecture

- Single SQLite file with sqlite-vec extension for vector search
- Local embeddings: all-MiniLM-L6-v2 (384 dims, downloads ~500MB on first run)
- Retrieval: O(log N) seed via sqlite-vec → recursive BFS graph walk (seeds=5, picks_per_hop=3, max_depth=3)
- No external services or API keys needed for the graph itself
- LLM needed only for extraction and think cycles (uses your Claude API key)

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `CASHEW_DB` | Path to graph database | `./data/graph.db` |
| `KMP_DUPLICATE_LIB_OK` | Set to `TRUE` for MKL/OpenMP errors | unset |
| `CASHEW_METRICS` | Set to `1` to enable metrics recording | `0` |
| `ANTHROPIC_API_KEY` | For extraction/think (uses Claude) | required for extract/think |
