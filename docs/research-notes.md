# Research Notes - Session Integration Best Practices

> **Scope note:** these are research notes from external papers and production-system writeups reviewed during cashew's design phase. They describe patterns from the literature (GraphRAG, ConversationKGMemory, hybrid retrieval), **not** what cashew actually implements. Specifically, things like learned ranking weights, temporal-decay coefficients in the score function, and TF-IDF fallbacks are paper concepts, not cashew behavior. For the implemented retrieval algorithm see [`docs/architecture.md`](architecture.md), and for the survival/decay model see PHILOSOPHY.md §9.

## 1. Knowledge Graph Retrieval Ranking

### Key Insights from Production Systems

**Microsoft GraphRAG & Similar Systems:**
- Use **hybrid ranking** combining multiple signals:
  - **Embedding similarity** (semantic relevance to query)
  - **Graph proximity** (structural distance from relevant entities)  
  - **Temporal proximity** (recency/relevance of temporal information)
  - **Structural relevance** (node importance in graph topology)

**Scoring Functions:**
- Linear combination with learned weights: `score = α×embedding_sim + β×(1/graph_distance) + γ×temporal_decay + δ×structural_importance`
- Typical weight distributions: embedding similarity (40-50%), graph proximity (20-30%), temporal (15-25%), structural (10-15%)
- **Token budget constraints** drive final ranking cutoffs

### Best Practices:
- Start with embedding similarity for entry points, expand via graph traversal
- Use **bidirectional traversal** (both parents and children)
- Apply **temporal decay** functions (exponential or linear based on use case)
- Consider **access frequency** as a proxy for importance
- Implement **subgraph extraction** rather than individual node retrieval

## 2. Session Context Injection Patterns

### Token Budget Guidelines

**Sweet Spot Research:**
- **1,500-2,500 tokens** is optimal for context injection (not overwhelming, sufficient detail)
- **Knowledge graphs** can reduce token usage by 40-60% vs raw conversation history
- **Structured context** (entities + relationships) more efficient than prose

### Memory Management Strategies:
- **Recent conversations** (last N exchanges) + **relevant graph facts** 
- **Buffer management**: Keep last 3-5 exchanges + graph-synthesized context
- **Hybrid approach**: Recent messages + KG summary + relevant historical facts
- **Progressive summarization**: Compress older context into graph nodes

### Implementation Patterns:
- Use **ConversationKGMemory** pattern (extract entities/relationships from recent history)
- **k-recent interactions** alongside KG-derived context (k=3-5 typical)
- **Context refresh** at session boundaries to prevent staleness
- **Relevance filtering** - only inject context above similarity threshold

## 3. Knowledge Extraction from Conversations

### Effective Extraction Techniques:

**Structured Extraction:**
- **Entity extraction**: People, places, decisions, facts
- **Relationship mapping**: Causal relationships, temporal sequences
- **Sentiment/emotional states**: Confidence levels, mood indicators
- **Key phrase extraction**: Domain-specific terminology, important concepts

**Content Categories:**
- **Observations** (what happened) 
- **Beliefs/Opinions** (what someone thinks)
- **Decisions** (commitments made)
- **Insights** (patterns discovered)
- **Facts** (objective information)

### Prompt Engineering for Extraction:
```
"Identify from this conversation:
1. New factual information (who, what, when, where)
2. Decisions or commitments made
3. Beliefs or opinions expressed  
4. Patterns or insights discovered
5. Emotional context or confidence levels
Format as structured data with confidence scores."
```

### Fallback Heuristics (when no LLM available):
- **Keyword extraction** (TF-IDF, named entities)
- **Decision markers** ("will", "should", "decided", "plan to")
- **Belief markers** ("think", "believe", "seems like", "probably")
- **Temporal markers** (dates, times, "next week", "yesterday")
- **Entity detection** (proper nouns, numbers, specific terms)

## 4. Implementation Recommendations

### Hybrid Retrieval Architecture:
1. **Embedding search** → entry points (top-K candidates)
2. **Graph walk** → expand context (1-2 hops)  
3. **Temporal filtering** → prefer recent/relevant
4. **Access-based ranking** → boost frequently-used nodes
5. **Token budget** → final cutoff

### Session Lifecycle:
- **Session start**: ~2000 tokens context budget, blend recent + relevant historical
- **Session end**: Extract 3-5 key insights, auto-link to existing knowledge
- **Think cycles**: Focus on underexplored regions, generate cross-domain connections

### Evaluation Metrics:
- **Context relevance** (human evaluation of retrieved context quality)
- **Knowledge retention** (how well extracted facts connect to existing graph)
- **Cross-domain discovery** (novel connections between domains)
- **Token efficiency** (information density per token used)