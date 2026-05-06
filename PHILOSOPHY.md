# PHILOSOPHY.md: The Cashew Brain Philosophy

Cashew is a thought graph that accumulates evidence about a specific human, a specific relationship, and a specific working context. An agent wired to a cashew brain can reason from accumulated evidence instead of reasoning purely from its base model's priors. This document explains the design choices behind that.

---

## 1. Specialization Through Evidence

Every LLM starts from identical weights and identical priors. What differentiates one running instance from another is the evidence it accumulates through real interaction. The graph is that evidence: extracted facts, observed patterns, recorded corrections, captured decisions.

Two agents on the same base model with different cashew brains will respond differently to the same prompt, because they have different evidence about the human in front of them. This divergence is the point of the system.

Cashew itself doesn't override anything. The agent's prompt decides how much weight to put on graph evidence versus model defaults. The graph just makes the evidence available, well-connected, and cheap to retrieve. How the agent uses it is a prompt-level concern.

---

## 2. Evidence Beats Priors For Local Questions

For questions about the world (physics, history, medicine), the model's training is the authority and the graph has nothing useful to add. For questions about this human, this relationship, and this working context, the model has no specific evidence and the graph does.

A node in the graph was extracted from real conversation, deduplicated against existing knowledge, embedded for semantic recall, and exposed to contradicting evidence over time. For local questions, that is more grounded than a trained prior.

---

## 3. Node Types Are Inference Hints

The graph is dumb. The reasoning layer is smart. Nodes carry a type tag (fact, observation, insight, belief, decision) but the type does nothing load-bearing. Retrieval ranks by recency, access count, edge degree, and semantic similarity. Decay treats all types the same. There is no privileged authority by type.

Why have types at all? They are descriptive metadata for the LLM at retrieval time. Knowing a node was tagged "decision" versus "observation" helps the model reason about how to use it. That is a hint to the consumer, not a knob in the graph engine. If the type vocabulary changed tomorrow, no graph code would change.

This is the central architectural commitment: dumb graph, smart reasoning layer. The graph stores connections and content, with no edge semantics, no node-type privileges, and no temporal scaffolding. The reasoning happens in the layer above.

---

## 4. Pressure-Test, Don't Amplify

Sycophancy is the default failure mode of LLMs. The agent should pressure-test claims rather than amplify them, including positive ones. When the human says "I think X improved," the appropriate response is to ask how to measure it, not to agree. When something might be vibes, the agent should say so.

The rule is to follow the evidence. The graph makes that easier because the evidence is right there.

---

## 5. Extraction Discipline

A brain-equipped agent absorbs as conversation happens. The extraction loop is the core value: the human and the agent talk, the agent extracts what matters, the next session starts smarter. Decisions, insights, corrections, commitments, and the agent's own learnings all go in.

Extract patterns rather than transcripts. Not "paid $50 at the restaurant" but the underlying tendency, if a pattern is actually visible. Corrections are the highest-priority extraction because they supersede earlier nodes.

More nodes is not a better brain. Aggressive extraction without quality gates pollutes retrieval. Cashew has two structural defenses: an n-plicate dedup pass that collapses near-duplicates, and an embedding diversity check that rejects extractions too close to existing knowledge. A 500-node graph with high-quality, well-connected nodes outperforms a 10,000-node graph of surface fragments.

---

## 6. Cross-Domain Retrieval

The graph's most useful behavior is surfacing connections the human cannot see directly. Retrieval is recursive BFS from a query embedding, walking edges across domain boundaries. A pattern in finance can pull in a related pattern from work or relationships if the embeddings put them near each other.

A flat memory file cannot do this. A vector store without graph structure cannot do this. The combination of semantic recall and edge traversal is what makes cross-domain insight cheap.

---

## 7. Patterns, Not Echoes

The agent's job is to reflect patterns back to the human with clarity the human cannot reach alone. Echoing means agreeing with whatever was just said ("you're right, that's a great insight"). Pattern reflection situates the claim against accumulated evidence ("this is the third time this shape has come up, here is where it connects, here is where it might be a blind spot"). The graph is what makes pattern reflection possible.

---

## 8. Privacy

The graph holds intimate knowledge. Nodes carry vault tags for sensitivity. The agent should query the graph for privacy state before sharing anything personal in a group context, and should never exfiltrate node content cross-context without explicit permission. The human owns their data. Export is always available.

---

## 9. Forgetting Is A Feature

Cashew has a sleep cycle that runs fitness functions over the graph and decays nodes that fail them. The deterministic survival gate is `access_count > 0 OR edge_degree > 0`. Anything that has been retrieved at least once or has at least one edge survives the cheap pass. Everything else goes to the fitness functions, which weigh recency, connectivity, and signal against age.

Decay is the forgetting mechanism. Don't fight it with structures that pin nodes artificially. If a node matters, it will be retrieved or connected, and the gate will keep it alive. If it stops mattering, it should fall out.

---

## 10. Constant Context Cost

Retrieval is the interface between a graph that grows without bound and a context window that does not. As the graph grows, the per-session context cost stays roughly bounded because each query pulls a bounded subgraph rather than scanning the whole store.

In practice, a context query against the author's current brain returns on the order of a few hundred words of structured output (measured at roughly 270 to 430 words per topic query across mixed hints), and that range does not move with total graph size. Boot context plus one or two topic queries plus a handful of extraction acks fits comfortably inside any session budget. Graph size and context cost are decoupled by design.
