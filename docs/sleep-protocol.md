# Sleep protocol integration

`core.sleep.run_sleep_cycle` is the bounded consolidation entry point. Existing
callers may continue to use the first six positional arguments; new adapter
options are keyword-only:

```python
run_sleep_cycle(
    db_path, limit=None, model_fn=None, background_dream=False,
    max_edges=100_000, cross_source_only=False, *, embedding_client=None,
    embedding_model=None, expected_dimension=None, auto_embed=True,
    journal_policy="manage", orphan_limit=None, orphan_batch_size=100,
)
```

By default, sleep lazily uses the configured model through the existing
`LocalBackend`. The model loads only when an orphan needs encoding; an empty
cycle or index repair from stored vectors does not load it. Existing CLI and
`SleepProtocol` callers retain automatic orphan embedding.

Adapters can supply all three of `embedding_client`, `embedding_model`, and
`expected_dimension`. The client must implement `encode(list[str])` and return
one finite, non-zero vector of the requested dimension per input. Tiny non-zero
vectors are accepted; exactly zero vectors are invalid for cosine similarity.
An injected client is authoritative: failure never triggers a local fallback.
Set `auto_embed=False` to prohibit automatic encoding when no client is supplied;
eligible unrepaired orphans then produce an unavailable/partial outcome. Process-
isolating hosts should set this explicitly, including when forwarding a client.
`SleepProtocol.run_sleep_cycle` forwards this option as well.

Invalid output is rejected before any node write. Each valid node is written
to `embeddings` and `vec_embeddings` inside one savepoint, and each batch is
committed independently. If the vec index is present and rejects a write, the
ordinary row is rolled back too. If the vec index is genuinely absent,
ordinary-only repair is reported by `orphan_vec_unavailable`. The default
repairs all eligible rows in batches of 100 for backward compatibility.
`orphan_limit` caps rows examined across ordinary and vec-index repair;
`orphan_batch_size` may reduce the encode/commit batch below 100. A table merely
named `vec_embeddings` is not a vec capability: it must be a `vec0` virtual
table that can be loaded and queried on the active connection.

Index repair preserves stored model identity and dimensions. It deliberately
rejects rows from a different model, even when dimensions match. Sleep is not a
model migration tool: complete Cashew's re-embedding migration before resuming
maintenance with a changed model. Matching rows can still be repaired in a mixed
batch; rejected rows are counted without relabeling their vectors.

When `limit` caps candidate discovery, sleep persists a private cursor and
rotates deterministically through `(timestamp, node_id)` pages. Cursor claims
commit before expensive phases, so a stopped cycle may defer its page until
the next wrap but cannot keep later pages permanently starved. Node timestamps
remain untouched. The private cursor table is checked on every capped run;
partial or malformed pre-release shapes are rebuilt transactionally, retaining
a single type-valid cursor when possible. Duplicate or invalid legacy cursor
rows reset to the deterministic origin. Capped orphan work has separate durable
ordinary and vec-repair cursors and alternates which phase receives the first
share of the cap. A repeatedly failing oldest batch therefore cannot starve
later rows or the other repair class forever.

The `_cashew_sleep_state` table belongs to Cashew sleep, not the host adapter.
It contains only disposable progress cursors, not memories. Uncapped calls do
not create it. Schema checking stays on the cycle connection so cursor claims
and recovery share its transaction; moving it to startup-only schema setup would
miss malformed state encountered by standalone callers. Future schema changes
must preserve valid cursors or reset them safely and keep the migration and
contention tests. Resetting this state can repeat work but must never delete
thoughts, embeddings, or graph edges.

## Supported work envelope

The public limits bound specific phases; they are not an end-to-end deadline:

- `limit=N` bounds the candidate embedding matrix to at most `N` active nodes.
  Similarity memory and arithmetic are quadratic in `N`.
- `max_edges=M` bounds newly completed or repaired unordered cross-link pairs.
  Existing, same-source, or failed candidates do not consume the budget, so
  candidate inspection can still visit every pair in the `N`-node matrix.
- deduplication consumes the same bounded candidate page, but maximal-clique
  enumeration and edge rewiring do not currently have a separate time or write
  budget.
- GC mutates at most 50 sampled nodes and orphan embedding examines at most
  `orphan_limit` rows when supplied. Metrics, permanence, core-memory ranking,
  audit cleanup, and vec compaction can still scan graph-wide state.
- a synchronous `model_fn` has no engine-enforced deadline. Integrations that
  need a wall-clock bound must own cancellation and SQLite admission outside
  this function.

Dream generation only receives cross-link pairs completed or repaired and
committed by the current cycle. Capped, filtered, failed, and already-complete
pairs are excluded. In particular, `max_edges=0` cannot invoke the dream model.

For reproducible local measurements, `scripts/benchmark_sleep_cycle.py` builds
an isolated synthetic database and reports wall time plus the public counters.
It never opens the configured Cashew database. `--hold-writer-ms` adds a
deterministic competing write transaction before the cycle starts. Results
characterize the chosen machine and fixture; they do not establish a
production deadline.

`journal_policy="manage"` retains the historical direct-call behavior.
Integrations that own SQLite admission should pass `journal_policy="preserve"`;
that mode performs no journal-mode assignment. The decay-audit table is created
idempotently on the cycle connection before any decay audit write.

The result is JSON-safe and includes `status`, bounded phase counters, dream
state (`skipped`, `pending`, `ran`, or `failed`), and explicit pair versus
directed-row counts. `cross_links_created` counts new unordered pairs only when
both directions are committed. A half-pair is repaired and counted separately;
existing complete pairs are skipped without consuming `max_edges`.
Cross-link counters and dream inputs advance only after each batch commit. A
known rolled-back suffix reports the exact committed prefix as `partial`; if a
commit's durability cannot be verified, the stable result instead reports
`uncertain` without claiming the attempted batch.
