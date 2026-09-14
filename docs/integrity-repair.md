# Connection-owned integrity repair

`core.integrity` provides a small repair boundary for applications that own
the SQLite lifecycle around a Cashew brain.  It is intended for adapters such
as Hermes that coordinate more than one process and therefore cannot safely
delegate connection or journal management to a path-based helper.

## Ownership contract

`inspect_integrity(conn, ...)` and `repair_integrity(conn, ...)` require an
open `sqlite3.Connection`.  They never open or close that connection, commit
or roll back an outer transaction, change journal mode, or acquire a process
lock.  The caller owns:

1. the profile-scoped backup;
2. an exclusive maintenance lease;
3. the SQLite journal and busy-timeout policy;
4. the outer transaction and final commit; and
5. the post-repair verification audit.

`repair_integrity` also requires the caller to have begun an outer
transaction.  It returns `status: "rejected"` with
`reason: "outer_transaction_required"` when `conn.in_transaction` is false;
the caller must explicitly begin the transaction before invoking it.

Each individual repair uses a savepoint.  A failed savepoint is rolled back
without discarding unrelated work in the caller's transaction.  A successful
repair is still uncommitted until the caller commits.  This is deliberate:
the caller must be able to classify an ambiguous commit and retain its backup
for recovery rather than having Cashew silently overwrite the live profile.

The result contains `transaction_owner: "caller"` and `committed: false`.
`completed` means that all selected work completed within the bound;
`partial` means that a repair was skipped or failed and the caller must audit
again.  The `remaining` mapping records actionable anomalies left after the
bounded pass, including work beyond `batch_size` or `max_items`.  Commit
uncertainty cannot be determined by this API because commit
belongs to the caller; adapters should report it as `uncertain` and preserve
their backup when their own commit or postcondition check is ambiguous.
If a savepoint rollback itself fails, the result sets `mutated: true`,
`mutation_uncertain: true`, and records `savepoint_rollback_failed`.

## Safe repair boundary

The default actions remove embeddings whose node no longer exists, remove
edges whose endpoint no longer exists, repair a missing vec row from a valid
ordinary embedding, remove stale vec rows, and re-embed a live node only when
the caller supplies a compatible embedding function, model name, and positive
dimension.  Every re-embedded node is written to the ordinary and vec tables
under one savepoint.  Invalid, nonfinite, zero-norm, wrong-dimension, or
model-mismatched output is rejected before a write.

Work is capped by `max_items` and each query is limited by `batch_size`.
Repeating the operation after a successful commit is safe and normally
returns zero additional repairs.

The vec table is never created, dropped, or recreated by this API.  Existing
vec rows are checked for schema dimension, finite/nonzero values, and exact
float32 parity with the ordinary embedding.  A missing, unloadable, or
dimension-incompatible vec table is reported as unavailable when parity is
required.  An ordinary-only repair may opt out with
`require_vec_parity=False`.
When an embedding model is supplied, a vec replacement may copy only an
ordinary embedding carrying that exact model; otherwise the row remains
skipped for the caller's model-specific repair path.

Permanence contradictions and self-edges are report-only by default.  A
caller must explicitly request `permanence_policy="preserve_permanent"` or
`"preserve_decay"` before either side of a contradictory flag is changed, and
must select `remove_self_edges` before self-edges are deleted.

## What cannot be reconstructed

Cashew does not infer deleted or merged facts from the current graph.  It does
not resurrect every decayed node, recreate the meaning of a historical
consolidation, or guess whether a contradictory permanence flag was the
intended one.  A backup and the audit report remain the recovery path for
those cases.

The embedding function is supplied by the host.  Cashew never constructs an
LLM or embedding model as part of repair, and it refuses to guess a model or
dimension from process-global configuration when the caller has not provided
one.
