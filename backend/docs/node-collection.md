# NodeCollection

Source: `src/graphs/api/collection.py` (`NodeCollection`,
`NodeCollectionRecord`, `get_default_embedder`)

## What it is

`NodeCollection` is an in-memory store of node **prototypes** — instances of
the `GraphNode` protocol (e.g. `TextNode` objects), **not** the LangGraph
callables `get_node(llm)` returns. Each prototype is keyed by a uuid4 `id`
(multiple nodes may share a `name`), usage/recency is tracked, and retrieval
supports combined metadata filters plus semantic (vector) search.

It does **no persistence and no graph wiring**: an external system persists
contents via `snapshot()` (deep-copied prototypes) or `records()` (metadata
view), and embeddings are rebuilt on re-add after a restore. The collection
also owns an internal **ephemeral** chromadb cosine index — `add` embeds
`"{name}\n{description}"`, `remove`/pruning delete the vectors, and
`semantic_search` embeds the query and ranks by similarity.

## `NodeCollectionRecord`

Frozen, read-only metadata view of a stored node.

| Field | Type | Meaning |
|-------|------|---------|
| `id` | `str` | uuid4 key assigned at `add()` |
| `name` | `str` | prototype's name (metadata; may repeat across nodes) |
| `description` | `str` | prototype's description |
| `use_counts` | `int` | bumped only by `get(id)` |
| `pinned` | `bool` | protected from auto-pruning |
| `created_at` | `datetime` | tz-aware UTC insertion time |
| `last_used_at` | `datetime \| None` | tz-aware UTC time of last `get(id)` |
| `run_counts` | `int` | bumped only by `record_run(id)` — execution outcomes, distinct from store reads |
| `tokens_count` | `int` | number of runs that reported an output-token count (subset of `run_counts`) |
| `tokens_mean` / `tokens_std` | `float \| None` | output-token mean/std over those runs (`None` while `tokens_count` is 0) |
| `time_count` | `int` | number of runs that reported elapsed time (subset of `run_counts`) |
| `time_mean` / `time_std` | `float \| None` | elapsed-time mean/std over those runs (`None` while `time_count` is 0) |

## API

`NodeCollection(*, max_nodes: int = 100, prune_to: int = 80, embedder: Callable[[str], list[float]] | None = None)`

Validates `0 < prune_to <= max_nodes` (`ValueError`).

| Method | Signature / behavior |
|--------|----------------------|
| `add(node, *, pinned=False) -> str` | Store a prototype, return its new uuid4 id; rejects an empty/missing `node.name`; auto-prunes on overflow; embeds the node (embedder failure degrades gracefully — node stays stored, marked unembedded) |
| `get(id) -> GraphNode` | Return a **deep copy** of the prototype, bumping `use_counts` + `last_used_at`; `KeyError` if unknown — the stored prototype is read-only from outside; mutate via `update`/`replace` |
| `record_run(id, *, tokens=None, elapsed_seconds=None) -> None` | Record one execution outcome on the entry's run stats: bump `run_counts`, fold each non-`None` measurement (must be finite and non-negative) into the Welford accumulators; `KeyError` if unknown, `ValueError` on NaN/±inf/negative |
| `restore_stats(id, *, run_counts, tokens, time) -> None` | Idempotently overwrite the entry's run telemetry with `OnlineStats` values; used by loaders (e.g. `load_collection`) to re-attach persisted stats; `KeyError` if unknown, `ValueError` for negative `run_counts` |
| `get_callable(id, llm) -> DualCallable` | Deep-copy the stored prototype, wrap `llm` in a recording proxy that sums output tokens (`usage_metadata`) and elapsed time across the run, and return an instrumented callable that reports them via `record_run` on every **successful** top-level run; bumps nothing itself; `KeyError` if unknown |
| `update(id, *, name=None, description=None, prompt=None, params=None, writes=None) -> None` | Field-level edit keyed by id; preserves all bookkeeping, never bumps; rebuilds via the stored node's `updated()` (re-validates placeholders ⊆ params); empty call is a no-op; `KeyError` if unknown; `TypeError` if the stored node has no `updated()` (use `replace`) |
| `replace(id, new_node) -> None` | Swap in a rebuilt prototype under the same id; preserves bookkeeping, never bumps; `ValueError` on empty name |
| `get_by_name(name) -> list[NodeCollectionRecord]` | All records with that name, insertion order; never bumps |
| `search(query=None, *, limit=10, sort_by='use', within_newest=None)` | Metadata retrieval: case-insensitive substring over name OR description; `within_newest` narrows candidates to the newest-`z`; `sort_by` `'use'` \| `'newest'`; never mutates |
| `top_by_use(k)` / `newest(k)` | Thin wrappers over `search` |
| `semantic_search(query, *, limit=10, within_newest=None, sort_by='similarity'\|'use'\|'newest')` | Vector retrieval; see Behavior |
| `remove(id)` | Drop the entry and its vector (allowed even when pinned); `KeyError` if unknown |
| `pin(id)` / `unpin(id)` | Protect / release from pruning (idempotent); `KeyError` if unknown |
| `records()` | All records, insertion order |
| `snapshot() -> list[GraphNode]` | Deep copies of the stored prototypes, insertion order — for persistence |
| `__len__` / `__contains__` | `len(coll)`, `id in coll` |

## Usage

```python
from graphs.api.collection import NodeCollection
from graphs.nodes.text_node import TextNode

coll = NodeCollection(max_nodes=50, prune_to=40)

summ = TextNode(
    name="summarizer",
    description="Condenses the previous step's output into one sentence.",
    prompt="Summarize this in one sentence: {user_message}",
    params={"user_message": str},
)
sid = coll.add(summ)            # uuid4 id
coll.pin(sid)

kept = coll.get(sid)            # bumps use_counts
coll.add(TextNode(
    name="questioner",
    description="Asks a follow-up question.",
    prompt="{user_message} -> ?",
    params={"user_message": str},
))

coll.search("summ")             # -> [NodeCollectionRecord(id=sid, ...)]
coll.semantic_search("condense text", limit=3, sort_by="use")
coll.records()                  # metadata views, insertion order
coll.snapshot()                 # deep copies for an external system to save
```

Read-modify-write an existing node (metadata edits never fail; only
params/writes/prompt interplay re-validates):

```python
rec = coll.get(sid)             # deep copy — the stored prototype is read-only
coll.update(sid, prompt="Summarize in at most two sentences: {user_message}")
coll.update(sid, description="Tight one-sentence summaries.")
coll.semantic_search("tight two-sentence summary")   # reflects the new text
coll.replace(sid, TextNode(name="tight-summarizer", description="...",
                           prompt="...", params={"user_message": str}))
```

Custom embedders are injected at construction; the default is a lazy
fastembed-backed helper (`BAAI/bge-small-en-v1.5`, one-time model download,
offline afterwards — never constructed in tests):

```python
def fake_embed(text: str) -> list[float]:
    return [1.0] if "graph" in text else [0.0]

coll = NodeCollection(embedder=fake_embed)
```

## Behavior

- **Keying** — every `add()` mints a uuid4 id; multiple nodes may share a
  `name`. `get_by_name` returns all matches in insertion order.
- **Metadata search** — `search` matches a single `query` as a
  case-insensitive substring against name **or** description, optionally over
  a candidate window of the `within_newest`-z most recently created entries,
  then sorts (`'use'`: use_counts desc, created_at desc tie-break; `'newest'`:
  created_at desc) and truncates to `limit`. Never bumps usage.
- **Vector search** — `semantic_search` embeds `query`, retrieves the top-n
  most similar entries from the collection-owned ephemeral chromadb cosine
  index (optionally narrowed by `within_newest`), then optionally re-sorts the
  top-n by `'use'`/`'newest'` before truncating to `limit`. Never bumps.
  Unembedded entries (embedder failure at `add`) are skipped — they have no
  vector.
- **Usage tracking** — `use_counts`/`last_used_at` bump **only** on
  `get(id)`. Searches, listings, `get_by_name`, `update` and `replace` never
  mutate usage.
- **Run telemetry** — `use_counts` measures store reads; `run_counts` and the
  token/time stats measure **executions**, and are bumped **only** by
  `record_run(id, …)` (via `get_callable`, or directly). `get_callable`
  deep-copies the prototype, wraps the supplied `llm` in a recording proxy
  that accumulates `usage_metadata["output_tokens"]` and call wall-time across
  the whole run — including nested `GraphNode` calls that thread the same llm
  — and records on **success only**: an exception propagates and records
  nothing. A run whose LLM responses carry no `usage_metadata` records time
  only (tokens skipped, no heuristic). `get_callable` itself bumps nothing.
  Per-node stats are folded in online (Welford) and stay on the store entry,
  so they are **preserved across `update`/`replace`**, come from **`records()`/
  `search`/etc.** as fields on the record, and are **persisted inline** in
  `collection.json` — legacy dumps load with empty stats (see `docs/data.md`).
- **Mutations don't bump** — `update`/`replace` rebuild the stored prototype
  under the same id and leave `use_counts`, `pinned`, `created_at` and
  `last_used_at` untouched. Only `get` records use/recency.
- **Copy-returning `get`** — `get(id)` hands out a **deep copy**; the stored
  prototype can never be mutated through it. To edit a stored node use
  `update` (same-class rebuild via `updated()`, validated before the swap —
  a failed edit leaves the store unchanged) or `replace` (swap in any
  prototype). This also kills reference aliasing: an LLM holding a `get` copy
  cannot corrupt the store or a graph sharing the same instance.
- **Embedding consistency** — `add` embeds `"{name}\n{description}"` into the
  vector index; `update`/`replace` **re-embed iff that text changed** (a
  previously-unembedded entry re-attempts embedding). A changed vector is
  swapped in place; failure degrades to `embedded=False` and the node change
  is kept — embedding never raises and never rolls back. Cost is ~tens of ms
  per node; only a whole-collection re-embed is worth avoiding.
- **Auto-pruning** — on `add`, while non-pinned count > `max_nodes`, evict
  lowest-use non-pinned entries (oldest `created_at` breaks ties) until
  non-pinned count <= `prune_to`. Pinned entries are excluded from the count
  **and** from eviction; pinning is mutable (`pin`/`unpin`).
- **Embedder failures degrade** — if embedding fails at `add` time the node is
  still stored (unembedded) and skipped by semantic search; a failed embedder
  never drops a node.
- **Semantics of `sort_by='similarity'`** — the raw index order (cosine
  ranking) is the default, so the returned list is already best-match-first.

## Tests

See `tests/test_node_collection.py` (73 tests): construction validation,
add/get/remove lifecycle, usage bumps, insertion order, combined-filter
metadata search, pruning + pinning, deep-copy snapshots, the vector index
(add/remove/prune vector lifecycle, cosine ranking, `within_newest`,
sort-by variants, degradation paths), field-level mutation
(`update`/`replace`: bookkeeping preservation, no-bump guarantee, re-embed on
text change, unembedded re-attempt, validation/error paths, copy-returning
`get`), and run telemetry (`record_run`/`restore_stats` measurement
validation + Welford folding, `get_callable` recording via the `llm` proxy —
token and time accumulation through nested runs, time-only on missing
`usage_metadata`, no bump, success-only recording on `invoke`/`ainvoke`).
The suite is pure-unit — it never touches the network; tests inject a
deterministic fake embedder, and the default fastembed embedder is verified
to stay unconstructed.