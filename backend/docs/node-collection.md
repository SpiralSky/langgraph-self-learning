# NodeCollection

Source: `src/graphs/api/collection.py` (`NodeCollection`,
`NodeCollectionRecord`, `get_default_embedder`)

## What it is

`NodeCollection` is an in-memory store of nodes — instances of the `GraphNode`
protocol (e.g. `TextNode` objects), **not** the LangGraph callables
`get_node(llm)` returns. Each node is keyed by a uuid4 `id` (multiple nodes
may share a `name`), usage/recency is tracked, and retrieval supports combined
metadata filters plus semantic (vector) search.

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
| `run_counts` | `int` | the stored node's own counter — bumped only when the node **runs** (its callable self-records), distinct from store reads |
| `tokens_count` | `int` | derived from the stored node: runs that reported an output-token count (subset of `run_counts`) |
| `tokens_mean` / `tokens_std` | `float \| None` | the node's `output_tokens` mean/std over those runs |
| `time_count` | `int` | derived from the stored node: runs that reported elapsed time (subset of `run_counts`) |
| `time_mean` / `time_std` | `float \| None` | the node's `output_time` mean/std over those runs |

## API

`NodeCollection(*, max_nodes: int = 100, prune_to: int = 80, embedder: Callable[[str], list[float]] | None = None)`

Validates `0 < prune_to <= max_nodes` (`ValueError`).

| Method | Signature / behavior |
|--------|----------------------|
| `add(node, *, pinned=False) -> str` | Store a node, return its new uuid4 id; rejects an empty/missing `node.name`; auto-prunes on overflow; embeds the node (embedder failure degrades gracefully — node stays stored, marked unembedded) |
| `get(id) -> GraphNode` | Return the stored node **by reference** (the live shared instance, no deep copy), bumping `use_counts` + `last_used_at`; `KeyError` if unknown. Aliasing is the flip side: mutating the returned node mutates the store — to change a stored node use `update_inplace`/`edit`/`replace` |
| `update_inplace(id, *, reset_stats=False, **changes) -> None` | Field-level edit keyed by id, rebuilding via the stored node's `updated()` (re-validates placeholders ⊆ params); preserves all bookkeeping and run stats unless `reset_stats=True` (also works for a stale `run_counts`); empty call is a no-op; `KeyError` if unknown; `TypeError` if the stored node has no `updated()` (use `replace`) |
| `edit(id, **changes) -> GraphNode` | Return a new, validated instance with the changes applied (via `updated()`, stats carried) — the store is **untouched**; `KeyError` if unknown, `TypeError` if no `updated()` |
| `replace(id, new_node) -> None` | Swap in a rebuilt node under the same id; preserves bookkeeping (`id`, `use_counts`, `pinned`, `created_at`, `last_used_at`) and never bumps; the incoming node's own stats come with it (no implicit carry); `ValueError` on empty name |
| `get_by_name(name) -> list[NodeCollectionRecord]` | All records with that name, insertion order; never bumps |
| `search(query=None, *, limit=10, sort_by='use', within_newest=None)` | Metadata retrieval: case-insensitive substring over name OR description; `within_newest` narrows candidates to the newest-`z`; `sort_by` `'use'` \| `'newest'`; never mutates |
| `top_by_use(k)` / `newest(k)` | Thin wrappers over `search` |
| `semantic_search(query, *, limit=10, within_newest=None, sort_by='similarity'\|'use'\|'newest')` | Vector retrieval; see Behavior |
| `remove(id)` | Drop the entry and its vector (allowed even when pinned); `KeyError` if unknown |
| `pin(id)` / `unpin(id)` | Protect / release from pruning (idempotent); `KeyError` if unknown |
| `records()` | All records, insertion order |
| `snapshot() -> list[GraphNode]` | Deep copies of the stored nodes, insertion order — for persistence |
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
rec = coll.get(sid)             # live reference — the stored node itself
coll.update_inplace(sid, prompt="Summarize in at most two sentences: {user_message}")
coll.update_inplace(sid, description="Tight one-sentence summaries.")
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
  `get(id)`. Searches, listings, `get_by_name`, `update_inplace`/`edit`/
  `replace` never mutate usage.
- **Run telemetry (node-owned)** — every stored `AbstractNode` carries
  `run_counts`, `output_tokens` and `output_time` (`OnlineStats`, Welford
  mean/std), and the LangGraph callables **self-record**: the text-node
  callable times its LLM call and reads `usage_metadata["output_tokens"]`
  from each result (`record_result`; a run with no `usage_metadata` records
  time only, tokens skipped — no heuristic), while tool and graph-node
  callables record elapsed time only. Storage/retrieval reads never bump.
  Because `get` (and a generator's `from_collection` pull) hand out the
  **live instance**, stats recorded while a graph runs fold straight back
  into the stored node. `NodeCollectionRecord` reproduces them
  (counts/mean/std) as a read-only view. Stats are **preserved across
  `update_inplace`/`edit`** (carried by `updated()`), **`replace`** (the
  incoming node brings its own), are **persisted inline** in
  `collection.json`, and are **restored straight onto the node** on load
  (see `docs/data.md`).
- **Mutations don't bump** — `update_inplace`/`replace` rebuild or swap the
  stored node under the same id and leave `use_counts`, `pinned`,
  `created_at` and `last_used_at` untouched. Only `get` records use/recency;
  `edit` touches nothing.
- **Reference-returning `get`** — `get(id)` returns the **stored node by
  reference** (no deep copy): zero copying cost and, importantly, run stats
  recorded by that node's callables are visible to the collection even when
  the node runs inside a generated graph. The cost is aliasing — mutating a
  `get` result mutates the store, so keep it read-only from outside and edit
  via `update_inplace` (same-class rebuild via `updated()`, validated before
  the swap — a failed edit leaves the store unchanged), `edit` (validated
  copy, store untouched), or `replace` (swap in any node). `snapshot()`
  remains the deep-copy escape hatch for persistence.
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

See `tests/test_node_collection.py` (68 tests, plus node-owned-stats
persistence in `tests/test_serialization.py`): construction validation,
add/get/remove lifecycle, usage bumps, insertion order, combined-filter
metadata search, pruning + pinning, deep-copy snapshots, the vector index
(add/remove/prune vector lifecycle, cosine ranking, `within_newest`,
sort-by variants, degradation paths), field-level mutation
(`update_inplace`/`edit`/`replace`: bookkeeping preservation, no-bump
guarantee, `reset_stats`, re-embed on text change, unembedded re-attempt,
validation/error paths, reference `get`), and node-owned run telemetry
(records derive their run/token/time fields from the stored node's own
stats).
The suite is pure-unit — it never touches the network; tests inject a
deterministic fake embedder, and the default fastembed embedder is verified
to stay unconstructed.