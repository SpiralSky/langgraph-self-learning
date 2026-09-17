# Graph Structure

Source: `src/graphs/graph.py` (`Graph`, `NodeView`, `GraphValidationError`,
`START`/`END`), `src/graphs/connections.py` (`Connection`,
`RoutingConnection`), `src/graphs/nodes/graph_node.py` (`GraphNode`)

## What it is

`Graph` is an editable, frameless graph structure: nodes (`AbstractNode`
instances) and connections are keyed by **manual, LLM-referenceable string
ids**, and the structure can be validated, rendered as plain text, and
compiled into a runnable langgraph `StateGraph`.

It is deliberately **not** a collection store — no vector search, no
auto-pruning, no usage tracking. Retrieval/persistence of node prototypes is
`NodeCollection`'s job (see `node-collection.md`); `Graph` owns the wiring.

- Nodes come from the `AbstractNode` family (`TextNode`, `GraphNode`, …);
  every graph declares an explicit pydantic `state_model` every node reads
  from / writes to (default `LearningGraphState`).
- Edges are `Connection` (plain) or `RoutingConnection` (LLM-chosen
  candidate); START/END are reserved, role-checked ids.
- Cycles and self-loops are allowed; removed node/edge ids are
  **tombstoned** (never re-addable) so LLM-held references stay safe.

## API

### Graph

`Graph(*, state_model: type[BaseModel] = LearningGraphState)`

Every graph takes an explicit `state_model`; custom schemas are intended
(`GraphNode` inner graphs declare their own).

| Method | Signature / behavior |
|--------|----------------------|
| `add_node(node_id, node, llm=None)` | Store a node under a manual id (optional per-node LLM); `ValueError` on reserved/non-string/duplicate/tombstoned id |
| `add_edge(edge_id, source, target, *, routing_prompt=None, model=None)` | Add a connection; single `target` without `routing_prompt` → `Connection`, otherwise `RoutingConnection`; validates id/role/endpoints/dedup |
| `remove_node(node_id)` | Remove the node and **CASCADE every incident edge**; `KeyError` if missing |
| `remove_edge(edge_id)` | Remove one edge; `KeyError` if missing |
| `update_node(node_id, new_node, llm=None)` | Replace the instance under the same id; no graph-wide revalidation |
| `update_edge(edge_id, **changes)` | Replace the stored edge, re-running add-edge checks; `KeyError` if missing |
| `validate()` | Full structural + schema validation; raises `GraphValidationError` listing **all** problems |
| `compile(llm=None) -> StateGraph` | Validate, then build a runnable langgraph `StateGraph` |
| `render(*, show_ids=False, show_descriptions=False, show_prompts=False) -> str` | Deterministic, LLM-parseable text |
| `nodes_by_name(name) -> list[NodeView]` | Read-only view of every node with that name |
| `nodes()` / `edges()` | Snapshot dicts `{id: item}` — nodes are **deep copies** |
| `get_node(id)` / `get_edge(id)` / `get_node_llm(id)` | Direct access (`KeyError` if missing) — `get_node` returns a **deep copy** of the stored node |
| `__len__` / `__contains__` | `len(graph)`, `id in graph` |

### Connections (`connections.py`)

| Class | Fields | Notes |
|-------|--------|-------|
| `Connection(source, target)` | `type = "standard"` | Plain one-to-one edge |
| `RoutingConnection(source, targets: tuple[str, ...], routing_prompt: str, model=None)` | `type = "routing"` | LLM-chosen fan-out: `source` reaches exactly one of `targets` at runtime |

- Dedup: no two edges may share an id or an equivalent
  `(source, frozenset(targets), type)`; standard and routing edges may coexist
  on the same endpoints.
- Role rules (enforced at `add_edge` **and** `validate()`): `START` may only
  be a source, `END` may only be a target; exactly **one** `START`-source edge
  is required, multiple `END` edges are allowed.
- Endpoint refs must name an existing node (or a reserved id).

### `NodeView`

Frozen, read-only `dataclass(id, name, description, prompt, params, writes)`
returned by `nodes_by_name`.

### `GraphNode` (`nodes/graph_node.py`)

Wraps a full inner `Graph` as one reusable `AbstractNode`:

```python
GraphNode(name, description, prompt, graph, input_map, output_map)
```

- `input_map: {inner_key: outer_key}` — which inner state keys this node
  sources from the outer state (required, non-empty; inner keys must exist on
  the inner graph's `state_model`).
- `output_map: {inner_key: outer_key}` — which inner terminal keys to surface
  to which outer keys; multiple entries produce a multi-output partial update.
- `params` (outer keys backed by `input_map` values, annotated with the inner
  field type) and `writes` (`{inner_key: outer_key}`) are derived
  automatically, so parent-graph validation just works.
- `get_node(llm)` yields a callable that reads/validates the outer params,
  builds the inner initial state, runs the inner graph (compiled lazily and
  **cached per bound LLM**), and surfaces the mapped outputs.
- Fits `NodeCollection` unchanged (proven by tests; no store changes).

## Usage

```python
from graphs.graph import START, END, Graph
from graphs.nodes.text_node import TextNode
from graphs.llm import get_chat_model

graph = Graph()

graph.add_node(
    "summarizer",
    TextNode(
        name="summarizer",
        description="Condenses the previous step's output into one sentence.",
        prompt="Summarize this in one sentence: {user_message}",
        params={"user_message": str},
    ),
    llm=get_chat_model("fast"),      # per-node override; else pass one to compile()
)

graph.add_edge("start->sum", START, "summarizer")
graph.add_edge("sum->end", "summarizer", END)

graph.validate()
print(graph.render(show_ids=True))
# START -> summarizer (summarizer) -> END

compiled = graph.compile()
compiled.invoke({"user_message": "Long text to summarize..."})
```

Routing — one LLM picks the target at runtime:

```python
graph.add_node("q", TextNode("questioner", "Asks a question",
                             "One question about {user_message}", params={"user_message": str}))
graph.add_edge("sum->q", "summarizer", "q")
graph.add_edge(
    "q->route", "q", ["summarizer", END],   # fan-out to candidates
    routing_prompt="Answer 'summarizer' or 'END' for what comes next: {user_message}",
    model=get_chat_model("fast"),
)

graph.compile().invoke({"user_message": "What is a closure?"})
```

Nested graphs — a `GraphNode` is just another node:

```python
inner = Graph()
# ... add inner nodes/edges (own state_model) ...
inner_comp = GraphNode(
    name="math_tutor",
    description="Runs a tutoring subgraph.",
    prompt="Inner tutoring session.",
    graph=inner,
    input_map={"user_message": "user_message"},   # inner_key: outer_key
    output_map={"response": "response"},      # inner terminal -> outer state key
)
outer.add_node("tutor", inner_comp)
```

## Behavior

- **Manual ids** — nodes and edges carry LLM-referenceable string ids; removed
  ids are tombstoned and cannot be re-added. `START`/`END` are reserved: you
  can never `add_node` them, and their role (source / target only) is checked
  at every `add_edge`.
- **Cycles & self-loops** — allowed by design; `validate()` does not reject
  them (an LLM-authored graph is expected to be a DAG, but the structure does
  not enforce it).
- **Edit ops** — `remove_node` cascades to incident edges; `update_node`
  swaps instances in place without graph-wide revalidation (call `validate()`
  yourself after bulk edits); `update_edge` re-runs full add-edge checks.
- **Copy getters** — `get_node(id)` and `nodes()` return **deep copies**, so
  external code (e.g. an LLM or `NodeCollection`) can never mutate the stored
  structure through them. Edit ops (`update_node` lazy, `update_edge`
  per-edge checks) are unchanged — invalid intermediate states are caught by
  `validate()`/`compile()`, which runs first in `compile()`.
- **Validation** — `GraphValidationError.errors` lists every problem found
  (`"; ".join` in the message). Checks: id rules, exact-one-START + END-role,
  endpoint refs, `params` (reads) and `writes.values()` (writes) are fields of
  `state_model`, prompt placeholders ⊆ `params`. `compile()` validates first.
- **compile()** — LLM resolution: per-node `add_node` llm wins, else the
  `compile(llm)` arg; a node or routing edge with neither raises `ValueError`
  naming it. Node ids become langgraph node names; `START`/`END` alias to the
  framework constants (`__start__`/`__end__`). Each routing edge becomes a
  conditional edge whose path callable runs its `routing_prompt` through an
  LLM and returns the chosen candidate id.
- **render()** — deterministic line list, one per edge
  (`n1 -> n2`, routing `n1 ->? (n2 | n3)`); reserved ids render as
  themselves. With `show_ids`/`show_descriptions`/`show_prompts`,
  per-node blocks are appended.
- **GraphNode execution** — the inner graph is compiled lazily on first use
  and cached per bound LLM; the outer callable validates by-name outer reads
  exactly like `_TextNodeFn`. `invoke()` returns a plain dict with defaults
  filled, so a partial inner initial state works.

## Tests

- `tests/test_graph.py` (2 smoke tests) + `tests/test_graph_structure.py`
  (37 tests) — construction, edit ops, cascade, tombstoning, dedup, role
  rules, endpoint checks, `validate()` aggregates, `render()`, `nodes_by_name`.
- `tests/test_graph_compile.py` (8 tests) — `compile()` to langgraph
  `StateGraph`: LLM resolution order, routing conditional edges (START/END
  aliasing), invokable artifact.
- `tests/test_graph_node.py` (8 tests) — inner-graph wrapping, input/output
  maps, multi-output surfaces, nesting, NodeCollection fit.

The suite is pure-unit — fake LLMs only, no network. Run from `backend/`:
`.venv/bin/python -m pytest tests/ -q`.