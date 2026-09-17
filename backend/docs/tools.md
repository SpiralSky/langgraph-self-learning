# Tools

Source: `src/graphs/nodes/tool_node.py` (`ToolCallNode`),
`src/graphs/tools/registry.py` (`ToolRegistry`, `default_registry`,
`ToolError`, `require_arg`, `int_arg`), `src/graphs/tools/ddgs.py`
(`ddgs_search`), `src/graphs/tools/mem0.py` (`mem0_remember`,
`mem0_retrieve`)

## What it is

`ToolCallNode` is a node that runs a **whitelisted tool** instead of an LLM: it
reads a tool name and its JSON-style arguments from state, dispatches the call
through a `ToolRegistry`, and writes the stringified result to its declared
`writes` keys. It is deliberately the *only* way a graph can reach outside the
model — there is **no arbitrary-function surface**:

- Every tool lives in a `ToolRegistry`, an explicit `name -> callable` map
  where the callable is a plain `dict -> str` function. Only registered names
  are callable.
- `ToolCallNode` validates its tool against the registry **at construction
  time** (build-time whitelist), so a node (and therefore any generated graph)
  can never reference an unregistered tool.
- The node's `prompt` and `params` are **fixed**: `params = {"tool": str,
  "args": dict}` and a template prompt naming the tool. There is no free-form
  prompt to steer.
- The `llm` argument to `get_node(llm)` is ignored — it exists only to satisfy
  the `GraphNode` protocol.

Importing any tool module constructs **nothing**: no DDGS client, no mem0
`Memory` store. Clients are built lazily on first actual tool run, so tests
and import-time are side-effect free.

## API

### `ToolRegistry` (`tools/registry.py`)

```python
registry = ToolRegistry()
registry.register("my_tool", my_fn)          # name -> callable (dict -> str)
@registry.register("decorated")              # or use as a decorator
def decorated(args: dict) -> str: ...
registry.get("my_tool")                      # the callable; KeyError if unknown
registry.names()                             # sorted tuple of registered names
"my_tool" in registry                        # membership check
```

- `register` rejects empty names, non-callables, and **duplicates** (re-registering
  a name is a whitelist smell and raises `ValueError`).
- `get` raises `KeyError` for unknown names — there is no fallback delegation.
- Helpers: `require_arg(args, key)` (raise `ValueError` when a required arg is
  missing), `int_arg(args, key, default)` (coerce an arg to `int`).
- `ToolError` distinguishes "a whitelisted tool was dispatched but failed to
  complete" from programming errors.

### `default_registry()`

The shared registry of the three builtin tools, built **lazily once**:

| Tool name | Callable | Purpose |
|-----------|----------|---------|
| `ddgs` | `ddgs_search(args)` | DuckDuckGo web search → numbered `title / URL / snippet` digest |
| `mem0_remember` | `mem0_remember(args)` | Store a raw memory text in the local mem0 store |
| `mem0_retrieve` | `mem0_retrieve(args)` | Top-k vector retrieval of raw memories |

- `ddgs_search`: `{"query": str (required), "max_results": int (default 5,
  capped 10)}`. Returns `"No results."` on an empty match; raises `ToolError`
  on a `DDGSException`.
- `mem0_remember`: `{"text": str (required), "user_id": str (default
  "default")}` → `"Remembered: <text>"`. Stores with `infer=False`.
- `mem0_retrieve`: `{"query": str (required), "top_k": int (default 5),
  "user_id": str (default "default")}` → one `- <memory>` line per hit,
  `"No memories found."` when empty.
- mem0 is **fully local** — fastembed embedder + Chroma vector store + SQLite
  history under `backend/data/` — no API key, no LLM provider invocation.

### `ToolCallNode` (`nodes/tool_node.py`)

```python
ToolCallNode(
    name: str,
    description: str,
    tool: str,                          # whitelisted name, e.g. "ddgs"
    registry: ToolRegistry | None = None,  # default: default_registry()
    *,
    writes: dict[str, str] | None = None,  # {"result": "response"}
)
```

- `tool` must be a non-empty string registered in the registry; an unknown name
  raises `ValueError` at construction.
- Fixed `params={"tool": str, "args": dict}` — both are read from state by
  name. The effective tool is the state's `tool` value, falling back to the
  constructor's `tool`.
- `args` coercion: `None` → `{}`, a dict → as-is, a JSON string → must decode
  to an object. Anything else (scalars, raw lists, malformed JSON) raises
  `TypeError`/`ValueError`.
- `get_node(llm)` returns a LangGraph-ready callable that runs
  `registry.get(tool)(args)` and spreads `str(result)` over the `writes` state
  keys with the same plain-value / list-wrap rule as `TextNode`.
- `updated(**changes)` accepts only `name`, `description`, `tool`, `writes` —
  `prompt`/`params` are fixed.
- `to_dict`/`from_dict` (`"type": "tool"`, see `data.md`): carries metadata,
  the tool name, and writes. The **registry is not serialized** — a restored
  node dispatches through `default_registry()`.

## Usage

```python
from graphs.nodes.tool_node import ToolCallNode
from graphs.tools import default_registry

node = ToolCallNode(
    name="search",
    description="Searches the web with DuckDuckGo.",
    tool="ddgs",
    writes={"result": "response"},
)

fn = node.get_node(llm=None)               # llm is ignored
fn.invoke({"tool": "ddgs", "args": {"query": "what is a closure",
                                    "max_results": 3}})
# -> {"response": "1. Title\n   URL\n   Snippet\n\n2. ..."}
```

Args typically arrive from an upstream `TextNode` that writes the `"args"` key
as a JSON object string:

```python
from graphs.nodes.text_node import TextNode

planner = TextNode(
    name="plan_query",
    description="Picks the search query.",
    prompt='Output JSON: {"query": "<search terms>"}',
    params={"user_message": str},
    writes={"result": "args"},
)
```

## Behavior

- **Whitelist enforced at build time** — the registry check runs in the
  `ToolCallNode` constructor, so a bad tool name fails immediately with a
  message listing `registry.names()`.
- **Args come from state, not construction** — there is deliberately no way to
  bake fixed args into the node; arguments flow through the `"args"` state key
  every turn.
- **Runtime failures are `ToolError`** — network/transport problems surface
  distinctly from programming errors.
- **Lazy clients** — importing `graphs.tools` or the tool modules opens no
  connections; the DDGS context manager and the mem0 `Memory` are created only
  when a tool actually runs.

## Tests

- `tests/test_registry.py` (14 tests) — registration, decorator form,
  whitelist/duplicate rejection, `get`/`names`, arg helpers.
- `tests/test_tool_node.py` (24 tests) — construction whitelist, fixed
  params/prompt, args coercion, state-vs-ctor tool precedence, writes
  list-wrap rule, `updated`/serialization.
- `tests/test_tools.py` (12 tests) — builtin tool argument validation with
  fakes injected at the registry level (no network, no mem0 construction).

The suite is pure-unit — run from `backend/`:
`.venv/bin/python -m pytest tests/ -q`.