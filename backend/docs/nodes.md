# Node System

Source: `src/graphs/state.py` (`LearningGraphState`), `src/graphs/nodes/base.py`
(`AbstractNode` ABC + `GraphNode` protocol), `src/graphs/nodes/text_node.py`
(`TextNode`)

## What it is

The node system gives every graph node a uniform shape: a **retrievable node
object** that carries its metadata and produces a **LangGraph-ready callable**
on demand.

- `AbstractNode` — the ABC concrete nodes extend: metadata (`name`,
  `description`, `prompt`), by-name inputs (`params`), an explicit output map
  (`writes`), and the factory method `get_node(llm)`.
- `GraphNode` — a `typing.Protocol` describing the same overall shape; it lives
  on for structural typing (duck-typed nodes that don't extend `AbstractNode`
  still satisfy it).
- `TextNode` — the first concrete node: reads its declared params from state by
  name, fills the prompt, runs an LLM, and writes the result via `writes`.

State is a formal **pydantic model** (`LearningGraphState`) with exactly two
fields — `user_message: str | None` (the turn's input) and
`response: str | None` (the terminal answer). Nodes receive the full state
(model instance or dict) and return a **partial update** — the whole state
flows through; there is no stack reading and no reducer accumulation.

It is a *foundation*: the node modules import **nothing from `langgraph`** —
`base.py`/`text_node.py` are loosely coupled to the graph runtime.

## API

### `AbstractNode` (`base.py`)

```python
class AbstractNode(ABC):
    name: str
    description: str
    prompt: str
    params: dict[str, type]     # by-name reads from state
    writes: dict[str, str]      # node-local field -> state key
    run_counts: int             # successful runs so far
    output_tokens: OnlineStats  # output-token stats (Welford mean/std)
    output_time: OnlineStats    # wall-clock-per-run stats (Welford mean/std)

    def __init__(self, name, description, prompt,
                 params: dict[str, type] | None = None,
                 writes: dict[str, str] | None = None): ...
    def updated(self, **changes) -> Self: ...
    def get_node(self, llm) -> DualCallable: ...
```

- **`updated(**changes) -> Self`** — returns a new, validated instance of the
  same class with the given `name`/`description`/`prompt`/`params`/`writes`
  applied (rebuilt through the constructor, so every provided value is
  re-validated; the current instance is never mutated). Unknown kwarg →
  `TypeError`. `GraphNode` overrides it to accept **only**
  `name`/`description`/`prompt` — its `params`/`writes` are derived from the
  `input_map`/`output_map` bridge, so editing them raises `ValueError` (swap
  in a fresh node via the collection's `replace()` instead). This is the
  primitive behind `NodeCollection.update_inplace`/`edit`.
- **Run stats are carried across `updated()`** — the constructor re-inits the
  stats to zero, so every `updated()` explicitly carries them over via
  `_carry_stats` (copies `run_counts` and shares the same `OnlineStats`
  instances, so later folds accumulate in place). All `updated()` overrides
  (`TextNode`/`ToolCallNode`/`GraphNode`/`GeneratorNode`) do this.

- **`params`** — by-name inputs: each key is a state field the node reads, the
  value is the annotation it must match. `{}` (default) is a **generator**
  node that reads nothing.
- **`writes`** — explicit output map `{field: state_key}`; each entry targets a
  state key the node writes to. Defaults to `{"result": "response"}`.
- **Construction validation**: `name`, `description`, `prompt` non-empty;
  prompt placeholders `{word}` (regex `\{([A-Za-z_]\w*)\}`) must be a **subset
  of `params`** keys — an orphan placeholder raises `ValueError` (prevents a
  KeyError at run time).

`DualCallable` is the runtime contract `get_node(llm)` fulfills — an object with
a sync `invoke(state) -> dict` and an awaitable `ainvoke(state) -> dict`. The
callable also carries `__name__ == node.name`, which is how LangGraph derives
the node name in `add_node(...)`.

### `TextNode` (`text_node.py`)

```python
TextNode(
    name: str,
    description: str,
    prompt: str,
    params: dict[str, type] | None = None,   # {} = generator
    writes: dict[str, str] | None = None,    # {"result": "response"}
)
```

## Usage

```python
from graphs.nodes.text_node import TextNode
from graphs.llm import get_chat_model

node = TextNode(
    name="summarizer",
    description="Condenses the user message into one sentence.",
    prompt="Summarize this in one sentence: {user_message}",
    params={"user_message": str},
)

fn = node.get_node(get_chat_model("fast"))
fn.invoke({"user_message": "Long text to summarize..."})
# -> {"response": "A one-sentence summary..."}

gen = TextNode("joke", "Tells a joke", "Tell a joke about programming")
gen.get_node(get_chat_model("fast")).invoke({})   # generator: no params

builder = StateGraph(LearningGraphState)
builder.add_node(fn)               # node name = "summarizer" (fn.__name__)
builder.add_edge(START, "summarizer")
graph = builder.compile()
graph.invoke({"user_message": "Hey"})
```

## Behavior

- **By-name param reads** — for each `p in params`, the callable reads
  `state[p]` (plain dict) or `getattr(state, p)` (pydantic model state) and
  validates it against the param annotation via a `TypeAdapter` (built once per
  `get_node`). A mismatch raises pydantic `ValidationError`. Generator nodes
  read nothing and run with an empty fill.
- **Prompt templating** — every `{param}` placeholder is replaced by the
  validated value's `str()`. Unrelated braces are left alone; orphans were
  already rejected at construction.
- **LLM invocation** — the LLM is a langchain-core `Runnable` passed into
  `get_node(llm)` (not stored on the node). The result is taken from
  `.content` and surfaced as a `str`.
- **Writes / plain values** — the callable returns
  `{state_key: str(result.content) for ... in writes.values()}` (typically one
  entry). Values are written **plain** — there is no reducer and no
  accumulation. A state key is list-wrapped **only when** the target pydantic
  field is declared list-typed (via `is_list_annotation` in `text_node.py`); a
  plain-dict caller's state has no schema, so values stay unwrapped there.
- **State** — `LearningGraphState` is a pydantic `BaseModel` with
  `user_message: str | None` and `response: str | None`. In langgraph the node
  callable receives the **model instance**; the abstract node also renders
  dicts for plain callers.
- **Sync + async** — `invoke` is synchronous; `ainvoke` is a coroutine using
  `await llm.ainvoke(...)`.
- **Self-recording stats** — every node carries `run_counts` (successful
  runs) plus `output_tokens`/`output_time` (`OnlineStats`, Welford mean/std).
  The callables `get_node(llm)` returns **self-record** on each run: the
  text-node callable times its LLM invocation and reads
  `usage_metadata["output_tokens"]` from the result (`record_result` — a
  missing/non-numeric usage records time only), while tool and graph-node
  callables record elapsed time only. Stats live on the node, so when the
  same instance is shared (a `NodeCollection.get`, or a generator
  `from_collection` pull), runs recorded anywhere fold back into it.
  `record_run(*, tokens=..., elapsed_seconds=...)` / `record_result(...)` are
  public for external recording; both reject NaN/negative values.

## Tests

See `tests/test_nodes_base.py` (ABC + protocol/contract shape + stats
carry-over via `updated()`) and `tests/test_text_node.py` (construction,
by-name params, pydantic-model state, plain-value + list-wrap writes,
generator nodes, sync/async, self-recording stats, integration against
`langgraph==1.2.11`).