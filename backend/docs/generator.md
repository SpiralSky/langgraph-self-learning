# Generator Node

Source: `src/graphs/generator.py` (`GeneratorNode`, `decode_builder_calls`,
`build_state_model`, `wrap_reused`, `save_reused`)

## What it is

`GeneratorNode` is the runtime counterpart to a catalog of reusable steps:
given a user request plus the sectioned behavior points, it asks an LLM for a
**single-pass list of builder tool calls** (`add_node` / `add_edge`), applies
them to a fresh `Graph` with an automatically derived inner state model,
**validates** the structure (retrying the whole spec a bounded number of times
with error feedback), and executes the result as a **nested subgraph in-turn**.
The node's own LangGraph output is the final `response`.

Safety and control come from the edges being closed:

- The node catalog is **fixed** to `TextNode` (type `"text"`) and
  `ToolCallNode` (type `"tool"`, whitelisted through a `ToolRegistry`) — no
  arbitrary class can be instantiated.
- The **core prompt template is a module constant** and never rewritten;
  behavior points are data rendered into it per node.
- The observed builder contract is tiny: `add_node(id, type, name,
  description, prompt?, params?, writes?, tool?, reuse?)` and
  `add_edge(source, target)`. `START`/`END` are the reserved graph endpoints.
- Tool-node args are **not** bakeshable into the node — fixed `tool_args`
  are rejected, and arguments always flow through the `"args"` state key from
  an upstream text step (see `tools.md`).

Reuse: nodes whose `add_node` call passes `reuse: true` are collected as the
run's save-ready artifacts — a single flagged id saves that node directly, a
chain saves as a generated `GraphNode` — and, with auto-save enabled (the
default), persisted into a dedicated `NodeCollection` at **end of turn**
(`data.md` describes the file layer). `auto_save=False` keeps everything
transient.

## API

```python
GeneratorNode(
    name: str,
    description: str,
    *,
    behaviors: list[BehaviorGroup] | str | Path | None = None,  # YAML path or objects
    retries: int = 3,          # whole-spec regeneration attempts after a failure
    registry: ToolRegistry | None = None,   # forwarded to generated tool nodes
    auto_save: bool = True,    # save reuse-flagged artifacts at end of turn
    collection: NodeCollection | None = None,  # dedicated artifact store
    save_path: str | Path = COLLECTION_PATH,   # backend/data/collection.json
    reuse_pinned: bool = False,# protect saved artifacts from auto-pruning
    writes: dict[str, str] | None = None,      # {"result": "response"}
)
```

- Reads `user_message` from state; `params` is fixed to
  `{"user_message": str}`.
- `behaviors` accepts pre-built groups, a `str`/`Path` to a behaviors YAML, or
  `None` for `backend/behaviors.yaml`. The rendered result is exposed as
  `node.behaviors_text`.
- `updated(**changes)` allows **only** `name`/`description` — the prompt,
  params, writes, behaviors, and retries are structural for a generator node.

### Builder call decoding

`decode_builder_calls(response) -> list[{"name", "args"}]` normalizes an LLM
response to the builder call list: langchain `tool_calls` when present,
otherwise a JSON `{"calls": [{name, args}, ...]}` blob parsed from `content`.
Malformed output raises `TypeError`/`ValueError`, which the retry loop feeds
back as error feedback.

### Inner state auto-build

`build_state_model(nodes) -> type[BaseModel]` derives the inner graph's pydantic
state model as the union of every node's declared `params` keys (typed by the
node's actual annotation) plus every `writes` target key (defaulting to `str`)
plus the always-present `user_message` and `response` fields — so the generated
structure always validates against its own schema.

## Execution flow

1. Bind the builder tool definitions (`llm.bind_tools(BUILDER_TOOL_DEFS)`)
   when the LLM supports it; render the fixed template with the user message
   and append the rendered behaviors.
2. For each attempt (initial + `retries` whole-spec re-generations): invoke the
   LLM, decode the builder calls into nodes/edges, assemble the `Graph`, and
   run `graph.validate()`. Any `GraphValidationError` (or decode/build error)
   is appended to the prompt as `- <error>` feedback lines and the spec is
   regenerated in full.
3. After `retries + 1` total attempts without a valid graph, raise
   `RuntimeError` naming the last errors.
4. On success, wrap the inner `Graph` as a `GraphNode`
   (`input_map={"user_message": ...}`, `output_map={"response": ...}`) and run
   it through the existing nested-subgraph machinery; the node returns
   `{"response": <terminal output>}`.
5. `_GeneratorFn` exposes `last_graph` (the generated `Graph`) and `reuse_ids`
   after a successful run, and auto-saves the reuse artifacts at end of turn
   when enabled.

## Usage

```python
from graphs.generator import GeneratorNode
from graphs.llm import get_chat_model

gen = GeneratorNode(
    name="answer",
    description="Answers the user's request with a generated workflow.",
)

fn = gen.get_node(get_chat_model("fast"))
result = fn.invoke({"user_message": "Compare closures and classes"})
# -> {"response": "..."}
# fn.last_graph   -> the generated Graph (re-rendered/re-compiled after)
# fn.reuse_ids    -> add_node ids flagged reuse: true
```

## Behavior

- **Single-pass contract** — one LLM invoke returns the whole spec; there is
  no multi-call dialog loop. Retries regenerate the *entire* spec, never
  partial edits.
- **Validation is the gate** — the internal `Graph.validate()`/`compile` path
  (structure, roles, exactly-one START, param/write fields, duplicate ids)
  decides success; the LLM sees only aggregate error text between attempts.
- **Nested execution** — the generated graph is not compiled into the outer
  graph; it runs in-turn through the `GraphNode` bridge, so it composes
  exactly like any other node.
- **Reuse auto-save** — `wrap_reused(reuse_ids, graph, ...)` produces the
  save-ready prototypes (single node, or a `GraphNode` wrapping the full
  graph); `save_reused(nodes, collection, path, pinned=...)` adds them to the
  dedicated collection and persists it atomically. Non-serializable prototypes
  raise `ValueError` before anything is stored.

## Tests

- `tests/test_generator.py` (26 tests) — template/behaviors assembly,
  decode paths, inner state auto-build, retry loop with error feedback, nested
  execution, failure exhaustion.
- `tests/test_reuse_save.py` (15 tests) — reuse id collection, single-node vs
  chain wrapping, collection persistence, transient (`auto_save=False`) mode.

The suite is pure-unit — FakeLLM responses only, no network. Run from
`backend/`: `.venv/bin/python -m pytest tests/ -q`.