# Prompt Node (`with_prompt`)

Source: `src/graphs/prompt_node.py`

## What it is

A small, standalone factory used to build LangGraph node callables. It binds a
prompt string as the first argument of a node function, so the node definition
can take the prompt explicitly while LangGraph calls it with just the state
(plus any injectable runtime kwargs).

It is a **single public function**; there is no class, no config, and it
imports **nothing from `langgraph`** (loose coupling — safe to use anywhere).

## API

```python
with_prompt(prompt: str) -> Callable[[Callable[Concatenate[str, P], R]], Callable[P, R]]
```

A factory that returns a decorator. Given a function:

```python
fn(prompt: str, state, *rest) -> Rt
```

it returns a wrapper callable as:

```python
wrapper(state, *rest) -> Rt
```

where `prompt` is bound to the string passed to `with_prompt`.

## Usage

```python
from graphs.prompt_node import with_prompt


@with_prompt("Grade the answer for correctness.")
def grade_node(prompt: str, state: State, config: RunnableConfig | None) -> State:
    ...  # prompt is the bound string; state and config are supplied at runtime
```

```python
builder = StateGraph(State)
builder.add_node(grade_node)          # node name = "grade_node" (fn __name__)
builder.add_edge(START, "grade_node")
graph = builder.compile()
graph.invoke({"answer": "..."})       # bound prompt + state + injected config
```

The function's post-prompt parameters (`state`, `config`, …) are re-exposed on
the wrapper via `__signature__`, which is what lets LangGraph inject runtime
kwargs such as `config: RunnableConfig` at call time.

## Behavior

- **Validation** (decoration time): raises `TypeError` unless the function has
  at least 2 parameters and its first parameter is annotated exactly `str`.
- **Prompt binding**: the first argument is substituted at decoration time
  (eager). The prompt is a plain `str` literal — no compiler/config registry.
- **Return passthrough**: the wrapper returns `fn`'s value untouched (a partial
  state update dict, full state, or anything else).
- **Metadata**: `functools.wraps` preserves `__name__`, `__doc__`, `__module__`,
  etc. — important because LangGraph derives the node name from the callable's
  `__name__`.
- **Signature**: `inspect.signature(wrapper)` equals `fn`'s signature minus the
  first (`prompt`) parameter. Tested against `langgraph==1.2.11` — state is
  passed positionally, injectable kwargs by keyword.
- **Sync only**: no async variants.

## Tests

See `tests/test_prompt_node.py` — covers prompt binding, arg forwarding, return
passthrough, metadata/signature preservation, validation errors, and a
LangGraph integration regression.