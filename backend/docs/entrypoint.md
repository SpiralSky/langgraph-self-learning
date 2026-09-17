# Entrypoint

Source: `src/graphs/entrypoint.py` (`graph`), config: `backend/langgraph.json`

## What it is

The compiled outer graph that makes the package runnable as a langgraph
agent. Before it existed, `langgraph.json` pointed at `graph.py`, which only
defines the `Graph` **class** — there was no module-level `graph` variable, so
`langgraph dev` failed to boot with `KeyError: 'graph'`.

The module builds a minimal **START → GeneratorNode → END** graph and exports
the compiled result as `graph` (a langgraph `CompiledStateGraph`). That export
is what `langgraph.json` names:

```json
{ "agent": "./src/graphs/entrypoint.py:graph" }
```

## Structure

- Outer graph: `START -> generator ("answer") -> END`, carrying a
  `LearningGraphState` (`user_message` in, `response` out).
- The single `GeneratorNode` does the real work: it asks the LLM for builder
  tool calls, assembles a **nested** graph from the reusable node catalog, and
  runs it in-turn as a subgraph. The inner graph is never compiled into the
  outer one — see `docs/generator.md` for how that works.
- Model: the generator node binds `models.fast` from `model_settings.yaml`
  (via `get_chat_model`).

## Running

```commandline
cd backend
uv sync
uv run langgraph dev --port 2024
```

Needs `OPENAI_API_KEY` (or the configured `models.fast` provider) in the
environment/`.env`. On the first message the generator builds + runs the
nested graph for the request.

## Tests

`tests/test_entrypoint.py` — import + compile smoke tests that stay fully
offline (fake embedder + test key; model construction does not hit the
network). Full suite: `uv run pytest`.