# Persistence & Data

Source: `src/graphs/persistence/serialization.py` (`serialize_annotation`,
`deserialize_annotation`, `node_from_dict`, `connection_from_dict`),
`src/graphs/persistence/storage.py` (`ensure_data_dir`, `read_json`, `write_json`,
`dump_collection`, `load_collection`)

## What it is

Graph artifacts persist as **JSON via explicit serializers** — every node
class, the connection types, and `Graph` implement `to_dict`/`from_dict`
(`Connection`/`RoutingConnection` have `as_dict` and rebuild via
`connection_from_dict`). A small storage layer under `backend/data/` writes
atomic files and round-trips whole `NodeCollection` instances.

Two design points matter:

- **Type-tagged node dispatch** — every serialized node carries a `"type"`
  tag (`"text"`, `"tool"`, `"graph"`). `node_from_dict` dispatches on the tag
  to the owning module's `from_dict`, so nested payloads rebuild without the
  caller knowing the concrete class. Registration is lazy: modules
  self-register on import.
- **Per-node run stats are persisted; other bookkeeping is not** — each node
  carries inline `{"node": …, "stats": …}`, where `stats` holds `run_counts`
  plus the output-token and elapsed-time summaries (count/mean/std). The
  per-entry bookkeeping that cannot be rebuilt — ids, `pinned`, timestamps,
  use counts — is deliberately **not** stored. `load_collection` restores
  prototypes and re-embeds them on `add` with the supplied (or default)
  embedder; an embedding failure degrades to `embedded=False` exactly like a
  normal `add`.

## API

### Annotations (`persistence/serialization.py`)

```python
serialize_annotation(str)            # -> "str"
serialize_annotation(list[str])      # -> "list"      (generics normalize)
serialize_annotation(str | None)     # -> "optional:str"
deserialize_annotation("optional:list")  # -> list | None
```

- Supported set: `{str, int, float, bool, list, dict}`. `Annotated` wrappers
  are stripped, `X | None`/`Optional[X]` unions normalize onto the core type
  with an `"optional:"` prefix, generic origins (`list[str]`) normalize to the
  bare name. Anything else raises `ValueError`.

### Nodes & connections

| Type | Tag | Carries | Notes |
|------|-----|---------|-------|
| `TextNode` | `"text"` | metadata, prompt, params (serialized annotations), writes | `from_dict` restores annotations |
| `ToolCallNode` | `"tool"` | metadata, tool name, writes | params/prompt structural; **registry not serialized** — restored nodes use `default_registry()` |
| `GraphNode` | `"graph"` | metadata, prompt, inner `Graph` payload, input/output maps | restores as a wrapper around the rebuilt inner graph |
| `Connection` | `"standard"` | source, target | — |
| `RoutingConnection` | `"routing"` | source, targets, routing_prompt | `model` is not serializable and is **dropped** — restored graphs need an `llm` at compile |

`Graph.to_dict`/`from_dict` store nodes and edges plus the state model; a
dynamic (generated) inner state model is encoded as `{"kind": "dynamic",
"fields": {name: annotation-name}}` and rebuilt via pydantic `create_model`.
Unknown fields/annotations raise rather than silently round-trip garbage.

### Storage (`persistence/storage.py`)

```python
ensure_data_dir() -> Path          # idempotently create backend/data/
read_json(path)                    # FileNotFoundError when missing
write_json(path, obj)              # atomic: temp sibling + os.replace
dump_collection(collection, path=COLLECTION_PATH)
load_collection(path=COLLECTION_PATH, embedder=None) -> NodeCollection
```

- `write_json` writes a `*.tmp` sibling first, then `os.replace` — a failed
  write never corrupts an existing file; parent dirs are created on demand.
- `dump_collection` persists each serializable prototype as an inline entry —
  `{"node": <type-tagged dict>, "stats": {…}}` under `{"nodes": [...]}` — where
  `stats` carries `run_counts` and the token/time (count/mean/std) summaries. A
  node without `to_dict` raises `ValueError` naming it.
- `load_collection` restores the file (missing file → empty collection),
  rebuilding every node through `node_from_dict` and applying each entry's
  `stats` directly onto the deserialized node (run stats live on nodes, not on
  collection entries). Legacy `{"nodes": [<node dict>]}` dumps (no `stats`)
  still load, with stats defaulting to empty.

## Layout

- `backend/data/` — runtime artifacts root (`DATA_ROOT`).
- `backend/data/collection.json` — the saved reuse-artifact collection
  (`COLLECTION_PATH`).
- `backend/data/mem0/` — the local mem0 Chroma vector store.
- `backend/data/mem0_history.db` — the local mem0 SQLite history.

```json
{
  "nodes": [
    {
      "node": {
        "type": "graph",
        "name": "gen",
        "description": "...",
        "prompt": "Reusable graph generated at runtime from the request.",
        "graph": {
          "type": "graph",
          "state_model": {"kind": "dynamic", "fields": {"user_message": "str"}},
          "nodes": {"a": {"type": "text", "name": "first", "prompt": "Start {user_message}", "params": {"user_message": "str"}, "writes": {"result": "notes"}}},
          "edges": {"e1": {"type": "standard", "source": "START", "target": "a"}}
        },
        "input_map": {"user_message": "user_message"},
        "output_map": {"response": "response"}
      },
      "stats": {
        "run_counts": 3,
        "tokens": {"count": 2, "mean": 15.0, "std": 2.0},
        "time": {"count": 3, "mean": 0.0012, "std": 0.0003}
      }
    }
  ]
}
```

## Behavior

- **Explicit over implicit** — serializers are hand-written on each class, not
  derived from introspection; unknown tags or annotations raise instead of
  half-restoring.
- **Atomic writes** — the temp-then-rename pattern is the one mutation path
  for data files.
- **Runtime deps stay out of the payload** — tool registries and routing
  models are runtime wiring, restored from the default/environment instead of
  the file.

## Tests

- `tests/test_serialization.py` (28 tests) — annotation registry,
  type-tagged node round-trips, connections, dynamic/static `Graph`
  serialization, storage helpers (incl. inline per-node `stats` dump/load
  round-trip and legacy-format back-compat).

The suite is pure-unit — run from `backend/`:
`.venv/bin/python -m pytest tests/ -q`.