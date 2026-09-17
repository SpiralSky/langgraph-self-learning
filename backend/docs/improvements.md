# Improvements

Source: `src/graphs/feedback.py` (`Patch`, `parse_regenerated`,
`diff_behaviors`, `compute_improvement_patches`), `src/graphs/patches.py`
(`save_behaviors`, `load_patches`, `append_patches`, `apply_patches`),
`src/api/feedback.py` (`create_app`, `FeedbackRequest`)

## What it is

The feedback loop that closes the circle on LLM-generated graphs: a learner
attaches a suggestion (a comment on a Q/A turn) and the system turns it into
**edits on the behavior data** — not on code. The LLM **regenerates the
affected behavior entries whole** (every entry keeps its stable key), then the
difference between the regenerated registry and the snapshot taken at the time
of the suggestion is computed **programmatically** as a structured patch list.
Patches are persisted to a JSONL log and applied deterministically,
most-recent-wins. The core generator prompt template is never rewritten — only
the behavior *data* changes, and the prompt is re-assembled from data each run.

The whole pipeline is exposed both as a library function and as a thin FastAPI
`POST /feedback` endpoint.

## Why whole-entry regeneration + programmatic diff

- Asking the LLM to emit a diff yields worse results; asking it to restate the
  full registry (keeping valid entries and their keys intact) is more robust.
- The **diff step is a pure function** (`diff_behaviors`) — it matches
  `(title, point_key)` between old and new, so it is deterministic and
  unit-testable without an LLM.
- The output is an ordered list of `Patch` objects whose semantics are the
  lowest common denominator: `add` / `update` / `remove` on a behavior point.

### The prompt contract

The LLM is asked for ONE JSON object (and nothing else):

```json
{"groups": [{"title": "...", "points": [{"key": "...", "text": "..."}]}]}
```

Every carried-over point must keep its **exact current key**; new points get a
new short key. The prompt renders the suggestion (question + prior responses +
comment), a **compact graph-state summary** (metadata-only, truncated at 800
chars), and the current behavior registry (`## title` / `- [key] text`).
Malformed regenerated output raises `ValueError` naming the failure.

## Patch format

A `Patch` is a pydantic model with a globally monotonic `seq` (apply order =
precedence) plus one of two shapes:

- **Behavior-targeted** (default): `title`, `point_key`, `content`.
  `add`/`update` need non-empty content; `remove` names the key only.
- **Node-targeted** (optional): `node_id` targeting a `NodeCollection` entry
  plus a `node` payload — field edits `{name, description, prompt, params,
  writes}` for `update`, or a full type-tagged serialized node (a `"type"`
  key) for `add`/`replace`. `remove`-on-node is unsupported.

Shape validation lives in a pydantic validator; invalid patches raise
`ValueError` at construction.

```python
Patch(seq=3, action="update", title="Answer directly",
      point_key="cite-uncertainty",
      content="When unsure, say so and list the missing piece.")
```

## The diff

`diff_behaviors(old_groups, new_groups, *, seq_prefix=0)` matches by
`(title, point_key)` and returns patches in `seq` order:

| Case | Patch |
|------|-------|
| Same key, different text | `update` |
| Key only in the regenerated set | `add` |
| Key only in the snapshot | `remove` |
| Same key, same text | none |

Duplicate `(title, point_key)` in the regenerated set, or a bad `seq_prefix`,
raise `ValueError`.

## Storage & application (`patches.py`)

- **`PATCHES_PATH`** = `backend/data/patches.jsonl`, one patch JSON object per
  line, written atomically (temp sibling + `os.replace`).
- `append_patches(patches, path?)` merges the existing log with new patches and
  rewrites it atomically — an append can never leave a half-written line.
- `load_patches(path?)` reads the log, seq-sorted; a missing/empty file yields
  `[]`.
- `apply_patches(patches, *, behaviors_path?, collection?)` applies in `seq`
  order (later seq wins):
  - behavior patches upsert/delete by `(title, point_key)`, creating groups on
    demand and dropping them when empty; the behaviors file is written back
    **only** when a behavior patch applied;
  - node-targeted patches route to `collection` (`add`/`update`/`replace`)
    and never touch the behaviors file;
  - application is **idempotent** — re-running the same `seq` range reproduces
    the same state.
- `save_behaviors(groups, path?)` writes every point with an explicit `id` so
  stable keys survive a `load_behaviors` round-trip losslessly.

## Precedence

Most recent change wins. Because the regeneration step already re-emits the
*whole* registry, superseded entries simply fall out of the regenerated set —
obsolete patches are effectively neutralized by later removes/updates against
the same key. `seq` is the total order that makes this well-defined: the next
patch starts after the log's last `seq`.

## Behaviors-as-data rule

Application never touches the generator prompt template: the template is a
module constant (see `generator.md`) and behavior changes only alter the data
that is rendered into it on the next run. Peer docs: `behaviors.md` for the
data format and stable keys, `generator.md` for the node that consumes them,
`data.md` for the storage/serialization layer this builds on.

## The `POST /feedback` endpoint

`create_app(*, llm=None, behaviors_path=None, patches_path=None,
collection=None)` builds a FastAPI app:

```
POST /feedback
{"question": "...", "responses": [...], "comment": "...", "graph_state": ...}
```

Flow: load current behaviors → `compute_improvement_patches(..., seq_prefix=
<log max + 1>)` (monotonic across requests) → `append_patches` +
`apply_patches` → `200 {"applied": [patch dicts, exclude_none]}`.

- The LLM resolves **lazily** to `get_chat_model("fast")` on first request —
  importing or building the app never touches the network, environment, or
  model settings.
- Compute failures (malformed regenerated registry) → `400`; application
  failures → `500`, both with a `detail` string. A body missing
  `question`/`responses`/`comment` → `422`.

Run it from `backend/` with `uvicorn api.app:app`.

## Behavior

- **Regenerate, don't diff with the LLM** — the model restates the registry
  whole; the programmatic diff is the deterministic backbone.
- **Data in, data out** — feedback edits behavior entries (and optionally the
  node store), never prompt text.
- **Deterministic application** — `seq`-ordered, idempotent, atomic writes.
- **Lazy LLM binding** — the endpoint is import-safe and testable with a
  `TestClient` plus a stub LLM (or no LLM at all for application tests).

## Tests

- `tests/test_feedback.py` (20 tests) — patch validation, regenerated parsing,
  pure diff, template assembly.
- `tests/test_patches.py` (20 tests) — atomic JSONL round-trip, seq-sorted
  loads, deterministic/idempotent application, node-targeted routing,
  behaviors file untouched by node patches.
- `tests/test_feedback_api.py` (6 tests) — happy path, seq continuation, 422
  on missing fields, 400 compute error, 500 apply error (behaviors untouched).

The suite is pure-unit — FakeLLM responses only, no live LLM, no network. Run
from `backend/`: `.venv/bin/python -m pytest tests/ -q`.