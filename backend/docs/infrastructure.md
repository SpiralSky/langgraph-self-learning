# Infrastructure & Utilities

Supporting modules used across the graph nodes.

## Observability (`observability.py`)

### `timed(name)` — decorator factory

Wraps a graph node callable to log its wall-clock duration on every invocation.

- On success: logs ``node=<name> elapsed_ms=<n>`` at INFO level.
- On exception: logs ``node=<name> error=<exc> elapsed_ms=<n>`` at WARNING level, then re-raises.
- Useful for spotting footprint regressions in node timing.

**Usage:** ``@timed("decision_maker")`` applied to a node function.

## Text Utils (`text_utils.py`)

### `message_to_text(message)` → `str`

Extracts plain-text content from a LangChain ``BaseMessage``, handling both formats:

1. ``str`` content — returned directly.
2. OpenAI-style block lists — iterates blocks, collecting ``{"type": "text", "text": ...}`` dicts and objects with a ``.text`` attribute; joins with spaces.
3. Fallback — ``str(message.content)`` for anything else.

Shared by ``retrieve_memory``, ``save_memory``, and ``response_builder``.

## Session Types (`session_types.py`)

### `SessionType` model

A single declared session template loaded from ``graph_config.yaml`` (under the ``session_types`` key). **YAML is guidance only** — the model may author its own session freely; the catalog only anchors the input analyzer's trigger vocabulary and optional hint block.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `label` | `str` | — | Human-readable display name |
| `summary` | `str` | — | One-line description used in prompts |
| `triggers` | `str` | `""` | Comma-separated cue phrases |
| `state_keys` | `list[str]` | `[]` | Suggested state keys (not enforced) |
| `system_block` | `str` | `""` | Template instruction block |
| `is_main` | `bool` | `True` | Whether offered to the input analyzer |

The ``main`` YAML key is aliased to ``is_main`` via ``Field(alias="main")``.

### `_load_session_types(path)` → `dict[str, SessionType]`

Parses the YAML catalog at ``path``. Skips the ``customizations`` top-level key and malformed entries; returns ``{}`` if the file is missing; raises ``ValueError`` on parse failures for known types.

### `analyzer_catalog_text()` → `str`

Renders the compact main-type catalog for the input analyzer prompt. Only ``is_main`` types are listed as ``id: summary; cues: triggers``. Returns ``"(none)"`` when empty.

## Execute-Code Infrastructure (`nodes/execute_code.py`)

### Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `SSH_KEY_PATH` | ``"/shared/keys/graph_to_code"`` | Ed25519 key for SSH auth |
| `CODE_RUNNER_USER` | ``"execuser"`` | SSH user on the code-runner |
| `CODE_NET_NETWORK` | ``"code-net"`` | Docker network for code execution |
| `MAX_OUTPUT_LINES` | ``100`` | Max stdout/stderr lines kept |
| `MAX_OUTPUT_BYTES` | ``10000`` | Max stdout/stderr bytes kept |

### `_ssh_exec(target_host, command, timeout=30)` → `dict`

Executes a command via SSH to a code-runner or session container using the graph_to_code ed25519 key (``StrictHostKeyChecking=no``). Returns ``{"stdout", "stderr", "returncode"}``; on timeout returns ``returncode=-1`` with a timeout message in stderr.

### `_parse_session_target(state)` → `str | None`

Resolves the SSH target hostname from ``/shared/ledger/sessions.json`` registry (returns the ``host`` of any active session), or falls back to ``"code-runner"`` pool when no session is found or the file is missing/unreadable.

### `execute_code(state, config=None)` (node)

The graph node (intent ``"execute_code"``). Scans ``state.teaching_strategy.execution_plan`` for a step containing ``"execute"`` or ``"code"``. Extracts the ``python3 -c "..."`` command (or falls back to the raw step), runs it via ``_ssh_exec`` with a 60s timeout, caps output at ``MAX_OUTPUT_LINES``/``MAX_OUTPUT_BYTES``, and returns ``{"final_output": <capped text>}``. Returns ``{}`` when no code step is found (pass-through).
