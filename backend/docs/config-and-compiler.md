# Config & Compiler

## Config System

### Config files

| File | Role |
|------|------|
| `config.yaml` | Runtime config: model providers, API keys (via env vars), reasoning defaults |
| `config.template.yaml` | Template with example provider configurations |
| `graph_config.yaml` | Pedagogical customizations (graph-level + per-node) and session type templates |

### Resolution order

1. `config.yaml` is loaded and env-var placeholders (`${VAR}`) are resolved
2. `graph_config.yaml` is merged in under the `customizations` key
3. The `Config` model validates the result

### `read_config(path)` → `Config`

Reads `config.yaml`, resolves `${VAR_NAME}` placeholders from the environment (and `.env` file via `python-dotenv`), merges `graph_config.yaml` customizations, and validates with the `Config` model.

### `Config` model

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `provider_data` | `dict[str, ProviderData]` | — | Per-provider endpoint/key/model |
| `model_providers` | `dict[str, str]` | — | Logical name → provider key mapping |
| `memory_top_k` | `int` | `5` | Max memory search results |
| `memory_max_chars` | `int` | `300` | Max chars per memory snippet |
| `session_min_overlap` | `int` | `1` | Minimum turn overlap for session continuity |
| `customizations` | `CustomizationsConfig` | default | Node + graph customization settings |
| `reasoning_defaults` | `dict[str, str]` | `{}` | Per-node reasoning effort overrides |
| `model_override` | `str \| None` | `None` | Global model override for all LLM calls |

### `ProviderData`

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `api_endpoint` | `str` | — | Provider endpoint URL |
| `api_key` | `str` | — | API key |
| `model_id` | `str` | — | Model identifier |
| `thinking_level` | `str \| None` | `None` | Legacy single thinking level |
| `reasoning_levels` | `list[str] \| None` | `None` | Accepted `reasoning_effort` values |
| `reasoning_default` | `str \| None` | `None` | Default reasoning effort |

### `CustomizationsConfig` — graph + per-node

**Graph-level** (`customizations.graph`, `ConfigDict(extra="allow")`):

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `pedagogical_style` | `str \| None` | `None` | `'socratic'`, `'direct'`, `'scaffolding'`, etc. |
| `verbosity` | `str \| None` | `None` | `'concise'`, `'detailed'`, `'balanced'` |
| `tone` | `str \| None` | `None` | Default tone for all nodes |
| *(free-form)* | — | — | Other blocks (e.g. `tone_calibration`) |

**Per-node** (`customizations.nodes`, each `NodeCustomization`, `extra="allow"`):

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `tone` | `str \| None` | `None` | Override the node's tone/style |
| `constraints` | `str \| None` | `None` | Additional hard rules |
| `extra_instructions` | `str \| None` | `None` | Extra instructions appended to the node's prompt |
| `prompt_overrides` | `dict \| None` | `None` | Key-value substitutions for the node's prompt |
| `formatting_rules` | `str \| None` | `None` | Markdown/formatting rules appended to the node's prompt |

## Prompt Compiler

### `compile_prompts(nodes, customizations)` → `dict[str, str \| None]`

Builds the compiled prompt for every node. Stores results in the module registry (read via `get_node_prompt`). Precedence order:

1. **Base prompt** — `Node.prompt`, after `prompt_overrides` key-value substitution
2. **Graph free-form blocks** — `customizations.graph` entries whose intent targets match the node's intent (declared in `INTENT_TARGETS`)
3. **Structured style directives** — `pedagogical_style`, `verbosity`, `tone` for teaching intents (`analyze`, `decide`, `generate`)
4. **Per-node settings** — `tone`, `constraints`, `formatting_rules`, `extra_instructions`

### Key constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `INTENT_TARGETS` | `dict[str, tuple[str,...]]` | Graph blocks → node intents they broadcast to |
| `STYLE_INTENTS` | `("analyze", "decide", "generate")` | Intents that receive structured style directives |
| `_COMPILED` | `dict[str, str \| None]` | Module registry of compiled prompts |

### `get_node_prompt(name, default=None)`

Returns the compiled prompt for a node. Falls back to `default` when the registry hasn't been populated (e.g., module imported without building the full graph).

## LLM Init

### `get_chat_model(name, temperature=0.0, model_override=None)`

Builds a configured chat model for a node/phase.

- Reads provider data for `name` from the shared config
- Reasoning effort: per-node override (`Config.reasoning_defaults`) → provider's `default_reasoning` → none for non-reasoning models
- `model_override` takes precedence over configured `model_id` (useful for eval runs with a different model)
- Returns an `init_chat_model` runnable (openai provider)

## Memory

### `memory.py` setup

- Creates `data/mem0_db/` collection directory
- Loads `memory_llm` and `memory_embedder` provider data from config
- Builds a `mem0.Memory` instance with Qdrant vector store (`collection_name: "user_memories"`)

### Knowledge model vs session storage

| Storage | What it holds | Persistence |
|---------|---------------|-------------|
| `memory_results` (Qdrant) | Compacted memory search hits | Across turns |
| `knowledge_model` (state) | Per-thread mastery model | Across turns |
| `SessionRecord` (state) | Full session state + control | Session lifetime |