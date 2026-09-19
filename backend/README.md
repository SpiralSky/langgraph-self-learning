#### Warning: Mainly Vibe-Coded

The base graph was written completely by me, but everything after revision `ef63dd40ebc4a8aa1b5d78fbdc67199cfdcdb675`
is vibe-coded.

Feel free to use this project as you wish, fork or make a better one. I honestly really hope someone makes a *actually good implementation* of this.

Please.

#### Details
Basically tries to reduce shortcoming of AI learning:
AI does not feed you information directly, instead asking questions to lead the learner.

Planned to add support with code so the learner can run code for learning. After all, to properly learn something,
you need to either: explain it in your own words (possibly given a scenario) or build it yourself.

#### Documentation

See ``docs/`` for developer/operator documentation:

| Doc | Covers |
|-----|--------|
| ``docs/prompt-node.md`` | `with_prompt` factory: bind a prompt to a LangGraph node callable |
| ``docs/nodes.md`` | Node system: ``AbstractNode``/``GraphNode`` + ``TextNode`` — metadata, by-name ``params``/``writes``, prompt templating, pydantic state, plain-value writes |
| ``docs/node-collection.md`` | ``NodeCollection``: in-memory prototype store — uuid ids, usage-tracked retrieval, update/replace mutation, metadata + semantic search, pruning, records/snapshot views |
| ``docs/graphs.md`` | ``Graph``: editable graph structure — manual-id nodes/edges, connections, validation, render, compile, ``GraphNode`` nesting |
| ``docs/tools.md`` | ``ToolCallNode`` + whitelist ``ToolRegistry`` — local mem0 memory + ddgs search tools, args validation, build-time whitelist |
| ``docs/behaviors.md`` | Behaviors: sectioned ``backend/behaviors.yaml`` data, stable point keys, rendering into the fixed generator template |
| ``docs/generator.md`` | ``GeneratorNode``: single-pass ``add_node``/``add_edge`` graph generation, validate + retry, nested in-turn execution, reuse auto-save |
| ``docs/data.md`` | Persistence: explicit JSON serializers, atomic storage helpers, ``backend/data/`` layout |
| ``docs/improvements.md`` | Improvement loop: suggestion → structured patches over behavior entries (whole-entry regeneration + programmatic diff), JSONL storage, most-recent-wins application, node-targeted edits, ``POST /feedback`` API |

## Running

Everything is `uv`-managed (see `pyproject.toml`). From the repository root:

```commandline
# 1. LangGraph server (the `agent` + `chat` graphs) — port 2024
cd backend
uv sync
uv run langgraph dev --port 2024

# 2. Feedback FastAPI (separate process) — port 8000
cd backend
uv run uvicorn api.app:app --port 8000

# 3. Frontend (assistant-ui) — port 3000
cd agent-chat-ui
npm install
npm run dev
```

Environment / config:

- `backend/.env` — `OPENAI_API_KEY` (or the configured `models.fast` provider)
  for the LangGraph server plus the feedback LLM.
- `agent-chat-ui/.env*` — see `.env.example`: `NEXT_PUBLIC_API_URL`
  (`http://localhost:2024` locally, or `<site>/api` to use the built-in proxy),
  `NEXT_PUBLIC_ASSISTANT_ID` (default `chat`), and
  `NEXT_PUBLIC_FEEDBACK_API_URL` (default `http://localhost:8000`) — the
  frontend forwards same-origin `POST /api/feedback` to the feedback service.
- Ledger approvals page (`/approvals`): file-based ledger under `LEDGER_PATH`
  (default `/shared/ledger`); the frontend route is `app/api/ledger/route.ts`.