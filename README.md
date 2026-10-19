# Self Learning

A **LangGraph** project built around two ideas: **Socratic pedagogy** and **runtime graph generation**.

Instead of feeding answers directly, the agent asks guiding questions to lead you to the answer yourself. For each request, an LLM assembles a custom directed graph of reasoning steps at runtime — the shape of the reasoning adapts to every question. A feedback loop lets the system refine its own behavior over time by applying structured patches to a YAML behavior registry.

> Note: Prompt results may vary.

## How it works

```
Frontend (Next.js :3000)
    │  streaming messages / feedback
    ▼
Backend — LangGraph agent (:2024)
    │  GeneratorNode → LLM builds + runs inner subgraph per turn
    │  NodeCollection — saves and reuses steps across turns
    ▼
Feedback service (FastAPI :8000)
    └─ POST /feedback → LLM diff → patches behaviors.yaml
```

## Layout

- `backend/` — LangGraph server (`agent` entrypoint + `chat` messages-looping graphs),
  node/collection system, persistence, tools, and a separate `/feedback` FastAPI app.
- `agent-chat-ui/` — assistant-ui frontend (Next.js, React): setup form, thread
  sidebar, generative-UI artifacts, agent inbox (human-in-the-loop interrupts),
  widget fenced blocks, file upload, the approvals ledger page, and per-message feedback.
- `shared/` — docker-compose code-exec sandbox, approval ledger volume, workspace
  volume, deployment keys.

## Features

- **Runtime graph generation** — `GeneratorNode` calls the LLM with a builder-tool schema (`add_node` / `add_edge`), assembles a DAG of `TextNode` / `ToolCallNode` steps, validates it, and executes it in-turn as a nested LangGraph subgraph.
- **Node collection & reuse** — steps marked `reuse` are saved to `data/collection.json` with run telemetry (token counts, latency mean/std); future turns retrieve and replay them via semantic search.
- **Behavior-driven prompts** — agent behaviors live in `backend/behaviors.yaml` as plain data, not hardcoded prompts. Edit the file to change behavior without touching Python.
- **Self-improvement loop** — `POST /feedback` with a suggestion → LLM regenerates the full behavior registry → structured diff → JSONL patch log → applied to `behaviors.yaml`.
- **Built-in tools** — DuckDuckGo search and Mem0 long-term memory (fully local: ChromaDB + fastembed, no external API key required).
- **Generative-UI artifacts** — the backend pushes arbitrary UI to a side panel via the `ui` state channel; the frontend renders it live.
- **Widget fence renderer** — custom `:::widget` fenced blocks in markdown map to registered React components.
- **Human-in-the-loop approvals** — the agent can pause and route a decision to the ledger; the frontend `/approvals` page surfaces pending items.
- **Workspace file browser** — file tree browsing of the shared workspace volume with composer attachments.
- **Code-execution sandbox** — a Docker `code-runner` service reachable over SSH (localhost:22022) for safe execution of generated code.
- **Thread sidebar** — persisted LangGraph thread list backed by the LangGraph SDK.
- **Per-message feedback** — thumbs up/down on any message, proxied from the frontend to the feedback service.

## Prerequisites

- Python 3.11+ and [`uv`](https://github.com/astral-sh/uv)
- Node.js 20+ and `npm`
- `OPENAI_API_KEY` set in `backend/.env`
- Docker + Docker Compose _(optional — required only for the code-runner sandbox and full-stack compose setup)_

## Running

### 1. Backend (LangGraph server) — port 2024

```commandline
cd backend
uv sync
uv run langgraph dev --port 2024
```

### 2. Feedback service (separate FastAPI app) — port 8000

```commandline
cd backend
uv run uvicorn api.app:app --port 8000
```

### 3. Frontend (assistant-ui) — port 3000

```commandline
cd agent-chat-ui
npm install
npm run dev
```

### Without docker — one command

Instead of starting the three services above separately (or `docker compose up`
in `shared/`), `shared/dev.sh` starts the whole core stack — backend, feedback
service, and frontend — with a single command:

```commandline
shared/dev.sh
```

Same ports as above (2024, 8000, 3000). It installs dependencies if needed and
runs all three in the foreground, stopping them together on Ctrl-C. Requires
`uv` (and `OPENAI_API_KEY` in `backend/.env`), plus `node`/`npm` for the
frontend. Excludes the docker-only services (`code-runner`, `manager`).

For the full stack including the code-runner sandbox:

```commandline
docker compose -f shared/docker-compose.yml up
```

## Frontend configuration

Copy `agent-chat-ui/.env.example` to `agent-chat-ui/.env.local` and set:

| Variable | Purpose | Default |
|----------|---------|---------|
| `NEXT_PUBLIC_API_URL` | LangGraph server URL (or `<site>/api` to use the built-in proxy) | `http://localhost:2024` |
| `NEXT_PUBLIC_ASSISTANT_ID` | graph/assistant id — use the messages-mode `chat` graph | `chat` |
| `NEXT_PUBLIC_AUTH_SCHEME` | optional; `langsmith-api-key` for Agent Builder deployments | _(empty)_ |
| `NEXT_PUBLIC_FEEDBACK_API_URL` | feedback FastAPI base URL; the frontend forwards same-origin `POST /api/feedback` to it | `http://localhost:8000` |
| `LANGSMITH_API_KEY` | optional, for deployed LangGraph servers | _(empty)_ |
| `LEDGER_PATH` | ledger directory backing the `/approvals` page | `/shared/ledger` |

See `backend/README.md` for backend configuration, `backend/docs/` for the
node/collection system, and `agent-chat-ui/.env.example` for the full frontend
variable list with production examples.

## Further reading

| Doc | Topic |
|-----|-------|
| `backend/README.md` | Backend config and environment |
| `backend/docs/generator.md` | LLM-driven graph generation |
| `backend/docs/nodes.md` | Node system (TextNode, AbstractNode) |
| `backend/docs/node-collection.md` | NodeCollection and reuse |
| `backend/docs/graphs.md` | Graph structure and compilation |
| `backend/docs/behaviors.md` | Behavior YAML format |
| `backend/docs/improvements.md` | Feedback loop and patches |
| `backend/docs/tools.md` | Built-in tools (ddgs, mem0) |
