# Self Learning

A **LangGraph** project that facilitates learning. Instead of feeding you answers
directly, the agent asks questions to lead you to the answer yourself. The
composition of multiple models is used to improve AI output for learning.

> Note: Prompt results may vary.

## Layout

- `backend/` — LangGraph server (`agent` entrypoint + `chat` messages-looping graphs),
  node/collection system, persistence, tools, and a separate `/feedback` FastAPI app.
- `agent-chat-ui/` — assistant-ui frontend (Next.js, React): setup form, thread
  sidebar, generative-UI artifacts, agent inbox (human-in-the-loop interrupts),
  widget fenced blocks, file upload, the approvals ledger page, and per-message feedback.
- `shared/` — docker-compose code-exec sandbox, approval ledger volume, workspace
  volume, deployment keys.

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