# AGENTS.md — llm-learning (app repo)

Project-specific rules for this repo. Generic opencode/orchestrator rules live in `/workspace/AGENTS.md`.

## Orientation

- Repo: `/workspace/llm-learning` (git; origin `SpiralSky/langgraph-self-learning`, upstream `langchain-ai/agent-chat-ui`). Branch rule: feature work goes on a `dev/<change>` branch off `dev`, merged into `dev` first (rebase/squash), never branched off `main`.
- Tests: `uv run pytest` from `backend/` (pytest.ini: `testpaths=tests`, `pythonpath=src`, `asyncio_mode=auto`).
- Codebase-memory project: `workspace-llm-learning-backend` (root `/workspace/llm-learning/backend`).
- Package root: `backend/` (`pyproject.toml`, `src/`, `tests/`, `docs/`, `data/collection.json`).

## Repo layout (shared/ sits outside the backend package)

- `shared/`: docker-compose `code-exec-sandbox` (frontend:3000, backend:2024, code-runner on internal `code-net`), `manager-entrypoint.sh` (approval ledger processor), `ledger/` (pending/approvals/rejected + installed.json + sessions.json), `workspace/` (empty dir — intended target of the frontend `/api/workspace/files` browser), `keys/` (graph_to_code SSH key mounted read-only into backend).
- More context: `plans/INFO.md`.