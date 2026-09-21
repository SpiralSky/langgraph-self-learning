#!/usr/bin/env bash
# dev.sh — run the core stack (backend :2024, feedback-api :8000, frontend :3000)
# without docker. Non-docker alternative to shared/docker-compose.yml.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

pids=()

cleanup() {
  trap - EXIT INT TERM
  kill "${pids[@]}" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

usage() {
  cat <<'EOF'
Usage: dev.sh [-h|--help]

Start the core stack without docker (ports match shared/docker-compose.yml):

  backend       LangGraph server     http://localhost:2024
  feedback-api  uvicorn              http://localhost:8000
  frontend      assistant-ui (Next)  http://localhost:3000

Prerequisites:
  uv        backend (OPENAI_API_KEY in backend/.env)
  node/npm  frontend (agent-chat-ui/.env.local only if the defaults need changing)

All three run in the foreground; Ctrl-C stops them.
EOF
}

for arg in "$@"; do
  case "$arg" in
    -h|--help) usage; exit 0 ;;
    *) echo "dev.sh: unknown option: $arg" >&2; usage >&2; exit 1 ;;
  esac
done

command -v uv >/dev/null 2>&1 || { echo "dev.sh: 'uv' not found — install uv (https://docs.astral.sh/uv) to run the backend." >&2; exit 1; }
command -v npm >/dev/null 2>&1 || { echo "dev.sh: 'npm' not found — install Node.js to run the frontend." >&2; exit 1; }
[ -f "$ROOT/backend/.env" ] || { echo "dev.sh: $ROOT/backend/.env missing — required for OPENAI_API_KEY." >&2; exit 1; }

# Shared state dirs (approval ledger, file-browser workspace).
mkdir -p "$ROOT/shared/ledger/pending" "$ROOT/shared/ledger/approvals" \
  "$ROOT/shared/ledger/rejected" "$ROOT/shared/workspace"

echo "backend: uv sync && langgraph dev on :2024"
(cd "$ROOT/backend" && uv sync && exec uv run langgraph dev --port 2024) &
pids+=("$!")
echo "feedback-api: uvicorn on :8000"
(cd "$ROOT/backend" && exec uv run uvicorn api.app:app --port 8000) &
pids+=("$!")
echo "frontend: npm install && npm run dev on :3000"
(cd "$ROOT/agent-chat-ui" && npm install && exec npm run dev) &
pids+=("$!")

wait