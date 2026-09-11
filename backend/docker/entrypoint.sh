#!/bin/sh
set -e

if [ -f /app/.ssh/graph_to_code ]; then
    install -m 600 /app/.ssh/graph_to_code /app/.ssh/id_ed25519
fi

if [ -n "$LANGGRAPH_BIND_HOST" ]; then
    HOST="$LANGGRAPH_BIND_HOST"
elif command -v hostname >/dev/null 2>&1; then
    HOST="$(hostname -I 2>/dev/null | awk '{print $1}')"
fi

if [ -z "$HOST" ]; then
    HOST="$(python3 -c 'import socket;print(socket.gethostbyname(socket.gethostname()))' 2>/dev/null || true)"
fi

: "${HOST:=0.0.0.0}"

exec uv run langgraph dev --host "$HOST" --port 2024 --no-browser --no-reload