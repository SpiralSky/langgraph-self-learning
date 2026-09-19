#!/bin/sh
set -e

if [ -f /app/.ssh/graph_to_code ]; then
    install -m 600 /app/.ssh/graph_to_code /app/.ssh/id_ed25519
fi

# Bind deterministically. 0.0.0.0 keeps the API reachable from every compose
# network this container is attached to (frontend-net DNS name `backend`, and
# host access via the published 2024:2024 port). The old `hostname -I` eth0
# guess was fragile when a container is on multiple networks. The code-runner
# service has no live backend integration yet, so there is no code-net reachability
# to gate by bind address here; revisit via LANGGRAPH_BIND_HOST if that changes.
: "${HOST:=0.0.0.0}"

exec uv run langgraph dev --host "$HOST" --port 2024 --no-browser --no-reload