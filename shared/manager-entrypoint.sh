#!/bin/sh
set -e

LEDGER_DIR="/shared/ledger"
APPROVALS_DIR="$LEDGER_DIR/approvals"
PENDING_DIR="$LEDGER_DIR/pending"
INSTALLED_FILE="$LEDGER_DIR/installed.json"
SESSIONS_FILE="$LEDGER_DIR/sessions.json"
COMPOSE_FILE="/shared/docker-compose.yml"
CODE_RUNNER_IMAGE="code-runner:base"

# Ensure directories exist
mkdir -p "$APPROVALS_DIR"
mkdir -p "$PENDING_DIR"

# Initialize installed.json if empty/missing
if [ ! -s "$INSTALLED_FILE" ]; then
    echo '{"packages":[]}' > "$INSTALLED_FILE"
fi

# Initialize sessions.json if empty/missing
if [ ! -s "$SESSIONS_FILE" ]; then
    echo '{}' > "$SESSIONS_FILE"
fi

# --- Approval processing ---

process_approval() {
    local approval_file="$1"
    local pkg_set
    pkg_set=$(jq -r '.packages // [] | .[]' "$approval_file" 2>/dev/null | sort | uniq)

    if [ -z "$pkg_set" ]; then
        echo "No packages in $approval_file, skipping"
        mv "$approval_file" "$LEDGER_DIR/rejected/"
        return 0
    fi

    # Load existing installed packages as a sorted list
    local existing_pkgs
    existing_pkgs=$(jq -r '.packages[]' "$INSTALLED_FILE" 2>/dev/null | sort | uniq)

    # Determine which packages are not yet installed
    local new_pkgs=""
    for pkg in $pkg_set; do
        if echo "$existing_pkgs" | grep -qx "$pkg"; then
            echo "Package $pkg already installed, skipping"
        else
            if [ -n "$new_pkgs" ]; then
                new_pkgs="$new_pkgs $pkg"
            else
                new_pkgs="$pkg"
            fi
        fi
    done

    if [ -z "$new_pkgs" ]; then
        echo "All packages already installed in $approval_file, moving to rejected"
        mv "$approval_file" "$LEDGER_DIR/rejected/"
        return 0
    fi

    echo "Building code-runner image with packages: $new_pkgs"
    docker build -t "$CODE_RUNNER_IMAGE" -f /shared/dockerfile/code-runner/Dockerfile --build-arg APT_PKGS="$new_pkgs" /shared/

    echo "Recreating code-runner container"
    docker compose -f "$COMPOSE_FILE" up -d --force-recreate code-runner

    # Update installed.json: add new packages to existing list
    local tmp_file
    tmp_file=$(mktemp)

    local all_pkgs="$existing_pkgs $new_pkgs"
    local unique_pkgs
    unique_pkgs=$(echo "$all_pkgs" | tr ' ' '\n' | sort -u | tr '\n' ' ' | sed 's/ $//')

    jq --argjson pkgs "$(echo "$unique_pkgs" | jq -R . | jq -s .)" '.packages = $pkgs' "$INSTALLED_FILE" > "$tmp_file"
    mv "$tmp_file" "$INSTALLED_FILE"

    # Move approval file to rejected (processed)
    mv "$approval_file" "$LEDGER_DIR/rejected/"
    echo "Processed $approval_file: installed $new_pkgs, rebuilt code-runner"
}

# --- Session management ---

spawn_session() {
    local session_id="$1"
    local keyref="$2"

    if [ -z "$session_id" ]; then
        echo "ERROR: session_id required"
        return 1
    fi

    # Check if session already exists
    local existing
    existing=$(jq -r --arg sid "$session_id" '.[$sid]' "$SESSIONS_FILE" 2>/dev/null)
    if [ "$existing" != "null" ] && [ -n "$existing" ]; then
        echo "Session $session_id already running"
        return 0
    fi

    echo "Spawning session container: session-$session_id"

    # Spawn container on code-net, isolated workspace dir
    local container_name="session-$session_id"

    docker run -d \
        --name "$container_name" \
        --network code-net \
        --restart unless-stopped \
        -v "$LEDGER_DIR:/shared:ro" \
        -v "/shared/workspace:/workspaces:rw" \
        -e SESSION_ID="$session_id" \
        "$CODE_RUNNER_IMAGE" \
        sleep infinity

    # Generate/derive SSH key for this session
    local session_key="/shared/keys/session_${session_id}_ed25519"
    mkdir -p "$(dirname "$session_key")"

    if [ -n "$keyref" ] && [ "$keyref" != "null" ]; then
        # Use provided key reference
        jq -r --arg kr "$keyref" '. + {keyref: $kr}' <<< "{}" > /dev/null
        # In a full implementation, we'd copy/import the key here
        echo "Using provided keyref: $keyref"
    else
        # Derive from graph_to_code keypair baked into image
        echo "Using graph_to_code key from image"
    fi

    # Register session in sessions.json
    local tmp_sessions
    tmp_sessions=$(mktemp)

    jq --arg sid "$session_id" \
       --arg host "$container_name" \
       --arg user "execuser" \
       --arg kr "$keyref" \
       '. + {($sid): {host: $host, user: $user, keyref: $kr}}' "$SESSIONS_FILE" > "$tmp_sessions"
    mv "$tmp_sessions" "$SESSIONS_FILE"

    echo "Session $session_id started: session-$container_name, registered in sessions.json"
}

cleanup_session() {
    local session_id="$1"

    if [ -z "$session_id" ]; then
        echo "ERROR: session_id required"
        return 1
    fi

    local container_name="session-$session_id"

    # Check if session exists in registry
    local exists
    exists=$(jq -r --arg sid "$session_id" 'has($sid)' "$SESSIONS_FILE" 2>/dev/null)
    if [ "$exists" != "true" ]; then
        echo "Session $session_id not found in registry"
        return 0
    fi

    echo "Cleaning up session: $session_id"

    # Stop and remove container
    if docker container inspect "$container_name" >/dev/null 2>&1; then
        docker container stop "$container_name" 2>/dev/null || true
        docker container rm "$container_name" 2>/dev/null || true
    fi

    # Remove from registry
    local tmp_sessions
    tmp_sessions=$(mktemp)

    jq --arg sid "$session_id" 'del(.'"$sid"')' "$SESSIONS_FILE" > "$tmp_sessions"
    mv "$tmp_sessions" "$SESSIONS_FILE"

    echo "Session $session_id cleaned up"
}

# --- Main loop ---

echo "Manager starting... watching $APPROVALS_DIR"

while true; do
    # Process approvals first
    for approval_file in "$APPROVALS_DIR"/*.json; do
        [ -f "$approval_file" ] || continue

        if jq -e '.status == "approved"' "$approval_file" >/dev/null 2>&1; then
            process_approval "$approval_file"
        else
            echo "Not an approval file: $approval_file, moving to rejected"
            mv "$approval_file" "$LEDGER_DIR/rejected/"
        fi
    done

    # Check for session control files
    # If /shared/ledger/start-session-<id> exists, spawn session
    for signal in "$LEDGER_DIR"/start-session-*; do
        [ -f "$signal" ] || continue
        local session_id
        session_id=$(basename "$signal" | sed 's/^start-//')
        if [ -n "$session_id" ]; then
            spawn_session "$session_id" ""
            rm -f "$signal"
        fi
    done

    # Check for session stop signals
    for signal in "$LEDGER_DIR"/stop-session-*; do
        [ -f "$signal" ] || continue
        local session_id
        session_id=$(basename "$signal" | sed 's/^stop-//')
        if [ -n "$session_id" ]; then
            cleanup_session "$session_id"
            rm -f "$signal"
        fi
    done

    sleep 5
done