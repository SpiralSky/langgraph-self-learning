import json
import os
import subprocess
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from graphs.learning_graph.nodes.node import node
from graphs.learning_graph.pydantic_models import (
    ExecuteCodeInput,
    ExecuteCodeOutput,
    PackageProposal,
)
from graphs.learning_graph.state import LearningGraphState
from graphs.learning_graph.text_utils import message_to_text
from graphs.learning_graph.config import config


SSH_KEY_PATH = "/shared/keys/graph_to_code"
CODE_RUNNER_USER = "execuser"
CODE_NET_NETWORK = "code-net"
MAX_OUTPUT_LINES = 100
MAX_OUTPUT_BYTES = 10000
WORKSPACE_ROOT = "/home/exec/workspace"
MAX_FILE_LIST = 500


def _ssh_exec(target_host: str, command: list[str], timeout: int = 30) -> dict:
    """Execute a command via SSH to a code-runner or session container.

    Uses the graph_to_code ed25519 key for authentication.
    Returns {stdout, stderr, returncode}.
    """
    key_path = SSH_KEY_PATH
    if not os.path.exists(key_path):
        raise FileNotFoundError(f"SSH key not found at {key_path}")

    ssh_cmd = [
        "ssh",
        "-i", key_path,
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=10",
        f"{CODE_RUNNER_USER}@{target_host}",
    ] + command

    try:
        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": f"SSH command timed out after {timeout}s",
            "returncode": -1,
        }


def _parse_session_target(state: LearningGraphState) -> Optional[str]:
    """Resolve the SSH target from sessions.json registry.

    Returns hostname (container name) if a session is active, else None.
    Falls back to 'code-runner' pool when no session is found.
    """
    import json

    sessions_file = "/shared/ledger/sessions.json"
    if not os.path.exists(sessions_file):
        return None

    try:
        with open(sessions_file) as f:
            sessions = json.load(f)
        # Check for any active session
        for session_id, info in sessions.items():
            if info and isinstance(info, dict):
                host = info.get("host")
                if host:
                    return host
    except (json.JSONDecodeError, IOError):
        pass

    # Fall back to code-runner pool
    return "code-runner"


def _scan_workspace() -> list[dict[str, Any]]:
    """Scan the code-runner workspace directory recursively.

    Returns a list of file entries with path, type, host_path, and size.
    Only includes files under the workspace root (security). Caps at 500 entries.
    """
    import os

    entries = []
    container_root = Path(WORKSPACE_ROOT).resolve()
    host_root = Path(config.workspace_host_path).resolve()

    for dirpath, _dirnames, filenames in os.walk(WORKSPACE_ROOT):
        for filename in filenames:
            file_path = Path(dirpath) / filename
            try:
                rel_path = file_path.resolve().relative_to(container_root)
            except ValueError:
                continue
            entries.append({
                "path": str(rel_path),
                "type": "file",
                "host_path": str(host_root / rel_path),
                "size": file_path.stat().st_size if file_path.exists() else 0,
            })
            if len(entries) >= MAX_FILE_LIST:
                return entries

    return entries


@node(prompt=None, intent="execute_code")
def execute_code(state: LearningGraphState, config: Any = None) -> dict[str, Any]:
    """
    Execute code in a code-runner container via SSH.

    Resolves the SSH target from ``sessions.json`` registry (active session
    container), or falls back to the ``code-runner`` pool. Extracts the
    ``python3 -c "..."`` command from the execution plan step; falls back to
    the raw step text if the pattern is not found. Runs the command with a
    60s timeout, then caps stdout/stderr at :data:`MAX_OUTPUT_LINES` lines
    and :data:`MAX_OUTPUT_BYTES` bytes.

    Output capping: if stdout or stderr exceeds the line limit, truncation
    is appended with a ``... (output capped at N lines)`` note; if bytes
    exceed the limit, the output is truncated with a ``... (output capped
    at N bytes)`` note.

    :param state: Current graph state carrying ``teaching_strategy``
        (the ``execution_plan`` is scanned for a code-execution step).
    :type state: LearningGraphState
    :param config: LangGraph run configuration (for thread_id, etc.).
    :type config: Any
    :return: When a code step is found, maps ``"final_output"`` to the
        capped execution result string; otherwise returns an empty dict
        (pass-through).
    :rtype: dict[str, Any]
    :raises ValueError: If no code execution request is found in state.
    """
    # Check for code execution request in state
    # The execution plan may contain a code execution step
    execution_plan = state.teaching_strategy.execution_plan if state.teaching_strategy else []

    code_request = None
    for step in execution_plan:
        if "execute" in step.lower() or "code" in step.lower():
            # Extract the python command from the step
            code_request = step
            break

    if code_request is None:
        # No code execution request, pass through
        return {}

    # Resolve SSH target (session container or code-runner pool)
    target_host = _parse_session_target(state)

    # Extract the actual Python command from the execution plan step
    # Look for python3 -c "..." or similar patterns
    cmd_match = re.search(r'python3\s+-c\s+"([^"]*)"', code_request)
    if not cmd_match:
        # Try other patterns
        cmd_match = re.search(r'python3\s+"([^"]*)"', code_request)

    python_cmd = cmd_match.group(1) if cmd_match else code_request

    # Execute via SSH
    ssh_result = _ssh_exec(target_host, ["bash", "-c", python_cmd], timeout=60)

    # Cap output
    stdout = ssh_result["stdout"]
    stderr = ssh_result["stderr"]

    # Line/byte capping
    stdout_lines = stdout.split("\n")
    if len(stdout_lines) > MAX_OUTPUT_LINES:
        stdout = "\n".join(stdout_lines[:MAX_OUTPUT_LINES]) + "\n... (output capped at %d lines)" % MAX_OUTPUT_LINES
    stderr_lines = stderr.split("\n")
    if len(stderr_lines) > MAX_OUTPUT_LINES:
        stderr = "\n".join(stderr_lines[:MAX_OUTPUT_LINES]) + "\n... (stderr capped at %d lines)" % MAX_OUTPUT_LINES

    if len(stdout.encode()) > MAX_OUTPUT_BYTES:
        stdout = stdout[:MAX_OUTPUT_BYTES] + "\n... (output capped at %d bytes)" % MAX_OUTPUT_BYTES
    if len(stderr.encode()) > MAX_OUTPUT_BYTES:
        stderr = stderr[:MAX_OUTPUT_BYTES] + "\n... (stderr capped at %d bytes)" % MAX_OUTPUT_BYTES

    return_code = ssh_result["returncode"]

    # Build the output message
    if return_code == 0:
        output_text = f"Execution output:\n\nSTDOUT:\n```\n{stdout}\n```\n\nSTDERR:\n```\n{stderr}\n```"
    else:
        output_text = f"Execution exited with code {return_code}.\n\nSTDOUT:\n```\n{stdout}\n```\n\nSTDERR:\n```\n{stderr}\n```"

    # Scan workspace for files created/modified
    files = _scan_workspace()

    # Return final_output and files so it flows to model_output
    return {
        "final_output": output_text,
        "success": return_code == 0,
        "files": files,
    }