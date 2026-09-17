"""Local mem0 memory tools — fully local, no API key.

A single mem0 ``Memory`` store is built lazily on first use and shared by
``mem0_remember`` / ``mem0_retrieve``. The store is *embedded*: the
fastembed embedder + Chroma vector store + SQLite history all live under
``backend/data/``. The configured LLM provider is never invoked — remember
stores raw text (``infer=False``) and retrieve is a pure vector search —
so no key or server is needed.

Importing this module constructs nothing: the ``Memory`` instance is
created only when a tool actually runs. Tests must never trigger that;
they inject fakes at the registry level.
"""

from __future__ import annotations

from graphs.tools.registry import int_arg, require_arg

_DEFAULT_USER_ID = "default"
_DEFAULT_TOP_K = 5

_memory = None


def _build_config() -> dict:
    """A fully-local mem0 config rooted under ``backend/data/``."""
    from graphs.persistence.storage import DATA_ROOT

    return {
        "llm": {
            # Never invoked (infer=False on add, no-LLM search); present only
            # because mem0 requires an LLM provider at construction.
            "provider": "lmstudio",
            "config": {
                "model": "local",
                "temperature": 0,
                "base_url": "http://localhost:1234/v1",
            },
        },
        "embedder": {
            "provider": "fastembed",
            "config": {"model": "BAAI/bge-small-en-v1.5"},
        },
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": "mem0",
                "path": str(DATA_ROOT / "mem0"),
            },
        },
        "history_db_path": str(DATA_ROOT / "mem0_history.db"),
    }


def get_memory():
    """Return (and lazily build once) the shared local ``Memory`` store."""
    global _memory
    if _memory is None:
        from mem0 import Memory

        _memory = Memory.from_config(_build_config())
    return _memory


def mem0_remember(args: dict) -> str:
    """Store a raw memory text in the local store (no LLM inference).

    :param dict args: ``{"text": str, "user_id": str (default "default")}``.
    :type args: dict
    :return: A one-line confirmation of what was stored.
    :rtype: str
    """
    text = require_arg(args, "text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("tool argument 'text' must be a non-empty string")
    user_id = str(args.get("user_id", _DEFAULT_USER_ID))
    get_memory().add(text, user_id=user_id, infer=False)
    return f"Remembered: {text.strip()}"


def mem0_retrieve(args: dict) -> str:
    """Retrieve the top-k relevant raw memories for a query.

    :param dict args: ``{"query": str, "top_k": int (default 5),
        "user_id": str (default "default")}``.
    :type args: dict
    :return: One ``- <memory>`` line per result; ``"No memories found."``
        when the store has nothing similar.
    :rtype: str
    """
    query = require_arg(args, "query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("tool argument 'query' must be a non-empty string")
    top_k = max(int_arg(args, "top_k", _DEFAULT_TOP_K), 1)
    user_id = str(args.get("user_id", _DEFAULT_USER_ID))
    results = get_memory().search(query, top_k=top_k, filters={"user_id": user_id})
    entries = [item["memory"] for item in results.get("results", [])]
    if not entries:
        return "No memories found."
    return "\n".join(f"- {entry}" for entry in entries)