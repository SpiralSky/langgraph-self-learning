import logging
import time
from typing import Any

from langchain_core.runnables import RunnableConfig

from graphs.learning_graph.config import config as app_config
from graphs.learning_graph.memory import memory
from graphs.learning_graph.nodes.node import node
from graphs.learning_graph.pydantic_models import SessionRecord, KnowledgeComponent
from graphs.learning_graph.state import LearningGraphState
from graphs.learning_graph.text_utils import message_to_text



logger = logging.getLogger(__name__)

SESSION_METADATA_KEY = "kind"
SESSION_METADATA_VALUE = "session"
KNOWLEDGE_METADATA_VALUE = "knowledge_model"


def _extract_knowledge_model(entries: list[dict]) -> list[KnowledgeComponent]:
    """
    Parse the thread's persisted mastery model entries.

    Each :class:`KnowledgeComponent` is stored as its own mem0 entry flagged
    with ``kind == "knowledge_model"``. Malformed payloads are skipped so a
    partial write never breaks a turn.

    :param entries: Normalized mem0 entries for the thread.
    :type entries: list[dict]
    :return: Parsed knowledge components.
    :rtype: list[KnowledgeComponent]
    """
    components: list[KnowledgeComponent] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        metadata = entry.get("metadata") or {}
        if metadata.get(SESSION_METADATA_KEY) != KNOWLEDGE_METADATA_VALUE:
            continue
        try:
            components.append(KnowledgeComponent.model_validate_json(entry.get("memory") or ""))
        except Exception:
            continue
    return components


def _compact_results(results: list[dict], top_k: int, max_chars: int) -> list[dict]:
    """
    Trim raw memory hits to a compact, token-lean list.

    Drops self-produced bookkeeping entries (the session record and the
    mastery-model components — surfaced separately via ``active_session`` and
    ``knowledge_model``) and truncates every remaining hit's text to
    ``max_chars``.

    :param results: Raw memory search hits.
    :type results: list[dict]
    :param top_k: Maximum number of hits to keep.
    :type top_k: int
    :param max_chars: Maximum characters kept per hit.
    :type max_chars: int
    :return: Compacted hit dicts with only ``id``, ``text``, and ``metadata``.
    :rtype: list[dict]
    """
    compact = []
    for r in results:
        metadata = (r.get("metadata") or {}) if isinstance(r, dict) else {}
        if metadata.get(SESSION_METADATA_KEY) in (SESSION_METADATA_VALUE, KNOWLEDGE_METADATA_VALUE):
            continue
        compact.append({
            "id": r.get("id"),
            "text": (r.get("memory") or "")[:max_chars],
            "metadata": metadata,
        })
    return compact[:top_k]


def _normalize_entries(resp: Any) -> list[dict]:
    """
    Normalize a mem0 ``get_all`` response into a flat result list.

    Handles dict, object-with-``results``, and raw-list shapes so the session
    read stays robust across mem0 versions.

    :param resp: Response from ``Memory.get_all``.
    :type resp: Any
    :return: Flat list of memory entries.
    :rtype: list[dict]
    """
    if isinstance(resp, dict):
        return resp.get("results") or []
    if hasattr(resp, "results"):
        return list(resp.results) or []
    if isinstance(resp, list):
        return resp
    return []


def _extract_active_session(entries: list[dict]) -> SessionRecord | None:
    """
    Find the thread's session record among raw mem0 entries.

    Parses each entry flagged as a session record and returns the most recently
    updated one, ignoring malformed payloads.

    :param entries: Normalized mem0 entries for the thread.
    :type entries: list[dict]
    :return: The active :class:`SessionRecord`, or ``None`` if none parses.
    :rtype: SessionRecord | None
    """
    best: SessionRecord | None = None
    for e in entries:
        if not isinstance(e, dict):
            continue
        metadata = e.get("metadata") or {}
        if metadata.get(SESSION_METADATA_KEY) != SESSION_METADATA_VALUE:
            continue
        try:
            record = SessionRecord.model_validate_json(e.get("memory") or "")
        except Exception:
            continue
        if best is None or record.updated_at > best.updated_at:
            best = record
    return best


@node(prompt=None, intent="memory_read")
def retrieve_memory(state: LearningGraphState, config: RunnableConfig) -> dict[str, Any]:
    """
    Retrieve relevant past memories and the active learning session for the thread.

    Extracts plain-text content from the user message (handling both ``str`` and
    block-list content), scopes the search to the thread id from the run config,
    compacts the hits to ``memory_top_k``/``memory_max_chars``, and loads the
    thread's active session record (``active_session``) and persistent mastery
    model (``knowledge_model``). Emits no widget markup.

    Normalizes mem0 response shapes (dict, object-with-``results``, raw list)
    via :func:`_normalize_entries` so the read is robust across mem0 versions.
    Bookkeeping entries (session record + knowledge model components) are
    surfaced separately via ``active_session`` and ``knowledge_model``, not
    included in ``memory_results``.

    :param state: Current graph state carrying ``user_message``.
    :type state: LearningGraphState
    :param config: LangGraph run configuration, read for the ``thread_id``.
    :type config: RunnableConfig
    :return: Mapping of ``"memory_results"`` to compacted hits,
        ``"active_session"`` to the session record (or ``None``), and
        ``"knowledge_model"`` to the persisted knowledge components (possibly empty).
    :rtype: dict[str, Any]
    :raises ValueError: If the user message is missing or empty.
    """
    if state.user_message is None:
        raise ValueError("Missing required field(s): user_message")

    message_content = message_to_text(state.user_message)
    if not message_content:
        raise ValueError("User message content is empty!")

    thread_id = config.get("configurable", {}).get("thread_id", "default_thread")

    started = time.perf_counter()
    results = memory.search(
        message_content,
        limit=app_config.memory_top_k,
        filters={"user_id": thread_id},
    )["results"]
    results = [] if results is None else results
    search_ms = (time.perf_counter() - started) * 1e3

    compacted = _compact_results(results, app_config.memory_top_k, app_config.memory_max_chars)

    session: SessionRecord | None = None
    knowledge_model: list[KnowledgeComponent] = []
    try:
        all_resp = memory.get_all(filters={SESSION_METADATA_KEY: SESSION_METADATA_VALUE, "user_id": thread_id})
        entries = _normalize_entries(all_resp)
        session = _extract_active_session(entries)
        knowledge_model = _extract_knowledge_model(entries)
    except Exception as exc:
        logger.warning("Could not load learning session for thread %s: %s", thread_id, exc)

    logger.info(
        "memory thread=%s hits=%d capped=%d session=%s knowledge=%d search_ms=%.1f",
        thread_id,
        len(results),
        len(compacted),
        session.session_id if session else None,
        len(knowledge_model),
        search_ms,
    )

    return {
        "memory_results": compacted,
        "active_session": session,
        "knowledge_model": knowledge_model,
    }