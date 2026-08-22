import logging
import time
import uuid
from typing import Any

from langchain_core.runnables import RunnableConfig

from graphs.learning_graph.config import config
from graphs.learning_graph.memory import memory
from graphs.learning_graph.pydantic_models import SessionRecord
from graphs.learning_graph.state import LearningGraphState
from graphs.learning_graph.text_utils import message_to_text

logger = logging.getLogger(__name__)

SESSION_METADATA_KEY = "kind"
SESSION_METADATA_VALUE = "session"


def _merge_ledger(existing: list[str], updates: list[str], cap: int) -> list[str]:
    """
    Merge new ledger entries into the existing ledger, capped in length.

    Preserves the existing entries, appends only non-empty, not-yet-present
    updates (in order), and trims to the newest ``cap`` entries.

    :param existing: Current session ledger entries.
    :type existing: list[str]
    :param updates: New ledger entries from the current exchange.
    :type updates: list[str]
    :param cap: Maximum number of ledger entries to keep.
    :type cap: int
    :return: The merged, capped ledger.
    :rtype: list[str]
    """
    merged = list(existing)
    for entry in updates:
        entry = (entry or "").strip()
        if entry and entry not in merged:
            merged.append(entry)
    if len(merged) > cap:
        merged = merged[-cap:]
    return merged


def _build_or_rotate_session(
    existing: SessionRecord | None,
    delta,
) -> SessionRecord:
    """
    Build the session record to persist for the current exchange.

    Reuses the existing active session when the model keeps it going
    (``continue_session``), rotating to a fresh session otherwise.

    :param existing: Previously persisted session, if any.
    :type existing: SessionRecord | None
    :param delta: Session delta produced by the response improver.
    :type delta: SessionDelta
    :return: The session record to write back to the memory store.
    :rtype: SessionRecord
    """
    now = time.time()
    continue_session = delta.continue_session if delta.continue_session is not None else True
    keep = existing is not None and continue_session and existing.status == "active"

    if keep:
        record = existing.model_copy(deep=True)
        record.topic = delta.topic
        record.turn_count += 1
        record.updated_at = now
    else:
        record = SessionRecord(
            session_id=uuid.uuid4().hex[:8],
            thread_id=existing.thread_id if existing is not None else "",
            topic=delta.topic or "general",
            turn_count=1,
            created_at=now,
            updated_at=now,
        )

    record.ledger = _merge_ledger(
        record.ledger,
        delta.ledger_updates,
        config.session_max_ledger_entries,
    )
    return record


def save_memory(state: LearningGraphState, config_: RunnableConfig) -> dict:
    """
    Persist the current exchange and the learning-session record to long-term
    memory.

    Saves the direct conversation (the user's message and the formatted final
    response) into mem0, scoped to the thread id from the run config, and
    upserts the condensed :class:`SessionRecord` (capped learner ledger) for the
    thread. Runs synchronously after ``model_output``, so the reply is already
    streamed to the user before the writes begin.

    :param state: Current graph state carrying ``user_message``,
        ``final_output``, and ``improved_response``.
    :type state: LearningGraphState
    :param config_: LangGraph run configuration, read for the ``thread_id``.
    :type config_: RunnableConfig
    :return: An empty update; the node only writes to the memory store.
    :rtype: dict
    :raises ValueError: If the user message is missing.
    """
    if state.user_message is None:
        raise ValueError("Missing required field(s): user_message")

    thread_id = config_.get("configurable", {}).get("thread_id", "default_thread")

    exchange = [
        {"role": "user", "content": message_to_text(state.user_message)},
    ]

    if state.final_output is not None:
        exchange.append({"role": "assistant", "content": state.final_output})

    memory.add(exchange, user_id=thread_id)

    delta = getattr(state.improved_response, "session_delta", None) if state.improved_response is not None else None
    if delta is None:
        return {}

    record = _build_or_rotate_session(state.active_session, delta)
    if record.thread_id != thread_id:
        record.thread_id = thread_id

    try:
        _replace_session_record(record)
    except Exception as exc:
        logger.warning("Could not persist session for thread %s: %s", thread_id, exc)

    return {}


def _replace_session_record(record: SessionRecord) -> None:
    """
    Replace the thread's persisted session record with ``record``.

    Best-effort: removes any previously stored session entries via ``get_all`` /
    ``delete`` and writes the record fresh so at most one session entry exists
    per thread.

    :param record: The session record to persist.
    :type record: SessionRecord
    :raises Exception: Any underlying mem0 store error (caller logs and keeps going).
    """
    try:
        all_resp = memory.get_all(
            user_id=record.thread_id, filters={SESSION_METADATA_KEY: SESSION_METADATA_VALUE}
        )
        entries: list[Any] = []
        if isinstance(all_resp, dict):
            entries = all_resp.get("results") or []
        elif hasattr(all_resp, "results"):
            entries = list(all_resp.results) or []

        for entry in entries:
            metadata = (entry.get("metadata") if isinstance(entry, dict) else None) or {}
            if metadata.get(SESSION_METADATA_KEY) != SESSION_METADATA_VALUE:
                continue
            mid = entry.get("id") if isinstance(entry, dict) else getattr(entry, "id", None)
            if mid:
                memory.delete(memory_id=mid)
    except Exception as exc:
        logger.warning("While clearing prior sessions for thread %s: %s", record.thread_id, exc)

    memory.add(
        [{"role": "system", "content": record.model_dump_json()}],
        user_id=record.thread_id,
        metadata={SESSION_METADATA_KEY: SESSION_METADATA_VALUE},
    )