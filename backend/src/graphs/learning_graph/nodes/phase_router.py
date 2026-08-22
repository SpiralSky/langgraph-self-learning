import re
from typing import Any

from graphs.learning_graph.config import config
from graphs.learning_graph.pydantic_models import QueryResult
from graphs.learning_graph.state import LearningGraphState


def _topic_tokens(text: str | None) -> set[str]:
    """
    Tokenize free text into a set of alphanumeric topic tokens.

    :param text: Text to tokenize.
    :type text: str | None
    :return: Lowercased alphanumeric token set.
    :rtype: set[str]
    """
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _topic_overlap(session_topic: str, key_points) -> int:
    """
    Count shared tokens between the session topic and the extracted key points.

    :param session_topic: Current consolidated session topic.
    :type session_topic: str
    :param key_points: Key concepts extracted from the user's question.
    :type key_points: list[str]
    :return: Number of distinct tokens shared by both sets.
    :rtype: int
    """
    if not key_points:
        return 0
    session_tokens = _topic_tokens(session_topic)
    points_tokens: set[str] = set()
    for point in key_points:
        points_tokens |= _topic_tokens(point)
    return len(session_tokens & points_tokens)


def phase_router(state: LearningGraphState) -> dict[str, Any]:
    """
    Decide the execution phase for the current turn.

    A turn stays in the steady-state phase when an active learning session
    shares enough topic overlap with the current question; otherwise it runs the
    full discovery phase. Steady-state turns skip web search and the downstream
    formatting pass, relying on the response improver for polish. Also seeds an
    empty :class:`QueryResult` on the steady-state path so downstream nodes can
    read ``search_results`` unconditionally.

    :param state: Current graph state carrying ``analysis_results``,
        ``memory_results``, and ``active_session``.
    :type state: LearningGraphState
    :return: Mapping with ``use_web_search``, ``skip_format``, and optionally a
        placeholder ``search_results``.
    :rtype: dict[str, Any]
    :raises ValueError: If ``analysis_results`` or ``memory_results`` is missing.
    """
    if state.analysis_results is None:
        raise ValueError("Missing required field(s): analysis_results")
    if state.memory_results is None:
        raise ValueError("Missing required field(s): memory_results")

    analysis = state.analysis_results
    session = state.active_session

    overlap = 0
    if session is not None:
        overlap = _topic_overlap(session.topic, analysis.key_points)

    use_web_search = session is None or overlap < config.session_min_overlap
    skip_format = not use_web_search

    updates: dict[str, Any] = {
        "use_web_search": use_web_search,
        "skip_format": skip_format,
    }

    if not use_web_search and state.search_results is None:
        updates["search_results"] = QueryResult(
            confidence=0.0,
            information="",
            queries=[],
            reasoning="steady-state session response",
        )

    return updates