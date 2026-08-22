import textwrap
from concurrent.futures import ThreadPoolExecutor

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import SystemMessage, HumanMessage

from graphs.learning_graph.llm import get_structured_model
from graphs.learning_graph.pydantic_models import FetcherOutput, QueryResult
from graphs.learning_graph.state import LearningGraphState
from graphs.learning_graph.text_utils import message_to_text

# TODO: Update to non-hardcoded search tool
search_tool = DuckDuckGoSearchRun()

INFORMATION_FETCHER_SYSTEM_PROMPT = """
You are an Information Retrieval Specialist. Decide what external info is needed to answer the user question and, if needed, generate optimal search queries.

Input: user question, intent, key points, memory context.

Tasks:
1. Assess whether your training data alone suffices to answer confidently.
2. If external info helps, create 1-3 keyword-rich, search-engine-friendly queries (no natural-language questions).
3. Rate confidence of answering *without* search.

Confidence scale:
- 0.8-1.0 High: standard facts, common concepts, basic code syntax. No search.
- 0.5-0.79 Medium: niche/recent/complex interdisciplinary topics. Search recommended.
- 0.0-0.49 Low: highly specific, very recent, or ambiguous queries. Search mandatory.

Output JSON:
- knowledge: str, summary of what you already know (empty if none)
- confidence: float (0.0-1.0)
- search_queries: list[str], optimized queries (empty only if confidence >= 0.7)
- reasoning: str, brief rationale

Always provide `search_queries` when confidence < 0.7. Keep them concise and keyword-focused. Do not answer the user's question here; only plan the retrieval strategy.
"""


def _memory_context_for_llm(state: LearningGraphState) -> str:
    """
    Render compact memory context lines for a prompt.

    :param state: Current graph state carrying compacted ``memory_results``.
    :type state: LearningGraphState
    :return: Newline-joined bullet points, or ``"None"`` when empty.
    :rtype: str
    """
    memory_results = state.memory_results or []
    if not memory_results:
        return "None"
    return "\n".join(f"- {m.get('text')}" for m in memory_results if isinstance(m, dict))


def information_fetcher(state: LearningGraphState) -> dict[str, QueryResult]:
    """
    Determine what external information (if any) is needed to answer the user
    question and run the generated search queries.

    Uses the ``state.analysis_results`` intent/key points and the condensed
    session ledger plus compacted memory context to decide confidence and, when
    confidence is low, invoke the web search tool per query. Emits no widget
    markup.

    :param state: Current graph state carrying ``analysis_results``,
        ``user_message``, ``active_session``, and ``memory_results``.
    :type state: LearningGraphState
    :return: Mapping the state key ``"search_results"`` to a
        :class:`QueryResult` holding any retrieved information plus confidence,
        queries, and reasoning.
    :rtype: dict[str, QueryResult]
    :raises ValueError: If ``analysis_results`` or ``user_message`` is missing
        from state.
    """
    if state.analysis_results is None:
        raise ValueError("Missing required field(s): analysis_results")
    if state.user_message is None:
        raise ValueError("Missing required field(s): user_message")

    model = get_structured_model("information_fetcher", FetcherOutput)

    analysis = state.analysis_results
    user_text = message_to_text(state.user_message)

    session_ledger = state.active_session.ledger if state.active_session is not None else []
    session_context = "None" if not session_ledger else "; ".join(session_ledger)

    information_str = textwrap.dedent(f"""
        User Input: {user_text}
        Intent: {analysis.intent}
        Key Points: {analysis.key_points}
        Learning Session: {session_context}
        Recent Memory: {_memory_context_for_llm(state)}
    """)

    result = model.invoke(
        [
            SystemMessage(INFORMATION_FETCHER_SYSTEM_PROMPT),
            HumanMessage(information_str)
        ]
    )

    query_results_str = ""

    def _run_search(query: str) -> str:
        try:
            res = search_tool.invoke(query)
            return f"\n[SUCCESS for Query: {query}]: {str(res)}\n"
        except Exception as e:
            return f"\n[FAILED for Query: {query}]: {str(e)}\n"

    # noinspection unresolved-references
    if queries := result.search_queries:
        with ThreadPoolExecutor() as executor:
            query_results_str = "".join(executor.map(_run_search, queries))

    # noinspection unresolved-references
    return {
        "search_results": QueryResult(
            information=query_results_str,
            confidence=result.confidence,
            queries=result.search_queries,
            reasoning=result.reasoning
        )
    }