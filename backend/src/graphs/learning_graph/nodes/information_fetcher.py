import textwrap

from langchain.chat_models import init_chat_model
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import SystemMessage, HumanMessage

from graphs.learning_graph.config import config
from graphs.learning_graph.pydantic_models import FetcherOutput, QueryResult
from graphs.learning_graph.state import LearningGraphState

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


def information_fetcher(state: LearningGraphState) -> dict[str, QueryResult]:
    """
    Determine what external information (if any) is needed to answer the user
    question and run the generated search queries.

    Uses the ``state.analysis_results`` intent/key points and the retrieved
    memory context to decide confidence and, when confidence is low, invoke the
    web search tool per query. Emits no widget markup.

    :param state: Current graph state carrying ``analysis_results``,
        ``user_message``, and ``memory_results``.
    :type state: LearningGraphState
    :return: Mapping the state key ``"search_results"`` to a
        :class:`QueryResult` holding any retrieved information plus confidence,
        queries, and reasoning.
    :rtype: dict[str, QueryResult]
    """
    model_config = config.get_model_data("information_fetcher")

    model = init_chat_model(
        model_config.model_id,
        api_key=model_config.api_key,
        base_url=model_config.api_endpoint,
        temperature=0,
        model_provider="openai"
    ).with_structured_output(FetcherOutput)

    analysis = state.analysis_results
    user_input = state.user_message

    information_str = textwrap.dedent(f"""
        User Input: {user_input}
        Intent: {analysis.intent}
        Key Points: {analysis.key_points}
        Memory Context: {state.memory_results}
    """)

    result = model.invoke(
        [
            SystemMessage(INFORMATION_FETCHER_SYSTEM_PROMPT),
            HumanMessage(information_str)
        ]
    )

    query_results_str = ""

    # noinspection unresolved-references
    if queries := result.search_queries:
        for query in queries:
            try:
                res = search_tool.invoke(query)
                query_results_str += f"\n[SUCCESS for Query: {query}]: {str(res)}\n"
            except Exception as e:
                query_results_str += f"\n[FAILED for Query: {query}]: {str(e)}\n"

    # noinspection unresolved-references
    return {
        "search_results": QueryResult(
            information=query_results_str,
            confidence=result.confidence,
            queries=result.search_queries,
            reasoning=result.reasoning
        )
    }