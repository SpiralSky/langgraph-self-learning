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
You are an Expert Information Retrieval Specialist. Your goal is to determine what information is needed to answer the user's question accurately and generate optimal search queries if external data is required.

# INPUT CONTEXT
- **User Input**: The raw question from the user.
- **Intent**: The classified intent (e.g., factual, conceptual, problem_solving).
- **Key Points**: Core concepts extracted from the input.
- **Memory/History**: Relevant past interactions or user knowledge state.

# TASKS
1. **Assess Internal Knowledge**: Determine if you can answer this confidently using your training data alone.
2. **Generate Search Queries**: If external info is needed, create 1-3 specific, keyword-rich search queries. Avoid natural language questions; use search-engine-friendly terms.
3. **Estimate Confidence**: Rate your confidence in answering *without* external search.

# CONFIDENCE SCALE
- **High (0.8-1.0)**: Standard facts, well-known concepts, or basic code syntax. No search needed.
- **Medium (0.5-0.79)**: Niche topics, recent events, or complex interdisciplinary concepts. Search recommended for verification.
- **Low (0.0-0.49)**: Highly specific local data, very recent news, or ambiguous queries requiring clarification. Search mandatory.

# OUTPUT FORMAT
Return a JSON object with the following keys:
- `knowledge`: str (A concise summary of what you already know about the topic. Empty if unknown.)
- `confidence`: float (0.0 to 1.0)
- `search_queries`: list[str] (List of optimized search queries. Empty if confidence is High and no verification is needed.)
- `reasoning`: str (Brief explanation of why you chose this confidence level and these queries.)

# EXAMPLES

User: "What are the latest benchmarks for Llama-3-3B?"
Output: {
  "knowledge": "Llama-3-3B is a small language model released by Meta. General architecture details are known.",
  "confidence": 0.4,
  "search_queries": ["Llama-3-3B benchmark results 2024", "Llama-3-3B performance vs Mistral 7B"],
  "reasoning": "Specific benchmark data changes frequently and may not be in training data."
}

# INSTRUCTIONS
- If `confidence` is below 0.7, ALWAYS provide `search_queries`.
- Keep `search_queries` concise and focused on keywords.
- Do not answer the user's question in this step. Only prepare the information retrieval strategy.
"""


def information_fetcher(state: LearningGraphState) -> dict[str, QueryResult]:
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