from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage

from graphs.learning_graph.config import config
from graphs.learning_graph.pydantic_models import InputAnalysisResult
from graphs.learning_graph.state import LearningGraphState

RECENT_HISTORY_WINDOW = 3

INPUT_ANALYSER_PROMPT = """You are a Learning Input Analyst. Analyze the user's question for clarity, intent, and key learning concepts.

Goals:
1. Decide if the question is clear enough to answer directly.
2. Identify learning intent: 'factual' (definition/date), 'conceptual' (mechanism/theory), 'problem_solving' (task or code), 'unclear'.
3. Extract the most relevant nouns/concepts; ignore filler.
4. Comment encouragingly but honestly; push vague questions to be more specific.

The message list may contain up to 3 preceding conversation turns (earlier user questions and AI tutor replies) before the current user question. Use this history to resolve pronouns, ellipsis, and references to earlier topics when judging clarity and extracting key concepts. The final message is always the current question to analyze.

Vague questions (e.g. "Hey, can you show me again?") => is_clear=false.

Return a valid JSON object matching the provided schema. No markdown fences.
"""

def input_analyzer(state: LearningGraphState) -> dict[str, Any]:
    """
    Analyze the user's question for clarity, intent, and key learning concepts.

    Calls the configured LLM with a structured-output schema and stores the
    resulting :class:`InputAnalysisResult` under ``"analysis_results"``. Prior
    conversation turns (up to :data:`RECENT_HISTORY_WINDOW`) are passed to the
    model before the current message so pronouns and implicit references in the
    follow-up are resolved. Emits no widget markup.

    :param state: Current graph state carrying ``user_message`` and the
        ``messages`` channel (used for the preceding turns).
    :type state: LearningGraphState
    :return: Mapping the state key ``"analysis_results"`` to an
        :class:`InputAnalysisResult`.
    :rtype: dict[str, Any]
    """
    user_message = state.user_message

    model_config = config.get_model_data("input_analyzer")

    model = init_chat_model(
        model_config.model_id,
        api_key=model_config.api_key,
        base_url=model_config.api_endpoint,
        temperature=0,
        model_provider="openai"
    ).with_structured_output(InputAnalysisResult)

    recent_history = state.messages[-RECENT_HISTORY_WINDOW - 1:-1]

    analysis = model.invoke(
        [
            SystemMessage(INPUT_ANALYSER_PROMPT),
            *recent_history,
            user_message
        ]
    )

    return {
        "analysis_results": analysis
    }