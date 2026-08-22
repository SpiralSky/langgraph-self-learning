from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage

from graphs.learning_graph.config import config
from graphs.learning_graph.pydantic_models import InputAnalysisResult
from graphs.learning_graph.state import LearningGraphState

from typing import Any

INPUT_ANALYSER_PROMPT = """
You are a Pedagogical Input Analyst. Analyze the user question for clarity, intent, and key concepts.

Goals:
1. Decide if the question is clear enough to answer directly.
2. Identify learning intent: 'factual' (definition/date), 'conceptual' (mechanism/theory), 'problem_solving' (task or code), 'unclear'.
3. Extract the most relevant nouns/concepts; ignore filler.
4. Comment encouragingly but honestly; push vague questions to be more specific.

Vague questions (e.g. "How does it work?") => is_clear=false.

Return a valid JSON object matching the provided schema. No markdown fences.
"""

def input_analyzer(state: LearningGraphState) -> dict[str, Any]:
    """
    Analyze the user's question for clarity, intent, and key concepts.

    Calls the configured LLM with a structured-output schema and stores the
    resulting :class:`InputAnalysisResult` under ``"analysis_results"``. Emits
    no widget markup.

    :param state: Current graph state carrying ``user_message``.
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

    analysis = model.invoke(
        [
            SystemMessage(INPUT_ANALYSER_PROMPT),
            user_message
        ]
    )

    return {
        "analysis_results": analysis
    }