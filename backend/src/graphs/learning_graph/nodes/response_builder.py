import json
import time

from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage
from langchain_core.runnables import Runnable

from graphs.learning_graph.llm import get_structured_model
from graphs.learning_graph.pydantic_models import ResponseBuilderOutput
from graphs.learning_graph.state import LearningGraphState
from graphs.learning_graph.text_utils import message_to_text

RECENT_HISTORY_WINDOW = 3
STRUCTURED_OUTPUT_RETRY_ATTEMPTS = 3
STRUCTURED_OUTPUT_RETRY_BACKOFF_SECONDS = 1.0

RESPONSE_BUILDER_SYSTEM_PROMPT = """
You are an AI Tutor. Synthesize the provided context into a helpful, accurate, pedagogically sound response.

Input JSON keys: `user_message` (learner's question), `conversation_history` (the last 2-3 turns preceding the question, human + AI), `analysis_results` (intent/clarity/key concepts), `learning_session` (current topic + condensed ledger of what the learner knows/struggles with), `memory_results` (past interactions, known gaps/strengths), `search_results` (fresh web info, if any).

Guidelines:
- Accuracy: use `search_results` to verify or update internal knowledge; trust them on conflict.
- Personalize: extra scaffolding if the user struggled with a concept before; more concise if advanced.
- Continuity: use `conversation_history` to answer follow-up questions coherently, resolving pronouns and references to earlier topics instead of repeating them.
- Intent: align style (direct for 'factual', analogies for 'conceptual').
- Ambiguity: if clarity is low, state the assumption you're making before answering.
- Markdown only: keep `response_content` as plain Markdown; never use HTML tags or widget markup (e.g. `<div data-widget="...">` or `:::` fences) — widget formatting is applied downstream.

Strategy by intent:
- Factual: answer clearly, then add "Why it matters".
- Conceptual: use analogies, break down complex ideas, avoid undefined jargon.
- Problem solving: show the logic/steps; don't just hand the answer.
- Unclear: give a best-guess answer and explicitly state assumptions.

Output JSON:
- `response_content`: str, main educational content in Markdown
- `tone`: str, e.g. encouraging/formal/direct/socratic
- `sources_used`: list[str], sources or key facts from search_results that informed the answer
- `follow_up_suggestion`: str, optional question/topic to guide further learning

Never mention "search results" or "memory" in `response_content`; integrate them naturally. Keep language accessible but precise.
"""

def _invoke_structured_with_retry(
    model: Runnable[list[BaseMessage], ResponseBuilderOutput],
    messages: list[BaseMessage]
) -> ResponseBuilderOutput:
    """
    Invoke a structured-output model, retrying transient JSON parse failures.

    The SDK's streaming accumulator can surface a ``ValueError`` when the
    upstream endpoint emits an empty or unparseable delta for a single chunk;
    re-invoking the model gives it another chance to produce valid JSON.

    :param model: Runnable configured with ``with_structured_output``.
    :type model: Runnable
    :param messages: Prompt messages to pass to the model.
    :type messages: list[BaseMessage]
    :return: The parsed structured output.
    :rtype: ResponseBuilderOutput
    :raises ValueError: If every attempt fails to produce parseable output.
    """
    last_error: ValueError | None = None
    for attempt in range(STRUCTURED_OUTPUT_RETRY_ATTEMPTS):
        try:
            return model.invoke(messages)
        except ValueError as e:
            last_error = e
            if attempt < STRUCTURED_OUTPUT_RETRY_ATTEMPTS - 1:
                time.sleep(STRUCTURED_OUTPUT_RETRY_BACKOFF_SECONDS * (attempt + 1))
    if last_error is not None:
        raise last_error


def response_builder(state: LearningGraphState) -> dict[str, ResponseBuilderOutput]:
    """
    Synthesise the available context into a draft educational response.

    Combines the user message, recent conversation turns (up to
    :data:`RECENT_HISTORY_WINDOW`), analysis, the condensed learning-session
    ledger, compacted memory, and search results and asks the configured LLM for
    a draft response via structured output. Produces plain Markdown only —
    widget formatting is deferred to ``format_output``.

    :param state: Current graph state carrying ``user_message``, ``messages``
        channel, ``analysis_results``, ``memory_results``, ``search_results``,
        and ``active_session``.
    :type state: LearningGraphState
    :return: Mapping the state key ``"draft_response"`` to a
        :class:`ResponseBuilderOutput`.
    :rtype: dict[str, ResponseBuilderOutput]
    :raises ValueError: If any of the required state fields are missing.
    """
    if state.user_message is None:
        raise ValueError("Missing required field(s): user_message")
    if state.analysis_results is None:
        raise ValueError("Missing required field(s): analysis_results")
    if state.memory_results is None:
        raise ValueError("Missing required field(s): memory_results")
    if state.search_results is None:
        raise ValueError("Missing required field(s): search_results")

    user_message = state.user_message
    analysis_results = state.analysis_results
    memory_results = state.memory_results
    search_results = state.search_results

    conversation_history = [
        {
            "role": "assistant" if isinstance(message, AIMessage) else "user",
            "content": message_to_text(message)
        }
        for message in state.messages[-RECENT_HISTORY_WINDOW - 1:-1]
    ]

    session_context = None
    if state.active_session is not None:
        session_context = {
            "topic": state.active_session.topic,
            "ledger": state.active_session.ledger,
        }

    context = {
        "user_message": message_to_text(user_message),
        "conversation_history": conversation_history,
        "analysis_results": analysis_results.model_dump(),
        "learning_session": session_context,
        "memory_results": [
            m.get("text") for m in memory_results if isinstance(m, dict)
        ] if memory_results else [],
        "search_results": search_results.model_dump() if hasattr(search_results, 'dict') else search_results
    }

    context_json = json.dumps(context, indent=2, default=str)

    model = get_structured_model("response_builder", ResponseBuilderOutput)

    res = _invoke_structured_with_retry(
        model,
        [
            SystemMessage(RESPONSE_BUILDER_SYSTEM_PROMPT),
            HumanMessage(f"Current Learning Context:\n{context_json}")
        ]
    )

    return {
        "draft_response": res
    }

