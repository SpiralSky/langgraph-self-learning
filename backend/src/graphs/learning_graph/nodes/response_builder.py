import json

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage

from graphs.learning_graph.config import config
from graphs.learning_graph.pydantic_models import ResponseBuilderOutput
from graphs.learning_graph.state import LearningGraphState
from util.validation import requires_present

RECENT_HISTORY_WINDOW = 3

RESPONSE_BUILDER_SYSTEM_PROMPT = """
You are an AI Tutor. Synthesize the provided context into a helpful, accurate, pedagogically sound response.

Input JSON keys: `user_message` (learner's question), `conversation_history` (the last 2-3 turns preceding the question, human + AI), `analysis_results` (intent/clarity/key concepts), `memory_results` (past interactions, known gaps/strengths), `search_results` (fresh web info, if any).

Guidelines:
- Accuracy: use `search_results` to verify or update internal knowledge; trust them on conflict.
- Personalize: extra scaffolding if the user struggled with a concept before; more concise if advanced.
- Continuity: use `conversation_history` to answer follow-up questions coherently, resolving pronouns and references to earlier topics instead of repeating them.
- Intent: align style (direct for 'factual', analogies for 'conceptual').
- Ambiguity: if clarity is low, state the assumption you're making before answering.

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

def _message_to_text(message: BaseMessage) -> str:
    """
    Extract the plain-text content of a message.

    Handles both ``str`` content and openai-style block lists, mirroring the
    extraction logic used in ``retrieve_memory``.

    :param message: Conversation message to serialize.
    :type message: BaseMessage
    :return: The message text as a single string.
    :rtype: str
    """
    match message.content:
        case str(text):
            return text

        case list(content_blocks):
            text_parts = []
            for block in content_blocks:
                match block:
                    case {"type": "text", "text": str(text)}:
                        text_parts.append(text)
                    case _ if hasattr(block, "text"):
                        text_parts.append(block.text)
            return " ".join(text_parts)

        case _:
            return str(message.content)


def response_builder(state: LearningGraphState) -> dict[str, ResponseBuilderOutput]:
    """
    Synthesise the available context into a draft educational response.

    Combines the user message, recent conversation turns (up to
    :data:`RECENT_HISTORY_WINDOW`), analysis, memory, and search results and
    asks the configured LLM for a draft response via structured output. Produces
    plain Markdown only — widget formatting is deferred to ``format_output``.

    :param state: Current graph state carrying ``user_message``, ``messages``
        channel, ``analysis_results``, ``memory_results``, and
        ``search_results``.
    :type state: LearningGraphState
    :return: Mapping the state key ``"draft_response"`` to a
        :class:`ResponseBuilderOutput`.
    :rtype: dict[str, ResponseBuilderOutput]
    :raises ValueError: If any of the required state fields are missing.
    """
    requires_present(
        user_message=state.user_message,
        analysis_results=state.analysis_results,
        memory_results=state.memory_results,
        search_results=state.search_results,
    )

    conversation_history = [
        {
            "role": "assistant" if isinstance(message, AIMessage) else "user",
            "content": _message_to_text(message)
        }
        for message in state.messages[-RECENT_HISTORY_WINDOW - 1:-1]
    ]

    context = {
        "user_message": state.user_message.content,
        "conversation_history": conversation_history,
        "analysis_results": state.analysis_results.model_dump(),
        "memory_results": state.memory_results,
        "search_results": state.search_results.model_dump() if hasattr(state.search_results, 'dict') else state.search_results
    }

    context_json = json.dumps(context, indent=2, default=str)

    model_config = config.get_model_data("response_builder")

    model = init_chat_model(
        model_config.model_id,
        api_key=model_config.api_key,
        base_url=model_config.api_endpoint,
        temperature=0,
        model_provider="openai"
    ).with_structured_output(ResponseBuilderOutput)

    res = model.invoke(
        [
            SystemMessage(RESPONSE_BUILDER_SYSTEM_PROMPT),
            HumanMessage(f"Current Learning Context:\n{context_json}")
        ]
    )

    return {
        "draft_response": res
    }

