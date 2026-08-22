from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig

from graphs.learning_graph.memory import memory
from graphs.learning_graph.state import LearningGraphState


def _message_to_text(message: BaseMessage) -> str:
    """
    Extract the plain-text content of a message.

    Handles both ``str`` content and openai-style block lists, mirroring the
    extraction logic used in ``retrieve_memory`` and ``response_builder``.

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


def save_memory(state: LearningGraphState, config: RunnableConfig) -> dict:
    """
    Persist the current exchange to the long-term memory store.

    Saves the direct conversation (the user's message and the formatted final
    response) into mem0, scoped to the thread id from the run config. Runs
    synchronously after ``model_output``, so the reply is already streamed to
    the user before the write begins.

    :param state: Current graph state carrying ``user_message`` and
        ``final_output``.
    :type state: LearningGraphState
    :param config: LangGraph run configuration, read for the ``thread_id``.
    :type config: RunnableConfig
    :return: An empty update; the node only writes to the memory store.
    :rtype: dict
    :raises ValueError: If the user message is missing.
    """
    if state.user_message is None:
        raise ValueError("Missing required field(s): user_message")

    thread_id = config.get("configurable", {}).get("thread_id", "default_thread")

    exchange = [
        {"role": "user", "content": _message_to_text(state.user_message)},
    ]

    if state.final_output is not None:
        exchange.append({"role": "assistant", "content": state.final_output})

    memory.add(exchange, user_id=thread_id)

    return {}