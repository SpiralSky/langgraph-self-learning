from pathlib import Path
from typing import Any

import chromadb
from langchain_core.runnables import RunnableConfig

from graphs.learning_graph.state import LearningGraphState
from graphs.learning_graph.memory import memory

PROJECT_ROOT = Path(__file__).resolve().parents[3] / "data"

client = chromadb.PersistentClient()

def retrieve_memory(state: LearningGraphState, config: RunnableConfig) -> dict[str, Any]:
    """
    Retrieve relevant past memories from the vector store for the current thread.

    Extracts plain-text content from the user message (handling both ``str`` and
    block-list content), scopes the search to the thread id from the run config,
    and returns the ranked memory hits. Emits no widget markup.

    :param state: Current graph state carrying ``user_message``.
    :type state: LearningGraphState
    :param config: LangGraph run configuration, read for the ``thread_id``.
    :type config: RunnableConfig
    :return: Mapping the state key ``"memory_results"`` to the list of matched
        memory snippets for the thread.
    :rtype: dict[str, Any]
    :raises ValueError: If the user message is missing or empty.
    """
    if state.user_message is None:
        raise ValueError("Missing required field(s): user_message")
    message = state.user_message

    message_content = ""

    thread_id = config.get("configurable", {}).get("thread_id", "default_thread")

    match message.content:
        case str():
            message_content = message.content

        case list(content_blocks):
            text_parts = []
            for block in content_blocks:
                match block:
                    case {"type": "text", "text": str(text)}:
                        text_parts.append(text)
                    case _ if hasattr(block, "text"):
                        text_parts.append(block.text)
            message_content = " ".join(text_parts)

    if not message_content:
        raise ValueError("User message content is empty!")

    results = memory.search(message_content, filters={"user_id": thread_id})["results"]
    results = [] if results is None else results

    return {
        "memory_results": results
    }