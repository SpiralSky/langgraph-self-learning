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
    Used to retrieve memories from memory collection in chromadb.
    :param state: Graph Node.
    :param config: RunnableConfig provided by LangGraph.
    :return:
    """
    message_content = ""

    thread_id = config.get("configurable", {}).get("thread_id", "default_thread")

    if message := state.user_message:
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
    else:
        raise ValueError("Missing user message!")

    return {
        "memory_results": results
    }