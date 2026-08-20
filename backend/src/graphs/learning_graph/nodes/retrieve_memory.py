from pathlib import Path
from typing import Any

import chromadb

from graphs.learning_graph.state import LearningGraphState
from graphs.learning_graph.memory import memory

PROJECT_ROOT = Path(__file__).resolve().parents[3] / "data"

client = chromadb.PersistentClient()

def retrieve_memory(state: LearningGraphState) -> dict[str, Any]:
    """
    Used to retrieve memories from memory collection in chromadb.
    :param state: Graph Node
    :return:
    """
    results = memory.search(state["user_message"])["results"]

    return {
        "memory_results": results
    }