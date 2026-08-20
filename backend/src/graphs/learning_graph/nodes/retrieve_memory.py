from pathlib import Path

import chromadb

from graphs.learning_graph.state import LearningGraphState

PROJECT_ROOT = Path(__file__).resolve().parents[3] / "data"

client = chromadb.PersistentClient()

def retrieve_memory(state: LearningGraphState):
    """
    Used to retrieve memories from memory collection in chromadb.
    :param state: Graph Node
    :return:
    """
    collection = client.get_or_create_collection(name="memories")
