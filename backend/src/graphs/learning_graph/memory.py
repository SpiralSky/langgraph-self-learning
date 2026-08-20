from pathlib import Path

from mem0 import Memory

from graphs.learning_graph.config import config

memory = Memory()

collection_dir = str(Path(__file__).resolve().parents[3] / "data" / "mem0_db")

models = config.model_providers[""]

config = {
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "user_memories",
            "path": collection_dir,
        }
    },
    "llm": {
        "provider": "openai",
        "config": {
            "model": "",
            "temperature": 0.1,
            "api_key": "your-openai-key",
            # "openai_base_ url": "http://localhost:11434/v1"  # Optional custom endpoint
        }
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small",
            "api_key": "your-openai-key",
        }
    },
    "version": "v1.1"
}