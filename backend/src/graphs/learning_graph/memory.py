"""mem0 + Qdrant memory setup for the LearningGraph.

Creates the ``data/mem0_db/`` collection directory, loads provider data
from the shared config, and builds the :class:`mem0.Memory` instance
used by ``retrieve_memory`` and ``save_memory``.
"""

import os
from pathlib import Path

from mem0 import Memory

from graphs.learning_graph.config import config as models_config

collection_dir = os.environ.get(
    "MEM0_COLLECTION_DIR",
    str(Path(__file__).resolve().parents[3] / "data" / "mem0_db"),
)
Path(collection_dir).mkdir(parents=True, exist_ok=True)

# Provider data loaded from config.
llm_config = models_config.get_model_data("memory_llm")
embedder_config = models_config.get_model_data("memory_embedder")
llm_reasoning = models_config.reasoning_effort_for("memory_llm")

llm_llm_config = {
    "model": llm_config.model_id,
    "temperature": 0.1,
    "api_key": llm_config.api_key,
    "openai_base_url": llm_config.api_endpoint,
}

config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "user_memories",
            "path": collection_dir,
        }
    },
    "llm": {
        "provider": "openai",
        "config": llm_llm_config
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": embedder_config.model_id,
            "api_key": embedder_config.api_key,
            "openai_base_url": embedder_config.api_endpoint
        }
    },
    "version": "v1.1",
    "enable_telemetry": False
}

memory = Memory.from_config(config)