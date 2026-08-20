from pathlib import Path

from mem0 import Memory

from graphs.learning_graph.config import config as models_config

collection_dir = str(Path(__file__).resolve().parents[3] / "data" / "mem0_db")

llm_config = models_config.get_model_data("memory_llm")
embedder_config = models_config.get_model_data("memory_embedder")

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
            "model": llm_config.model_id,
            "temperature": 0.1,
            "api_key": llm_config.api_key,
            "openai_base_url": llm_config.api_endpoint
        }
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": llm_config.model_id,
            "api_key": llm_config.api_key,
            "openai_base_url": llm_config.api_endpoint
        }
    },
    "version": "v1.1"
}

memory = Memory.from_config(config)
