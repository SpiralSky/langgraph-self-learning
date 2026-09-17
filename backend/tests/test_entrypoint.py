"""Smoke tests for the langgraph entrypoint (``graphs.entrypoint``).

The module-level ``Graph`` is assembled at import time, so these tests patch
the network-facing seams *before* the import: ``OPENAI_API_KEY`` feeds the
``ChatOpenAI`` constructor (which never calls out) and
``get_default_embedder`` stands in for the fastembed download that a real
embedder would trigger while ``GeneratorNode`` restores the persisted
collection. Everything else is pure structure — no LLM is invoked.
"""

import pytest


@pytest.fixture
def entrypoint_import(monkeypatch):
    """Import ``graphs.entrypoint`` with all network seams stubbed."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "graphs.api.collection.get_default_embedder",
        lambda: lambda s: [0.0] * 384,
    )
    import graphs.entrypoint as entrypoint
    return entrypoint


def test_entrypoint_exports_compiled_graph(entrypoint_import):
    graph = entrypoint_import.graph
    assert type(graph).__name__ == "CompiledStateGraph"
    assert callable(graph.invoke) and callable(graph.ainvoke)


def test_entrypoint_graph_has_generator_node_only(entrypoint_import):
    nodes = entrypoint_import.graph.nodes
    assert set(nodes) == {"__start__", "generator"}


def test_entrypoint_state_model_is_learning_graph_state(entrypoint_import):
    graph = entrypoint_import.graph
    from graphs.structure.state import LearningGraphState

    assert graph.builder.state_schema is LearningGraphState
    fields = dict(graph.get_input_schema().model_fields)
    assert set(fields) == {"user_message", "response"}