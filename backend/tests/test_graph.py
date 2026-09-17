"""Smoke tests for the ``Graph`` structure (full coverage in
``test_graph_structure.py``). Keeps the original intent: a graph that can be
built, validated, and — once ``compile()`` lands — executed.
"""

from graphs.structure.graph import END, START, Graph
from graphs.nodes.text_node import TextNode
from graphs.structure.state import LearningGraphState


def test_graph_constructs_with_default_state_model():
    assert Graph().state_model is LearningGraphState


def test_graph_builds_and_validates_simple_chain():
    graph = Graph()
    graph.add_node("n", TextNode("node", "a node", "Do {user_message}", params={"user_message": str}))
    graph.add_edge("e1", START, "n")
    graph.add_edge("e2", "n", END)
    graph.validate()
    assert len(graph) == 1
    assert "n" in graph