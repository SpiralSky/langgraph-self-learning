"""Runnable outer graph — the langgraph entrypoint.

The compiled graph is a single generator node (``START -> generator -> END``)
that reads the user's ``user_message`` and returns a ``response``. The
generator asks the LLM for builder tool calls, assembles a nested graph, and
runs it as a subgraph; the outer ``LearningGraphState`` just carries the
message in and the final answer out.

The module-level ``graph`` export is what ``langgraph.json`` points at, so
``langgraph dev`` can boot the agent.
"""

from graphs.nodes.generator import GeneratorNode
from graphs.runtime.llm import get_chat_model
from graphs.structure.graph import END, START, Graph

_generator = GeneratorNode(
    name="answer",
    description="Generates and runs a nested graph for the user request.",
)

_graph = Graph()
_graph.add_node(
    "generator",
    _generator,
    llm=get_chat_model("fast"),
)
_graph.add_edge("start->generator", START, "generator")
_graph.add_edge("generator->end", "generator", END)

graph = _graph.compile()