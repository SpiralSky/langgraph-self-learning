"""Learning Graph — LangGraph Socratic tutoring system.

Defines the ``StateGraph`` that orchestrates the teaching loop:
user_input → retrieve_memory + input_analyzer → decision_maker →
response_builder → save_memory → format_output → model_output → END,
with an optional execute_code branch from decision_maker.

Exports the compiled graph, node registry, and compiled prompts via
:data:`LearningGraphExport`.
"""

from typing import TypedDict

from langgraph.constants import START, END
from langgraph.graph import StateGraph

from graphs.learning_graph.compiler import compile_prompts
from graphs.learning_graph.config import config
from graphs.learning_graph.nodes.node import Node
from graphs.learning_graph.nodes.user_input import user_input
from graphs.learning_graph.nodes.input_analyzer import input_analyzer
from graphs.learning_graph.nodes.retrieve_memory import retrieve_memory
from graphs.learning_graph.nodes.decision_maker import decision_maker
from graphs.learning_graph.nodes.response_builder import response_builder
from graphs.learning_graph.nodes.save_memory import save_memory
from graphs.learning_graph.nodes.format_output import format_output
from graphs.learning_graph.nodes.model_output import model_output
from graphs.learning_graph.nodes.execute_code import execute_code
from graphs.learning_graph.observability import timed
from graphs.learning_graph.state import LearningGraphState


class LearningGraphExport(TypedDict):
    graph: StateGraph
    nodes: dict[str, Node]
    prompts: dict[str, str | None]


_NODES: dict[str, Node] = {
    "user_input": user_input,
    "input_analyzer": input_analyzer,
    "retrieve_memory": retrieve_memory,
    "decision_maker": decision_maker,
    "response_builder": response_builder,
    "save_memory": save_memory,
    "format_output": format_output,
    "model_output": model_output,
    "execute_code": execute_code,
}


def _route_after_decision(state: LearningGraphState) -> str:
    """Determine the next node based on the teaching strategy.

    Returns the name of the next node to visit.
    """
    move = state.teaching_strategy.pedagogical_move if state.teaching_strategy else "explain_concept"

    if move == "execute_code":
        return "execute_code"

    # Default flow
    return "response_builder"


builder = StateGraph(LearningGraphState)

for name, node_obj in _NODES.items():
    fn = timed(name)(node_obj)
    builder.add_node(name, fn)

builder.add_edge(START, "user_input")
builder.add_edge("user_input", "retrieve_memory")
builder.add_edge("user_input", "input_analyzer")
builder.add_edge("retrieve_memory", "decision_maker")
builder.add_edge("input_analyzer", "decision_maker")

# Conditional edge from decision_maker based on pedagogical move
builder.add_conditional_edges(
    "decision_maker",
    _route_after_decision,
    {
        "execute_code": "execute_code",
        "response_builder": "response_builder",
    },
)

# After execute_code, flow back to response_builder to continue
builder.add_edge("execute_code", "response_builder")
builder.add_edge("response_builder", "save_memory")
builder.add_edge("save_memory", "format_output")
builder.add_edge("format_output", "model_output")
builder.add_edge("model_output", END)

graph = builder.compile()

compiled_prompts = compile_prompts(_NODES, config.customizations)

export: LearningGraphExport = {
    "graph": graph,
    "nodes": _NODES,
    "prompts": compiled_prompts,
}