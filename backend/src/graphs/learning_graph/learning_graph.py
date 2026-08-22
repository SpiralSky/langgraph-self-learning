from langgraph.constants import START, END
from langgraph.graph import StateGraph

from graphs.learning_graph.observability import timed
from graphs.learning_graph.nodes.format_output import format_output
from graphs.learning_graph.nodes.model_output import model_output
from graphs.learning_graph.nodes.user_input import user_input
from graphs.learning_graph.nodes.information_fetcher import information_fetcher
from graphs.learning_graph.nodes.input_analyzer import input_analyzer
from graphs.learning_graph.nodes.response_builder import response_builder
from graphs.learning_graph.nodes.response_improver import response_improver
from graphs.learning_graph.nodes.retrieve_memory import retrieve_memory
from graphs.learning_graph.nodes.save_memory import save_memory
from graphs.learning_graph.nodes.phase_router import phase_router
from graphs.learning_graph.state import LearningGraphState

# noinspection bad-argument-type
builder = StateGraph(LearningGraphState)

builder.add_node("user_input", timed("user_input")(user_input))
builder.add_node("model_output", timed("model_output")(model_output))

builder.add_node("information_fetcher", timed("information_fetcher")(information_fetcher))
builder.add_node("input_analyzer", timed("input_analyzer")(input_analyzer))
builder.add_node("retrieve_memory", timed("retrieve_memory")(retrieve_memory))
builder.add_node("phase_router", timed("phase_router")(phase_router))
builder.add_node("response_builder", timed("response_builder")(response_builder))
builder.add_node("response_improver", timed("response_improver")(response_improver))
builder.add_node("format_output", timed("format_output")(format_output))
builder.add_node("save_memory", timed("save_memory")(save_memory))

builder.add_edge(START, "user_input")
builder.add_edge("user_input", "retrieve_memory")
builder.add_edge("user_input", "input_analyzer")

# Steady-state sessions skip external search and go straight to the response.
builder.add_edge("retrieve_memory", "phase_router")
builder.add_edge("input_analyzer", "phase_router")
builder.add_conditional_edges(
    "phase_router",
    lambda s: "response_builder" if not s.use_web_search else "information_fetcher",
    {"response_builder": "response_builder", "information_fetcher": "information_fetcher"},
)
builder.add_edge("information_fetcher", "response_builder")

builder.add_edge("response_builder", "response_improver")

# Short/steady-state turns skip the extra formatting LLM pass.
builder.add_conditional_edges(
    "response_improver",
    lambda s: "model_output" if s.skip_format else "format_output",
    {"model_output": "model_output", "format_output": "format_output"},
)
builder.add_edge("format_output", "model_output")

builder.add_edge("model_output", "save_memory")
builder.add_edge("save_memory", END)

graph = builder.compile()