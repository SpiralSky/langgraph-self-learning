from langgraph.constants import START, END
from langgraph.graph import StateGraph

from graphs.learning_graph.nodes.model_output import model_output
from graphs.learning_graph.nodes.user_input import user_input
from graphs.learning_graph.nodes.information_fetcher import information_fetcher
from graphs.learning_graph.nodes.input_analyzer import input_analyzer
from graphs.learning_graph.nodes.response_builder import response_builder
from graphs.learning_graph.nodes.response_improver import response_improver
from graphs.learning_graph.nodes.retrieve_memory import retrieve_memory
from graphs.learning_graph.state import LearningGraphState

# noinspection bad-argument-type
builder = StateGraph(LearningGraphState)

builder.add_node("user_input", user_input)
builder.add_node("model_output", model_output)

builder.add_node("information_fetcher", information_fetcher)
builder.add_node("input_analyzer", input_analyzer)
builder.add_node("retrieve_memory", retrieve_memory)
builder.add_node("response_builder", response_builder)
builder.add_node("response_improver", response_improver)

builder.add_edge(START, "user_input")
builder.add_edge("user_input", "retrieve_memory")
builder.add_edge("retrieve_memory", "input_analyzer")
builder.add_edge("input_analyzer", "information_fetcher")
builder.add_edge("information_fetcher", "response_builder")
builder.add_edge("response_builder", "response_improver")
builder.add_edge("response_improver", "model_output")
builder.add_edge("model_output", END)

graph = builder.compile()