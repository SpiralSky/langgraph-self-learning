from langgraph.constants import START, END
from langgraph.graph import StateGraph

from graphs.learning_graph.nodes.information_fetcher import information_fetcher
from graphs.learning_graph.nodes.input_analyzer import input_analyzer
from graphs.learning_graph.nodes.response_builder import response_builder
from graphs.learning_graph.nodes.response_improver import response_improver
from graphs.learning_graph.nodes.retrieve_memory import retrieve_memory
from graphs.learning_graph.state import LearningGraphState

# noinspection bad-argument-type
agent_builder = StateGraph(LearningGraphState)

agent_builder.add_node("information_fetcher", information_fetcher)
agent_builder.add_node("input_analyzer", input_analyzer)
agent_builder.add_node("retrieve_memory", retrieve_memory)
agent_builder.add_node("response_builder", response_builder)
agent_builder.add_node("response_improver", response_improver)

agent_builder.add_edge(START, "retrieve_memory")
agent_builder.add_edge("retrieve_memory", "input_analyzer")
agent_builder.add_edge("input_analyzer", "information_fetcher")
agent_builder.add_edge("information_fetcher", "response_builder")
agent_builder.add_edge("response_builder", "response_improver")
agent_builder.add_edge("response_improver", END)