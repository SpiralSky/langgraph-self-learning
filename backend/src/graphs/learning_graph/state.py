from typing import TypedDict

from langchain_core.messages import HumanMessage

from graphs.learning_graph.nodes.input_analyzer import InputAnalysisResult


class LearningGraphState(TypedDict):
    user_message: HumanMessage
    memory_results: list[dict]
    analysis_results: InputAnalysisResult


