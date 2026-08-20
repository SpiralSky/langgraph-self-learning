from typing import TypedDict

from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from graphs.learning_graph.nodes.information_fetcher import QueryResult
from graphs.learning_graph.nodes.input_analyzer import InputAnalysisResult
from graphs.learning_graph.nodes.response_builder import ResponseBuilderOutput


class LearningGraphState(BaseModel):
    user_message: HumanMessage
    memory_results: list[dict]
    analysis_results: InputAnalysisResult
    search_results: QueryResult
    draft_response: ResponseBuilderOutput
    improved_response: ResponseBuilderOutput


