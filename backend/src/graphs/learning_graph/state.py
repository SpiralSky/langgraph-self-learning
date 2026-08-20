from typing import TypedDict, Annotated, List, Optional

from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field

from graphs.learning_graph.pydantic_models import InputAnalysisResult, ResponseBuilderOutput, ResponseImproverOutput, \
    QueryResult


class LearningGraphState(BaseModel):
    messages: Annotated[List[BaseMessage], add_messages] = []

    user_message: Optional[BaseMessage] = None

    memory_results: List[dict] | None = Field(default_factory=list)
    analysis_results: Optional[InputAnalysisResult] = None
    search_results: Optional[QueryResult] = None
    draft_response: Optional[ResponseBuilderOutput] = None
    improved_response: Optional[ResponseImproverOutput] = None



