from typing import Annotated, List, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field

from graphs.learning_graph.pydantic_models import InputAnalysisResult, ResponseBuilderOutput, ResponseImproverOutput, \
    QueryResult, SessionRecord


class LearningGraphState(BaseModel):
    messages: Annotated[List[BaseMessage], add_messages] = []

    user_message: Optional[BaseMessage] = None

    memory_results: List[dict] | None = Field(default_factory=list)
    analysis_results: Optional[InputAnalysisResult] = None
    search_results: Optional[QueryResult] = None
    active_session: Optional[SessionRecord] = None
    use_web_search: bool = True
    skip_format: bool = False
    draft_response: Optional[ResponseBuilderOutput] = None
    improved_response: Optional[ResponseImproverOutput] = None
    final_output: Optional[str] = None



