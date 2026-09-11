"""LearningGraphState — the central state object threaded through every graph turn.

All nodes read from and write to this state. Fields are populated
sequentially by the graph nodes and consumed by downstream nodes.
"""

from typing import Annotated, List, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field

from graphs.learning_graph.pydantic_models import InputAnalysisResult, ResponseBuilderOutput, \
    QueryResult, SessionRecord, KnowledgeComponent, TeachingStrategy


class LearningGraphState(BaseModel):
    """Central state object threaded through every graph turn.

    All nodes read from and write to this state. Fields are populated
    sequentially by the graph nodes and consumed by downstream nodes.

    :param messages: Conversation history; LangGraph-managed channel.
    :vartype messages: Annotated[list[BaseMessage], add_messages]
    :param user_message: Latest user message, extracted by user_input.
    :vartype user_message: BaseMessage | None
    :param memory_results: Compacted memory search hits from retrieve_memory.
    :vartype memory_results: list[dict] | None
    :param analysis_results: Intent/clarity/key-concepts analysis from input_analyzer.
    :vartype analysis_results: InputAnalysisResult | None
    :param search_results: Web search results, if use_web_search is set.
    :vartype search_results: QueryResult | None
    :param active_session: Current learning session state.
    :vartype active_session: SessionRecord | None
    :param knowledge_model: Persistent per-thread mastery model.
    :vartype knowledge_model: list[KnowledgeComponent]
    :param use_web_search: Whether to query web search.
    :vartype use_web_search: bool
    :param skip_format: Skip formatting for this turn.
    :vartype skip_format: bool
    :param assess_session: Whether to assess the session this turn.
    :vartype assess_session: bool
    :param teaching_strategy: Pedagogical decision from decision_maker.
    :vartype teaching_strategy: TeachingStrategy | None
    :param draft_response: Draft response from response_builder.
    :vartype draft_response: ResponseBuilderOutput | None
    :param final_output: Formatted output ready for the chat UI.
    :vartype final_output: str | None
    """

    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    user_message: BaseMessage | None = None
    memory_results: list[dict] | None = None
    analysis_results: InputAnalysisResult | None = None
    search_results: QueryResult | None = None
    active_session: SessionRecord | None = None
    knowledge_model: list[KnowledgeComponent] = Field(default_factory=list)
    use_web_search: bool = False
    skip_format: bool = False
    assess_session: bool = False
    teaching_strategy: TeachingStrategy | None = None
    draft_response: ResponseBuilderOutput | None = None
    final_output: str | None = None



