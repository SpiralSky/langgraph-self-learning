from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph
from langgraph.pregel.protocol import RunnableConfig

from graphs.learning_graph.config import Config, ProviderData, read_config
from graphs.learning_graph.pydantic_models import (
    InputAnalysisResult,
    PrerequisiteGap,
    QueryResult,
    ResponseBuilderOutput,
    SessionControl,
    SessionRecord,
    TeachingStrategy,
)
from graphs.learning_graph.state import LearningGraphState


@pytest.fixture(scope="session")
def config() -> Config:
    root = Path(__file__).resolve().parents[1]
    return read_config(root / "config.yaml")


@pytest.fixture
def state() -> LearningGraphState:
    return LearningGraphState()


@pytest.fixture
def state_with_message(state) -> LearningGraphState:
    state.messages = [HumanMessage("hello")]
    state.user_message = HumanMessage("hello")
    return state


@pytest.fixture
def state_with_analysis(state_with_message) -> LearningGraphState:
    state_with_message.analysis_results = InputAnalysisResult(
        is_clear=True,
        intent="conceptual",
        key_points=["conditional probability"],
        prerequisite_gaps=[
            PrerequisiteGap(
                concept="Bayes theorem",
                status="rusty",
                evidence="learner said 'I forget Bayes'",
            )
        ],
        comments="User has a clear conceptual question.",
    )
    return state_with_message


@pytest.fixture
def state_with_strategy(state_with_analysis) -> LearningGraphState:
    state_with_analysis.teaching_strategy = TeachingStrategy(
        pedagogical_move="explain_concept",
        focus_concept="conditional probability",
        tone="socratic",
        execution_plan=[
            "Define conditional probability",
            "Give concrete example",
            "Ask a guiding question",
        ],
    )
    return state_with_analysis


@pytest.fixture
def state_full_turn(state_with_strategy) -> LearningGraphState:
    s = state_with_strategy
    s.draft_response = ResponseBuilderOutput(
        draft_response="# Conditional Probability\n\nP(A|B) = P(A∩B) / P(B)",
        tone="socratic",
        sources_used=[],
    )
    s.final_output = "Final formatted response"
    s.search_results = QueryResult(
        confidence=0.9,
        information="Conditional probability is the probability of an event occurring given another event.",
        queries=["conditional probability definition"],
        reasoning="Direct definition match.",
    )
    s.active_session = SessionRecord(
        session_id="test_session_1",
        thread_id="test_thread",
        description="A test learning session",
        control=SessionControl(awaiting_user_input=False),
    )
    return s


@pytest.fixture
def runnable_config() -> RunnableConfig:
    return RunnableConfig(configurable={"thread_id": "test_thread"})


@pytest.fixture
def mock_mem0():
    with patch("graphs.learning_graph.memory.memory") as mock:
        mock.search.return_value = {"results": []}
        mock.add.return_value = {"message": "ok"}
        yield mock


@pytest.fixture
def mock_llm():
    with patch("graphs.learning_graph.llm.get_chat_model") as mock:
        yield mock
