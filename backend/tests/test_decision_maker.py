import json
from unittest.mock import patch

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from graphs.learning_graph.nodes.decision_maker import (
    PEDAGOGICAL_DIRECTOR_SYSTEM_PROMPT,
    decision_maker,
)
from graphs.learning_graph.pydantic_models import (
    InputAnalysisResult,
    SessionRecord,
    TeachingStrategy,
)
from graphs.learning_graph.state import LearningGraphState


class TestDecisionMaker:
    PATCH_PATH = "graphs.learning_graph.nodes.decision_maker.get_chat_model"

    @pytest.fixture
    def mock_get_chat_model(self):
        with patch(self.PATCH_PATH) as mock:
            yield mock

    @pytest.fixture
    def fake_strategy(self):
        return TeachingStrategy(
            pedagogical_move="explain_concept",
            focus_concept="conditional probability",
            tone="socratic",
            execution_plan=[
                "Define conditional probability",
                "Give concrete example",
                "Ask a guiding question",
            ],
        )

    def _invocation_context(self, mock_get_chat_model):
        args = (
            mock_get_chat_model.return_value.with_structured_output.return_value.invoke.call_args
        )
        messages = args[0][0]
        context_msg = messages[1]
        assert isinstance(context_msg, SystemMessage)
        prefix = "Current Learning Context:\n"
        assert context_msg.content.startswith(prefix)
        return json.loads(context_msg.content[len(prefix):])

    def test_llm_invoked_with_correct_model_name(
        self, mock_get_chat_model, state_with_analysis, fake_strategy
    ):
        mock_get_chat_model.return_value.with_structured_output.return_value.invoke.return_value = (
            fake_strategy
        )
        decision_maker(state_with_analysis)
        mock_get_chat_model.assert_called_once_with("decision_maker")

    def test_structured_output_configured(
        self, mock_get_chat_model, state_with_analysis, fake_strategy
    ):
        mock_get_chat_model.return_value.with_structured_output.return_value.invoke.return_value = (
            fake_strategy
        )
        decision_maker(state_with_analysis)
        mock_get_chat_model.return_value.with_structured_output.assert_called_once_with(
            TeachingStrategy, method="function_calling"
        )

    def test_context_contains_user_message(
        self, mock_get_chat_model, state_with_message, fake_strategy
    ):
        mock_get_chat_model.return_value.with_structured_output.return_value.invoke.return_value = (
            fake_strategy
        )
        decision_maker(state_with_message)
        ctx = self._invocation_context(mock_get_chat_model)
        assert "user_message" in ctx

    def test_context_contains_analysis_results(
        self, mock_get_chat_model, state_with_analysis, fake_strategy
    ):
        mock_get_chat_model.return_value.with_structured_output.return_value.invoke.return_value = (
            fake_strategy
        )
        decision_maker(state_with_analysis)
        ctx = self._invocation_context(mock_get_chat_model)
        assert "analysis_results" in ctx
        assert ctx["analysis_results"] is not None

    def test_context_contains_active_session(
        self, mock_get_chat_model, state_full_turn, fake_strategy
    ):
        mock_get_chat_model.return_value.with_structured_output.return_value.invoke.return_value = (
            fake_strategy
        )
        decision_maker(state_full_turn)
        ctx = self._invocation_context(mock_get_chat_model)
        assert "active_session" in ctx
        assert ctx["active_session"] is not None
        assert ctx["active_session"]["session_id"] == "test_session_1"

    def test_null_active_session(
        self, mock_get_chat_model, state_with_analysis, fake_strategy
    ):
        state_with_analysis.active_session = None
        mock_get_chat_model.return_value.with_structured_output.return_value.invoke.return_value = (
            fake_strategy
        )
        decision_maker(state_with_analysis)
        ctx = self._invocation_context(mock_get_chat_model)
        assert ctx["active_session"] is None

    def test_null_analysis_results(
        self, mock_get_chat_model, state_with_message, fake_strategy
    ):
        state_with_message.analysis_results = None
        mock_get_chat_model.return_value.with_structured_output.return_value.invoke.return_value = (
            fake_strategy
        )
        decision_maker(state_with_message)
        ctx = self._invocation_context(mock_get_chat_model)
        assert ctx["analysis_results"] is None

    def test_teaching_strategy_returned_in_state_key(
        self, mock_get_chat_model, state_with_analysis, fake_strategy
    ):
        mock_get_chat_model.return_value.with_structured_output.return_value.invoke.return_value = (
            fake_strategy
        )
        result = decision_maker(state_with_analysis)
        assert "teaching_strategy" in result
        assert result["teaching_strategy"] == fake_strategy

    def test_non_dict_memory_results_filtered(
        self, mock_get_chat_model, state_with_message, fake_strategy
    ):
        state_with_message.memory_results = [
            {"text": "valid memory", "id": "1"},
            "not a dict",
            None,
            42,
            {"text": "another valid", "id": "2"},
        ]
        mock_get_chat_model.return_value.with_structured_output.return_value.invoke.return_value = (
            fake_strategy
        )
        decision_maker(state_with_message)
        ctx = self._invocation_context(mock_get_chat_model)
        assert ctx["memory_results"] == ["valid memory", "another valid"]