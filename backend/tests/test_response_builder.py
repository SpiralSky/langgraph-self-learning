import json
from unittest.mock import patch

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from graphs.learning_graph.nodes.response_builder import (
    RESPONSE_BUILDER_OUTPUT_SCHEMA_HINT,
    RESPONSE_BUILDER_SYSTEM_PROMPT,
    response_builder,
)
from graphs.learning_graph.pydantic_models import ResponseBuilderOutput


class TestResponseBuilder:
    PATCH_PATH = "graphs.learning_graph.nodes.response_builder.get_chat_model"

    @pytest.fixture
    def mock_get_chat_model(self):
        with patch(self.PATCH_PATH) as mock:
            yield mock

    @pytest.fixture
    def fake_output(self):
        return ResponseBuilderOutput(
            draft_response="# Conditional Probability\n\nP(A|B) = P(A∩B) / P(B)",
            tone="socratic",
            sources_used=[],
        )

    def _invocation_messages(self, mock_get_chat_model):
        args = (
            mock_get_chat_model.return_value.with_structured_output.return_value.invoke.call_args
        )
        return args[0][0]

    def test_missing_user_message_raises_value_error(self, state, state_with_strategy):
        state_with_strategy.user_message = None
        with pytest.raises(ValueError, match="user_message"):
            response_builder(state_with_strategy)

    def test_missing_teaching_strategy_raises_value_error(self, state_with_message):
        state_with_message.teaching_strategy = None
        with pytest.raises(ValueError, match="decision_maker must run first"):
            response_builder(state_with_message)

    def test_llm_invoked_with_correct_model_name(
        self, mock_get_chat_model, state_with_strategy, fake_output
    ):
        mock_get_chat_model.return_value.with_structured_output.return_value.invoke.return_value = (
            fake_output
        )
        response_builder(state_with_strategy)
        mock_get_chat_model.assert_called_once_with("response_builder")

    def test_structured_output_configured(
        self, mock_get_chat_model, state_with_strategy, fake_output
    ):
        mock_get_chat_model.return_value.with_structured_output.return_value.invoke.return_value = (
            fake_output
        )
        response_builder(state_with_strategy)
        mock_get_chat_model.return_value.with_structured_output.assert_called_once_with(
            ResponseBuilderOutput, method="function_calling"
        )

    def test_context_json_contains_all_strategy_fields(
        self, mock_get_chat_model, state_with_strategy, fake_output
    ):
        mock_get_chat_model.return_value.with_structured_output.return_value.invoke.return_value = (
            fake_output
        )
        response_builder(state_with_strategy)
        messages = self._invocation_messages(mock_get_chat_model)
        human_msg = messages[2]
        assert isinstance(human_msg, HumanMessage)
        prefix = "Current Teaching Context:\n"
        assert human_msg.content.startswith(prefix)
        ctx = json.loads(human_msg.content[len(prefix):])
        strategy = state_with_strategy.teaching_strategy
        assert ctx["user_message"] == "hello"
        assert ctx["focus_concept"] == strategy.focus_concept
        assert ctx["tone"] == strategy.tone
        assert ctx["execution_plan"] == strategy.execution_plan
        assert isinstance(ctx["session_context"], dict)

    def test_message_history_structure(
        self, mock_get_chat_model, state_with_strategy, fake_output
    ):
        mock_get_chat_model.return_value.with_structured_output.return_value.invoke.return_value = (
            fake_output
        )
        response_builder(state_with_strategy)
        messages = self._invocation_messages(mock_get_chat_model)
        assert len(messages) == 3
        assert isinstance(messages[0], SystemMessage)
        assert messages[0].content.startswith(RESPONSE_BUILDER_SYSTEM_PROMPT)
        assert isinstance(messages[1], SystemMessage)
        assert messages[1].content == RESPONSE_BUILDER_OUTPUT_SCHEMA_HINT
        assert isinstance(messages[2], HumanMessage)

    def test_result_stored_in_draft_response(
        self, mock_get_chat_model, state_with_strategy, fake_output
    ):
        mock_get_chat_model.return_value.with_structured_output.return_value.invoke.return_value = (
            fake_output
        )
        result = response_builder(state_with_strategy)
        assert "draft_response" in result
        assert result["draft_response"] == fake_output