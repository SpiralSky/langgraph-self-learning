from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from graphs.learning_graph.nodes.input_analyzer import (
    RECENT_HISTORY_WINDOW,
    INPUT_ANALYSER_PROMPT,
    input_analyzer,
)
from graphs.learning_graph.pydantic_models import InputAnalysisResult
from graphs.learning_graph.state import LearningGraphState


class TestInputAnalyzer:
    PATCH_PATH = "graphs.learning_graph.nodes.input_analyzer.get_chat_model"

    @pytest.fixture
    def mock_get_chat_model(self):
        with patch(self.PATCH_PATH) as mock:
            yield mock

    def test_llm_invoked_with_correct_model_name(self, mock_get_chat_model, state_with_message):
        mock_get_chat_model.return_value.with_structured_output.return_value.invoke.return_value = (
            InputAnalysisResult(is_clear=True, intent="factual", key_points=[], comments="ok")
        )
        input_analyzer(state_with_message)
        mock_get_chat_model.assert_called_once_with("input_analyzer")

    def test_structured_output_configured(self, mock_get_chat_model, state_with_message):
        mock_get_chat_model.return_value.with_structured_output.return_value.invoke.return_value = (
            InputAnalysisResult(is_clear=True, intent="factual", key_points=[], comments="ok")
        )
        input_analyzer(state_with_message)
        mock_get_chat_model.return_value.with_structured_output.assert_called_once_with(
            InputAnalysisResult, method="function_calling"
        )

    def test_system_prompt_contains_catalog(self, mock_get_chat_model, state_with_message):
        fake_result = InputAnalysisResult(
            is_clear=True, intent="factual", key_points=[], comments="ok"
        )
        mock_get_chat_model.return_value.with_structured_output.return_value.invoke.return_value = (
            fake_result
        )

        input_analyzer(state_with_message)

        invocation_args = (
            mock_get_chat_model.return_value.with_structured_output.return_value.invoke.call_args
        )
        call_kwargs = invocation_args[0][0]
        system_msg = call_kwargs[0]
        assert isinstance(system_msg, SystemMessage)
        assert "Session hint detection" in system_msg.content

    def test_history_window_captured_correctly(self, mock_get_chat_model):
        messages = [
            AIMessage("response 1"),
            HumanMessage("question 2"),
            AIMessage("response 2"),
            HumanMessage("question 3"),
            AIMessage("response 3"),
            HumanMessage("current question"),
        ]
        state = LearningGraphState(messages=messages)
        state.user_message = messages[-1]

        fake_result = InputAnalysisResult(
            is_clear=True, intent="factual", key_points=[], comments="ok"
        )
        mock_get_chat_model.return_value.with_structured_output.return_value.invoke.return_value = (
            fake_result
        )

        input_analyzer(state)

        invocation_args = (
            mock_get_chat_model.return_value.with_structured_output.return_value.invoke.call_args
        )
        call_kwargs = invocation_args[0][0]
        # expected: [SystemMessage, *recent_history, user_message]
        # recent_history = messages[-4:-1] = messages[1:5]
        expected_count = 1 + RECENT_HISTORY_WINDOW + 1
        assert len(call_kwargs) == expected_count
        assert call_kwargs[1] == messages[-4]
        assert call_kwargs[2] == messages[-3]
        assert call_kwargs[3] == messages[-2]
        assert call_kwargs[4] == messages[-1]

    def test_result_stored_in_analysis_results(self, mock_get_chat_model, state_with_message):
        fake_result = InputAnalysisResult(
            is_clear=True, intent="conceptual", key_points=["Bayes"], comments="Clear question."
        )
        mock_get_chat_model.return_value.with_structured_output.return_value.invoke.return_value = (
            fake_result
        )

        result = input_analyzer(state_with_message)
        assert "analysis_results" in result
        assert result["analysis_results"] == fake_result

    def test_no_session_catalog(self, mock_get_chat_model, state_with_message):
        catalog_text = "no session catalog text"
        fake_result = InputAnalysisResult(
            is_clear=True, intent="factual", key_points=[], comments="ok"
        )
        mock_get_chat_model.return_value.with_structured_output.return_value.invoke.return_value = (
            fake_result
        )

        with patch(
            "graphs.learning_graph.nodes.input_analyzer.analyzer_catalog_text",
            return_value=catalog_text,
        ):
            result = input_analyzer(state_with_message)

        assert "analysis_results" in result

        invocation_args = (
            mock_get_chat_model.return_value.with_structured_output.return_value.invoke.call_args
        )
        system_msg = invocation_args[0][0][0]
        assert catalog_text in system_msg.content

    def test_empty_history_window(self, mock_get_chat_model, state_with_message):
        fake_result = InputAnalysisResult(
            is_clear=True, intent="factual", key_points=[], comments="ok"
        )
        mock_get_chat_model.return_value.with_structured_output.return_value.invoke.return_value = (
            fake_result
        )

        result = input_analyzer(state_with_message)

        invocation_args = (
            mock_get_chat_model.return_value.with_structured_output.return_value.invoke.call_args
        )
        call_kwargs = invocation_args[0][0]
        # Only SystemMessage + user_message (no history since only 1 message)
        assert len(call_kwargs) == 2
        assert isinstance(call_kwargs[0], SystemMessage)
        assert call_kwargs[1] == state_with_message.user_message

    def test_hides_messages_before_window(self, mock_get_chat_model):
        messages = [
            AIMessage("old response"),
            HumanMessage("old question"),
            AIMessage("mid response"),
            HumanMessage("mid question"),
            AIMessage("recent response"),
            HumanMessage("current question"),
        ]
        state = LearningGraphState(messages=messages)
        state.user_message = messages[-1]

        fake_result = InputAnalysisResult(
            is_clear=True, intent="factual", key_points=[], comments="ok"
        )
        mock_get_chat_model.return_value.with_structured_output.return_value.invoke.return_value = (
            fake_result
        )

        input_analyzer(state)

        invocation_args = (
            mock_get_chat_model.return_value.with_structured_output.return_value.invoke.call_args
        )
        call_kwargs = invocation_args[0][0]
        # messages[0] and messages[1] are before the window and should not appear
        assert call_kwargs[1] == messages[2]
        assert call_kwargs[2] == messages[3]
        assert call_kwargs[3] == messages[4]
        assert call_kwargs[4] == messages[5]