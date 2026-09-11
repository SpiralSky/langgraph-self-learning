import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from graphs.learning_graph.nodes.format_output import (
    FORMAT_OUTPUT_SYSTEM_PROMPT,
    format_output,
    normalize_widget_fences,
)


class TestFormatOutput:
    PATCH_PATH = "graphs.learning_graph.nodes.format_output.get_chat_model"

    @pytest.fixture
    def mock_model(self):
        with patch(self.PATCH_PATH) as mock:
            chat = MagicMock()
            mock.return_value = chat
            yield mock

    @pytest.fixture
    def fake_ai_message(self):
        return AIMessage(
            content="# Conditional Probability\n\nP(A|B) = P(A∩B) / P(B)"
        )

    def _set_invoke(self, mock_model, ai_message):
        mock_model.return_value.invoke.return_value = ai_message

    def test_missing_draft_response_raises_value_error(self, state):
        with pytest.raises(ValueError, match="draft_response"):
            format_output(state)

    def test_llm_invoked_with_correct_model_name_and_temp(
        self, mock_model, state_full_turn, fake_ai_message
    ):
        self._set_invoke(mock_model, fake_ai_message)
        format_output(state_full_turn)
        mock_model.assert_called_once_with("format_output", temperature=0.2)

    def test_context_json_includes_draft_tone_strategy(
        self, mock_model, state_full_turn, fake_ai_message
    ):
        self._set_invoke(mock_model, fake_ai_message)
        format_output(state_full_turn)
        args = mock_model.return_value.invoke.call_args
        messages = args[0][0]
        assert isinstance(messages[0], SystemMessage)
        assert messages[0].content.startswith(FORMAT_OUTPUT_SYSTEM_PROMPT)
        assert isinstance(messages[1], HumanMessage)
        ctx = json.loads(messages[1].content)
        draft = state_full_turn.draft_response
        assert ctx["response_to_format"] == draft.draft_response
        assert ctx["original_tone"] == draft.tone
        assert ctx["strategy_used"] == "direct"

    def test_normalization_applied_when_result_is_str(
        self, mock_model, state_full_turn
    ):
        with patch(
            "graphs.learning_graph.nodes.format_output.normalize_widget_fences"
        ) as mock_norm:
            msg = AIMessage(content="# Raw markdown output")
            mock_model.return_value.invoke.return_value = msg
            mock_norm.return_value = "# Normalized output"
            result = format_output(state_full_turn)
            mock_norm.assert_called_once_with("# Raw markdown output")
            assert result["final_output"] == "# Normalized output"

    def test_normalization_not_applied_on_non_string_result(
        self, mock_model, state_full_turn
    ):
        with patch(
            "graphs.learning_graph.nodes.format_output.normalize_widget_fences"
        ) as mock_norm:
            msg = AIMessage(content=[{"type": "text", "text": "hello"}])
            mock_model.return_value.invoke.return_value = msg
            result = format_output(state_full_turn)
            mock_norm.assert_not_called()
            assert result["final_output"] == [{"type": "text", "text": "hello"}]

    def test_result_stored_in_final_output(
        self, mock_model, state_full_turn, fake_ai_message
    ):
        self._set_invoke(mock_model, fake_ai_message)
        result = format_output(state_full_turn)
        assert result == {"final_output": fake_ai_message.content}


class TestNormalizeWidgetFences:
    def test_converts_text_widget_div_to_fence(self):
        html = '<div data-widget="text" data-widget-title="Note" data-widget-color="blue">Important info</div>'
        expected = ':::text title="Note" color="blue"\nImportant info\n:::'
        assert normalize_widget_fences(html) == expected

    def test_converts_code_widget_div_to_fence(self):
        html = '<div data-widget="code" data-widget-language="python" data-widget-title="Example">print("hello")</div>'
        expected = ':::code title="Example" language="python"\nprint("hello")\n:::'
        assert normalize_widget_fences(html) == expected

    def test_preserves_attributes_correctly(self):
        html = '<div data-widget="text" data-widget-title="Warning" data-widget-color="amber">Careful!</div>'
        expected = ':::text title="Warning" color="amber"\nCareful!\n:::'
        assert normalize_widget_fences(html) == expected

    def test_noop_when_no_div_markup(self):
        markdown = "# Hello\n\nThis is plain markdown."
        assert normalize_widget_fences(markdown) == markdown

    def test_handles_multiline_body(self):
        html = '<div data-widget="code" data-widget-language="python" data-widget-title="multiline">line1\nline2\nline3</div>'
        expected = ':::code title="multiline" language="python"\nline1\nline2\nline3\n:::'
        assert normalize_widget_fences(html) == expected