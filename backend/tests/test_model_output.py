import pytest
from langchain_core.messages import AIMessage

from graphs.learning_graph.nodes.model_output import model_output
from graphs.learning_graph.pydantic_models import ResponseBuilderOutput


class TestModelOutput:
    def test_final_output_used_when_present(self, state_full_turn):
        result = model_output(state_full_turn)
        msg = result["messages"][0]
        assert isinstance(msg, AIMessage)
        assert msg.content == state_full_turn.final_output

    def test_final_output_none_falls_back_to_draft_response(self, state):
        state.draft_response = ResponseBuilderOutput(
            draft_response="fallback text", tone="neutral", sources_used=[]
        )
        result = model_output(state)
        msg = result["messages"][0]
        assert msg.content == "fallback text"

    def test_both_none_raises_validation_error(self, state):
        import pydantic_core
        with pytest.raises(pydantic_core._pydantic_core.ValidationError):
            model_output(state)

    def test_message_is_appended(self, state_full_turn):
        result = model_output(state_full_turn)
        assert "messages" in result
        assert isinstance(result["messages"], list)
        assert len(result["messages"]) == 1