from langchain_core.messages import AIMessage, HumanMessage

from graphs.learning_graph.nodes.user_input import user_input
from graphs.learning_graph.state import LearningGraphState


class TestUserInput:
    def test_extracts_last_message_from_multiple(self):
        state = LearningGraphState(
            messages=[AIMessage("first"), HumanMessage("second")]
        )
        result = user_input(state)
        assert result == {"user_message": HumanMessage("second")}

    def test_single_message(self):
        state = LearningGraphState(messages=[HumanMessage("only")])
        result = user_input(state)
        assert result == {"user_message": HumanMessage("only")}

    def test_empty_messages_raises_index_error(self):
        state = LearningGraphState(messages=[])
        try:
            user_input(state)
            assert False, "Expected IndexError"
        except IndexError:
            pass