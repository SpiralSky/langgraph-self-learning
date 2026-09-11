import pytest
from langgraph.pregel.protocol import RunnableConfig

from graphs.learning_graph.nodes.save_memory import save_memory


class TestSaveMemory:
    PATCH_PATH = "graphs.learning_graph.nodes.save_memory.memory"

    @pytest.fixture
    def mock_memory(self):
        with pytest.MonkeyPatch.context() as mp:
            from graphs.learning_graph.nodes import save_memory as sm
            from unittest.mock import MagicMock
            fake = MagicMock()
            mp.setattr(sm, "memory", fake)
            yield fake

    def test_missing_user_message_raises_value_error(self, state, runnable_config):
        with pytest.raises(ValueError, match="user_message"):
            save_memory(state, runnable_config)

    def test_mem0_add_called_with_correct_args(
        self, state_full_turn, runnable_config, mock_memory
    ):
        save_memory(state_full_turn, runnable_config)
        expected_exchange = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "Final formatted response"},
        ]
        mock_memory.add.assert_called_once_with(
            expected_exchange, user_id="test_thread"
        )

    def test_thread_id_falls_back_to_default(
        self, state_full_turn, mock_memory
    ):
        config = RunnableConfig(configurable={})
        save_memory(state_full_turn, config)
        mock_memory.add.assert_called_once()
        _, kwargs = mock_memory.add.call_args
        assert kwargs["user_id"] == "default_thread"

    def test_no_final_output_excludes_assistant_message(
        self, state_with_message, runnable_config, mock_memory
    ):
        save_memory(state_with_message, runnable_config)
        expected_exchange = [
            {"role": "user", "content": "hello"},
        ]
        mock_memory.add.assert_called_once_with(
            expected_exchange, user_id="test_thread"
        )

    def test_returns_empty_dict(
        self, state_full_turn, runnable_config, mock_memory
    ):
        result = save_memory(state_full_turn, runnable_config)
        assert result == {}