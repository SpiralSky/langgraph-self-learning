from unittest.mock import patch

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from graphs.learning_graph.nodes.retrieve_memory import (
    _compact_results,
    _extract_active_session,
    _extract_knowledge_model,
    _normalize_entries,
    retrieve_memory,
)
from graphs.learning_graph.pydantic_models import KnowledgeComponent, SessionRecord


class TestRetrieveMemoryValidation:
    PATCH_PATH = "graphs.learning_graph.nodes.retrieve_memory.memory"

    @pytest.fixture
    def mock_memory(self):
        with patch(self.PATCH_PATH) as mock:
            yield mock

    def test_missing_user_message_raises_value_error(self, mock_memory, state):
        with pytest.raises(ValueError, match="Missing required field"):
            retrieve_memory(state, RunnableConfig())

    def test_empty_user_message_raises_value_error(self, mock_memory, state):
        state.user_message = HumanMessage(content="")
        with pytest.raises(ValueError, match="empty"):
            retrieve_memory(state, RunnableConfig())


class TestCompactResults:
    def test_filters_out_session_entries(self):
        entries = [
            {"id": "1", "memory": "good result", "metadata": {"kind": "session"}},
            {"id": "2", "memory": "keep this", "metadata": {"kind": "fact"}},
        ]
        result = _compact_results(entries, top_k=5, max_chars=300)
        assert len(result) == 1
        assert result[0]["id"] == "2"

    def test_filters_out_knowledge_model_entries(self):
        entries = [
            {"id": "1", "memory": "knowledge data", "metadata": {"kind": "knowledge_model"}},
            {"id": "2", "memory": "keep this", "metadata": {"kind": "fact"}},
        ]
        result = _compact_results(entries, top_k=5, max_chars=300)
        assert len(result) == 1
        assert result[0]["id"] == "2"

    def test_truncates_text_to_max_chars(self):
        long_text = "a" * 500
        entries = [{"id": "1", "memory": long_text, "metadata": {"kind": "fact"}}]
        result = _compact_results(entries, top_k=5, max_chars=10)
        assert len(result[0]["text"]) == 10

    def test_respects_top_k_limit(self):
        entries = [{"id": str(i), "memory": f"entry {i}", "metadata": {"kind": "fact"}} for i in range(10)]
        result = _compact_results(entries, top_k=3, max_chars=300)
        assert len(result) == 3

    def test_skip_entries_with_missing_metadata(self):
        entries = [
            {"id": "1", "memory": "good", "metadata": None},
            {"id": "2", "memory": "good too"},
        ]
        result = _compact_results(entries, top_k=5, max_chars=300)
        assert len(result) == 2


class TestNormalizeEntries:
    def test_dict_with_results(self):
        resp = {"results": [{"id": "1"}, {"id": "2"}]}
        assert _normalize_entries(resp) == [{"id": "1"}, {"id": "2"}]

    def test_object_with_results_attr(self):
        class FakeResp:
            results = [{"id": "1"}]
        assert _normalize_entries(FakeResp()) == [{"id": "1"}]

    def test_raw_list(self):
        resp = [{"id": "1"}, {"id": "2"}]
        assert _normalize_entries(resp) == [{"id": "1"}, {"id": "2"}]

    def test_missing_results_returns_empty(self):
        assert _normalize_entries({}) == []

    def test_empty_results_returns_empty(self):
        assert _normalize_entries({"results": []}) == []

    def test_none_results_returns_empty(self):
        assert _normalize_entries({"results": None}) == []


class TestExtractActiveSession:
    def test_picks_entry_with_session_kind(self):
        entries = [
            {"id": "1", "memory": '{"session_id": "s1", "updated_at": 100.0}', "metadata": {"kind": "session"}},
        ]
        result = _extract_active_session(entries)
        assert result is not None
        assert result.session_id == "s1"

    def test_returns_most_recently_updated_when_multiple(self):
        entries = [
            {"id": "1", "memory": '{"session_id": "old", "updated_at": 50.0}', "metadata": {"kind": "session"}},
            {"id": "2", "memory": '{"session_id": "new", "updated_at": 200.0}', "metadata": {"kind": "session"}},
        ]
        result = _extract_active_session(entries)
        assert result.session_id == "new"

    def test_skips_malformed_json_payloads(self):
        entries = [
            {"id": "1", "memory": "not json", "metadata": {"kind": "session"}},
            {"id": "2", "memory": '{"session_id": "good", "updated_at": 100.0}', "metadata": {"kind": "session"}},
        ]
        result = _extract_active_session(entries)
        assert result is not None
        assert result.session_id == "good"

    def test_returns_none_when_no_session_found(self):
        entries = [
            {"id": "1", "memory": "some memory", "metadata": {"kind": "fact"}},
        ]
        assert _extract_active_session(entries) is None

    def test_skips_non_dict_entries(self):
        entries = ["string", None, 42]
        assert _extract_active_session(entries) is None


class TestExtractKnowledgeModel:
    def test_parses_knowledge_model_entries(self):
        entries = [
            {"memory": '{"concept": "Bayes", "status": "rusty", "evidence": "said so", "turn": 1}',
             "metadata": {"kind": "knowledge_model"}},
        ]
        result = _extract_knowledge_model(entries)
        assert len(result) == 1
        assert result[0].concept == "Bayes"
        assert result[0].status == "rusty"

    def test_skips_malformed_memory_field(self):
        entries = [
            {"memory": "not valid json", "metadata": {"kind": "knowledge_model"}},
            {"memory": '{"concept": "Bayes", "status": "rusty", "evidence": "said so", "turn": 1}',
             "metadata": {"kind": "knowledge_model"}},
        ]
        result = _extract_knowledge_model(entries)
        assert len(result) == 1

    def test_returns_empty_list_when_none_found(self):
        entries = [
            {"memory": "some memory", "metadata": {"kind": "fact"}},
        ]
        assert _extract_knowledge_model(entries) == []

    def test_skips_non_dict_entries(self):
        entries = ["string", None, 42]
        assert _extract_knowledge_model(entries) == []


class TestRetrieveMemoryNode:
    PATCH_PATH = "graphs.learning_graph.nodes.retrieve_memory.memory"

    @pytest.fixture
    def mock_memory(self):
        with patch(self.PATCH_PATH) as mock:
            mock.search.return_value = {"results": []}
            yield mock

    def test_mem0_search_called_with_correct_args(self, mock_memory, state_with_message, runnable_config):
        retrieve_memory(state_with_message, runnable_config)
        mock_memory.search.assert_called_once_with(
            "hello",
            limit=5,
            filters={"user_id": "test_thread"},
        )

    def test_thread_id_from_run_config(self, mock_memory, state_with_message):
        config = RunnableConfig(configurable={"thread_id": "custom_thread"})
        retrieve_memory(state_with_message, config)
        mock_memory.search.assert_called_once_with(
            "hello",
            limit=5,
            filters={"user_id": "custom_thread"},
        )

    def test_thread_id_falls_back_to_default(self, mock_memory, state_with_message):
        config = RunnableConfig(configurable={})
        retrieve_memory(state_with_message, config)
        mock_memory.search.assert_called_once_with(
            "hello",
            limit=5,
            filters={"user_id": "default_thread"},
        )

    def test_get_all_graceful_failure(self, mock_memory, state_with_message, runnable_config, caplog):
        mock_memory.get_all.side_effect = Exception("connection error")
        result = retrieve_memory(state_with_message, runnable_config)
        assert result["active_session"] is None
        assert result["knowledge_model"] == []
        assert "connection error" in caplog.text

    def test_returns_expected_keys(self, mock_memory, state_with_message, runnable_config):
        result = retrieve_memory(state_with_message, runnable_config)
        assert "memory_results" in result
        assert "active_session" in result
        assert "knowledge_model" in result

    def test_returns_compacted_results(self, mock_memory, state_with_message, runnable_config):
        mock_memory.search.return_value = {
            "results": [
                {"id": "1", "memory": "relevant memory", "metadata": {"kind": "fact"}},
                {"id": "2", "memory": "session record", "metadata": {"kind": "session"}},
            ]
        }
        result = retrieve_memory(state_with_message, runnable_config)
        assert len(result["memory_results"]) == 1
        assert result["memory_results"][0]["id"] == "1"