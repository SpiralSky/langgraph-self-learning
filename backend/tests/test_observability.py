import logging
import time
from unittest.mock import patch

import pytest

from graphs.learning_graph.observability import timed


class TestTimedDecorator:
    def test_successful_call_logs_info(self, caplog):
        caplog.set_level(logging.INFO)
        @timed("my_node")
        def my_func():
            return 42

        result = my_func()

        assert result == 42
        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.levelname == "INFO"
        assert "node=my_node" in record.getMessage()
        assert "elapsed_ms=" in record.getMessage()

    def test_exception_re_raises_and_logs_warning(self, caplog):
        @timed("failing_node")
        def failing():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            failing()

        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.levelname == "WARNING"
        msg = record.getMessage()
        assert "node=failing_node" in msg
        assert "error=boom" in msg
        assert "elapsed_ms=" in msg

    def test_timing_measurement(self):
        with patch.object(time, "perf_counter", side_effect=[10.0, 15.0]):
            @timed("timed_node")
            def my_func():
                return "done"

            result = my_func()

        assert result == "done"

    def test_wrapt_preservation(self):
        @timed("preserved")
        def my_func():
            """My docstring."""
            ...

        assert my_func.__name__ == "my_func"
        assert my_func.__doc__ == "My docstring."

    def test_multiple_nodes_get_unique_loggers(self, caplog):
        caplog.set_level(logging.INFO)
        @timed("node_a")
        def fn_a():
            return "a"

        @timed("node_b")
        def fn_b():
            return "b"

        fn_a()
        fn_b()

        assert len(caplog.records) == 2
        messages = [r.getMessage() for r in caplog.records]
        assert any("node=node_a" in m for m in messages)
        assert any("node=node_b" in m for m in messages)