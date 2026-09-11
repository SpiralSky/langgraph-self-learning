import pytest
from langchain_core.messages import HumanMessage

from graphs.learning_graph.text_utils import message_to_text


class TestMessageToText:
    def test_string_content(self):
        msg = HumanMessage(content="plain text")
        assert message_to_text(msg) == "plain text"

    def test_single_text_block(self):
        msg = HumanMessage(content=[{"type": "text", "text": "Hello"}])
        assert message_to_text(msg) == "Hello"

    def test_multiple_text_blocks_joined(self):
        msg = HumanMessage(content=[{"type": "text", "text": "A"}, {"type": "text", "text": "B"}])
        assert message_to_text(msg) == "A B"

    def test_text_block_with_attr(self):
        msg = HumanMessage(content="")
        msg.content = [type("Block", (), {"text": "from_attr"})()]
        assert message_to_text(msg) == "from_attr"

    def test_mixed_blocks(self):
        msg = HumanMessage(content="")
        msg.content = [
            {"type": "text", "text": "Hello"},
            type("Block", (), {"text": "attr_text"})(),
        ]
        assert message_to_text(msg) == "Hello attr_text"

    def test_non_text_blocks_skipped(self):
        msg = HumanMessage(content=[{"type": "image", "image_url": "x.png"}])
        assert message_to_text(msg) == ""

    def test_fallback_numeric(self):
        msg = HumanMessage(content="")
        msg.content = 42
        assert message_to_text(msg) == "42"

    def test_empty_list(self):
        msg = HumanMessage(content=[])
        assert message_to_text(msg) == ""

    def test_empty_string(self):
        msg = HumanMessage(content="")
        assert message_to_text(msg) == ""
