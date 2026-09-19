"""Smoke tests for the messages-looping chat entrypoint (``graphs.chat``).

Like ``test_entrypoint``, the ``agent`` graph is imported at module load, so
the network seams are stubbed *before* the import; ``agent_graph`` is then
patched with a fake so no LLM is ever invoked while testing the chat loop's
mapping (last human message -> user_message -> appended assistant reply).
"""

import pytest

from graphs.chat import ChatState, _last_human_message


@pytest.fixture
def stub_agent_graph(monkeypatch):
    """Import ``graphs.chat`` with network seams + a fake nested agent."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "graphs.api.collection.get_default_embedder",
        lambda: lambda s: [0.0] * 384,
    )

    class FakeAgentGraph:
        def invoke(self, state):
            return {"user_message": state["user_message"], "response": f"reply to {state['user_message']}"}

    monkeypatch.setattr("graphs.chat.agent_graph", FakeAgentGraph())

    import graphs.chat as chat

    return chat


def test_chat_exports_compiled_messages_graph(stub_agent_graph):
    chat = stub_agent_graph
    assert type(chat.graph).__name__ == "CompiledStateGraph"
    assert callable(chat.graph.invoke) and callable(chat.graph.ainvoke)


def test_chat_state_schema_is_chat_state(stub_agent_graph):
    chat = stub_agent_graph
    assert chat.graph.builder.state_schema is ChatState
    assert "messages" in ChatState.__annotations__


def test_last_human_message_returns_most_recent_content():
    from langchain_core.messages import AIMessage, HumanMessage

    messages = [HumanMessage(content="first"), AIMessage(content="reply")]
    assert _last_human_message(messages) == "first"
    assert _last_human_message([AIMessage(content="reply")]) == ""


def test_run_turn_appends_assistant_reply(stub_agent_graph):
    chat = stub_agent_graph
    result = chat.graph.invoke({"messages": [{"role": "user", "content": "hello"}]})
    messages = result["messages"]
    assert len(messages) == 2
    assert messages[0].content == "hello"
    assert messages[1].content == "reply to hello"
    assert messages[1].type == "ai"


def test_run_turn_forwards_nested_agent_ui_channel(stub_agent_graph):
    """The values-state ``ui`` channel passes through the wrapper (generative
    UI from the nested agent reaches the frontend via ``state.values.ui``)."""
    chat = stub_agent_graph

    class AgentWithUI:
        def invoke(self, state):
            return {
                "user_message": state["user_message"],
                "response": "reply",
                "ui": [{"type": "ui", "id": "u1", "name": "text", "props": {"title": "Note"}}],
            }

    chat.agent_graph = AgentWithUI()
    result = chat.graph.invoke({"messages": [{"role": "user", "content": "hello"}]})
    assert result["ui"] == [
        {"type": "ui", "id": "u1", "name": "text", "props": {"title": "Note"}}
    ]