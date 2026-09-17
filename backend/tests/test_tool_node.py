"""Unit tests for ``ToolCallNode`` — fake whitelisted tools only.

``ToolCallNode`` never touches an LLM and never runs a real tool here: every
test injects a tiny ``ToolRegistry`` with a fake ``dict -> str`` tool so no
network / mem0 / ddgs code can run. Serialization round-trips rebuild
through the default registry, which only holds plain functions.
"""

import pytest
from langgraph.graph import START, StateGraph
from pydantic import BaseModel, Field

from graphs.api.collection import NodeCollection
from graphs.graph import END as G_END
from graphs.graph import START as G_START
from graphs.graph import Graph
from graphs.nodes.text_node import TextNode
from graphs.nodes.tool_node import ToolCallNode
from graphs.serialization import node_from_dict
from graphs.storage import dump_collection, load_collection
from graphs.tools import ToolRegistry, require_arg


class DummyEmbedder:
    """Fixed-vector embedder so collection adds never touch fastembed."""

    def __call__(self, text):
        return [1.0, 0.0, 0.0, 0.0]


class FakeLLM:
    """Std-in for the protocol-only ``llm`` argument of ``get_node``."""

    def invoke(self, prompt):
        raise AssertionError("ToolCallNode must never invoke an LLM")

    async def ainvoke(self, prompt):
        raise AssertionError("ToolCallNode must never invoke an LLM")


def _registry():
    registry = ToolRegistry()

    registry.register("echo", lambda args: f"echo:{args.get('text')}")
    registry.register("higher", lambda args: args["value"] + 1)
    registry.register("strict", lambda args: f"strict:{require_arg(args, 'text')}")
    return registry


class ToolState(BaseModel):
    """State model carrying the fields a ``ToolCallNode`` reads/writes."""

    user_message: str | None = None
    response: str | None = None
    tool: str | None = None
    args: dict | None = None


class ListResponseState(BaseModel):
    """A state model whose ``response`` field is declared list-typed."""

    response: list[str] = Field(default_factory=list)
    args: dict | None = None


# ------------------------------------------------------------- construction


def test_construction_requires_non_empty_tool():
    for bad in ("", None, 0):
        with pytest.raises(ValueError, match="non-empty string"):
            ToolCallNode("n", "d", bad)


def test_construction_unknown_tool_rejected_from_default_registry():
    with pytest.raises(ValueError, match="unknown tool") as exc_info:
        ToolCallNode("n", "d", "not_a_real_tool")
    assert "not_a_real_tool" in str(exc_info.value)


def test_construction_unknown_tool_rejected_from_custom_registry():
    with pytest.raises(ValueError, match="unknown tool 'nope'"):
        ToolCallNode("n", "d", "nope", _registry())


def test_construction_validates_metadata_like_base():
    with pytest.raises(ValueError):
        ToolCallNode("", "d", "echo", _registry())


def test_fixed_params_writes_and_prompt():
    node = ToolCallNode("search", "web search", "echo", _registry())
    assert node.params == {"tool": str, "args": dict}
    assert node.writes == {"result": "response"}
    assert node.tool == "echo"
    assert "echo" in node.prompt


# ---------------------------------------------------------- fake invocation


def test_invoke_reads_args_from_state_and_writes_result():
    node = ToolCallNode("search", "web search", "echo", _registry())
    fn = node.get_node(FakeLLM())
    assert fn.invoke({"args": {"text": "cats"}}) == {"response": "echo:cats"}


def test_invoke_falls_back_to_configured_tool_without_state_tool():
    node = ToolCallNode("search", "web search", "echo", _registry())
    fn = node.get_node(FakeLLM())
    assert fn.invoke({"args": {"text": "x"}}) == {"response": "echo:x"}


def test_invoke_state_tool_overrides_configured_tool():
    node = ToolCallNode("search", "web search", "echo", _registry())
    fn = node.get_node(FakeLLM())
    assert fn.invoke({"tool": "higher", "args": {"value": 41}}) == {
        "response": "42"
    }


def test_invoke_missing_args_defaults_to_empty_dict():
    node = ToolCallNode("search", "web search", "echo", _registry())
    fn = node.get_node(FakeLLM())
    assert fn.invoke({}) == {"response": "echo:None"}


def test_invoke_accepts_json_string_args():
    node = ToolCallNode("search", "web search", "echo", _registry())
    fn = node.get_node(FakeLLM())
    assert fn.invoke({"args": '{"text": "cats"}'}) == {"response": "echo:cats"}


def test_invoke_invalid_json_args_raises():
    node = ToolCallNode("search", "web search", "echo", _registry())
    fn = node.get_node(FakeLLM())
    with pytest.raises(ValueError, match="args JSON is invalid"):
        fn.invoke({"args": "{not json"})


def test_invoke_json_args_encoding_non_object_raises():
    node = ToolCallNode("search", "web search", "echo", _registry())
    fn = node.get_node(FakeLLM())
    with pytest.raises(TypeError, match="must encode an object"):
        fn.invoke({"args": '["cats"]'})


def test_invoke_non_dict_args_raises():
    node = ToolCallNode("search", "web search", "echo", _registry())
    fn = node.get_node(FakeLLM())
    with pytest.raises(ValueError, match="args must be a dict"):
        fn.invoke({"args": [1, 2, 3]})


def test_invoke_unknown_runtime_tool_raises_key_error():
    node = ToolCallNode("search", "web search", "echo", _registry())
    fn = node.get_node(FakeLLM())
    with pytest.raises(KeyError, match="unknown tool: 'bogus'"):
        fn.invoke({"tool": "bogus", "args": {"text": "x"}})


def test_tool_missing_required_arg_error_propagates():
    node = ToolCallNode("search", "web search", "strict", _registry())
    fn = node.get_node(FakeLLM())
    with pytest.raises(ValueError, match="missing required tool argument"):
        fn.invoke({"args": {}})


def test_multi_key_writes_spread_result():
    node = ToolCallNode(
        "twin", "d", "echo", _registry(), writes={"first": "a_out", "second": "b_out"}
    )
    fn = node.get_node(FakeLLM())
    assert fn.invoke({"args": {"text": "x"}}) == {
        "a_out": "echo:x",
        "b_out": "echo:x",
    }


async def test_async_ainvoke():
    node = ToolCallNode("search", "web search", "echo", _registry())
    fn = node.get_node(FakeLLM())
    assert await fn.ainvoke({"args": {"text": "cats"}}) == {"response": "echo:cats"}


def test_node_name_attribute():
    node = ToolCallNode("my_tool_node", "d", "echo", _registry())
    assert node.get_node(FakeLLM()).__name__ == "my_tool_node"


def test_pydantic_model_state_reads_and_writes():
    node = ToolCallNode("search", "web search", "echo", _registry())
    fn = node.get_node(FakeLLM())
    assert fn.invoke(ToolState(args={"text": "cats"})) == {"response": "echo:cats"}


# ---------------------------------------------------------------- write shape


def test_list_typed_field_gets_wrapped_value():
    node = ToolCallNode("search", "web search", "echo", _registry())
    fn = node.get_node(FakeLLM())
    assert fn.invoke(ListResponseState(args={"text": "x"})) == {
        "response": ["echo:x"]
    }


def test_plain_dict_state_never_wraps():
    node = ToolCallNode("search", "web search", "echo", _registry())
    fn = node.get_node(FakeLLM())
    assert fn.invoke({"args": {"text": "x"}, "response": "seed"}) == {
        "response": "echo:x"
    }


def test_default_writes_target_response():
    assert ToolCallNode("n", "d", "echo", _registry()).writes == {
        "result": "response"
    }


# --------------------------------------------------------- updated() mutation


def test_updated_allows_tool_swap_with_revalidation():
    node = ToolCallNode("n", "d", "echo", _registry())
    swapped = node.updated(tool="higher")
    assert swapped.tool == "higher"
    assert swapped.name == node.name
    assert swapped.get_node(FakeLLM()).invoke({"args": {"value": 1}}) == {
        "response": "2"
    }
    with pytest.raises(ValueError, match="unknown tool"):
        node.updated(tool="nope")


def test_updated_rejects_fixed_and_unknown_fields():
    node = ToolCallNode("n", "d", "echo", _registry())
    with pytest.raises(ValueError, match="fixed"):
        node.updated(prompt="customize me")
    with pytest.raises(TypeError, match="unsupported fields"):
        node.updated(bogus=1)


# ------------------------------------------------------------ langgraph graphs


def test_langgraph_integration_direct_state_graph():
    builder = StateGraph(ToolState)
    node = ToolCallNode("search", "web search", "echo", _registry())
    builder.add_node(node.get_node(FakeLLM()))
    builder.add_edge(START, "search")
    graph = builder.compile()
    result = graph.invoke({"tool": "higher", "args": {"value": 1}})
    assert result["response"] == "2"


def test_graph_validate_compile_and_invoke():
    graph = Graph(state_model=ToolState)
    node = ToolCallNode("search", "web search", "echo", _registry())
    graph.add_node("t", node)
    graph.add_edge("e1", G_START, "t")
    graph.add_edge("e2", "t", G_END)
    graph.validate()
    compiled = graph.compile(FakeLLM())
    result = compiled.invoke({"args": {"text": "cats"}})
    assert result["response"] == "echo:cats"


# ------------------------------------------------------------ round trips


def test_round_trip_preserves_tool_and_writes():
    node = ToolCallNode(
        "search",
        "web search",
        "ddgs",
        writes={"result": "analysis"},
    )
    data = node.to_dict()
    assert data["type"] == "tool"
    assert data["name"] == "search"
    assert data["tool"] == "ddgs"
    assert data["writes"] == {"result": "analysis"}

    restored = ToolCallNode.from_dict(data)
    assert restored.name == node.name
    assert restored.description == node.description
    assert restored.tool == node.tool
    assert restored.writes == node.writes
    assert restored.prompt == node.prompt
    assert restored.to_dict() == data


def test_round_trip_via_type_tag_dispatch():
    node = ToolCallNode("search", "web search", "ddgs")
    restored = node_from_dict(node.to_dict())
    assert isinstance(restored, ToolCallNode)
    assert restored.tool == "ddgs"


def test_round_trip_with_unknown_tool_revalidates():
    data = {
        "type": "tool",
        "name": "n",
        "description": "d",
        "tool": "nope",
        "writes": {"result": "response"},
    }
    with pytest.raises(ValueError, match="unknown tool 'nope'"):
        ToolCallNode.from_dict(data)


def test_collection_dump_load_round_trip(tmp_path):
    path = tmp_path / "collection.json"
    sub = TextNode("plain", "d", "p")
    tool = ToolCallNode("search", "web search", "ddgs")
    coll = NodeCollection(embedder=DummyEmbedder())
    coll.add(sub)
    coll.add(tool)

    dump_collection(coll, path)
    restored = load_collection(path, embedder=DummyEmbedder())

    assert len(restored) == 2
    rebuilt = restored.snapshot()[1]
    assert isinstance(rebuilt, ToolCallNode)
    assert rebuilt.tool == "ddgs"
    assert rebuilt.writes == tool.writes