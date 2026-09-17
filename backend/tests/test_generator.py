"""Unit tests for ``GeneratorNode`` — runtime graph generation + nested run.

Pure-unit: the generator LLM is scripted with canned ``tool_calls`` (or JSON
content) and inner text nodes echo their prompt through the same fake; every
tool runs against a tiny injected whitelist registry. No network, no real
tools, no default-registry client construction.
"""

import pytest
import yaml
from langchain_core.messages import AIMessage

from graphs.behaviors import BehaviorGroup, BehaviorPoint
from graphs.generator import (
    BUILDER_TOOL_DEFS,
    GeneratorNode,
    build_state_model,
    decode_builder_calls,
)
from graphs.graph import Graph
from graphs.nodes.text_node import TextNode
from graphs.nodes.tool_node import ToolCallNode
from graphs.tools import ToolRegistry


def tool_call(call_id, name, args):
    return {"name": name, "args": args, "id": call_id}


def calls_message(*specs):
    return AIMessage(content="", tool_calls=list(specs))


def valid_text_calls():
    """A valid single-text-node chain writing the final ``response``."""
    return [
        tool_call(
            "c1",
            "add_node",
            {
                "id": "a",
                "type": "text",
                "name": "answer",
                "description": "answers",
                "prompt": "Answer {user_message}",
                "params": {"user_message": "str"},
                "writes": {"result": "response"},
            },
        ),
        tool_call("c2", "add_edge", {"source": "START", "target": "a"}),
        tool_call("c3", "add_edge", {"source": "a", "target": "END"}),
    ]


def _registry():
    registry = ToolRegistry()
    registry.register("echo", lambda args: f"echo:{args.get('text')}")
    return registry


class GeneratorLLM:
    """Scripted LLM: generator prompts pop canned responses; others echo.

    ``script`` holds one entry per generator invocation (an ``AIMessage``).
    Inner text nodes never see a generator prompt, so their invocations return
    a deterministic ``inner:<prompt>`` echo instead of consuming the script.
    """

    def __init__(self, script):
        self.script = list(script)
        self.prompts = []

    @property
    def gen_prompts(self):
        return [p for p in self.prompts if self._is_generator(p)]

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    @staticmethod
    def _is_generator(prompt):
        return prompt.startswith("Build a single-pass graph")

    def _pop(self):
        if not self.script:
            raise AssertionError("generator consumed more prompts than scripted")
        return self.script.pop(0)

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if self._is_generator(prompt):
            return self._pop()
        return AIMessage(content=f"inner:{prompt}")

    async def ainvoke(self, prompt):
        self.prompts.append(prompt)
        if self._is_generator(prompt):
            return self._pop()
        return AIMessage(content=f"inner:{prompt}")


# ------------------------------------------------------------------ decoding


def test_decode_tool_calls_preferred():
    msg = calls_message(tool_call("c1", "add_node", {"id": "a"}))
    assert decode_builder_calls(msg) == [{"name": "add_node", "args": {"id": "a"}}]


def test_decode_json_fallback():
    msg = AIMessage(
        content='{"calls": [{"name": "add_edge", "args": {"source": "x", "target": "y"}}]}'
    )
    assert decode_builder_calls(msg) == [
        {"name": "add_edge", "args": {"source": "x", "target": "y"}}
    ]


def test_decode_without_any_calls_raises():
    with pytest.raises(ValueError, match="no builder tool calls"):
        decode_builder_calls(AIMessage(content=""))


def test_decode_malformed_json_raises():
    with pytest.raises(ValueError, match="not valid JSON"):
        decode_builder_calls(AIMessage(content="{not json"))


def test_decode_json_without_calls_key_raises():
    with pytest.raises(TypeError, match='"calls"'):
        decode_builder_calls(AIMessage(content='{"foo": []}'))


# --------------------------------------------------------------- state model


def test_build_state_model_unions_params_writes_and_defaults():
    nodes = {
        "a": TextNode(
            "a",
            "d",
            "P {user_message}",
            params={"user_message": str},
            writes={"result": "notes"},
        ),
        "b": ToolCallNode("b", "d", "echo", _registry()),
    }
    model = build_state_model(nodes)
    fields = model.model_fields
    assert set(fields) == {"user_message", "notes", "response", "tool", "args"}
    assert fields["user_message"].annotation is str
    assert fields["notes"].annotation is str
    assert fields["tool"].annotation is str
    assert fields["args"].annotation is dict


def test_bind_builder_tools_attached():
    llm = GeneratorLLM([calls_message(*valid_text_calls())])
    fn = GeneratorNode("gen", "d", behaviors=[], registry=_registry()).get_node(llm)
    fn.invoke({"user_message": "hi"})
    assert llm.bound_tools == BUILDER_TOOL_DEFS


# ------------------------------------------------------------ happy path run


def test_single_pass_spec_builds_and_runs_nested():
    llm = GeneratorLLM([calls_message(*valid_text_calls())])
    fn = GeneratorNode("gen", "d", behaviors=[], registry=_registry()).get_node(llm)
    result = fn.invoke({"user_message": "hi"})

    assert result == {"response": "inner:Answer hi"}
    assert isinstance(fn.last_graph, Graph)
    fields = fn.last_graph.state_model.model_fields
    assert set(fields) == {"user_message", "response"}
    assert fn.reuse_ids == []


async def test_async_ainvoke_runs_nested():
    llm = GeneratorLLM([calls_message(*valid_text_calls())])
    fn = GeneratorNode("gen", "d", behaviors=[], registry=_registry()).get_node(llm)
    result = await fn.ainvoke({"user_message": "hi"})
    assert result == {"response": "inner:Answer hi"}


# ------------------------------------------------------------- retry behavior


def test_invalid_spec_retries_with_feedback_then_succeeds():
    bad = calls_message(
        tool_call(
            "c1",
            "add_node",
            {"id": "x", "type": "text", "name": "n", "description": "d"},
        ),
        tool_call("c2", "add_edge", {"source": "x", "target": "ghost"}),
    )
    llm = GeneratorLLM([bad, calls_message(*valid_text_calls())])
    fn = GeneratorNode("gen", "d", behaviors=[], registry=_registry()).get_node(llm)

    assert fn.invoke({"user_message": "hi"}) == {"response": "inner:Answer hi"}
    assert len(llm.gen_prompts) == 2
    second = llm.gen_prompts[1]
    assert "rejected" in second
    assert "non-empty 'prompt'" in second
    assert "endpoint node 'x' does not exist" in second


def test_persistent_invalid_raises_with_aggregate_errors():
    bad = calls_message(
        tool_call("c1", "add_node", {"id": "x", "type": "bogus", "name": "n", "description": "d"})
    )
    llm = GeneratorLLM([bad] * 4)
    fn = GeneratorNode("gen", "d", behaviors=[], retries=3, registry=_registry()).get_node(llm)

    with pytest.raises(RuntimeError, match="after 4 attempts") as exc_info:
        fn.invoke({"user_message": "hi"})
    assert "unknown node type 'bogus'" in str(exc_info.value)
    assert len(llm.gen_prompts) == 4
    assert "unknown node type 'bogus'" in llm.gen_prompts[3]


def test_malformed_response_retries_then_succeeds():
    llm = GeneratorLLM([AIMessage(content="not json"), calls_message(*valid_text_calls())])
    fn = GeneratorNode("gen", "d", behaviors=[], registry=_registry()).get_node(llm)

    assert fn.invoke({"user_message": "hi"}) == {"response": "inner:Answer hi"}
    assert "not valid JSON" in llm.gen_prompts[1]


def test_unknown_builder_call_is_retry_feedback():
    bad = AIMessage(
        content='{"calls": [{"name": "explode", "args": {}}]}'
    )
    llm = GeneratorLLM([bad, calls_message(*valid_text_calls())])
    fn = GeneratorNode("gen", "d", behaviors=[], registry=_registry()).get_node(llm)

    fn.invoke({"user_message": "hi"})
    assert "unknown builder call 'explode'" in llm.gen_prompts[1]


# ------------------------------------------------------------------ tool path


def test_tool_node_path_runs_through_fake_registry():
    spec = [
        tool_call("c1", "add_node", {"id": "t", "type": "tool", "name": "search", "description": "searches", "tool": "echo"}),
        tool_call("c2", "add_edge", {"source": "START", "target": "t"}),
        tool_call("c3", "add_edge", {"source": "t", "target": "END"}),
    ]
    llm = GeneratorLLM([calls_message(*spec)])
    fn = GeneratorNode("gen", "d", behaviors=[], registry=_registry()).get_node(llm)

    assert fn.invoke({"user_message": "hi"}) == {"response": "echo:None"}


def test_unknown_tool_name_is_retry_feedback():
    bad = calls_message(
        tool_call("c1", "add_node", {"id": "t", "type": "tool", "name": "n", "description": "d", "tool": "nope"})
    )
    llm = GeneratorLLM([bad, calls_message(*valid_text_calls())])
    fn = GeneratorNode("gen", "d", behaviors=[], registry=_registry()).get_node(llm)

    fn.invoke({"user_message": "hi"})
    assert "unknown tool 'nope'" in llm.gen_prompts[1]


def test_fixed_tool_args_rejected_with_feedback():
    bad = calls_message(
        tool_call(
            "c1",
            "add_node",
            {"id": "t", "type": "tool", "name": "n", "description": "d", "tool": "echo", "tool_args": {"text": "hi"}},
        )
    )
    llm = GeneratorLLM([bad, calls_message(*valid_text_calls())])
    fn = GeneratorNode("gen", "d", behaviors=[], registry=_registry()).get_node(llm)

    fn.invoke({"user_message": "hi"})
    assert "fixed 'tool_args' are not supported" in llm.gen_prompts[1]


# ------------------------------------------------------------- inner state typing


def test_inner_state_model_uses_declared_list_type():
    spec = [
        tool_call("c1", "add_node", {"id": "a", "type": "text", "name": "produce", "description": "d", "prompt": "Write notes", "writes": {"result": "notes"}}),
        tool_call("c2", "add_node", {"id": "b", "type": "text", "name": "count", "description": "d", "prompt": "Count {notes}", "params": {"notes": "list"}}),
        tool_call("c3", "add_edge", {"source": "START", "target": "a"}),
        tool_call("c4", "add_edge", {"source": "a", "target": "b"}),
        tool_call("c5", "add_edge", {"source": "b", "target": "END"}),
    ]
    llm = GeneratorLLM([calls_message(*spec)])
    fn = GeneratorNode("gen", "d", behaviors=[], registry=_registry()).get_node(llm)

    fn.invoke({"user_message": "hi"})
    fields = fn.last_graph.state_model.model_fields
    assert fields["notes"].annotation is list
    assert fields["user_message"].annotation is str
    assert fields["response"].annotation is str


# ----------------------------------------------------------------- behaviors


def test_behaviors_injected_as_objects_render_into_prompt():
    groups = [
        BehaviorGroup(title="Explain concepts", points=[BehaviorPoint(id="p1", text="be direct")])
    ]
    llm = GeneratorLLM([calls_message(*valid_text_calls())])
    fn = GeneratorNode("gen", "d", behaviors=groups, registry=_registry()).get_node(llm)

    fn.invoke({"user_message": "hi"})
    first = llm.gen_prompts[0]
    assert "## Explain concepts" in first
    assert "- be direct" in first


def test_behaviors_loaded_from_path(tmp_path):
    path = tmp_path / "behaviors.yaml"
    path.write_text(
        yaml.safe_dump(
            [{"title": "Answer directly", "points": ["no chatter"]}],
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    node = GeneratorNode("gen", "d", behaviors=str(path), registry=_registry())
    assert "## Answer directly" in node.behaviors_text
    assert "- no chatter" in node.behaviors_text


def test_behaviors_default_path_loads_committed_file():
    node = GeneratorNode("gen", "d")
    assert "Explain concepts" in node.behaviors_text


def test_behaviors_rejects_non_group_items():
    with pytest.raises(TypeError, match="BehaviorGroup"):
        GeneratorNode("gen", "d", behaviors=[{"title": "T", "points": []}])


# ---------------------------------------------------------------- reuse flag


def test_reuse_ids_collected_from_spec():
    spec = [
        tool_call("c1", "add_node", {"id": "a", "type": "text", "name": "n", "description": "d", "prompt": "Answer {user_message}", "params": {"user_message": "str"}, "reuse": True}),
        tool_call("c2", "add_node", {"id": "b", "type": "text", "name": "m", "description": "d", "prompt": "Beep"}),
        tool_call("c3", "add_edge", {"source": "START", "target": "a"}),
        tool_call("c4", "add_edge", {"source": "a", "target": "b"}),
        tool_call("c5", "add_edge", {"source": "b", "target": "END"}),
    ]
    llm = GeneratorLLM([calls_message(*spec)])
    fn = GeneratorNode("gen", "d", behaviors=[], registry=_registry()).get_node(llm)

    fn.invoke({"user_message": "hi"})
    assert fn.reuse_ids == ["a"]


# ------------------------------------------------------------ construction


def test_ctor_rejects_bad_retries():
    for bad in (-1, "3", True):
        with pytest.raises(ValueError, match="retries"):
            GeneratorNode("gen", "d", retries=bad)


def test_ctor_rejects_bad_behaviors_type():
    with pytest.raises(TypeError, match="behaviors must be"):
        GeneratorNode("gen", "d", behaviors=42)


def test_updated_only_allows_metadata():
    node = GeneratorNode("gen", "d", behaviors=[], registry=_registry())
    renamed = node.updated(name="new")
    assert renamed.name == "new"
    assert renamed.description == node.description
    with pytest.raises(ValueError, match="structural"):
        node.updated(prompt="custom")
    with pytest.raises(TypeError, match="unsupported fields"):
        node.updated(bogus=1)


def test_metadata_and_template_fixed():
    node = GeneratorNode("gen", "explains", behaviors=[], registry=_registry())
    assert node.params == {"user_message": str}
    assert node.writes == {"result": "response"}
    assert node.prompt.startswith("Build a single-pass graph")
    assert node.get_node(GeneratorLLM([])).__name__ == "gen"