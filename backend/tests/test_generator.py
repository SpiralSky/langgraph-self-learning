"""Unit tests for ``GeneratorNode`` — runtime graph generation + nested run.

Pure-unit: the generator LLM is scripted with canned ``tool_calls`` (or JSON
content) and inner text nodes echo their prompt through the same fake; every
tool runs against a tiny injected whitelist registry. No network, no real
tools, no default-registry client construction.
"""

from dataclasses import FrozenInstanceError

import pytest
import yaml
from langchain_core.messages import AIMessage

from graphs.api.collection import NodeCollection
from graphs.learning.behaviors import BehaviorGroup, BehaviorPoint
from graphs.nodes.generator import (
    BUILDER_TOOL_DEFS,
    GeneratorNode,
    Rejection,
    _GENERATOR_TEMPLATE,
    build_state_model,
    classify_rejection,
    decode_builder_calls,
    validate_produced_inputs,
)
from graphs.structure.graph import Graph
from graphs.nodes.text_node import TextNode
from graphs.nodes.tool_node import ToolCallNode
from graphs.tools import ToolRegistry


def _embed(text: str) -> list[float]:
    return [0.1, 0.2, 0.3]


def _empty_collection() -> NodeCollection:
    return NodeCollection(embedder=_embed)


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
                "prompt": "Answer {input}",
                "params": {"input": "str"},
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
            "P {input}",
            params={"input": str},
            writes={"result": "notes"},
        ),
        "b": ToolCallNode("b", "d", "echo", _registry()),
    }
    model = build_state_model(nodes)
    fields = model.model_fields
    assert set(fields) == {"input", "notes", "response", "tool", "args"}
    assert fields["input"].annotation is str
    assert fields["notes"].annotation is str
    assert fields["tool"].annotation is str
    assert fields["args"].annotation is dict


def test_bind_builder_tools_attached():
    llm = GeneratorLLM([calls_message(*valid_text_calls())])
    fn = GeneratorNode("gen", "d", behaviors=[], registry=_registry(), collection=_empty_collection()).get_node(llm)
    fn.invoke({"user_message": "hi"})
    assert llm.bound_tools == BUILDER_TOOL_DEFS


# ------------------------------------------------------------ happy path run


def test_single_pass_spec_builds_and_runs_nested():
    llm = GeneratorLLM([calls_message(*valid_text_calls())])
    fn = GeneratorNode("gen", "d", behaviors=[], registry=_registry(), collection=_empty_collection()).get_node(llm)
    result = fn.invoke({"user_message": "hi"})

    assert result == {"response": "inner:Answer hi"}
    assert isinstance(fn.last_graph, Graph)
    fields = fn.last_graph.state_model.model_fields
    assert set(fields) == {"input", "response"}
    assert fn.reuse_ids == []


async def test_async_ainvoke_runs_nested():
    llm = GeneratorLLM([calls_message(*valid_text_calls())])
    fn = GeneratorNode("gen", "d", behaviors=[], registry=_registry(), collection=_empty_collection()).get_node(llm)
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
    fn = GeneratorNode("gen", "d", behaviors=[], registry=_registry(), collection=_empty_collection()).get_node(llm)

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
    fn = GeneratorNode("gen", "d", behaviors=[], retries=3, registry=_registry(), collection=_empty_collection()).get_node(llm)

    with pytest.raises(RuntimeError, match="after 4 attempts") as exc_info:
        fn.invoke({"user_message": "hi"})
    assert "unknown node type 'bogus'" in str(exc_info.value)
    assert len(llm.gen_prompts) == 4
    assert "unknown node type 'bogus'" in llm.gen_prompts[3]


def test_malformed_response_retries_then_succeeds():
    llm = GeneratorLLM([AIMessage(content="not json"), calls_message(*valid_text_calls())])
    fn = GeneratorNode("gen", "d", behaviors=[], registry=_registry(), collection=_empty_collection()).get_node(llm)

    assert fn.invoke({"user_message": "hi"}) == {"response": "inner:Answer hi"}
    assert "not valid JSON" in llm.gen_prompts[1]


def test_unknown_builder_call_is_retry_feedback():
    bad = AIMessage(
        content='{"calls": [{"name": "explode", "args": {}}]}'
    )
    llm = GeneratorLLM([bad, calls_message(*valid_text_calls())])
    fn = GeneratorNode("gen", "d", behaviors=[], registry=_registry(), collection=_empty_collection()).get_node(llm)

    fn.invoke({"user_message": "hi"})
    assert "unknown builder call 'explode'" in llm.gen_prompts[1]


# -------------------------------------- rejection framework / reserved node ids


def _reserved_id_bad_call():
    return calls_message(
        tool_call(
            "c0",
            "add_node",
            {
                "id": "START",
                "type": "text",
                "name": "first",
                "description": "starts",
                "prompt": "Intro {input}",
                "params": {"input": "str"},
                "writes": {"result": "response"},
            },
        )
    )


def _gen(registry=None) -> GeneratorNode:
    return GeneratorNode(
        "gen",
        "d",
        behaviors=[],
        retries=3,
        registry=registry or _registry(),
        collection=_empty_collection(),
    )


def test_reserved_node_id_feedback_recovers():
    llm = GeneratorLLM([_reserved_id_bad_call(), calls_message(*valid_text_calls())])
    fn = _gen().get_node(llm)

    assert fn.invoke({"user_message": "hi"}) == {"response": "inner:Answer hi"}
    assert len(llm.gen_prompts) == 2
    second = llm.gen_prompts[1]
    assert "node id is reserved: 'START'" in second
    assert "framework ids" in second
    assert "Fix:" in second


def test_persistent_reserved_id_failure_reproduces_prod_error():
    llm = GeneratorLLM([_reserved_id_bad_call()] * 4)
    fn = _gen().get_node(llm)

    with pytest.raises(
        RuntimeError, match="after 4 attempts: node id is reserved: 'START'"
    ):
        fn.invoke({"user_message": "hi"})
    assert len(llm.gen_prompts) == 4
    assert all("Fix:" in prompt for prompt in llm.gen_prompts[1:])


def _unproduced_input_bad_call():
    return calls_message(
        tool_call(
            "c0",
            "add_node",
            {
                "id": "first_step",
                "type": "text",
                "name": "first_step",
                "description": "starts",
                "prompt": "You are an assistant. {user_request}",
                "params": {"user_request": "str"},
                "writes": {"result": "response"},
            },
        ),
        tool_call("c1", "add_edge", {"source": "START", "target": "first_step"}),
        tool_call("c2", "add_edge", {"source": "first_step", "target": "END"}),
    )


def test_unproduced_input_feedback_recovers():
    llm = GeneratorLLM([_unproduced_input_bad_call(), calls_message(*valid_text_calls())])
    fn = _gen().get_node(llm)

    assert fn.invoke({"user_message": "hi"}) == {"response": "inner:Answer hi"}
    assert len(llm.gen_prompts) == 2
    second = llm.gen_prompts[1]
    assert "reads state field 'user_request'" in second
    assert "no upstream step writes" in second
    assert "Fix:" in second
    assert "input" in second


def test_persistent_unproduced_input_failure_reproduces_prod_error():
    llm = GeneratorLLM([_unproduced_input_bad_call()] * 4)
    fn = _gen().get_node(llm)

    with pytest.raises(
        RuntimeError,
        match="after 4 attempts: reads state field 'user_request' that no "
        "upstream step writes",
    ):
        fn.invoke({"user_message": "hi"})
    assert len(llm.gen_prompts) == 4
    assert all("Fix:" in prompt for prompt in llm.gen_prompts[1:])


def test_validate_produced_inputs_flags_only_unproduced_reads():
    nodes = {
        "a": TextNode(
            "a",
            "d",
            "Do {input}",
            params={"input": str},
            writes={"result": "notes"},
        ),
        "b": TextNode("b", "d", "Use {notes}", params={"notes": str}),
    }
    graph = Graph(state_model=build_state_model(nodes))
    graph.add_node("a", nodes["a"])
    graph.add_node("b", nodes["b"])
    graph.add_edge("e1", "START", "a")
    graph.add_edge("e2", "a", "b")
    graph.add_edge("e3", "b", "END")
    assert validate_produced_inputs(graph) == []

    bad = TextNode(
        "c",
        "d",
        "See {missing}",
        params={"missing": str},
        writes={"result": "response"},
    )
    graph.add_node("c", bad)
    graph.add_edge("e4", "b", "c")
    errors = validate_produced_inputs(graph)
    assert errors == [
        "reads state field 'missing' that no upstream step writes (node 'c')"
    ]


def test_classify_rejection_unproduced_input_hint():
    r = classify_rejection(
        "reads state field 'user_request' that no upstream step writes (node 'first_step')"
    )
    assert isinstance(r, Rejection)
    assert r.code == "unproduced-input"
    assert r.hint
    assert "input" in r.hint


def test_template_and_tool_defs_teach_produced_inputs():
    assert "The only field present when the graph starts is" in _GENERATOR_TEMPLATE
    assert "upstream step's \"writes\"" in _GENERATOR_TEMPLATE
    assert "input" in _GENERATOR_TEMPLATE
    assert "the first step" in _GENERATOR_TEMPLATE
    assert '{{"input": "str"}}' in _GENERATOR_TEMPLATE
    params_def = next(
        d for d in BUILDER_TOOL_DEFS if d["function"]["name"] == "add_node"
    )["function"]["parameters"]["properties"]["params"]["description"]
    assert "input" in params_def
    assert "upstream step's writes" in params_def


def test_input_is_the_only_starting_field_regression():
    """The inner graph starts at ``input``; the old ``user_message`` is now
    an unproduced field and the spec must be rejected with feedback."""
    good = [
        tool_call(
            "c1",
            "add_node",
            {
                "id": "a",
                "type": "text",
                "name": "answer",
                "description": "answers",
                "prompt": "Answer {input}",
                "params": {"input": "str"},
            },
        ),
        tool_call("c2", "add_edge", {"source": "START", "target": "a"}),
        tool_call("c3", "add_edge", {"source": "a", "target": "END"}),
    ]
    llm = GeneratorLLM([calls_message(*good)])
    fn = _gen().get_node(llm)

    assert fn.invoke({"user_message": "hi"}) == {"response": "inner:Answer hi"}
    assert fn.last_graph is not None
    assert validate_produced_inputs(fn.last_graph) == []

    stale = [
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
            },
        ),
        tool_call("c2", "add_edge", {"source": "START", "target": "a"}),
        tool_call("c3", "add_edge", {"source": "a", "target": "END"}),
    ]
    llm = GeneratorLLM([calls_message(*stale)] * 4)
    fn = _gen().get_node(llm)
    with pytest.raises(
        RuntimeError,
        match="after 4 attempts: reads state field 'user_message' that no "
        "upstream step writes",
    ):
        fn.invoke({"user_message": "hi"})


def test_template_and_tool_defs_forbid_reserved_node_ids():
    assert "add_node" in _GENERATOR_TEMPLATE
    assert "START" in _GENERATOR_TEMPLATE
    assert "END" in _GENERATOR_TEMPLATE
    assert "reserved" in _GENERATOR_TEMPLATE
    assert "never created with add_node" in _GENERATOR_TEMPLATE
    add_node_def = next(
        d for d in BUILDER_TOOL_DEFS if d["function"]["name"] == "add_node"
    )
    id_desc = add_node_def["function"]["parameters"]["properties"]["id"]["description"]
    assert "reserved" in id_desc


def test_classify_rejection_hint_and_fallback():
    r = classify_rejection("node id is reserved: 'START'")
    assert isinstance(r, Rejection)
    assert r.code == "reserved-id"
    assert r.hint
    assert r.detail == "node id is reserved: 'START'"

    generic = classify_rejection("text node needs a non-empty 'prompt'")
    assert generic.code == "error"
    assert generic.hint is None
    assert generic.detail == "text node needs a non-empty 'prompt'"

    with pytest.raises(FrozenInstanceError):
        r.hint = "mutated"


def test_decode_error_gets_decode_hint():
    llm = GeneratorLLM([AIMessage(content="not json"), calls_message(*valid_text_calls())])
    fn = _gen().get_node(llm)

    fn.invoke({"user_message": "hi"})
    assert "Fix:" in llm.gen_prompts[1]
    assert "valid JSON" in llm.gen_prompts[1]


# ------------------------------------------------------------------ tool path


def test_tool_node_path_runs_through_fake_registry():
    spec = [
        tool_call("c1", "add_node", {"id": "t", "type": "tool", "name": "search", "description": "searches", "tool": "echo"}),
        tool_call("c2", "add_edge", {"source": "START", "target": "t"}),
        tool_call("c3", "add_edge", {"source": "t", "target": "END"}),
    ]
    llm = GeneratorLLM([calls_message(*spec)])
    fn = GeneratorNode("gen", "d", behaviors=[], registry=_registry(), collection=_empty_collection()).get_node(llm)

    assert fn.invoke({"user_message": "hi"}) == {"response": "echo:None"}


def test_unknown_tool_name_is_retry_feedback():
    bad = calls_message(
        tool_call("c1", "add_node", {"id": "t", "type": "tool", "name": "n", "description": "d", "tool": "nope"})
    )
    llm = GeneratorLLM([bad, calls_message(*valid_text_calls())])
    fn = GeneratorNode("gen", "d", behaviors=[], registry=_registry(), collection=_empty_collection()).get_node(llm)

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
    fn = GeneratorNode("gen", "d", behaviors=[], registry=_registry(), collection=_empty_collection()).get_node(llm)

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
    fn = GeneratorNode("gen", "d", behaviors=[], registry=_registry(), collection=_empty_collection()).get_node(llm)

    fn.invoke({"user_message": "hi"})
    fields = fn.last_graph.state_model.model_fields
    assert fields["notes"].annotation is list
    assert fields["input"].annotation is str
    assert fields["response"].annotation is str


# ----------------------------------------------------------------- behaviors


def test_behaviors_injected_as_objects_render_into_prompt():
    groups = [
        BehaviorGroup(title="Explain concepts", points=[BehaviorPoint(id="p1", text="be direct")])
    ]
    llm = GeneratorLLM([calls_message(*valid_text_calls())])
    fn = GeneratorNode("gen", "d", behaviors=groups, registry=_registry(), collection=_empty_collection()).get_node(llm)

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
    node = GeneratorNode("gen", "d", behaviors=str(path), registry=_registry(), collection=_empty_collection())
    assert "## Answer directly" in node.behaviors_text
    assert "- no chatter" in node.behaviors_text


def test_behaviors_default_path_loads_committed_file():
    node = GeneratorNode("gen", "d", collection=_empty_collection())
    assert "Explain concepts" in node.behaviors_text


def test_behaviors_rejects_non_group_items():
    with pytest.raises(TypeError, match="BehaviorGroup"):
        GeneratorNode(
            "gen", "d", behaviors=[{"title": "T", "points": []}], collection=_empty_collection()
        )


# ---------------------------------------------------------------- reuse flag


def test_reuse_ids_collected_from_spec(tmp_path):
    spec = [
        tool_call("c1", "add_node", {"id": "a", "type": "text", "name": "n", "description": "d", "prompt": "Answer {input}", "params": {"input": "str"}, "reuse": True}),
        tool_call("c2", "add_node", {"id": "b", "type": "text", "name": "m", "description": "d", "prompt": "Beep"}),
        tool_call("c3", "add_edge", {"source": "START", "target": "a"}),
        tool_call("c4", "add_edge", {"source": "a", "target": "b"}),
        tool_call("c5", "add_edge", {"source": "b", "target": "END"}),
    ]
    llm = GeneratorLLM([calls_message(*spec)])
    fn = GeneratorNode(
        "gen", "d", behaviors=[], save_path=tmp_path / "collection.json",
        registry=_registry(), collection=_empty_collection(),
    ).get_node(llm)

    fn.invoke({"user_message": "hi"})
    assert fn.reuse_ids == ["a"]


# ------------------------------------------------------------ construction


def test_ctor_rejects_bad_retries():
    for bad in (-1, "3", True):
        with pytest.raises(ValueError, match="retries"):
            GeneratorNode("gen", "d", retries=bad, collection=_empty_collection())


def test_ctor_rejects_bad_behaviors_type():
    with pytest.raises(TypeError, match="behaviors must be"):
        GeneratorNode("gen", "d", behaviors=42, collection=_empty_collection())


def test_updated_only_allows_metadata():
    node = GeneratorNode("gen", "d", behaviors=[], registry=_registry(), collection=_empty_collection())
    renamed = node.updated(name="new")
    assert renamed.name == "new"
    assert renamed.description == node.description
    with pytest.raises(ValueError, match="structural"):
        node.updated(prompt="custom")
    with pytest.raises(TypeError, match="unsupported fields"):
        node.updated(bogus=1)


def test_metadata_and_template_fixed():
    node = GeneratorNode("gen", "explains", behaviors=[], registry=_registry(), collection=_empty_collection())
    assert node.params == {"user_message": str}
    assert node.writes == {"result": "response"}
    assert node.prompt.startswith("Build a single-pass graph")
    assert node.get_node(GeneratorLLM([])).__name__ == "gen"