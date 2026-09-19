"""Unit tests for task-06: reuse auto-save + collection persistence.

Covers ``wrap_reused`` / ``save_reused`` and the end-of-turn hook on
:class:`GeneratorNode`: single-node vs GraphNode wrapping, the auto-save
opt-out, collection pruning (with the ``pinned`` passthrough), and JSON
round-trips through the storage layer. Pure-unit: the generator LLM is
scripted with canned ``tool_calls``, the collection gets a fake embedder, and
persistence targets ``tmp_path`` — no network, no default-embedder
construction, no real tools.
"""

from langchain_core.messages import AIMessage

from graphs.api.collection import NodeCollection
from graphs.nodes.generator import GeneratorNode, save_reused, wrap_reused
from graphs.nodes.graph_node import GraphNode
from graphs.nodes.text_node import TextNode
from graphs.nodes.tool_node import ToolCallNode
from graphs.persistence.serialization import node_from_dict
from graphs.persistence.storage import load_collection
from graphs.tools import ToolRegistry


def _embed(text: str) -> list[float]:
    return [0.1, 0.2, 0.3]


def _registry():
    registry = ToolRegistry()
    registry.register("echo", lambda args: f"echo:{args.get('text')}")
    return registry


def tool_call(call_id, name, args):
    return {"name": name, "args": args, "id": call_id}


def calls_message(*specs):
    return AIMessage(content="", tool_calls=list(specs))


def reuse_single_spec():
    """One reuse-flagged text step writing the final ``response``."""
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
                "reuse": True,
            },
        ),
        tool_call("c2", "add_edge", {"source": "START", "target": "a"}),
        tool_call("c3", "add_edge", {"source": "a", "target": "END"}),
    ]


def reuse_chain_spec():
    """Two reuse-flagged text steps wired START -> a -> b -> END."""
    return [
        tool_call(
            "c1",
            "add_node",
            {
                "id": "a",
                "type": "text",
                "name": "first",
                "description": "first step",
                "prompt": "Start {input}",
                "params": {"input": "str"},
                "writes": {"result": "notes"},
                "reuse": True,
            },
        ),
        tool_call(
            "c2",
            "add_node",
            {
                "id": "b",
                "type": "text",
                "name": "second",
                "description": "second step",
                "prompt": "Continue {notes}",
                "params": {"notes": "str"},
                "writes": {"result": "response"},
                "reuse": True,
            },
        ),
        tool_call("c3", "add_edge", {"source": "START", "target": "a"}),
        tool_call("c4", "add_edge", {"source": "a", "target": "b"}),
        tool_call("c5", "add_edge", {"source": "b", "target": "END"}),
    ]


def plain_spec():
    """A valid spec with no ``reuse`` flags at all."""
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


class GeneratorLLM:
    """Scripted LLM: generator prompts pop canned responses; others echo."""

    def __init__(self, script):
        self.script = list(script)

    def bind_tools(self, tools):
        return self

    @staticmethod
    def _is_generator(prompt):
        return prompt.startswith("Build a single-pass graph")

    def _pop(self):
        if not self.script:
            raise AssertionError("generator consumed more prompts than scripted")
        return self.script.pop(0)

    def invoke(self, prompt):
        if self._is_generator(prompt):
            return self._pop()
        return AIMessage(content=f"inner:{prompt}")

    async def ainvoke(self, prompt):
        if self._is_generator(prompt):
            return self._pop()
        return AIMessage(content=f"inner:{prompt}")


def _node(**kwargs):
    kwargs.setdefault("collection", NodeCollection(embedder=_embed))
    return GeneratorNode("gen", "d", behaviors=[], registry=_registry(), **kwargs)


# ----------------------------------------------------------------- reconstruction


def test_wrap_reused_single_returns_node_deep_copy():
    llm = GeneratorLLM([calls_message(*reuse_single_spec())])
    fn = _node().get_node(llm)
    fn.invoke({"user_message": "hi"})
    graph = fn.last_graph

    single = wrap_reused(["a"], graph, name="gen", description="d")
    assert len(single) == 1
    assert isinstance(single[0], TextNode)
    assert single[0].name == "answer"
    single[0].name = "mutated"
    assert graph.get_node("a").name == "answer"


def test_wrap_reused_chain_returns_graph_node():
    llm = GeneratorLLM([calls_message(*reuse_chain_spec())])
    fn = _node().get_node(llm)
    fn.invoke({"user_message": "hi"})
    graph = fn.last_graph

    chain = wrap_reused(["a", "b"], graph, name="gen", description="d")
    assert len(chain) == 1
    assert isinstance(chain[0], GraphNode)
    assert chain[0].input_map == {"input": "user_message"}
    assert chain[0].output_map == {"response": "response"}
    assert set(chain[0].graph.nodes()) == {"a", "b"}


def test_wrap_reused_empty_returns_nothing():
    llm = GeneratorLLM([calls_message(*plain_spec())])
    fn = _node().get_node(llm)
    fn.invoke({"user_message": "hi"})
    assert wrap_reused([], fn.last_graph, name="gen", description="d") == []


def test_save_reused_rejects_node_without_to_dict():
    class NoDict:
        name = "nope"

    collection = NodeCollection(embedder=_embed)
    try:
        save_reused([NoDict()], collection, pinned=False)
    except ValueError as exc:
        assert "without to_dict" in str(exc)
    else:
        raise AssertionError("expected ValueError for a node without to_dict()")
    assert len(collection) == 0


# ----------------------------------------------------------- single-node reuse


def test_reuse_single_node_stored_and_persisted(tmp_path):
    collection = NodeCollection(embedder=_embed)
    path = tmp_path / "collection.json"
    llm = GeneratorLLM([calls_message(*reuse_single_spec())])
    fn = _node(collection=collection, save_path=path).get_node(llm)

    fn.invoke({"user_message": "hi"})

    assert fn.reuse_ids == ["a"]
    assert len(collection) == 1
    stored = collection.snapshot()[0]
    assert isinstance(stored, TextNode)
    assert stored.name == "answer"

    restored = load_collection(path, embedder=_embed)
    assert len(restored) == 1
    node = restored.snapshot()[0]
    assert isinstance(node, TextNode)
    assert node.name == "answer"
    assert node.prompt == "Answer {input}"
    assert node.writes == {"result": "response"}


def test_each_run_saves_a_fresh_artifact(tmp_path):
    collection = NodeCollection(embedder=_embed)
    path = tmp_path / "collection.json"
    for _ in range(2):
        llm = GeneratorLLM([calls_message(*reuse_single_spec())])
        _node(collection=collection, save_path=path).get_node(llm).invoke(
            {"user_message": "hi"}
        )
    assert len(collection) == 2
    assert len(load_collection(path, embedder=_embed)) == 2


def test_reuse_single_tool_node_stored(tmp_path):
    spec = [
        tool_call(
            "c1",
            "add_node",
            {
                "id": "t",
                "type": "tool",
                "name": "search",
                "description": "searches",
                "tool": "echo",
                "reuse": True,
            },
        ),
        tool_call("c2", "add_edge", {"source": "START", "target": "t"}),
        tool_call("c3", "add_edge", {"source": "t", "target": "END"}),
    ]
    collection = NodeCollection(embedder=_embed)
    path = tmp_path / "collection.json"
    llm = GeneratorLLM([calls_message(*spec)])
    _node(collection=collection, save_path=path).get_node(llm).invoke(
        {"user_message": "hi"}
    )
    stored = collection.snapshot()[0]
    assert isinstance(stored, ToolCallNode)
    assert stored.tool == "echo"


# ---------------------------------------------------------------- chain reuse


def test_reuse_chain_stored_as_graph_node_round_trips(tmp_path):
    collection = NodeCollection(embedder=_embed)
    path = tmp_path / "collection.json"
    llm = GeneratorLLM([calls_message(*reuse_chain_spec())])
    fn = _node(collection=collection, save_path=path).get_node(llm)

    fn.invoke({"user_message": "hi"})

    assert fn.reuse_ids == ["a", "b"]
    assert len(collection) == 1
    stored = collection.snapshot()[0]
    assert isinstance(stored, GraphNode)
    assert set(stored.graph.nodes()) == {"a", "b"}

    again = node_from_dict(stored.to_dict())
    assert isinstance(again, GraphNode)
    assert again.name == stored.name
    assert set(again.graph.nodes()) == {"a", "b"}
    assert again.input_map == {"input": "user_message"}
    assert again.output_map == {"response": "response"}

    restored = load_collection(path, embedder=_embed)
    assert len(restored) == 1
    assert isinstance(restored.snapshot()[0], GraphNode)


# --------------------------------------------------------------- no-reuse path


def test_no_reuse_saves_nothing(tmp_path):
    collection = NodeCollection(embedder=_embed)
    path = tmp_path / "collection.json"
    llm = GeneratorLLM([calls_message(*plain_spec())])
    fn = _node(collection=collection, save_path=path).get_node(llm)

    fn.invoke({"user_message": "hi"})

    assert fn.reuse_ids == []
    assert len(collection) == 0
    assert not path.exists()


def test_auto_save_disabled_saves_nothing(tmp_path):
    collection = NodeCollection(embedder=_embed)
    path = tmp_path / "collection.json"
    llm = GeneratorLLM([calls_message(*reuse_single_spec())])
    fn = _node(auto_save=False, collection=collection, save_path=path).get_node(llm)

    fn.invoke({"user_message": "hi"})

    assert fn.reuse_ids == ["a"]
    assert len(collection) == 0
    assert not path.exists()


async def test_async_run_auto_saves_too(tmp_path):
    collection = NodeCollection(embedder=_embed)
    path = tmp_path / "collection.json"
    llm = GeneratorLLM([calls_message(*reuse_single_spec())])
    fn = _node(collection=collection, save_path=path).get_node(llm)

    result = await fn.ainvoke({"user_message": "hi"})

    assert result == {"response": "inner:Answer hi"}
    assert len(collection) == 1
    assert len(load_collection(path, embedder=_embed)) == 1


# ------------------------------------------------------------------ pruning


def test_pruning_governs_collection_and_persistence(tmp_path):
    collection = NodeCollection(max_nodes=2, prune_to=1, embedder=_embed)
    path = tmp_path / "collection.json"
    for _ in range(3):
        llm = GeneratorLLM([calls_message(*reuse_single_spec())])
        _node(collection=collection, save_path=path).get_node(llm).invoke(
            {"user_message": "hi"}
        )
    assert len(collection) == 1
    assert len(load_collection(path, embedder=_embed)) == 1


def test_pinned_reuse_artifacts_exempt_from_pruning(tmp_path):
    collection = NodeCollection(max_nodes=2, prune_to=1, embedder=_embed)
    path = tmp_path / "collection.json"
    for _ in range(3):
        llm = GeneratorLLM([calls_message(*reuse_single_spec())])
        _node(collection=collection, save_path=path, reuse_pinned=True).get_node(
            llm
        ).invoke({"user_message": "hi"})
    assert len(collection) == 3
    assert all(record.pinned for record in collection.records())
    assert len(load_collection(path, embedder=_embed)) == 3


# -------------------------------------------------------------- construction


def test_ctor_validates_auto_save_config():
    for kwargs in ({"auto_save": "yes"}, {"reuse_pinned": "yes"}):
        try:
            GeneratorNode("gen", "d", **kwargs)
        except TypeError as exc:
            assert "must be a bool" in str(exc)
        else:
            raise AssertionError(f"expected TypeError for {kwargs}")
    for kwargs in ({"save_path": 42}, {"collection": object()}):
        try:
            GeneratorNode("gen", "d", **kwargs)
        except TypeError as exc:
            assert "must be" in str(exc)
        else:
            raise AssertionError(f"expected TypeError for {kwargs}")


def test_updated_preserves_auto_save_config(tmp_path):
    collection = NodeCollection(embedder=_embed)
    node = _node(collection=collection, save_path=tmp_path / "c.json", reuse_pinned=True)
    renamed = node.updated(name="new")
    fn = renamed.get_node(GeneratorLLM([calls_message(*reuse_single_spec())]))
    fn.invoke({"user_message": "hi"})
    assert renamed.name == "new"
    assert len(collection) == 1