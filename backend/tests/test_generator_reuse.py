"""Unit tests for task-05: generator builder reuse of collection nodes.

Covers the internal retrieve step (catalog -> chosen ids, skipped on an empty
collection), the restore of the reuse collection from disk, ``add_node``'s
``from_collection`` path (live shared instance, stats fold back, no
double-add), and the error feedback for unknown ids / duplicate pulls / mixed
fresh+pulled / ``reuse``+pull conflicts. Pure-unit: every LLM call is scripted
(retrieve and build prompts pop canned responses; inner nodes echo), the
collection gets a fake embedder, and persistence writes under ``tmp_path``.
"""

import json

from langchain_core.messages import AIMessage

from graphs.api.collection import NodeCollection
from graphs.nodes.generator import GeneratorNode
from graphs.nodes.text_node import TextNode
from graphs.persistence.storage import dump_collection, load_collection
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


def retrieve_message(ids):
    return AIMessage(content=json.dumps({"ids": ids}))


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


def pull_calls(cid, node_id="a"):
    """A valid chain that pulls a collection node as the only step."""
    return [
        tool_call(
            "c1",
            "add_node",
            {
                "id": node_id,
                "type": "text",
                "name": "answer",
                "description": "answers",
                "from_collection": cid,
            },
        ),
        tool_call("c2", "add_edge", {"source": "START", "target": node_id}),
        tool_call("c3", "add_edge", {"source": node_id, "target": "END"}),
    ]


def _catalog_node(run_counts=0):
    node = TextNode(
        "answer",
        "answers the questions",
        "Answer {input}",
        params={"input": str},
        writes={"result": "response"},
    )
    node.run_counts = run_counts
    return node


class GenLLM:
    """Scripted LLM: retrieve/build prompts pop canned scripts; inner echoes."""

    def __init__(self, build=(), retrieve=()):
        self.build = list(build)
        self.retrieve = list(retrieve)
        self.prompts = []
        self.build_prompts = []
        self.retrieve_prompts = []

    def bind_tools(self, tools):
        return self

    @staticmethod
    def _is_build(prompt):
        return prompt.startswith("Build a single-pass graph")

    @staticmethod
    def _is_retrieve(prompt):
        return prompt.startswith("Select reusable steps")

    def _pop(self, script, label):
        if not script:
            raise AssertionError(
                f"generator consumed more {label} responses than scripted"
            )
        return script.pop(0)

    def _dispatch(self, prompt):
        if self._is_build(prompt):
            self.build_prompts.append(prompt)
            return self._pop(self.build, "build")
        if self._is_retrieve(prompt):
            self.retrieve_prompts.append(prompt)
            return self._pop(self.retrieve, "retrieve")
        return AIMessage(content=f"inner:{prompt}")

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return self._dispatch(prompt)

    async def ainvoke(self, prompt):
        self.prompts.append(prompt)
        return self._dispatch(prompt)


def _node(**kwargs):
    kwargs.setdefault("collection", NodeCollection(embedder=_embed))
    return GeneratorNode("gen", "d", behaviors=[], registry=_registry(), **kwargs)


# -------------------------------------------------------------- skip + catalog


def test_empty_collection_skips_retrieve():
    llm = GenLLM(build=[calls_message(*valid_text_calls())])
    fn = _node().get_node(llm)

    assert fn.invoke({"user_message": "hi"}) == {"response": "inner:Answer hi"}
    assert llm.retrieve_prompts == []
    assert llm.build_prompts == [llm.prompts[0]]


def test_retrieve_catalog_lists_live_stats():
    source = NodeCollection(embedder=_embed)
    source.add(_catalog_node(run_counts=3))
    llm = GenLLM(build=[calls_message(*valid_text_calls())], retrieve=[retrieve_message([])])
    fn = _node(collection=source).get_node(llm)

    fn.invoke({"user_message": "hi"})

    assert len(llm.retrieve_prompts) == 1
    assert "ran 3 times" in llm.retrieve_prompts[0]


# --------------------------------------------------------------- retrieve step


def test_retrieve_pulls_live_node_and_stats_fold_back():
    source = NodeCollection(embedder=_embed)
    shared = _catalog_node(run_counts=3)
    cid = source.add(shared)
    llm = GenLLM(build=[calls_message(*pull_calls(cid))], retrieve=[retrieve_message([cid])])
    fn = _node(collection=source).get_node(llm)

    result = fn.invoke({"user_message": "hi"})

    assert result == {"response": "inner:Answer hi"}
    assert llm.retrieve_prompts[0]
    assert fn._retrieved_ids == [cid]
    assert "Reusable steps selected for this request" in llm.build_prompts[0]
    assert shared.run_counts == 4
    assert source.get(cid) is shared


async def test_async_retrieve_runs_too():
    source = NodeCollection(embedder=_embed)
    shared = _catalog_node()
    cid = source.add(shared)
    llm = GenLLM(build=[calls_message(*pull_calls(cid))], retrieve=[retrieve_message([cid])])
    fn = _node(collection=source).get_node(llm)

    assert await fn.ainvoke({"user_message": "hi"}) == {"response": "inner:Answer hi"}
    assert len(llm.retrieve_prompts) == 1
    assert shared.run_counts == 1


def test_malformed_retrieve_response_degrades_gracefully():
    source = NodeCollection(embedder=_embed)
    source.add(_catalog_node())
    llm = GenLLM(build=[calls_message(*valid_text_calls())], retrieve=[AIMessage(content="not json")])
    fn = _node(collection=source).get_node(llm)

    assert fn.invoke({"user_message": "hi"}) == {"response": "inner:Answer hi"}
    assert "retrieve step was ignored" in llm.build_prompts[0]


def test_retrieve_unknown_ids_ignored_with_note():
    source = NodeCollection(embedder=_embed)
    source.add(_catalog_node())
    llm = GenLLM(build=[calls_message(*valid_text_calls())], retrieve=[retrieve_message(["bogus"])])
    fn = _node(collection=source).get_node(llm)

    fn.invoke({"user_message": "hi"})

    assert fn._retrieved_ids == []
    assert "unknown collection ids" in llm.build_prompts[0]


# ------------------------------------------------------------- build feedback


def test_from_collection_unknown_id_is_retry_feedback():
    source = NodeCollection(embedder=_embed)
    cid = source.add(_catalog_node())
    bad = calls_message(
        tool_call(
            "c1", "add_node", {"id": "a", "type": "text", "name": "n", "description": "d", "from_collection": "nope"}
        )
    )
    llm = GenLLM(build=[bad, calls_message(*valid_text_calls())], retrieve=[retrieve_message([cid])])
    fn = _node(collection=source).get_node(llm)

    fn.invoke({"user_message": "hi"})
    assert "unknown collection node 'nope'" in llm.build_prompts[1]


def test_duplicate_collection_pull_is_retry_feedback():
    source = NodeCollection(embedder=_embed)
    cid = source.add(_catalog_node())
    bad = calls_message(
        tool_call("c1", "add_node", {"id": "a", "type": "text", "name": "n", "description": "d", "from_collection": cid}),
        tool_call("c2", "add_node", {"id": "b", "type": "text", "name": "m", "description": "d", "from_collection": cid}),
    )
    llm = GenLLM(build=[bad, calls_message(*valid_text_calls())], retrieve=[retrieve_message([cid])])
    fn = _node(collection=source).get_node(llm)

    fn.invoke({"user_message": "hi"})
    assert "already added to this graph" in llm.build_prompts[1]


def test_from_collection_mixed_with_fresh_spec_is_retry_feedback():
    source = NodeCollection(embedder=_embed)
    cid = source.add(_catalog_node())
    bad = calls_message(
        tool_call("c1", "add_node", {"id": "a", "type": "text", "name": "n", "description": "d", "from_collection": cid, "prompt": "X"})
    )
    llm = GenLLM(build=[bad, calls_message(*valid_text_calls())], retrieve=[retrieve_message([cid])])
    fn = _node(collection=source).get_node(llm)

    fn.invoke({"user_message": "hi"})
    assert "cannot be combined with a fresh spec" in llm.build_prompts[1]


def test_reuse_and_from_collection_conflict_is_retry_feedback():
    source = NodeCollection(embedder=_embed)
    cid = source.add(_catalog_node())
    bad = calls_message(
        tool_call("c1", "add_node", {"id": "a", "type": "text", "name": "n", "description": "d", "from_collection": cid, "reuse": True})
    )
    llm = GenLLM(build=[bad, calls_message(*valid_text_calls())], retrieve=[retrieve_message([cid])])
    fn = _node(collection=source).get_node(llm)

    fn.invoke({"user_message": "hi"})
    assert "cannot set both 'reuse' and 'from_collection'" in llm.build_prompts[1]


# --------------------------------------------------------- reuse:true interplay


def test_fresh_reuse_and_pull_in_same_graph_no_double_add(tmp_path):
    source = NodeCollection(embedder=_embed)
    cid = source.add(_catalog_node())
    path = tmp_path / "collection.json"
    spec = [
        tool_call("c1", "add_node", {"id": "r", "type": "text", "name": "answer", "description": "answers", "from_collection": cid}),
        tool_call(
            "c2", "add_node", {"id": "b", "type": "text", "name": "another", "description": "fresh step", "prompt": "Beep {input}", "params": {"input": "str"}, "writes": {"result": "response"}, "reuse": True}
        ),
        tool_call("c3", "add_edge", {"source": "START", "target": "r"}),
        tool_call("c4", "add_edge", {"source": "r", "target": "b"}),
        tool_call("c5", "add_edge", {"source": "b", "target": "END"}),
    ]
    llm = GenLLM(build=[calls_message(*spec)], retrieve=[retrieve_message([cid])])
    fn = _node(collection=source, save_path=path).get_node(llm)

    result = fn.invoke({"user_message": "hi"})

    assert result == {"response": "inner:Beep hi"}
    assert fn.reuse_ids == ["b"]
    assert sum(1 for record in source.records() if record.id == cid) == 1
    assert len(source) == 2
    assert len(load_collection(path, embedder=_embed)) == 2


# ---------------------------------------------------- restore from disk (step 1)


def test_reuse_collection_restored_from_disk(tmp_path, monkeypatch):
    path = tmp_path / "collection.json"
    source = NodeCollection(embedder=_embed)
    source.add(_catalog_node())
    dump_collection(source, path)

    import graphs.nodes.generator as gen_mod

    seen = []
    monkeypatch.setattr(
        gen_mod,
        "load_collection",
        lambda p, **kw: seen.append(p) or source,
    )
    node = GeneratorNode("gen", "d", behaviors=[], registry=_registry(), save_path=path)
    assert seen == [path]
    assert node._reuse_collection is source
    assert len(node._reuse_collection) == 1


def test_restored_collection_nodes_feed_retrieve(tmp_path, monkeypatch):
    path = tmp_path / "collection.json"
    source = NodeCollection(embedder=_embed)
    nid = source.add(_catalog_node())
    dump_collection(source, path)

    import graphs.nodes.generator as gen_mod

    monkeypatch.setattr(gen_mod, "load_collection", lambda p, **kw: source)
    llm = GenLLM(build=[calls_message(*pull_calls(nid))], retrieve=[retrieve_message([nid])])
    fn = GeneratorNode("gen", "d", behaviors=[], registry=_registry(), save_path=path).get_node(llm)

    assert fn.invoke({"user_message": "hi"}) == {"response": "inner:Answer hi"}
    assert len(llm.retrieve_prompts) == 1