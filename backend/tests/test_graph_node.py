"""Unit tests for ``GraphNode`` — wrapping a full graph as a usable node.

Pure-unit: every LLM is a fake derived from the pattern in
``test_graph_compile.py``; nothing touches the network. Covers the
``input_map``/``output_map`` bridge, the multi-output surface, nesting inside
a parent graph, lazy inner validation, and constructor checks.
"""

import pytest
from pydantic import BaseModel

from graphs.graph import END, START, Graph, GraphValidationError
from graphs.nodes.graph_node import GraphNode
from graphs.nodes.text_node import TextNode


class FakeLLM:
    """Returns ``{content}:{prompt}`` so tests can tell which LLM ran."""

    def __init__(self, content="ok", tokens=None):
        self.content = content
        self.tokens = tokens

    def _result(self, content):
        if self.tokens is None:
            return type("R", (), {"content": content})()
        return type("R", (), {
            "content": content,
            "usage_metadata": {"output_tokens": self.tokens},
        })()

    def invoke(self, prompt):
        return self._result(f"{self.content}:{prompt}")

    async def ainvoke(self, prompt):
        return self._result(f"a:{prompt}")


def _text_node(node_id, name, prompt, *, params=None, writes=None):
    return TextNode(name, f"description of {node_id}", prompt, params=params, writes=writes)


def _single_output_subgraph(llm="I"):
    """A valid inner graph: START -> a -> END, writing ``response``."""
    inner = Graph()
    inner.add_node(
        "a",
        _text_node("a", "inner alpha", "Gen {user_message}",
                   params={"user_message": str}),
        llm=FakeLLM(llm),
    )
    inner.add_edge("e1", START, "a")
    inner.add_edge("e2", "a", END)
    return inner


def test_invoke_bridges_outer_state_into_inner_initial_state():
    node = GraphNode(
        "sub", "wraps the inner graph", "reusable subgraph",
        _single_output_subgraph(llm="I"),
        input_map={"user_message": "user_message"},
        output_map={"response": "response"},
    )

    result = node.get_node(FakeLLM("P")).invoke({"user_message": "hi"})

    assert result == {"response": "I:Gen hi"}


class MultiOutputState(BaseModel):
    """Inner state with two independent output fields."""

    user_message: str | None = None
    answer: str | None = None
    summary: str | None = None


def _multi_output_subgraph():
    inner = Graph(state_model=MultiOutputState)
    inner.add_node(
        "ans", _text_node("ans", "answerer", "Answer {user_message}",
                          params={"user_message": str},
                          writes={"result": "answer"}),
        llm=FakeLLM("A"),
    )
    inner.add_node(
        "sum", _text_node("sum", "summarizer", "Summary {user_message}",
                          params={"user_message": str},
                          writes={"result": "summary"}),
        llm=FakeLLM("S"),
    )
    inner.add_edge("e1", START, "ans")
    inner.add_edge("e2", "ans", "sum")
    inner.add_edge("e3", "sum", END)
    return inner


def test_invoke_surfaces_multiple_output_keys():
    node = GraphNode(
        "sub", "wraps the inner graph", "multi-output subgraph",
        _multi_output_subgraph(),
        input_map={"user_message": "user_message"},
        output_map={"answer": "outer_answer", "summary": "outer_summary"},
    )

    result = node.get_node(FakeLLM("P")).invoke({"user_message": "hi"})

    assert result == {
        "outer_answer": "A:Answer hi",
        "outer_summary": "S:Summary hi",
    }


def test_nested_graph_compiles_and_runs_end_to_end():
    node = GraphNode(
        "sub", "wraps the inner graph", "reusable subgraph",
        _single_output_subgraph(llm="I"),
        input_map={"user_message": "user_message"},
        output_map={"response": "response"},
    )
    parent = Graph()
    parent.add_node("sub", node, llm=FakeLLM("P"))
    parent.add_edge("e1", START, "sub")
    parent.add_edge("e2", "sub", END)

    result = parent.compile().invoke({"user_message": "hey"})

    assert result["response"] == "I:Gen hey"


def test_construction_rejects_empty_maps():
    inner = _single_output_subgraph()
    with pytest.raises(ValueError, match="input_map"):
        GraphNode(
            "g", "d", "p", inner,
            input_map={},
            output_map={"response": "response"},
        )
    with pytest.raises(ValueError, match="output_map"):
        GraphNode(
            "g", "d", "p", inner,
            input_map={"user_message": "user_message"},
            output_map={},
        )


def test_construction_rejects_non_string_map_entries():
    inner = _single_output_subgraph()
    with pytest.raises(ValueError, match="input_map"):
        GraphNode(
            "g", "d", "p", inner,
            input_map={"user_message": ""},
            output_map={"response": "response"},
        )


def test_construction_rejects_inner_keys_outside_inner_state():
    inner = _single_output_subgraph()
    with pytest.raises(ValueError, match="inner key 'nope"):
        GraphNode(
            "g", "d", "p", inner,
            input_map={"nope": "user_message"},
            output_map={"response": "response"},
        )
    with pytest.raises(ValueError, match="inner key 'nope"):
        GraphNode(
            "g", "d", "p", inner,
            input_map={"user_message": "user_message"},
            output_map={"nope": "response"},
        )


def test_parent_validation_checks_graph_node_writes_against_parent_state():
    node = GraphNode(
        "sub", "wraps the inner graph", "subgraph",
        _single_output_subgraph(),
        input_map={"user_message": "user_message"},
        output_map={"response": "does_not_exist"},
    )
    parent = Graph()
    parent.add_node("sub", node)

    with pytest.raises(GraphValidationError, match="writes state keys"):
        parent.validate()


def test_inner_graph_is_validated_lazily_on_first_invoke():
    inner = Graph()
    inner.add_node("a", _text_node("a", "alpha", "p", writes={"result": "ghost"}))
    inner.add_edge("e1", START, "a")
    inner.add_edge("e2", "a", END)
    node = GraphNode(
        "sub", "wraps the inner graph", "subgraph", inner,
        input_map={"user_message": "user_message"},
        output_map={"response": "response"},
    )

    with pytest.raises(GraphValidationError, match="writes state keys"):
        node.get_node(FakeLLM("P")).invoke({"user_message": "hi"})


def _sub_node():
    return GraphNode(
        "sub", "wraps the inner graph", "reusable subgraph",
        _single_output_subgraph(llm="I"),
        input_map={"user_message": "user_message"},
        output_map={"response": "response"},
    )


def test_updated_rebuilds_metadata_and_keeps_derived_maps():
    node = _sub_node()
    updated = node.updated(name="renamed", description="new description")
    assert updated is not node
    assert updated.name == "renamed"
    assert updated.description == "new description"
    assert updated.prompt == node.prompt
    assert updated.input_map == node.input_map
    assert updated.output_map == node.output_map
    assert updated.params == node.params
    assert updated.writes == node.writes
    assert updated.graph is node.graph
    with pytest.raises(ValueError, match="derived from input_map/output_map"):
        node.updated(params={"x": str})
    with pytest.raises(ValueError, match="derived from input_map/output_map"):
        node.updated(writes={"x": "y"})
    with pytest.raises(TypeError, match="unsupported"):
        node.updated(bogus=1)


def test_updated_prompt_placeholder_checked_against_input_map():
    node = _sub_node()
    with pytest.raises(ValueError, match="not declared in params"):
        node.updated(prompt="Use {ghost_key}")


# ---------------------------------------------------------------- run stats


def test_graph_node_records_time_while_inner_nodes_self_record_tokens():
    inner_text = _text_node(
        "a", "inner alpha", "Gen {user_message}", params={"user_message": str}
    )
    inner = Graph()
    inner.add_node("a", inner_text, llm=FakeLLM("I", tokens=9))
    inner.add_edge("e1", START, "a")
    inner.add_edge("e2", "a", END)
    node = GraphNode(
        "sub", "wraps the inner graph", "reusable subgraph", inner,
        input_map={"user_message": "user_message"},
        output_map={"response": "response"},
    )
    fn = node.get_node(FakeLLM("P"))
    fn.invoke({"user_message": "hi"})
    fn.invoke({"user_message": "hi"})

    assert node.run_counts == 2
    assert node.output_tokens.count == 0
    assert node.output_time.count == 2
    assert node.output_time.mean > 0.0
    assert inner_text.run_counts == 2
    assert inner_text.output_tokens.count == 2
    assert inner_text.output_tokens.mean == 9.0
    assert inner_text.output_time.count == 2


async def test_graph_node_ainvoke_records_time_once():
    inner = _single_output_subgraph(llm="I")
    node = GraphNode(
        "sub", "wraps the inner graph", "reusable subgraph", inner,
        input_map={"user_message": "user_message"},
        output_map={"response": "response"},
    )
    result = await node.get_node(FakeLLM("P")).ainvoke({"user_message": "hi"})
    assert result == {"response": "I:Gen hi"}
    assert node.run_counts == 1
    assert node.output_time.count == 1


def test_graph_node_updated_preserves_run_stats():
    node = _sub_node()
    node.get_node(FakeLLM("P")).invoke({"user_message": "hi"})
    renamed = node.updated(name="renamed")
    assert renamed.run_counts == 1
    assert renamed.output_time.count == 1
    assert renamed.output_time.mean > 0.0