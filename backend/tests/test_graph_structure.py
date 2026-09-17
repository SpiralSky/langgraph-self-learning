"""Unit tests for the editable ``Graph`` structure in ``graphs.graph``.

Pure-structure tests: no langgraph, no live LLM. Every node is a minimal
``AbstractNode`` stub; connections are exercised exactly as the edit ops
wire them up.
"""

import pytest
from pydantic import BaseModel

from graphs.structure.connections import Connection, RoutingConnection
from graphs.structure.graph import (
    END,
    START,
    Graph,
    GraphValidationError,
    NodeView,
)
from graphs.nodes.base import AbstractNode
from graphs.structure.state import LearningGraphState


class _MinNode(AbstractNode):
    """Minimal concrete node; only construction invariants are exercised."""

    def get_node(self, llm):
        raise NotImplementedError


def _node(name="alpha", prompt="p", params=None, writes=None):
    return _MinNode(name, "description", prompt, params=params, writes=writes)


def _chain_graph(nodes=("a", "b")):
    """``START -> a -> b -> END``."""
    graph = Graph()
    for node_id, name in zip(nodes, ("alpha", "beta")):
        graph.add_node(node_id, _node(name=name, prompt="do it"))
    graph.add_edge("e_start_a", START, nodes[0])
    graph.add_edge("e_a_b", nodes[0], nodes[1])
    graph.add_edge("e_b_end", nodes[1], END)
    return graph


# ---------------------------------------------------------------- add_node


@pytest.mark.parametrize("bad_id", ["", None, START, END])
def test_add_node_rejects_bad_ids(bad_id):
    graph = Graph()
    with pytest.raises(ValueError):
        graph.add_node(bad_id, _node())


def test_add_node_id_clash_rejected():
    graph = Graph()
    graph.add_node("a", _node())
    with pytest.raises(ValueError, match="already present"):
        graph.add_node("a", _node())


def test_add_node_tombstoned_id_rejected():
    graph = Graph()
    graph.add_node("a", _node())
    graph.remove_node("a")
    with pytest.raises(ValueError, match="cannot be re-added"):
        graph.add_node("a", _node())


def test_add_node_retrieval_and_length():
    graph = Graph()
    node = _node(name="alpha", prompt="do it")
    graph.add_node("a", node)
    stored = graph.get_node("a")
    assert stored is not node
    assert stored.name == "alpha"
    assert stored.description == "description"
    assert stored.prompt == "do it"
    assert len(graph) == 1
    assert "a" in graph
    assert "b" not in graph


def test_add_node_per_node_llm_override_stored():
    graph = Graph()
    llm = object()
    graph.add_node("a", _node(), llm=llm)
    graph.add_node("b", _node())
    assert graph.get_node_llm("a") is llm
    assert graph.get_node_llm("b") is None


# ---------------------------------------------------------------- add_edge


def test_add_edge_endpoint_must_exist():
    graph = Graph()
    graph.add_node("a", _node())
    with pytest.raises(ValueError, match="does not exist"):
        graph.add_edge("e1", "a", "missing")


def test_add_edge_role_rules():
    graph = Graph()
    graph.add_node("a", _node())
    with pytest.raises(ValueError, match="may only be a target"):
        graph.add_edge("e1", END, "a")
    with pytest.raises(ValueError, match="may only be a source"):
        graph.add_edge("e2", "a", START)
    with pytest.raises(ValueError, match="may only be a source"):
        graph.add_edge("e3", "a", [START, "END"], routing_prompt="pick")


def test_add_edge_standard_builds_connection():
    graph = Graph()
    graph.add_node("a", _node())
    graph.add_node("b", _node())
    graph.add_edge("e1", "a", "b")
    edge = graph.get_edge("e1")
    assert isinstance(edge, Connection)
    assert edge.source == "a"
    assert edge.target == "b"


def test_add_edge_routing_builds_routing_connection():
    graph = Graph()
    graph.add_node("a", _node())
    graph.add_node("b", _node())
    graph.add_node("c", _node())
    graph.add_edge("e1", "a", ["b", "c"], routing_prompt="pick one")
    edge = graph.get_edge("e1")
    assert isinstance(edge, RoutingConnection)
    assert edge.targets == ("b", "c")
    assert edge.routing_prompt == "pick one"


def test_add_edge_routing_requires_prompt_and_targets():
    graph = Graph()
    graph.add_node("a", _node())
    graph.add_node("b", _node())
    graph.add_node("c", _node())
    with pytest.raises(ValueError, match="routing_prompt"):
        graph.add_edge("e1", "a", ["b", "c"])
    with pytest.raises(ValueError, match="routing_prompt"):
        graph.add_edge("e2", "a", ["b"], routing_prompt="")
    with pytest.raises(ValueError, match="at least one"):
        graph.add_edge("e3", "a", [], routing_prompt="pick")


def test_add_edge_multi_target_without_routing_rejected():
    graph = Graph()
    graph.add_node("a", _node())
    graph.add_node("b", _node())
    graph.add_node("c", _node())
    with pytest.raises(ValueError, match="routing_prompt"):
        graph.add_edge("e1", "a", ["b", "c"])


def test_add_edge_dedup_by_equivalence():
    graph = Graph()
    graph.add_node("a", _node())
    graph.add_node("b", _node())
    graph.add_edge("e1", "a", "b")
    with pytest.raises(ValueError, match="already exists"):
        graph.add_edge("e2", "a", ["b"])
    with pytest.raises(ValueError, match="already exists"):
        graph.add_edge("e3", "a", "b")


def test_add_edge_standard_and_routing_coexist():
    graph = Graph()
    graph.add_node("a", _node())
    graph.add_node("b", _node())
    graph.add_edge("e1", "a", "b")
    graph.add_edge("e2", "a", ["b"], routing_prompt="pick one")
    assert not isinstance(graph.get_edge("e2"), Connection)


def test_add_edge_id_repeat_rejected():
    graph = Graph()
    graph.add_node("a", _node())
    graph.add_node("b", _node())
    graph.add_node("c", _node())
    graph.add_edge("e1", "a", "b")
    with pytest.raises(ValueError, match="already present"):
        graph.add_edge("e1", "b", "c")


# -------------------------------------------------------- remove/update ops


def test_remove_node_cascades_incident_edges():
    graph = Graph()
    graph.add_node("a", _node())
    graph.add_node("b", _node())
    graph.add_node("c", _node())
    graph.add_edge("e1", "a", "b")
    graph.add_edge("e2", "b", "c")
    graph.add_edge("e3", "a", "c")
    graph.remove_node("b")
    remaining = graph.nodes()
    assert set(remaining) == {"a", "c"}
    assert all(remaining[node_id].name == "alpha" for node_id in remaining)
    assert set(graph.edges()) == {"e3"}


def test_remove_node_absent_raises_key_error():
    with pytest.raises(KeyError):
        Graph().remove_node("nope")


def test_remove_edge_and_tombstone():
    graph = Graph()
    graph.add_node("a", _node())
    graph.add_node("b", _node())
    graph.add_edge("e1", "a", "b")
    graph.remove_edge("e1")
    assert "e1" not in graph.edges()
    with pytest.raises(ValueError, match="cannot be re-added"):
        graph.add_edge("e1", "a", "b")
    with pytest.raises(KeyError):
        graph.remove_edge("e1")


def test_remove_node_frees_equivalent_edges():
    graph = Graph()
    graph.add_node("a", _node())
    graph.add_node("b", _node())
    graph.add_edge("e1", "a", "b")
    graph.remove_node("a")
    graph.add_node("a2", _node())
    graph.add_edge("e2", "a2", "b")
    assert "e2" in graph.edges()


def test_update_node_replaces_instance_no_revalidation():
    graph = Graph()
    first = _node(name="alpha", prompt="do it")
    graph.add_node("a", first)
    second = _node(name="omega", prompt="do different", params={"topic": str})
    graph.update_node("a", second)
    stored = graph.get_node("a")
    assert stored is not second
    assert stored.name == "omega"
    assert stored.prompt == "do different"


def test_update_edge_reruns_checks():
    graph = Graph()
    graph.add_node("a", _node())
    graph.add_node("b", _node())
    graph.add_node("c", _node())
    graph.add_edge("e1", "a", "b")
    graph.update_edge("e1", target="c")
    edge = graph.get_edge("e1")
    assert isinstance(edge, Connection)
    assert edge.target == "c"
    with pytest.raises(ValueError, match="routing_prompt"):
        graph.update_edge("e1", targets=("c", "a"))


def test_update_edge_to_routing():
    graph = Graph()
    graph.add_node("a", _node())
    graph.add_node("b", _node())
    graph.add_node("c", _node())
    graph.add_edge("e1", "a", "b")
    graph.update_edge("e1", targets=("b", "c"), routing_prompt="pick")
    edge = graph.get_edge("e1")
    assert isinstance(edge, RoutingConnection)
    assert edge.targets == ("b", "c")


def test_update_absent_node_edge_raise_key_error():
    graph = Graph()
    with pytest.raises(KeyError):
        graph.update_node("x", _node())
    with pytest.raises(KeyError):
        graph.update_edge("x", target="a")


# ---------------------------------------------------------------- validate


def test_validate_pass_on_valid_chain():
    _chain_graph().validate()


def test_validate_requires_exactly_one_start_edge():
    graph = Graph()
    graph.add_node("a", _node())
    with pytest.raises(GraphValidationError, match="exactly one START-source edge"):
        graph.validate()
    graph.add_edge("e1", START, "a")
    graph.validate()

    graph2 = _chain_graph()
    graph2.add_node("c", _node(name="gamma"))
    graph2.add_edge("e_extra_start2", START, "c")
    with pytest.raises(GraphValidationError, match=r"START-source edge \(found 2\)"):
        graph2.validate()


def test_validate_is_aggregate():
    graph = Graph()
    with pytest.raises(GraphValidationError) as excinfo:
        graph.validate()
    assert len(excinfo.value.errors) >= 1


def test_validate_catches_missing_endpoint_reference():
    graph = _chain_graph()
    graph._nodes.pop("a")
    with pytest.raises(GraphValidationError, match="references missing node"):
        graph.validate()


def test_validate_state_schema_errors():
    graph = Graph()
    graph.add_node("a", _node(prompt="x {topic}", params={"topic": str}))
    with pytest.raises(GraphValidationError, match=r"reads state fields"):
        graph.validate()


def test_validate_writes_state_keys_checked():
    graph = Graph()
    graph.add_node("a", _node(prompt="p", writes={"r": "analysis"}))
    with pytest.raises(GraphValidationError, match=r"writes state keys"):
        graph.validate()


def test_orphan_placeholder_fails_at_construction():
    with pytest.raises(ValueError, match="not declared in params"):
        _node(prompt="Do {topic}")


def test_custom_state_model_validated():
    class Custom(BaseModel):
        topic: str = "x"
        result: str = "y"
        response: str | None = None

    graph = Graph(state_model=Custom)
    graph.add_node("a", _node(prompt="T {topic}", params={"topic": str}))
    graph.add_node("b", _node(prompt="W {topic}", params={"topic": str}, writes={"r": "result"}))
    graph.add_edge("e1", START, "a")
    graph.add_edge("e2", "a", "b")

    graph.validate()

    graph3 = Graph(state_model=LearningGraphState)
    graph3.add_node("a", _node(prompt="T {topic}", params={"topic": str}))
    with pytest.raises(GraphValidationError, match=r"reads state fields"):
        graph3.validate()


# ---------------------------------------------------------------- render


def test_render_default_name_chain():
    assert _chain_graph().render() == "START -> alpha\nalpha -> beta\nbeta -> END"


def test_render_show_ids():
    rendered = _chain_graph().render(show_ids=True)
    assert rendered == (
        "START -> alpha (a)\nalpha (a) -> beta (b)\nbeta (b) -> END"
    )


def test_render_routing_candidates():
    graph = Graph()
    graph.add_node("a", _node(name="first"))
    graph.add_node("b", _node(name="beta"))
    graph.add_node("c", _node(name="gamma"))
    graph.add_edge("e1", "a", ["b", "c"], routing_prompt="pick")
    assert graph.render() == "first ->? (beta | gamma)"
    assert graph.render(show_ids=True) == "first (a) ->? (beta (b) | gamma (c))"


def test_render_descriptions_and_prompts_blocks():
    graph = Graph()
    graph.add_node("a", _node(name="alpha", prompt="Do {topic}", params={"topic": str}))
    out = graph.render(show_descriptions=True, show_prompts=True)
    assert "alpha" in out
    assert "description: description" in out
    assert "prompt: Do {topic}" in out


def test_render_empty_graph():
    assert Graph().render() == ""


# ----------------------------------------------------------- nodes_by_name


def test_nodes_by_name_single_and_multiple_and_absent():
    graph = Graph()
    graph.add_node("a", _node(name="dup"))
    graph.add_node("b", _node(name="dup"))
    graph.add_node("c", _node(name="solo"))
    views = graph.nodes_by_name("dup")
    assert [v.id for v in views] == ["a", "b"]
    assert graph.nodes_by_name("nope") == []


def test_nodes_by_name_view_fields():
    graph = Graph()
    graph.add_node(
        "a",
        _node(name="alpha", prompt="T {topic}", params={"topic": str}),
    )
    view = graph.nodes_by_name("alpha")
    assert len(view) == 1
    assert isinstance(view[0], NodeView)
    assert view[0].id == "a"
    assert view[0].params == {"topic": str}
    assert view[0].writes == {"result": "response"}