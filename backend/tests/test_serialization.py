"""Unit tests for JSON serialization + the ``backend/data`` file layer.

Covers the annotation <-> name registry, type-tagged node dispatch, ``Graph``
state-model round-trips (default and dynamic), and the storage helpers.
Pure-unit: no LLM, no network — the default fastembed embedder must never be
constructed, so the autouse ``no_default_embedder`` fixture replaces it with a
hard failure and every collection test injects a dummy embedder.
"""

import pytest
from pydantic import create_model

import graphs.api.collection as collection_mod
import graphs.storage as storage_mod
from graphs.api.collection import NodeCollection
from graphs.connections import Connection, RoutingConnection
from graphs.graph import END, START, Graph
from graphs.nodes.graph_node import GraphNode
from graphs.nodes.text_node import TextNode
from graphs.serialization import (
    connection_from_dict,
    deserialize_annotation,
    node_from_dict,
    serialize_annotation,
)
from graphs.state import LearningGraphState


class RaisingDefaultEmbedder:
    """Replacement ``get_default_embedder`` that fails loudly if called."""

    @staticmethod
    def __call__():
        raise AssertionError(
            "default fastembed embedder must not be constructed in tests"
        )


@pytest.fixture(autouse=True)
def no_default_embedder(monkeypatch):
    """Guarantee the default embedder (needs a one-time model download) is
    never constructed anywhere in this test module."""
    monkeypatch.setattr(
        collection_mod, "get_default_embedder", RaisingDefaultEmbedder()
    )


class DummyEmbedder:
    """Fixed-vector embedder so collection adds never touch fastembed."""

    def __call__(self, text):
        return [1.0, 0.0, 0.0, 0.0]


class FailingEmbedder(DummyEmbedder):
    """Dummy embedding that raises for one specific input text."""

    def __init__(self, fail_on):
        self.fail_on = fail_on

    def __call__(self, text):
        if text == self.fail_on:
            raise RuntimeError("embedder down")
        return super().__call__(text)


class BareNode:
    """Minimal GraphNode stand-in with no ``to_dict`` implementation."""

    def __init__(self, name, description="d", prompt="p"):
        self.name = name
        self.description = description
        self.prompt = prompt

    def get_node(self, llm):
        raise NotImplementedError


def _chain() -> Graph:
    graph = Graph()
    graph.add_node(
        "a", TextNode("first", "d", "Gen {user_message}", params={"user_message": str})
    )
    graph.add_node(
        "b",
        TextNode("second", "d", "Refine {user_message}", params={"user_message": str}),
    )
    graph.add_edge("e0", START, "a")
    graph.add_edge("e1", "a", "b", routing_prompt="always pick b")
    graph.add_edge("e2", "b", END)
    return graph


# ---------------------------------------------------------------------------
# annotation <-> name registry
# ---------------------------------------------------------------------------


def test_annotation_round_trip_atomic_types():
    for annotation in (str, int, float, bool, list, dict):
        assert deserialize_annotation(serialize_annotation(annotation)) is annotation


def test_annotation_generics_normalize_to_origin():
    assert serialize_annotation(list[str]) == "list"
    assert serialize_annotation(dict[str, int]) == "dict"
    assert deserialize_annotation("list") is list
    assert deserialize_annotation("dict") is dict


def test_annotation_optional_prefix_round_trips():
    for annotation, name in (
        (str | None, "optional:str"),
        (list[str] | None, "optional:list"),
        (bool | None, "optional:bool"),
    ):
        assert serialize_annotation(annotation) == name
        restored = deserialize_annotation(name)
        assert serialize_annotation(restored) == name


def test_annotation_serialize_unknown_raises():
    class Custom:
        pass

    for bad in (set[str], tuple[str, ...], Custom):
        with pytest.raises(ValueError, match="unsupported annotation"):
            serialize_annotation(bad)


def test_annotation_rejects_multi_type_union():
    with pytest.raises(ValueError, match="unsupported union"):
        serialize_annotation(str | int)


def test_annotation_deserialize_unknown_raises():
    for bad in ("", "unknown", "tuple", "optional:unknown"):
        with pytest.raises(ValueError, match="unknown serialized annotation"):
            deserialize_annotation(bad)


# ---------------------------------------------------------------------------
# node + connection dispatch
# ---------------------------------------------------------------------------


def test_node_from_dict_unknown_tag_raises():
    with pytest.raises(ValueError, match="unknown node type tag"):
        node_from_dict({"type": "nope"})


def test_text_node_round_trip():
    node = TextNode(
        "outline",
        "plan the answer",
        "Outline {topic} with {others}",
        params={"topic": str | None, "others": list[str]},
        writes={"result": "response"},
    )
    data = node.to_dict()
    assert data["type"] == "text"
    assert data["params"] == {"topic": "optional:str", "others": "list"}

    restored = TextNode.from_dict(data)
    assert isinstance(restored, TextNode)
    assert restored.name == node.name
    assert restored.description == node.description
    assert restored.prompt == node.prompt
    assert restored.params["topic"] == str | None
    assert restored.params["others"] is list
    assert restored.writes == node.writes
    assert restored.to_dict()["params"] == data["params"]


def test_node_from_dict_dispatch_rebuilds_via_registry():
    restored = node_from_dict(TextNode("n", "d", "p").to_dict())
    assert isinstance(restored, TextNode)
    assert restored.name == "n"


def test_connection_dict_round_trip():
    standard = Connection(source="a", target="b")
    restored_standard = connection_from_dict(standard.as_dict())
    assert restored_standard == standard
    assert isinstance(restored_standard, Connection)

    routing = RoutingConnection(
        source="a", targets=("b", "c"), routing_prompt="pick one", model=object()
    )
    data = routing.as_dict()
    assert "model" not in data
    restored_routing = connection_from_dict(data)
    assert isinstance(restored_routing, RoutingConnection)
    assert restored_routing == RoutingConnection(
        source="a", targets=("b", "c"), routing_prompt="pick one"
    )
    assert restored_routing.model is None


# ---------------------------------------------------------------------------
# Graph round-trips
# ---------------------------------------------------------------------------


def test_graph_round_trip_default_state_model():
    graph = _chain()
    data = graph.to_dict()
    assert data["type"] == "graph"
    assert data["state_model"] == {"kind": "LearningGraphState"}

    restored = Graph.from_dict(data)
    assert restored.state_model is LearningGraphState
    assert restored.render(show_ids=True) == graph.render(show_ids=True)
    assert restored.get_edge("e1") == RoutingConnection(
        source="a", targets=("b",), routing_prompt="always pick b"
    )
    assert restored.get_edge("e0") == Connection(source=START, target="a")


def test_graph_round_trip_dynamic_state_model():
    dynamic = create_model(
        "GeneratedState",
        user_message=(str | None, None),
        response=(str | None, None),
        notes=(list[str] | None, None),
    )
    graph = Graph(state_model=dynamic)
    graph.add_node(
        "n", TextNode("n", "d", "Do {user_message}", params={"user_message": str})
    )
    graph.add_edge("e1", START, "n")
    graph.add_edge("e2", "n", END)

    data = graph.to_dict()
    assert data["state_model"]["kind"] == "dynamic"
    assert data["state_model"]["fields"] == {
        "user_message": "optional:str",
        "response": "optional:str",
        "notes": "optional:list",
    }

    restored = Graph.from_dict(data)
    assert restored.state_model is not dynamic
    assert set(restored.state_model.model_fields) == {
        "user_message",
        "response",
        "notes",
    }
    assert restored.render(show_ids=True) == graph.render(show_ids=True)
    assert restored.to_dict()["state_model"] == data["state_model"]


def test_graph_from_dict_unknown_state_kind_raises():
    with pytest.raises(ValueError, match="unknown state model kind"):
        Graph.from_dict({"state_model": {"kind": "bogus"}})


def test_graph_node_round_trip():
    sub = GraphNode(
        "subgraph",
        "wraps an inner chain",
        "subgraph node",
        _chain(),
        input_map={"user_message": "user_message"},
        output_map={"response": "response"},
    )
    data = sub.to_dict()
    assert data["type"] == "graph"
    assert data["input_map"] == {"user_message": "user_message"}
    assert data["output_map"] == {"response": "response"}

    restored = GraphNode.from_dict(data)
    assert isinstance(restored, GraphNode)
    assert restored.name == sub.name
    assert restored.input_map == sub.input_map
    assert restored.output_map == sub.output_map
    assert restored.params == sub.params
    assert restored.writes == sub.writes
    assert restored.graph.render(show_ids=True) == sub.graph.render(show_ids=True)
    assert restored.graph.state_model is LearningGraphState


# ---------------------------------------------------------------------------
# storage helpers
# ---------------------------------------------------------------------------


def test_ensure_data_dir_creates(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    monkeypatch.setattr(storage_mod, "DATA_ROOT", data_root)
    assert storage_mod.ensure_data_dir() == data_root
    assert data_root.is_dir()
    assert storage_mod.ensure_data_dir() == data_root


def test_write_read_json_round_trip(tmp_path):
    path = tmp_path / "sub" / "obj.json"
    payload = {"nodes": [{"type": "text", "name": "n"}], "ok": True}
    storage_mod.write_json(path, payload)
    assert path.is_file()
    assert storage_mod.read_json(path) == payload


def test_write_json_is_atomic_no_tmp_leftover(tmp_path):
    path = tmp_path / "obj.json"
    storage_mod.write_json(path, {"a": 1})
    assert not (tmp_path / "obj.json.tmp").exists()
    assert storage_mod.read_json(path) == {"a": 1}


def test_read_json_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        storage_mod.read_json(tmp_path / "missing.json")


# ---------------------------------------------------------------------------
# collection dump/load
# ---------------------------------------------------------------------------


def test_collection_dump_load_round_trip(tmp_path):
    path = tmp_path / "collection.json"
    coll = NodeCollection(embedder=DummyEmbedder())
    coll.add(TextNode("alpha", "d1", "p1"))
    coll.add(TextNode("beta", "d2", "p2"))
    pinned_id = coll.add(TextNode("alpha", "d3", "p3"))
    coll.pin(pinned_id)
    coll.get(pinned_id)

    storage_mod.dump_collection(coll, path)

    restored = storage_mod.load_collection(path, embedder=DummyEmbedder())
    assert [r.name for r in restored.records()] == ["alpha", "beta", "alpha"]
    assert [r.description for r in restored.records()] == ["d1", "d2", "d3"]
    assert all(r.pinned is False for r in restored.records())
    assert all(r.use_counts == 0 for r in restored.records())


def test_collection_dump_load_with_graph_node(tmp_path):
    path = tmp_path / "collection.json"
    sub = GraphNode(
        "subgraph",
        "wraps an inner chain",
        "subgraph node",
        _chain(),
        input_map={"user_message": "user_message"},
        output_map={"response": "response"},
    )
    coll = NodeCollection(embedder=DummyEmbedder())
    coll.add(sub)
    coll.add(TextNode("plain", "d", "p"))

    storage_mod.dump_collection(coll, path)
    restored = storage_mod.load_collection(path, embedder=DummyEmbedder())

    assert len(restored) == 2
    rebuilt = restored.snapshot()[0]
    assert isinstance(rebuilt, GraphNode)
    assert rebuilt.name == "subgraph"
    assert rebuilt.graph.render(show_ids=True) == sub.graph.render(show_ids=True)


def test_collection_dump_rejects_non_serializable_prototype(tmp_path):
    coll = NodeCollection(embedder=DummyEmbedder())
    coll.add(BareNode("bare"))
    with pytest.raises(ValueError, match="without to_dict"):
        storage_mod.dump_collection(coll, tmp_path / "collection.json")


def test_collection_load_missing_file_returns_empty(tmp_path):
    restored = storage_mod.load_collection(
        tmp_path / "missing.json", embedder=DummyEmbedder()
    )
    assert isinstance(restored, NodeCollection)
    assert len(restored) == 0


def test_collection_load_embedding_failure_degrades(tmp_path):
    path = tmp_path / "collection.json"
    coll = NodeCollection(embedder=DummyEmbedder())
    coll.add(TextNode("good", "d", "p"))
    coll.add(TextNode("bad", "d", "p"))
    storage_mod.dump_collection(coll, path)

    restored = storage_mod.load_collection(
        path, embedder=FailingEmbedder(fail_on="bad\nd")
    )
    assert {r.name for r in restored.records()} == {"good", "bad"}
    assert [r.name for r in restored.semantic_search("q", limit=5)] == ["good"]


def test_collection_load_unknown_node_tag_raises(tmp_path):
    path = tmp_path / "collection.json"
    storage_mod.write_json(path, {"nodes": [{"type": "nope"}]})
    with pytest.raises(ValueError, match="unknown node type tag"):
        storage_mod.load_collection(path, embedder=DummyEmbedder())