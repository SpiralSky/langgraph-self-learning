"""Unit tests for ``NodeCollection`` — in-memory store + ephemeral vector index.

Timing-sensitive tests patch the module-level ``datetime`` with a fake
``now()`` so ``created_at`` ordering is deterministic; everything else runs
against the real clock.

The default fastembed-backed embedder must NEVER be constructed here (no
network): the ``no_default_embedder`` autouse fixture replaces
``get_default_embedder`` with a hard failure, and every test that needs
embeddings injects a ``FakeEmbedder``.
"""

from datetime import UTC, datetime

import math

import pytest

import graphs.api.collection as collection_mod
from graphs.api.collection import NodeCollection
from graphs.api.stats import OnlineStats
from graphs.graph import END, START, Graph
from graphs.nodes.graph_node import GraphNode
from graphs.nodes.text_node import TextNode


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


class BareNode:
    """Minimal GraphNode stand-in with a controllable ``name``."""

    def __init__(self, name, description="d", prompt="p"):
        self.name = name
        self.description = description
        self.prompt = prompt
        self.run_counts = 0
        self.output_tokens = OnlineStats()
        self.output_time = OnlineStats()

    def get_node(self, llm):
        raise NotImplementedError


class FakeDatetime:
    """Replacement ``datetime`` with a monotonic, controllable ``now()``."""

    tick = 1_600_000_000

    @classmethod
    def now(cls, tz=UTC):
        cls.tick += 1
        return datetime.fromtimestamp(cls.tick, tz)


@pytest.fixture
def clock(monkeypatch):
    """Expose the fake clock and restore the real one after the test."""
    import graphs.api.collection as mod

    FakeDatetime.tick = 1_600_000_000
    monkeypatch.setattr(mod, "datetime", FakeDatetime)
    return FakeDatetime


def add(coll, name, *, pinned=False, use=0, description="d"):
    """Add a node by name; bump ``use_counts`` via ``get`` if asked."""
    node_id = coll.add(TextNode(name, description, "p"), pinned=pinned)
    for _ in range(use):
        coll.get(node_id)
    return node_id


def names(records):
    return [r.name for r in records]


def test_add_returns_str_id_distinct_per_add():
    coll = NodeCollection()
    one = coll.add(TextNode("n", "d", "p"))
    two = coll.add(TextNode("n", "d", "p"))
    assert isinstance(one, str) and one
    assert one != two
    assert len(coll) == 2


def test_get_returns_reference_and_raises_on_unknown():
    coll = NodeCollection()
    node = TextNode("n", "d", "p")
    node_id = coll.add(node)
    gotten = coll.get(node_id)
    assert gotten is node
    assert coll.get(node_id) is node
    assert coll.get(node_id).name == "n"
    with pytest.raises(KeyError):
        coll.get("missing")


def test_use_counts_bump_only_on_get():
    coll = NodeCollection()
    node_id = coll.add(TextNode("n", "d", "p"))
    first = coll.records()[0]
    assert first.use_counts == 0
    assert first.last_used_at is None

    coll.get(node_id)
    bumped = coll.records()[0]
    assert bumped.use_counts == 1
    assert bumped.last_used_at is not None
    assert bumped.last_used_at.tzinfo == UTC

    coll.get(node_id)
    assert coll.records()[0].use_counts == 2

    coll.search(query="n")
    coll.get_by_name("n")
    assert coll.records()[0].use_counts == 2


def test_get_by_name_insertion_order_and_non_bumping():
    coll = NodeCollection()
    first = coll.add(TextNode("dup", "d1", "p"))
    second = coll.add(TextNode("dup", "d2", "p"))
    coll.add(TextNode("other", "d3", "p"))
    coll.get(first)
    coll.get(second)

    recs = coll.get_by_name("dup")
    assert names(recs) == ["dup", "dup"]
    assert [r.id for r in recs] == [first, second]
    assert [r.description for r in recs] == ["d1", "d2"]
    assert [r.use_counts for r in recs] == [1, 1]
    assert coll.get_by_name("missing") == []


def test_search_keyword_matches_name_or_description_case_insensitive():
    coll = NodeCollection()
    coll.add(TextNode("AlphaBot", "summarizes things", "p"))
    coll.add(TextNode("beta", "handles ALPHA channel", "p"))

    assert sorted(names(coll.search(query="ALP"))) == ["AlphaBot", "beta"]
    assert sorted(names(coll.search(query="alp"))) == ["AlphaBot", "beta"]
    assert names(coll.search(query="summar")) == ["AlphaBot"]
    assert names(coll.search(query="channel")) == ["beta"]
    assert sorted(names(coll.search(query="alpha"))) == ["AlphaBot", "beta"]
    assert coll.search(query="xyzzy") == []


def test_search_within_newest_narrows_candidates(clock):
    coll = NodeCollection()
    for name in ("a", "b", "c", "d"):
        add(coll, name)
    assert names(coll.search(within_newest=2)) == ["d", "c"]
    assert names(coll.search(sort_by="newest", within_newest=3)) == ["d", "c", "b"]


def test_search_sorts_by_use_then_newest(clock):
    coll = NodeCollection()
    a = add(coll, "a")
    b = add(coll, "b")
    add(coll, "c")
    for _ in range(2):
        coll.get(a)
    coll.get(b)
    assert names(coll.search()) == ["a", "b", "c"]

    d = add(coll, "d")
    for _ in range(2):
        coll.get(d)
    assert names(coll.search()) == ["d", "a", "b", "c"]


def test_search_sorts_by_newest(clock):
    coll = NodeCollection()
    add(coll, "a")
    add(coll, "b")
    add(coll, "c")
    assert names(coll.search(sort_by="newest")) == ["c", "b", "a"]


def test_search_validation():
    coll = NodeCollection()
    coll.add(TextNode("n", "d", "p"))
    with pytest.raises(ValueError):
        coll.search(limit=0)
    with pytest.raises(ValueError):
        coll.search(sort_by="bogus")
    with pytest.raises(ValueError):
        coll.search(within_newest=0)


def test_top_by_use_and_newest_wrappers_match_search(clock):
    coll = NodeCollection()
    add(coll, "a")
    b = add(coll, "b")
    c = add(coll, "c")
    coll.get(b)
    coll.get(c)
    coll.get(c)

    assert names(coll.top_by_use(2)) == names(coll.search(sort_by="use", limit=2))
    assert names(coll.newest(2)) == names(coll.search(sort_by="newest", limit=2))
    assert names(coll.top_by_use(2)) == ["c", "b"]
    with pytest.raises(ValueError):
        coll.top_by_use(0)


def test_auto_prune_trims_to_prune_to():
    coll = NodeCollection(max_nodes=5, prune_to=3)
    for name in ("n1", "n2", "n3", "n4", "n5"):
        add(coll, name)
    for _ in range(2):
        coll.get(next(r for r in coll.records() if r.name == "n1").id)
    coll.get(next(r for r in coll.records() if r.name == "n2").id)
    assert len(coll) == 5

    add(coll, "n6")
    assert set(names(coll.records())) == {"n1", "n2", "n6"}
    assert len(coll) == 3

    add(coll, "n7")
    assert set(names(coll.records())) == {"n1", "n2", "n6", "n7"}

    by_name = {r.name: r.use_counts for r in coll.records()}
    assert by_name["n1"] == 2
    assert by_name["n2"] == 1
    assert by_name["n6"] == 0


def test_pinned_excluded_from_pruning():
    coll = NodeCollection(max_nodes=5, prune_to=3)
    for name in ("p1", "p2", "p3"):
        add(coll, name, pinned=True)
    for name in ("n1", "n2", "n3", "n4", "n5"):
        add(coll, name)
    assert len(coll) == 8

    add(coll, "n6")
    assert len(coll) == 6
    record_names = set(names(coll.records()))
    assert {"p1", "p2", "p3"} <= record_names
    assert {"n4", "n5", "n6"} <= record_names
    assert not {"n1", "n2", "n3"} & record_names


def test_pin_unpin_flip_flag_and_control_eviction():
    coll = NodeCollection(max_nodes=5, prune_to=3)
    p = coll.add(TextNode("p", "d", "p"))
    coll.pin(p)
    for name in ("n1", "n2", "n3", "n4", "n5"):
        add(coll, name)
    add(coll, "n6")
    assert len(coll) == 4
    assert "p" in set(names(coll.records()))
    assert next(r for r in coll.records() if r.id == p).pinned is True

    coll.unpin(p)
    assert next(r for r in coll.records() if r.id == p).pinned is False

    add(coll, "n7")
    add(coll, "n8")
    remaining = set(names(coll.records()))
    assert "p" not in remaining
    assert remaining == {"n6", "n7", "n8"}

    with pytest.raises(KeyError):
        coll.pin("missing")
    with pytest.raises(KeyError):
        coll.unpin("missing")


def test_remove_drops_entry_even_when_pinned():
    coll = NodeCollection()
    node_id = coll.add(TextNode("n", "d", "p"))
    coll.remove(node_id)
    assert node_id not in coll
    with pytest.raises(KeyError):
        coll.get(node_id)
    with pytest.raises(KeyError):
        coll.remove(node_id)

    pinned_id = coll.add(TextNode("q", "d", "p"), pinned=True)
    coll.remove(pinned_id)
    assert pinned_id not in coll


def test_snapshot_deep_copies_prototypes():
    coll = NodeCollection()
    node = TextNode("n", "d", "p")
    node_id = coll.add(node)
    snap = coll.snapshot()
    assert len(snap) == 1
    assert snap[0] is not node
    assert snap[0].name == "n"
    snap[0].description = "mutated"
    assert coll.get(node_id).description == "d"


def test_records_expose_correct_fields_in_insertion_order():
    coll = NodeCollection()
    coll.add(TextNode("a", "da", "p"), pinned=True)
    second = coll.add(TextNode("b", "db", "p"))
    coll.get(second)

    recs = coll.records()
    assert names(recs) == ["a", "b"]
    first, second_rec = recs
    assert first.pinned is True
    assert second_rec.pinned is False
    assert second_rec.id == second
    assert second_rec.name == "b"
    assert second_rec.description == "db"
    assert second_rec.use_counts == 1
    assert second_rec.last_used_at is not None
    assert first.created_at <= second_rec.created_at


def test_len_and_contains():
    coll = NodeCollection()
    node_id = coll.add(TextNode("n", "d", "p"))
    assert len(coll) == 1
    assert node_id in coll
    assert "missing" not in coll
    assert coll


def test_constructor_validation():
    with pytest.raises(ValueError):
        NodeCollection(max_nodes=5, prune_to=10)
    with pytest.raises(ValueError):
        NodeCollection(prune_to=0)
    with pytest.raises(ValueError):
        NodeCollection(max_nodes=0)
    assert isinstance(NodeCollection(), NodeCollection)


def test_add_validation_requires_non_empty_name():
    coll = NodeCollection()
    with pytest.raises(ValueError):
        coll.add(BareNode(""))
    with pytest.raises(ValueError):
        coll.add(BareNode(None))
    assert coll.add(BareNode("ok")) is not None


# ---------------------------------------------------------------------------
# semantic_search (vector index) — injected FakeEmbedder, no network.
# ---------------------------------------------------------------------------

VX = [1.0, 0.0, 0.0, 0.0]
VAB = [0.7071067811865476, 0.7071067811865476, 0.0, 0.0]
VY = [0.0, 1.0, 0.0, 0.0]
VZ = [0.0, 0.0, 1.0, 0.0]


def txt(name, desc="d"):
    return f"{name}\n{desc}"


class FakeEmbedder:
    """Maps embedded texts to fixed unit vectors for controlled ranking."""

    def __init__(self, table):
        self.table = table

    def __call__(self, text):
        try:
            return self.table[text]
        except KeyError:
            raise KeyError(f"no embedding for {text!r}") from None


def test_semantic_search_returns_most_similar_first():
    table = {txt("alpha"): VX, txt("beta"): VAB, txt("gamma"): VY, "q": VX}
    coll = NodeCollection(embedder=FakeEmbedder(table))
    for name in ("alpha", "beta", "gamma"):
        coll.add(TextNode(name, "d", "p"))

    recs = coll.semantic_search("q", limit=3)
    assert [r.name for r in recs] == ["alpha", "beta", "gamma"]


def test_semantic_search_ranks_by_cosine_similarity():
    table = {txt("a"): VX, txt("b"): VAB, txt("d"): VZ, "q": VAB}
    coll = NodeCollection(embedder=FakeEmbedder(table))
    for name in ("a", "b", "d"):
        coll.add(TextNode(name, "d", "p"))

    recs = coll.semantic_search("q", limit=3)
    assert [r.name for r in recs] == ["b", "a", "d"]


def test_semantic_search_limit_and_validation():
    table = {txt("a"): VX, txt("b"): VY, txt("c"): VZ, "q": VX}
    coll = NodeCollection(embedder=FakeEmbedder(table))
    for name in ("a", "b", "c"):
        coll.add(TextNode(name, "d", "p"))

    assert len(coll.semantic_search("q", limit=2)) == 2
    with pytest.raises(ValueError):
        coll.semantic_search("q", limit=0)
    with pytest.raises(ValueError):
        coll.semantic_search("q", within_newest=0)
    with pytest.raises(ValueError):
        coll.semantic_search("q", sort_by="bogus")


def test_semantic_search_within_newest_narrows_candidates(clock):
    table = {txt("old"): VX, txt("mid"): VAB, txt("new"): VY, "q": VX}
    coll = NodeCollection(embedder=FakeEmbedder(table))
    coll.add(TextNode("old", "d", "p"))
    coll.add(TextNode("mid", "d", "p"))
    coll.add(TextNode("new", "d", "p"))

    assert [r.name for r in coll.semantic_search("q", limit=3)] == [
        "old",
        "mid",
        "new",
    ]
    narrowed = coll.semantic_search("q", limit=3, within_newest=2)
    assert [r.name for r in narrowed] == ["mid", "new"]


def test_semantic_search_sort_by_use_newest_similarity(clock):
    table = {txt("a"): VX, txt("b"): VAB, txt("c"): VZ, "q": VAB}
    coll = NodeCollection(embedder=FakeEmbedder(table))
    a = coll.add(TextNode("a", "d", "p"))
    b = coll.add(TextNode("b", "d", "p"))
    coll.add(TextNode("c", "d", "p"))
    coll.get(a)
    coll.get(a)
    coll.get(b)

    by_sim = [r.name for r in coll.semantic_search("q", limit=3, sort_by="similarity")]
    assert by_sim == ["b", "a", "c"]
    by_use = [r.name for r in coll.semantic_search("q", limit=3, sort_by="use")]
    assert by_use == ["a", "b", "c"]
    newest = [r.name for r in coll.semantic_search("q", limit=3, sort_by="newest")]
    assert newest == ["c", "b", "a"]


def test_semantic_search_add_builds_vector_index():
    table = {txt("a"): VX, txt("b"): VY, txt("c"): VZ, "q": VX}
    coll = NodeCollection(embedder=FakeEmbedder(table))
    for name in ("a", "b", "c"):
        coll.add(TextNode(name, "d", "p"))

    assert coll._vector_collection.count() == len(coll) == 3


def test_semantic_search_remove_deletes_vector():
    table = {txt("alpha"): VX, txt("beta"): VY, txt("gamma"): VZ, "q": VX}
    coll = NodeCollection(embedder=FakeEmbedder(table))
    coll.add(TextNode("alpha", "d", "p"))
    beta = coll.add(TextNode("beta", "d", "p"))
    coll.add(TextNode("gamma", "d", "p"))

    coll.remove(beta)
    assert coll._vector_collection.count() == 2
    names = [r.name for r in coll.semantic_search("q", limit=5)]
    assert "alpha" in names and "beta" not in names


def test_semantic_search_prune_deletes_vectors(clock):
    table = {txt(f"n{i}"): VX for i in range(1, 7)}
    table["q"] = VX
    coll = NodeCollection(max_nodes=5, prune_to=3, embedder=FakeEmbedder(table))
    n1 = coll.add(TextNode("n1", "d", "p"))
    for i in (2, 3, 4, 5):
        coll.add(TextNode(f"n{i}", "d", "p"))
    coll.get(n1)
    coll.get(n1)
    coll.add(TextNode("n6", "d", "p"))

    assert len(coll) == 3
    assert coll._vector_collection.count() == 3
    names = {r.name for r in coll.semantic_search("q", limit=10)}
    assert names == {"n1", "n5", "n6"}


def test_semantic_search_embedder_failure_degrades():
    table = {txt("good"): VX, "q": VX}
    coll = NodeCollection(embedder=FakeEmbedder(table))
    coll.add(TextNode("good", "d", "p"))
    coll.add(TextNode("bad", "d", "p"))

    assert len(coll) == 2
    assert {r.name for r in coll.records()} == {"good", "bad"}
    assert [r.name for r in coll.semantic_search("q", limit=5)] == ["good"]
    assert [r.name for r in coll.search(query="bad")] == ["bad"]


def test_semantic_search_never_bumps_use_counts(clock):
    table = {txt("a"): VX, "q": VX}
    coll = NodeCollection(embedder=FakeEmbedder(table))
    a = coll.add(TextNode("a", "d", "p"))

    before = coll.records()[0]
    coll.semantic_search("q", limit=3)
    coll.semantic_search("q", limit=3, sort_by="use")
    after = coll.records()[0]
    assert after.id == a
    assert before.use_counts == after.use_counts == 0
    assert before.last_used_at is after.last_used_at is None


def test_semantic_search_empty_collection():
    coll = NodeCollection()
    assert coll.semantic_search("q") == []


def test_default_embedder_resolved_lazily_never_in_tests(monkeypatch):
    called = []

    def spy():
        called.append(1)
        raise AssertionError("default embedder must not be constructed")

    monkeypatch.setattr(collection_mod, "get_default_embedder", spy)
    coll = NodeCollection()
    coll.search(limit=1)
    assert called == []


def test_add_without_embedder_degrades_gracefully(clock):
    coll = NodeCollection()
    node_id = coll.add(TextNode("n", "d", "p"))
    assert node_id in coll
    assert len(coll) == 1
    assert coll.records()[0].name == "n"
    assert coll.search(query="n")[0].name == "n"


def test_graph_node_prototype_flows_through_collection():
    inner = Graph()
    inner.add_node("a", TextNode("a", "d", "Gen {user_message}",
                                params={"user_message": str}))
    inner.add_edge("e1", START, "a")
    inner.add_edge("e2", "a", END)
    sub = GraphNode("subgraph", "wraps an inner graph", "subgraph node", inner,
                    input_map={"user_message": "user_message"},
                    output_map={"response": "response"})

    table = {txt("subgraph", "wraps an inner graph"): VX, "q": VX}
    coll = NodeCollection(embedder=FakeEmbedder(table))
    nid = coll.add(sub)

    assert coll.get(nid).name == "subgraph"
    assert [r.name for r in coll.search("inner graph", limit=10)] == ["subgraph"]
    assert [r.name for r in coll.semantic_search("q", limit=10)] == ["subgraph"]
    assert [r.name for r in coll.records()] == ["subgraph"]
    assert coll.snapshot()[0].name == "subgraph"

    coll.remove(nid)
    assert nid not in coll


def test_graph_node_prototype_pinned_survives_prune_like_any_node():
    inner = Graph()
    inner.add_node("a", TextNode("a", "d", "Gen {user_message}",
                                params={"user_message": str}))
    inner.add_edge("e1", START, "a")
    inner.add_edge("e2", "a", END)
    sub = GraphNode("subgraph", "wraps an inner graph", "subgraph node", inner,
                    input_map={"user_message": "user_message"},
                    output_map={"response": "response"})

    table = {txt("subgraph", "wraps an inner graph"): VX, txt("alpha"): VY, txt("beta"): VZ, txt("gamma"): VAB}
    coll = NodeCollection(max_nodes=2, prune_to=1, embedder=FakeEmbedder(table))
    coll.add(sub, pinned=True)
    coll.add(TextNode("alpha", "d", "p"))
    coll.add(TextNode("beta", "d", "p"))
    coll.add(TextNode("gamma", "d", "p"))

    assert [r.name for r in coll.records()] == ["subgraph", "gamma"]
    assert coll.records()[0].pinned is True


# ---------------------------------------------------------------------------
# update() / replace() — granular mutation, no bump, embedding sync.
# ---------------------------------------------------------------------------


class _FailOnEmbedder(FakeEmbedder):
    """FakeEmbedder that raises ``RuntimeError`` for a specific text."""

    def __init__(self, table, fail_on):
        super().__init__(table)
        self.fail_on = fail_on

    def __call__(self, text):
        if text == self.fail_on:
            raise RuntimeError("embedder down")
        return super().__call__(text)


def test_update_inplace_prompt_only_keeps_id_and_bookkeeping_and_get_bumps(clock):
    coll = NodeCollection()
    node_id = coll.add(TextNode("n", "d", "say {topic}", params={"topic": str}))
    coll.pin(node_id)
    coll.get(node_id)
    before = next(r for r in coll.records())

    coll.update_inplace(node_id, prompt="say it loudly {topic}")

    after = next(r for r in coll.records())
    assert after.id == node_id
    assert after.name == "n"
    assert after.description == "d"
    assert after.pinned is True
    assert after.use_counts == before.use_counts == 1
    assert after.created_at == before.created_at
    assert after.last_used_at == before.last_used_at

    coll.get(node_id)
    assert coll.records()[0].use_counts == 2


def test_update_inplace_never_bumps(clock):
    coll = NodeCollection()
    node_id = coll.add(TextNode("n", "d", "p"))
    for _ in range(3):
        coll.update_inplace(node_id, description="d2")
    rec = coll.records()[0]
    assert rec.use_counts == 0
    assert rec.last_used_at is None


def test_update_inplace_name_description_change_reembeds():
    table = {txt("n", "d"): VX, txt("n", "d2"): VY, "q": VY}
    coll = NodeCollection(embedder=FakeEmbedder(table))
    node_id = coll.add(TextNode("n", "d", "p"))

    coll.update_inplace(node_id, description="d2")

    assert coll._vector_collection.count() == 1
    assert [r.name for r in coll.semantic_search("q", limit=5)] == ["n"]


def test_update_inplace_params_writes_change_revalidates():
    coll = NodeCollection()
    node_id = coll.add(TextNode("n", "d", "p"))

    coll.update_inplace(
        node_id,
        params={"topic": str},
        prompt="Do {topic}",
        writes={"result": "response"},
    )

    updated = coll.snapshot()[0]
    assert updated.prompt == "Do {topic}"
    assert updated.params == {"topic": str}
    assert updated.writes == {"result": "response"}


def test_update_inplace_orphan_placeholder_raises_and_entry_unchanged():
    coll = NodeCollection()
    node_id = coll.add(TextNode("n", "d", "say {topic}", params={"topic": str}))

    with pytest.raises(ValueError, match="not declared in params"):
        coll.update_inplace(node_id, prompt="Do {missing}", params={"topic": str})

    untouched = coll.snapshot()[0]
    assert untouched.prompt == "say {topic}"
    assert untouched.params == {"topic": str}


def test_update_inplace_empty_name_raises_and_entry_unchanged():
    coll = NodeCollection()
    node_id = coll.add(TextNode("n", "d", "p"))

    with pytest.raises(ValueError, match="non-empty"):
        coll.update_inplace(node_id, name="")

    assert coll.snapshot()[0].name == "n"


def test_update_inplace_and_replace_unknown_id_raise_key_error():
    coll = NodeCollection()
    with pytest.raises(KeyError):
        coll.update_inplace("missing", name="x")
    with pytest.raises(KeyError):
        coll.replace("missing", TextNode("x", "d", "p"))


def test_update_inplace_requires_updated_method():
    coll = NodeCollection()
    node_id = coll.add(BareNode("n"))
    with pytest.raises(TypeError, match="replace()"):
        coll.update_inplace(node_id, description="x")


def test_update_inplace_bogus_kwarg_raises_type_error():
    coll = NodeCollection()
    node_id = coll.add(TextNode("n", "d", "p"))
    with pytest.raises(TypeError):
        coll.update_inplace(node_id, bogus=1)


def test_update_inplace_zero_kwargs_is_noop(clock):
    coll = NodeCollection()
    node_id = coll.add(TextNode("n", "d", "p"))
    coll.update_inplace(node_id)
    rec = coll.records()[0]
    assert rec.id == node_id
    assert rec.use_counts == 0
    assert rec.last_used_at is None


def test_update_inplace_reembeds_previously_unembedded_entry():
    table = {txt("n", "d2"): VY, "q": VY}
    coll = NodeCollection(embedder=FakeEmbedder(table))
    node_id = coll.add(TextNode("n", "d", "p"))
    assert coll._vector_collection.count() == 0

    coll.update_inplace(node_id, description="d2")

    assert coll._vector_collection.count() == 1
    assert [r.name for r in coll.semantic_search("q", limit=5)] == ["n"]


def test_update_inplace_embedder_failure_keeps_change_and_degrades():
    table = {txt("n", "d"): VX, "q": VX}
    coll = NodeCollection(embedder=_FailOnEmbedder(table, fail_on=txt("n", "d2")))
    node_id = coll.add(TextNode("n", "d", "p"))
    assert coll._vector_collection.count() == 1

    coll.update_inplace(node_id, description="d2")

    assert coll.snapshot()[0].description == "d2"
    assert coll._vector_collection.count() == 0
    assert [r.name for r in coll.semantic_search("q", limit=5)] == []


def test_replace_preserves_bookkeeping(clock):
    coll = NodeCollection()
    node_id = coll.add(TextNode("n", "d", "p"))
    coll.pin(node_id)
    coll.get(node_id)
    before = next(r for r in coll.records())

    coll.replace(node_id, TextNode("replacement", "d2", "p2"))

    after = next(r for r in coll.records())
    assert after.id == node_id
    assert after.name == "replacement"
    assert after.description == "d2"
    assert after.pinned is True
    assert after.use_counts == before.use_counts == 1
    assert after.created_at == before.created_at
    assert after.last_used_at == before.last_used_at
    assert coll.snapshot()[0].name == "replacement"


def test_replace_reembeds_when_text_changes_and_keeps_vector_when_same():
    table = {txt("n", "d"): VX, txt("n", "d2"): VY, "q": VY}
    coll = NodeCollection(embedder=FakeEmbedder(table))
    first = coll.add(TextNode("n", "d", "old prompt"))
    coll.replace(first, TextNode("n", "d2", "new prompt"))
    assert coll._vector_collection.count() == 1
    assert [r.name for r in coll.semantic_search("q", limit=5)] == ["n"]

    coll.replace(first, TextNode("n", "d2", "other prompt"))
    assert coll._vector_collection.count() == 1


def test_replace_reembeds_previously_unembedded_entry():
    table = {txt("n", "newd"): VY, "q": VY}
    coll = NodeCollection(embedder=FakeEmbedder(table))
    node_id = coll.add(BareNode("n", description="oldd"))
    assert coll._vector_collection.count() == 0

    coll.replace(node_id, BareNode("n", description="newd"))

    assert coll._vector_collection.count() == 1
    assert [r.name for r in coll.semantic_search("q", limit=5)] == ["n"]


def test_replace_never_bumps(clock):
    coll = NodeCollection()
    node_id = coll.add(TextNode("n", "d", "p"))
    for _ in range(3):
        coll.replace(node_id, TextNode("x", "y", "z"))
    rec = coll.records()[0]
    assert rec.use_counts == 0
    assert rec.last_used_at is None


def test_replace_validates_empty_name():
    coll = NodeCollection()
    node_id = coll.add(TextNode("n", "d", "p"))
    with pytest.raises(ValueError, match="node.name"):
        coll.replace(node_id, BareNode(""))
    assert coll.snapshot()[0].name == "n"


# ---------------------------------------------------------------------------
# run telemetry — OnlineStats + node-owned stats (records derive from node).
# ---------------------------------------------------------------------------


def test_online_stats_single_value_mean_and_no_std():
    stats = OnlineStats()
    stats.update(5.0)
    assert stats.count == 1
    assert stats.mean == pytest.approx(5.0)
    assert stats.std is None


def test_online_stats_known_set_mean_and_std():
    stats = OnlineStats()
    for value in range(1, 6):
        stats.update(value)
    assert stats.count == 5
    assert stats.mean == pytest.approx(3.0)
    assert stats.std == pytest.approx(math.sqrt(2.5))


def test_online_stats_incremental_matches_batch():
    incremental = OnlineStats()
    for value in (1.0, 2.0, 3.0):
        incremental.update(value)
    batch = OnlineStats()
    for value in (1.0, 2.0, 3.0):
        batch.update(value)
    assert incremental.count == batch.count
    assert incremental.mean == pytest.approx(batch.mean)
    assert incremental.std == pytest.approx(batch.std)


def test_online_stats_round_trip_restores_m2_and_continues():
    stats = OnlineStats()
    for value in (1.0, 2.0, 3.0):
        stats.update(value)
    restored = OnlineStats.from_dict(stats.to_dict())
    assert restored.count == stats.count == 3
    assert restored.mean == pytest.approx(stats.mean)
    assert restored.std == pytest.approx(stats.std)

    restored.update(4.0)
    stats.update(4.0)
    assert restored.std == pytest.approx(stats.std)


def test_online_stats_round_trip_single_value_continues():
    stats = OnlineStats()
    stats.update(7.0)
    restored = OnlineStats.from_dict(stats.to_dict())
    assert restored.count == 1
    assert restored.std is None
    restored.update(9.0)
    assert restored.std is not None


def _record_node(coll, node_id, *, tokens=None, elapsed_seconds=None):
    """Record a run on the stored node via the reference getter."""
    coll.get(node_id).record_run(tokens=tokens, elapsed_seconds=elapsed_seconds)


def test_records_derive_run_stats_from_stored_node():
    coll = NodeCollection()
    node_id = coll.add(TextNode("n", "d", "p"))
    _record_node(coll, node_id, tokens=10, elapsed_seconds=0.5)
    _record_node(coll, node_id, tokens=20, elapsed_seconds=1.5)

    rec = coll.records()[0]
    assert rec.run_counts == 2
    assert rec.tokens_count == 2
    assert rec.tokens_mean == pytest.approx(15.0)
    assert rec.tokens_std == pytest.approx(math.sqrt(50.0))
    assert rec.time_count == 2
    assert rec.time_mean == pytest.approx(1.0)
    assert rec.time_std == pytest.approx(math.sqrt(0.5))


def test_records_reflect_none_metrics_skip_that_stat():
    coll = NodeCollection()
    node_id = coll.add(TextNode("n", "d", "p"))
    _record_node(coll, node_id)
    _record_node(coll, node_id, tokens=5)
    _record_node(coll, node_id, elapsed_seconds=2.0)

    rec = coll.records()[0]
    assert rec.run_counts == 3
    assert rec.tokens_count == 1
    assert rec.tokens_mean == pytest.approx(5.0)
    assert rec.time_count == 1
    assert rec.time_mean == pytest.approx(2.0)


def test_records_derive_empty_stats_by_default():
    coll = NodeCollection()
    node_id = coll.add(TextNode("n", "d", "p"))
    rec = coll.records()[0]
    assert rec.id == node_id
    assert rec.run_counts == 0
    assert rec.tokens_count == 0
    assert rec.tokens_mean is None
    assert rec.time_count == 0
    assert rec.time_mean is None


def test_get_search_semantic_never_mutate_node_stats():
    table = {txt("n", "d"): VX, "q": VX}
    coll = NodeCollection(embedder=FakeEmbedder(table))
    node_id = coll.add(TextNode("n", "d", "p"))
    _record_node(coll, node_id, tokens=10, elapsed_seconds=1.0)

    coll.search(query="n")
    coll.get_by_name("n")
    coll.semantic_search("q", limit=5)

    rec = coll.records()[0]
    assert rec.use_counts == 1
    assert rec.run_counts == 1
    assert rec.tokens_count == 1
    assert rec.time_count == 1
    assert rec.tokens_mean == pytest.approx(10.0)


def test_update_inplace_preserves_node_stats():
    coll = NodeCollection()
    node_id = coll.add(TextNode("n", "d", "p"))
    _record_node(coll, node_id, tokens=10, elapsed_seconds=1.0)

    coll.update_inplace(node_id, description="d2")
    node = coll.get(node_id)

    rec = coll.records()[0]
    assert rec.description == "d2"
    assert rec.run_counts == 1
    assert rec.tokens_count == 1
    assert rec.time_count == 1
    assert rec.tokens_mean == pytest.approx(10.0)
    assert rec.time_mean == pytest.approx(1.0)

    node.record_run(tokens=30, elapsed_seconds=0.5)
    assert coll.records()[0].run_counts == 2


def test_update_inplace_reset_stats_zeroes_node_stats():
    coll = NodeCollection()
    node_id = coll.add(TextNode("n", "d", "p"))
    _record_node(coll, node_id, tokens=10, elapsed_seconds=1.0)

    coll.update_inplace(node_id, description="d2", reset_stats=True)

    rec = coll.records()[0]
    assert rec.description == "d2"
    assert rec.run_counts == 0
    assert rec.tokens_count == 0
    assert rec.tokens_mean is None
    assert rec.time_count == 0
    assert rec.time_mean is None


def test_edit_returns_validated_copy_and_leaves_store_untouched():
    coll = NodeCollection()
    node_id = coll.add(TextNode("n", "d", "say {topic}", params={"topic": str}))
    _record_node(coll, node_id, tokens=7, elapsed_seconds=0.4)

    copy = coll.edit(node_id, description="d2", prompt="Do {topic}")

    assert copy.name == "n"
    assert copy.description == "d2"
    assert copy.prompt == "Do {topic}"
    assert copy.params == {"topic": str}
    assert copy.run_counts == 1
    assert copy.output_tokens.mean == pytest.approx(7.0)
    assert copy is not coll.get(node_id)
    rec = coll.records()[0]
    assert rec.description == "d"
    assert coll.snapshot()[0].prompt == "say {topic}"
    assert rec.run_counts == 1

    coll.replace(node_id, copy)
    assert coll.records()[0].description == "d2"


def test_edit_unknown_id_and_missing_updated_raise():
    coll = NodeCollection()
    with pytest.raises(KeyError):
        coll.edit("missing", name="x")
    node_id = coll.add(BareNode("n"))
    with pytest.raises(TypeError, match="replace()"):
        coll.edit(node_id, description="x")


def test_replace_swaps_incoming_nodes_own_stats():
    coll = NodeCollection()
    node_id = coll.add(TextNode("n", "d", "p"))
    _record_node(coll, node_id, tokens=10, elapsed_seconds=1.0)

    replacement = TextNode("renamed", "d2", "p2")
    replacement.record_run(tokens=5, elapsed_seconds=0.25)
    coll.replace(node_id, replacement)

    rec = coll.records()[0]
    assert rec.name == "renamed"
    assert rec.run_counts == 1
    assert rec.tokens_count == 1
    assert rec.tokens_mean == pytest.approx(5.0)
    assert rec.time_mean == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# reference get + use/recency bookkeeping.
# ---------------------------------------------------------------------------


def test_get_returns_same_object_across_calls():
    coll = NodeCollection()
    node_id = coll.add(TextNode("n", "d", "p"))
    assert coll.get(node_id) is coll.get(node_id)


def test_get_still_bumps_use_and_recency_not_run_stats():
    coll = NodeCollection()
    node_id = coll.add(TextNode("n", "d", "p"))
    rec = coll.records()[0]
    assert rec.use_counts == 0
    assert rec.last_used_at is None

    coll.get(node_id)
    coll.get(node_id)

    bumped = coll.records()[0]
    assert bumped.use_counts == 2
    assert bumped.last_used_at is not None
    assert bumped.run_counts == 0


def test_get_unknown_id_raises_key_error():
    coll = NodeCollection()
    with pytest.raises(KeyError):
        coll.get("missing")