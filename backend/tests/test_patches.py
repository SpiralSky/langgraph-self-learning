"""Unit tests for patch persistence + deterministic application.

Behavior patches are applied against ``tmp_path`` YAML files; node patches
route through a lightweight fake collection recording calls. No live LLM, no
network.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from graphs.behaviors import BehaviorGroup, BehaviorPoint, load_behaviors
from graphs.feedback import Patch
from graphs.nodes.text_node import TextNode
from graphs.patches import (
    append_patches,
    apply_patches,
    load_patches,
    save_behaviors,
)


class FakeCollection:
    def __init__(self) -> None:
        self.updated: list[tuple[str, dict]] = []
        self.replaced: list[tuple[str, object]] = []
        self.added: list[object] = []

    def update(self, node_id: str, **changes: object) -> None:
        self.updated.append((node_id, changes))

    def replace(self, node_id: str, new_node: object) -> None:
        self.replaced.append((node_id, new_node))

    def add(self, node: object) -> str:
        self.added.append(node)
        return "new-id"


def group(title: str, *points: tuple[str, str]) -> BehaviorGroup:
    return BehaviorGroup(
        title=title, points=[BehaviorPoint(id=key, text=text) for key, text in points]
    )


def write_behaviors(path: Path, groups: list[BehaviorGroup]) -> None:
    save_behaviors(groups, path)


TEXT_NODE = {
    "type": "text",
    "name": "Explain",
    "description": "explains things",
    "prompt": "Explain {user_message}",
    "params": {"user_message": "str"},
    "writes": {"result": "response"},
}


def test_apply_add_creates_missing_group(tmp_path):
    behaviors_path = tmp_path / "behaviors.yaml"
    patch = Patch(seq=0, action="add", title="New rules", point_key="n1", content="rule one")

    applied = apply_patches([patch], behaviors_path=behaviors_path)

    assert applied == [patch]
    assert load_behaviors(behaviors_path) == [group("New rules", ("n1", "rule one"))]


def test_apply_update_replaces_text(tmp_path):
    behaviors_path = tmp_path / "behaviors.yaml"
    write_behaviors(behaviors_path, [group("G", ("a", "old"))])

    apply_patches(
        [Patch(seq=0, action="update", title="G", point_key="a", content="new")],
        behaviors_path=behaviors_path,
    )

    assert load_behaviors(behaviors_path) == [group("G", ("a", "new"))]


def test_apply_update_missing_key_upserts(tmp_path):
    behaviors_path = tmp_path / "behaviors.yaml"
    write_behaviors(behaviors_path, [group("G", ("a", "old"))])

    apply_patches(
        [Patch(seq=0, action="update", title="G", point_key="z", content="fresh")],
        behaviors_path=behaviors_path,
    )

    assert load_behaviors(behaviors_path) == [group("G", ("a", "old"), ("z", "fresh"))]


def test_apply_update_missing_group_creates(tmp_path):
    behaviors_path = tmp_path / "behaviors.yaml"

    apply_patches(
        [Patch(seq=0, action="update", title="Brand new", point_key="k", content="v")],
        behaviors_path=behaviors_path,
    )

    assert load_behaviors(behaviors_path) == [group("Brand new", ("k", "v"))]


def test_apply_remove_deletes_point_and_empty_group(tmp_path):
    behaviors_path = tmp_path / "behaviors.yaml"
    write_behaviors(
        behaviors_path, [group("G", ("a", "x"), ("b", "y")), group("H", ("c", "z"))]
    )

    apply_patches(
        [
            Patch(seq=0, action="remove", title="G", point_key="a"),
            Patch(seq=1, action="remove", title="H", point_key="c"),
        ],
        behaviors_path=behaviors_path,
    )

    assert load_behaviors(behaviors_path) == [group("G", ("b", "y"))]


def test_apply_remove_missing_key_is_noop(tmp_path):
    behaviors_path = tmp_path / "behaviors.yaml"
    write_behaviors(behaviors_path, [group("G", ("a", "x"))])

    apply_patches(
        [Patch(seq=0, action="remove", title="G", point_key="nope")],
        behaviors_path=behaviors_path,
    )

    assert load_behaviors(behaviors_path) == [group("G", ("a", "x"))]


def test_precedence_later_seq_wins(tmp_path):
    behaviors_path = tmp_path / "behaviors.yaml"
    write_behaviors(behaviors_path, [group("G", ("a", "original"))])

    apply_patches(
        [
            Patch(seq=0, action="update", title="G", point_key="a", content="first edit"),
            Patch(seq=1, action="update", title="G", point_key="a", content="latest edit"),
        ],
        behaviors_path=behaviors_path,
    )

    assert load_behaviors(behaviors_path) == [group("G", ("a", "latest edit"))]


def test_apply_order_is_seq_not_input_order(tmp_path):
    behaviors_path = tmp_path / "behaviors.yaml"

    apply_patches(
        [
            Patch(seq=1, action="update", title="G", point_key="a", content="later"),
            Patch(seq=0, action="add", title="G", point_key="a", content="earlier"),
        ],
        behaviors_path=behaviors_path,
    )

    assert load_behaviors(behaviors_path) == [group("G", ("a", "later"))]


def test_apply_idempotent_reapply(tmp_path):
    behaviors_path = tmp_path / "behaviors.yaml"
    patches = [
        Patch(seq=0, action="add", title="G", point_key="a", content="one"),
        Patch(seq=1, action="add", title="G", point_key="b", content="two"),
        Patch(seq=2, action="remove", title="G", point_key="b"),
        Patch(seq=3, action="update", title="G", point_key="a", content="final"),
    ]

    apply_patches(patches, behaviors_path=behaviors_path)
    first = load_behaviors(behaviors_path)
    apply_patches(patches, behaviors_path=behaviors_path)

    assert first == [group("G", ("a", "final"))]
    assert load_behaviors(behaviors_path) == first


def test_apply_node_field_edit_calls_update(tmp_path):
    collection = FakeCollection()
    patch = Patch(
        seq=0,
        action="update",
        node_id="abc",
        node={"name": "New name", "description": "desc"},
    )

    apply_patches([patch], behaviors_path=tmp_path / "behaviors.yaml", collection=collection)

    assert collection.updated == [("abc", {"name": "New name", "description": "desc"})]
    assert not (tmp_path / "behaviors.yaml").exists()


def test_apply_node_serialized_replace(tmp_path):
    collection = FakeCollection()
    patch = Patch(seq=0, action="update", node_id="abc", node=dict(TEXT_NODE))

    apply_patches([patch], collection=collection)

    (node_id, new_node), = collection.replaced
    assert node_id == "abc"
    assert isinstance(new_node, TextNode)
    assert new_node.name == "Explain"
    assert collection.updated == []


def test_apply_node_add_stores_new_node(tmp_path):
    collection = FakeCollection()
    patch = Patch(seq=0, action="add", node=dict(TEXT_NODE))

    apply_patches([patch], collection=collection)

    (node,) = collection.added
    assert isinstance(node, TextNode)
    assert node.name == "Explain"


def test_apply_node_patch_requires_collection(tmp_path):
    with pytest.raises(ValueError, match="collection"):
        apply_patches(
            [Patch(seq=0, action="update", node_id="abc", node={"name": "N"})]
        )


def test_apply_node_patch_leaves_behaviors_file_untouched(tmp_path):
    behaviors_path = tmp_path / "behaviors.yaml"
    write_behaviors(behaviors_path, [group("G", ("a", "x"))])
    before = behaviors_path.read_text(encoding="utf-8")

    apply_patches(
        [Patch(seq=0, action="update", node_id="abc", node={"name": "N"})],
        behaviors_path=behaviors_path,
        collection=FakeCollection(),
    )

    assert behaviors_path.read_text(encoding="utf-8") == before


def test_apply_behavior_write_is_atomic(tmp_path):
    behaviors_path = tmp_path / "behaviors.yaml"

    apply_patches(
        [Patch(seq=0, action="add", title="G", point_key="a", content="one")],
        behaviors_path=behaviors_path,
    )

    assert list(tmp_path.iterdir()) == [behaviors_path]
    assert load_behaviors(behaviors_path) == [group("G", ("a", "one"))]


def test_append_and_load_round_trip(tmp_path):
    patches = [
        Patch(seq=0, action="add", title="G", point_key="a", content="one"),
        Patch(seq=5, action="remove", title="G", point_key="b"),
        Patch(seq=2, action="update", title="G", point_key="c", content="three"),
    ]
    log = tmp_path / "patches.jsonl"

    append_patches(patches[:2], path=log)
    append_patches(patches[2:], path=log)

    loaded = load_patches(log)
    assert loaded == sorted(patches, key=lambda patch: patch.seq)
    assert [p.seq for p in loaded] == [0, 2, 5]


def test_load_patches_missing_file_is_empty(tmp_path):
    assert load_patches(tmp_path / "none.jsonl") == []


def test_node_patch_round_trips_through_jsonl(tmp_path):
    patch = Patch(seq=1, action="update", node_id="abc", node={"name": "N"})
    log = tmp_path / "patches.jsonl"

    append_patches([patch], path=log)

    assert load_patches(log) == [patch]


def test_patch_node_validation():
    with pytest.raises(ValidationError):
        Patch(seq=0, action="remove", node_id="abc", node={"name": "N"})

    with pytest.raises(ValidationError):
        Patch(seq=0, action="update", node={"name": "N"})

    with pytest.raises(ValidationError):
        Patch(seq=0, action="add", node={"name": "N"})

    with pytest.raises(ValidationError):
        Patch(seq=0, action="update", node_id="abc", node="not a dict")


def test_patch_node_round_trips_through_validation():
    patch = Patch(seq=3, action="update", node_id="abc", node={"writes": {"r": "response"}})

    assert Patch.model_validate(patch.model_dump()) == patch