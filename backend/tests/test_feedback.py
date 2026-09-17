"""Unit tests for feedback patch computation.

Pure-unit: fake LLMs return fixed regenerated registries; no live model, no
network.
"""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from graphs.behaviors import BehaviorGroup, BehaviorPoint
from graphs.feedback import (
    Patch,
    RegeneratedGroup,
    RegeneratedPoint,
    compute_improvement_patches,
    diff_behaviors,
    parse_regenerated,
)


class FakeLLM:
    def __init__(self, content: str) -> None:
        self._content = content

    def invoke(self, prompt: str) -> SimpleNamespace:
        return SimpleNamespace(content=self._content)


def group(title: str, *points: tuple[str, str]) -> BehaviorGroup:
    return BehaviorGroup(
        title=title, points=[BehaviorPoint(id=key, text=text) for key, text in points]
    )


REGENERATED = (
    '{"groups": [{"title": "Explain concepts", "points": ['
    '{"key": "explain-why", "text": "reason first"},'
    '{"key": "new-point", "text": "new rule"}]}]}'
)


def test_diff_empty_old_emits_no_patches():
    assert diff_behaviors([], []) == []


def test_diff_all_adds_when_snapshot_empty():
    new = RegeneratedGroup(
        title="G",
        points=[
            RegeneratedPoint(key="a", text="one"),
            RegeneratedPoint(key="b", text="two"),
        ],
    )

    patches = diff_behaviors([], [new])

    assert [p.seq for p in patches] == [0, 1]
    assert [(p.action, p.title, p.point_key, p.content) for p in patches] == [
        ("add", "G", "a", "one"),
        ("add", "G", "b", "two"),
    ]


def test_diff_update_on_text_change():
    old = [group("G", ("a", "old text"))]
    new = [RegeneratedGroup(title="G", points=[RegeneratedPoint(key="a", text="new text")])]

    patches = diff_behaviors(old, new)

    assert [(p.action, p.point_key, p.content) for p in patches] == [
        ("update", "a", "new text")
    ]


def test_diff_unchanged_key_emits_no_patch():
    old = [group("G", ("a", "same"))]
    new = [RegeneratedGroup(title="G", points=[RegeneratedPoint(key="a", text="same")])]

    assert diff_behaviors(old, new) == []


def test_diff_remove_for_missing_key():
    old = [group("G", ("a", "gone"), ("b", "kept"))]
    new = [RegeneratedGroup(title="G", points=[RegeneratedPoint(key="b", text="kept")])]

    patches = diff_behaviors(old, new)

    assert [(p.action, p.point_key, p.content) for p in patches] == [
        ("remove", "a", None)
    ]


def test_diff_partial_edit_mixed_patch_set():
    old = [
        group("G", ("a", "old a"), ("b", "kept"), ("c", "to drop")),
        group("H", ("x", "old x")),
    ]
    new = [
        RegeneratedGroup(
            title="G",
            points=[
                RegeneratedPoint(key="a", text="new a"),
                RegeneratedPoint(key="b", text="kept"),
                RegeneratedPoint(key="d", text="brand new"),
            ],
        ),
        RegeneratedGroup(
            title="H", points=[RegeneratedPoint(key="x", text="updated x")]
        ),
    ]

    patches = diff_behaviors(old, new)

    assert [(p.seq, p.action, p.title, p.point_key, p.content) for p in patches] == [
        (0, "update", "G", "a", "new a"),
        (1, "add", "G", "d", "brand new"),
        (2, "update", "H", "x", "updated x"),
        (3, "remove", "G", "c", None),
    ]


def test_diff_removes_follow_snapshot_order_after_adds_updates():
    old = [group("G", ("a", "old a"), ("b", "old b"))]
    new = [RegeneratedGroup(title="G", points=[RegeneratedPoint(key="a", text="new a")])]

    patches = diff_behaviors(old, new)

    assert [(p.action, p.point_key) for p in patches] == [
        ("update", "a"),
        ("remove", "b"),
    ]


def test_diff_seq_contiguous_with_prefix():
    old = [group("G", ("a", "old"), ("b", "drop"))]
    new = [RegeneratedGroup(title="G", points=[RegeneratedPoint(key="a", text="new")])]

    patches = diff_behaviors(old, new, seq_prefix=5)

    assert [p.seq for p in patches] == [5, 6]


def test_diff_duplicate_new_key_raises():
    new = [
        RegeneratedGroup(
            title="G",
            points=[RegeneratedPoint(key="a", text="x"), RegeneratedPoint(key="a", text="y")],
        )
    ]

    with pytest.raises(ValueError, match="duplicate point"):
        diff_behaviors([], new)


def test_diff_negative_seq_prefix_raises():
    with pytest.raises(ValueError, match="seq_prefix"):
        diff_behaviors([], [], seq_prefix=-1)


def test_parse_regenerated_valid():
    groups = parse_regenerated(REGENERATED)

    assert len(groups) == 1
    assert groups[0].title == "Explain concepts"
    assert [(p.key, p.text) for p in groups[0].points] == [
        ("explain-why", "reason first"),
        ("new-point", "new rule"),
    ]


def test_parse_regenerated_bad_json_raises():
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_regenerated("{not json")


def test_parse_regenerated_missing_groups_raises():
    with pytest.raises(ValueError, match="malformed"):
        parse_regenerated('{"foo": []}')


def test_parse_regenerated_point_without_key_raises():
    with pytest.raises(ValueError, match="malformed"):
        parse_regenerated('{"groups": [{"title": "G", "points": [{"text": "x"}]}]}')


def test_compute_improvement_patches_produces_diff():
    behaviors = [group("Explain concepts", ("explain-why", "old reasoning"))]

    patches = compute_improvement_patches(
        question="why?",
        responses=[],
        graph_state=None,
        comment="please reason first",
        behaviors_at_time=behaviors,
        llm=FakeLLM(REGENERATED),
    )

    assert [(p.action, p.title, p.point_key, p.content) for p in patches] == [
        ("update", "Explain concepts", "explain-why", "reason first"),
        ("add", "Explain concepts", "new-point", "new rule"),
    ]


def test_compute_empty_behaviors_all_add():
    patches = compute_improvement_patches(
        question="hi",
        responses=["a", "b"],
        graph_state={"user_message": "hi", "response": "hi!"},
        comment="add a rule",
        behaviors_at_time=[],
        llm=FakeLLM(REGENERATED),
    )

    assert [p.action for p in patches] == ["add", "add"]


def test_compute_empty_llm_content_raises():
    with pytest.raises(ValueError, match="no regenerated behaviors content"):
        compute_improvement_patches(
            question="q",
            responses=[],
            graph_state=None,
            comment="c",
            behaviors_at_time=[],
            llm=FakeLLM("  "),
        )


def test_compute_uses_seq_prefix():
    behaviors = [group("G", ("a", "old"))]
    content = '{"groups": [{"title": "G", "points": [{"key": "a", "text": "new"}]}]}'

    patches = compute_improvement_patches(
        question="q",
        responses=[],
        graph_state=None,
        comment="c",
        behaviors_at_time=behaviors,
        llm=FakeLLM(content),
        seq_prefix=10,
    )

    assert [p.seq for p in patches] == [10]


def test_patch_model_validation():
    with pytest.raises(ValidationError):
        Patch(seq=0, action="update", title="G", point_key="a")

    with pytest.raises(ValidationError):
        Patch(seq=0, action="remove", title="G", point_key="a", content="x")

    with pytest.raises(ValidationError):
        Patch(seq=0, action="add", title="", point_key="a", content="x")


def test_patch_round_trips_through_validation():
    patch = Patch(seq=3, action="remove", title="G", point_key="a")

    restored = Patch.model_validate(patch.model_dump())

    assert restored == patch