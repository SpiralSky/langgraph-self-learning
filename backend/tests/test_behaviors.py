"""Unit tests for the behaviors config loader and renderer.

Pure-file tests: YAML fixtures are written to ``tmp_path``; the only test that
touches the repository is the default-path load, which reads the committed
``backend/behaviors.yaml``.
"""

from pathlib import Path

import pytest
import yaml

from graphs.learning.behaviors import (
    DEFAULT_BEHAVIORS_PATH,
    BehaviorGroup,
    BehaviorPoint,
    load_behaviors,
    render_behaviors,
)


def write_behaviors(tmp_path: Path, data: list) -> Path:
    path = tmp_path / "behaviors.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_load_default_file_has_content():
    groups = load_behaviors()

    assert isinstance(groups, list)
    assert len(groups) == 2
    assert [group.title for group in groups] == ["Explain concepts", "Answer directly"]
    assert all(group.points for group in groups)


def test_default_path_points_to_backend_behaviors():
    assert DEFAULT_BEHAVIORS_PATH.name == "behaviors.yaml"
    assert DEFAULT_BEHAVIORS_PATH.exists()


def test_missing_file_returns_empty_list(tmp_path):
    assert load_behaviors(tmp_path / "missing.yaml") == []


def test_empty_file_returns_empty_list(tmp_path):
    path = tmp_path / "behaviors.yaml"
    path.write_text("", encoding="utf-8")
    assert load_behaviors(path) == []


def test_explicit_ids_kept_and_auto_keys_assigned(tmp_path):
    path = write_behaviors(
        tmp_path,
        [
            {
                "title": "Explain concepts",
                "points": [{"id": "keep-me", "text": "reason first"}, "give examples"],
            }
        ],
    )

    groups = load_behaviors(path)

    assert len(groups) == 1
    points = groups[0].points
    assert points[0].id == "keep-me"
    assert points[1].id == "explain-concepts-p2"


def test_auto_key_stability_across_reloads(tmp_path):
    path = write_behaviors(
        tmp_path,
        [{"title": "My Guide", "points": ["first", "second"]}],
    )

    first = load_behaviors(path)
    second = load_behaviors(path)

    keys_a = [point.id for group in first for point in group.points]
    keys_b = [point.id for group in second for point in group.points]
    assert keys_a == ["my-guide-p1", "my-guide-p2"]
    assert keys_b == keys_a


def test_custom_path_override(tmp_path):
    path = write_behaviors(tmp_path, [{"title": "Only", "points": ["p"]}])

    groups = load_behaviors(path)

    assert [group.title for group in groups] == ["Only"]
    assert groups[0].points[0].text == "p"


def test_render_matches_section_format():
    groups = [
        BehaviorGroup(
            title="Explain concepts",
            points=[
                BehaviorPoint(id="a", text="reason first"),
                BehaviorPoint(id="b", text="give examples"),
            ],
        )
    ]

    rendered = render_behaviors(groups)

    assert rendered == "## Explain concepts\n- reason first\n- give examples"


def test_render_multiple_groups_in_order():
    groups = [
        BehaviorGroup(title="First", points=[BehaviorPoint(id="a", text="one")]),
        BehaviorGroup(title="Second", points=[BehaviorPoint(id="b", text="two")]),
    ]

    assert render_behaviors(groups) == "## First\n- one\n## Second\n- two"


def test_render_skips_empty_groups():
    groups = [
        BehaviorGroup(title="Empty", points=[]),
        BehaviorGroup(title="Full", points=[BehaviorPoint(id="a", text="one")]),
    ]

    assert render_behaviors(groups) == "## Full\n- one"


def test_render_empty_input_is_empty_string():
    assert render_behaviors([]) == ""


def test_duplicate_point_ids_rejected(tmp_path):
    path = write_behaviors(
        tmp_path,
        [
            {
                "title": "Group",
                "points": [{"id": "same", "text": "a"}, {"id": "same", "text": "b"}],
            }
        ],
    )

    with pytest.raises(ValueError, match="duplicate point id 'same'"):
        load_behaviors(path)


def test_auto_key_collision_with_explicit_id_rejected(tmp_path):
    path = write_behaviors(
        tmp_path,
        [
            {
                "title": "Explain concepts",
                "points": [{"id": "explain-concepts-p2", "text": "a"}, "b"],
            }
        ],
    )

    with pytest.raises(ValueError, match="duplicate point id"):
        load_behaviors(path)


def test_non_list_root_raises(tmp_path):
    path = tmp_path / "behaviors.yaml"
    path.write_text("just: a mapping\n", encoding="utf-8")

    with pytest.raises(TypeError, match="top-level list"):
        load_behaviors(path)


def test_group_without_title_raises(tmp_path):
    path = write_behaviors(tmp_path, [{"points": ["orphan"]}])

    with pytest.raises(ValueError, match="non-empty 'title'"):
        load_behaviors(path)


def test_point_mapping_without_text_raises(tmp_path):
    path = write_behaviors(tmp_path, [{"title": "G", "points": [{"id": "x"}]}])

    with pytest.raises(TypeError, match="must carry 'text'"):
        load_behaviors(path)