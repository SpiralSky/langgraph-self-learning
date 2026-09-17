"""Tests for the ``POST /feedback`` FastAPI endpoint.

Pure-unit: compute/apply are stubbed at the ``api.feedback`` module boundary
(no live LLM, no network, no model-settings access); behaviors data is
verified untouched on error paths via tmp YAML files.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api import feedback as api_feedback
from api.feedback import create_app
from graphs.behaviors import BehaviorGroup, BehaviorPoint
from graphs.feedback import Patch
from graphs.patches import save_behaviors


def group(title: str, *points: tuple[str, str]) -> BehaviorGroup:
    return BehaviorGroup(
        title=title, points=[BehaviorPoint(id=key, text=text) for key, text in points]
    )


def post_feedback(client: TestClient, **overrides) -> object:
    body = {
        "question": "why?",
        "responses": ["because"],
        "comment": "reason first, please",
    }
    body.update(overrides)
    return client.post("/feedback", json=body)


def write_behaviors(path: Path) -> None:
    save_behaviors([group("Explain concepts", ("explain-why", "old"))], path)


def patch(seq: int = 0) -> Patch:
    return Patch(
        seq=seq, action="update", title="Explain concepts",
        point_key="explain-why", content="reason first",
    )


def test_feedback_valid_request_returns_applied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    behaviors_path = tmp_path / "behaviors.yaml"
    patches_path = tmp_path / "patches.jsonl"
    write_behaviors(behaviors_path)
    expected = patch()

    monkeypatch.setattr(
        api_feedback, "compute_improvement_patches", lambda **kwargs: [expected]
    )
    monkeypatch.setattr(api_feedback, "apply_patches", lambda patches, **kwargs: list(patches))

    app = create_app(
        llm=object(), behaviors_path=behaviors_path, patches_path=patches_path
    )
    response = post_feedback(TestClient(app))

    assert response.status_code == 200
    assert response.json()["applied"] == [
        {
            "seq": 0,
            "action": "update",
            "title": "Explain concepts",
            "point_key": "explain-why",
            "content": "reason first",
        }
    ]


def test_feedback_continues_seq_from_existing_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patches_path = tmp_path / "patches.jsonl"
    save_behaviors([group("G", ("a", "x"))], tmp_path / "behaviors.yaml")
    api_feedback.append_patches([patch(seq=4)], path=patches_path)
    seen: dict = {}

    def fake_compute(**kwargs):
        seen["seq_prefix"] = kwargs["seq_prefix"]
        return []

    monkeypatch.setattr(api_feedback, "compute_improvement_patches", fake_compute)
    monkeypatch.setattr(api_feedback, "append_patches", lambda patches, path=None: None)
    monkeypatch.setattr(api_feedback, "apply_patches", lambda patches, **kwargs: list(patches))

    app = create_app(
        llm=object(), behaviors_path=tmp_path / "behaviors.yaml",
        patches_path=patches_path,
    )
    response = post_feedback(TestClient(app))

    assert response.status_code == 200
    assert seen["seq_prefix"] == 5


def test_feedback_missing_comment_returns_422(tmp_path: Path) -> None:
    app = create_app(
        llm=object(), behaviors_path=tmp_path / "behaviors.yaml",
        patches_path=tmp_path / "patches.jsonl",
    )
    response = post_feedback(TestClient(app), comment=None)

    assert response.status_code == 422


def test_feedback_missing_fields_returns_422(tmp_path: Path) -> None:
    app = create_app(
        llm=object(), behaviors_path=tmp_path / "behaviors.yaml",
        patches_path=tmp_path / "patches.jsonl",
    )
    response = TestClient(app).post("/feedback", json={"comment": "please fix"})

    assert response.status_code == 422


def test_feedback_compute_error_returns_400(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(**kwargs):
        raise ValueError("malformed regenerated registry")

    monkeypatch.setattr(api_feedback, "compute_improvement_patches", boom)

    app = create_app(
        llm=object(), behaviors_path=tmp_path / "behaviors.yaml",
        patches_path=tmp_path / "patches.jsonl",
    )
    response = post_feedback(TestClient(app))

    assert response.status_code == 400
    assert "invalid feedback" in response.json()["detail"]


def test_feedback_apply_failure_returns_500_and_leaves_behaviors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    behaviors_path = tmp_path / "behaviors.yaml"
    patches_path = tmp_path / "patches.jsonl"
    write_behaviors(behaviors_path)
    before = behaviors_path.read_text(encoding="utf-8")

    def boom(patches, **kwargs):
        raise RuntimeError("node-targeted patch needs a collection")

    monkeypatch.setattr(api_feedback, "compute_improvement_patches", lambda **kwargs: [patch()])
    monkeypatch.setattr(api_feedback, "apply_patches", boom)

    app = create_app(
        llm=object(), behaviors_path=behaviors_path, patches_path=patches_path
    )
    response = post_feedback(TestClient(app))

    assert response.status_code == 500
    assert "failed to apply feedback" in response.json()["detail"]
    assert behaviors_path.read_text(encoding="utf-8") == before