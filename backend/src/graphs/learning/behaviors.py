"""Sectioned behavior points, loaded from a data YAML file.

Behaviors are **data**, never prompt text. Each :class:`BehaviorGroup` holds a
title and a list of point strings, and every point carries a stable key — an
explicit ``id`` from the file or an auto-assigned ``<title-slug>-p<index>``
key. Keys are part of the loaded structure so later plans (Patch-deployment
work) can target individual entries. :func:`render_behaviors` is the *only*
surface that turns the data into the ``## <title>`` / ``- <point>`` prompt
lines consumed by the ``GeneratorNode``; the core template itself lives in
code, never in the YAML.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import yaml
from pydantic import BaseModel

from config import BACKEND_ROOT

DEFAULT_BEHAVIORS_PATH = BACKEND_ROOT / "behaviors.yaml"


class BehaviorPoint(BaseModel):
    """A single behavior point with its stable key.

    :param id: Stable key used to reference this point (explicit ``id`` from
        the YAML or a loader-assigned ``<title-slug>-p<index>`` key).
    :type id: str
    :param text: The point text rendered into the generator prompt.
    :type text: str
    """

    id: str
    text: str


class BehaviorGroup(BaseModel):
    """A titled section of behavior points, mirroring the YAML shape.

    :param title: Section title (rendered as ``## <title>``).
    :type title: str
    :param points: Ordered points, each with a stable key.
    :type points: list[BehaviorPoint]
    """

    title: str
    points: list[BehaviorPoint]


def _title_slug(title: str) -> str:
    """Lowercase alphanumerics joined by ``-``; never empty."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug or "group"


def _auto_key(title: str, index: int) -> str:
    """Stable positional key ``<title-slug>-p<index>`` (1-based index)."""
    return f"{_title_slug(title)}-p{index}"


def load_behaviors(path: str | Path | None = None) -> list[BehaviorGroup]:
    """Load behavior groups from a sectioned YAML file.

    The file shape is a list of ``{title, points[]}`` entries; each point is
    either a plain string (auto-key assigned) or a mapping ``{id, text}``.
    An empty or absent file yields an empty list; any other shape raises.
    Keys must be unique across the whole file.

    :param path: YAML location (default ``backend/behaviors.yaml``).
    :type path: str | Path | None
    :return: Loaded groups, never ``None``.
    :rtype: list[BehaviorGroup]
    :raises TypeError: Non-list root or malformed point entries.
    :raises ValueError: Missing/invalid titles, duplicate or empty ids.
    """
    source = Path(path) if path is not None else DEFAULT_BEHAVIORS_PATH
    if not source.exists():
        return []

    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise TypeError(f"{source} must contain a top-level list of groups")

    groups: list[BehaviorGroup] = []
    seen_ids: set[str] = set()
    for entry in raw:
        if not isinstance(entry, dict):
            raise TypeError(f"{source}: each group must be a mapping")
        title = entry.get("title")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"{source}: each group needs a non-empty 'title'")
        points_raw = entry.get("points", [])
        if not isinstance(points_raw, list):
            raise TypeError(f"{source}: group {title!r} 'points' must be a list")

        points: list[BehaviorPoint] = []
        for index, point in enumerate(points_raw, start=1):
            if isinstance(point, str):
                text, key = point, None
            elif isinstance(point, dict):
                key = point.get("id")
                text = point.get("text")
                if key is not None and (not isinstance(key, str) or not key):
                    raise ValueError(f"{source}: point ids must be non-empty strings")
                if not isinstance(text, str):
                    raise TypeError(
                        f"{source}: group {title!r} point {index} must carry 'text'"
                    )
            else:
                raise TypeError(
                    f"{source}: group {title!r} point {index} must be a string "
                    "or a {id, text} mapping"
                )
            point_id = key or _auto_key(title, index)
            if point_id in seen_ids:
                raise ValueError(f"{source}: duplicate point id {point_id!r}")
            seen_ids.add(point_id)
            points.append(BehaviorPoint(id=point_id, text=text))

        groups.append(BehaviorGroup(title=title.strip(), points=points))
    return groups


def render_behaviors(groups: Iterable[BehaviorGroup]) -> str:
    """Render groups to prompt lines: ``## <title>`` then ``- <point>``.

    Groups with no points are skipped; ordering follows the input. An empty
    input renders to an empty string.

    :param groups: Groups to render, in order.
    :type groups: Iterable[BehaviorGroup]
    :return: Rendered prompt fragment.
    :rtype: str
    """
    lines: list[str] = []
    for group in groups:
        if not group.points:
            continue
        lines.append(f"## {group.title}")
        lines.extend(f"- {point.text}" for point in group.points)
    return "\n".join(lines)