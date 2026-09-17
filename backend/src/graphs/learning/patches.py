"""Persistent patch storage + deterministic application with precedence.

Patches computed by :mod:`graphs.feedback` are appended to a JSONL log under
``backend/data/patches.jsonl`` and applied **in ``seq`` order** against the
behaviors data file: entries are upserted/deleted by ``(title, point_key)``,
groups are created on demand and dropped when they become empty, and a later
patch for the same key overwrites any earlier applied content (precedence =
seq). Application is deterministic and idempotent — re-running the same seq
range reproduces the same state. A patch may alternatively target a
``NodeCollection`` node: field edits route to ``update``, a full type-tagged
serialized node to ``replace`` (or ``add`` when untargeted).

Only behaviors **data** (and optionally the node store) is ever touched —
application never rewrites the generator prompt template, which lives in code
and is assembled from the current behaviors each run.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

import yaml

from graphs.learning.behaviors import (
    DEFAULT_BEHAVIORS_PATH,
    BehaviorGroup,
    BehaviorPoint,
    load_behaviors,
)
from graphs.learning.feedback import Patch
from graphs.persistence.serialization import node_from_dict
from graphs.persistence.storage import DATA_ROOT, ensure_data_dir

PATCHES_PATH = DATA_ROOT / "patches.jsonl"


def _atomic_write_text(target: Path, text: str) -> None:
    """Write text atomically (temp file + ``os.replace``).

    Mirrors ``storage.write_json``: a failed write can never corrupt an
    existing file because the replacement only happens after the temp file is
    fully written.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, target)


def save_behaviors(
    groups: Iterable[BehaviorGroup], path: str | Path | None = None
) -> None:
    """Persist groups to a sectioned YAML file, atomically.

    Every point is written with an explicit ``id`` so stable keys survive a
    ``load_behaviors`` round-trip losslessly. No group is dropped, even empty
    ones — preservation is the caller's choice.
    """
    payload = [
        {
            "title": group.title,
            "points": [
                {"id": point.id, "text": point.text} for point in group.points
            ],
        }
        for group in groups
    ]
    text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    _atomic_write_text(Path(path) if path is not None else DEFAULT_BEHAVIORS_PATH, text)


def load_patches(path: str | Path | None = None) -> list[Patch]:
    """Read the patch JSONL log, seq-sorted; missing/empty file -> ``[]``."""
    source = Path(path) if path is not None else PATCHES_PATH
    if not source.exists():
        return []
    patches: list[Patch] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        patches.append(Patch.model_validate_json(line))
    return sorted(patches, key=lambda patch: patch.seq)


def append_patches(
    patches: Iterable[Patch], path: str | Path | None = None
) -> None:
    """Append patches to the JSONL log (one patch per line, atomic rewrite).

    The merged log is written to a temp name and renamed so an append can
    never leave a half-written line behind; prior lines are preserved exactly.
    """
    target = Path(path) if path is not None else PATCHES_PATH
    combined = load_patches(target) + list(patches)
    if not combined:
        return
    ensure_data_dir()
    lines = [patch.model_dump_json(exclude_none=True) for patch in combined]
    _atomic_write_text(target, "".join(f"{line}\n" for line in lines))


def _find_group(groups: list[BehaviorGroup], title: str) -> BehaviorGroup | None:
    """The first group with ``title``, or ``None``."""
    for group in groups:
        if group.title == title:
            return group
    return None


def _apply_behavior_patch(groups: list[BehaviorGroup], patch: Patch) -> None:
    """Upsert/delete the targeted entry by ``(title, point_key)``."""
    title, key = patch.title, patch.point_key
    group = _find_group(groups, title)
    if patch.action == "remove":
        if group is None:
            return
        group.points = [point for point in group.points if point.id != key]
        if not group.points:
            groups.remove(group)
        return
    if group is None:
        group = BehaviorGroup(title=title, points=[])
        groups.append(group)
    for point in group.points:
        if point.id == key:
            point.text = patch.content
            return
    if patch.content is None:
        raise ValueError(f"{patch.action} patch needs 'content'")
    group.points.append(BehaviorPoint(id=key, text=patch.content))


def _is_node_patch(patch: Patch) -> bool:
    """True when the patch targets a collection node, not behaviors data."""
    return patch.node is not None or patch.node_id is not None


def _apply_node_patch(patch: Patch, collection: object) -> None:
    """Route a node-targeted patch through the collection.

    A payload with a ``"type"`` key is a full serialized node: ``add`` stores
    it, ``update`` swaps it in under ``node_id`` via ``replace``. A payload
    without ``"type"`` is field-level edits for ``update`` only.
    """
    if collection is None:
        raise ValueError("node-targeted patch needs a collection")
    if patch.node is None:
        raise ValueError("node patch needs a 'node' payload")
    if "type" in patch.node:
        node = node_from_dict(dict(patch.node))
        if patch.action == "add":
            collection.add(node)
        else:
            collection.replace(patch.node_id, node)
        return
    collection.update(patch.node_id, **patch.node)


def apply_patches(
    patches: Iterable[Patch],
    *,
    behaviors_path: str | Path | None = None,
    collection: object | None = None,
) -> list[Patch]:
    """Apply patches in ``seq`` order and return them (precedence = seq).

    Behavior-targeted patches mutate the behaviors data loaded from
    ``behaviors_path`` (default ``backend/behaviors.yaml``), which is written
    back only when at least one behavior patch applied. Node-targeted patches
    route to ``collection`` (``add``/``update``/``replace``) and never touch
    the behaviors file. Applying is idempotent: re-running the same patches
    reproduces the same state.

    :raises ValueError: a node-targeted patch without a ``collection``, or an
        ``add``/``update`` patch missing its required content.
    """
    ordered = sorted(patches, key=lambda patch: patch.seq)
    groups = load_behaviors(behaviors_path)
    changed = False
    for patch in ordered:
        if _is_node_patch(patch):
            _apply_node_patch(patch, collection)
        else:
            _apply_behavior_patch(groups, patch)
            changed = True
    if changed:
        save_behaviors(groups, behaviors_path)
    return list(ordered)