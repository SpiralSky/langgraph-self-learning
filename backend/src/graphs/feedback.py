"""Feedback loop: turn a user suggestion into structured behavior patches.

Given a suggestion attached to a Q/A turn, the LLM **regenerates the affected
behavior entries whole** (additions, rewrites, deletions all clearly keyed),
then the difference against the behaviors snapshot is computed
**programmatically** — the diff step (:func:`diff_behaviors`) is a pure
function, so it is unit-testable without an LLM. The output is an ordered
``Patch`` list that the storage/application layer persists and applies
most-recent-wins.

The core prompt template is a module constant (data never rewrites it); the
behaviors at the time of the suggestion are rendered into it from data, so no
prompt text is ever hand-edited.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Literal

from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field, ValidationError, model_validator

from graphs.behaviors import BehaviorGroup

_FEEDBACK_TEMPLATE = """A learner suggested an improvement to the behavior rules.
Rewrite the FULL behavior registry based on the suggestion: keep the rules that
still hold, edit the ones the suggestion addresses, drop obsolete ones, and add
new ones where needed.

Reply with ONE JSON object and nothing else, in this shape:
{{"groups": [{{"title": "...", "points": [{{"key": "...", "text": "..."}}]}}]}}

Every carried-over point MUST keep its exact current "key"; new points get a
new short key. Every point you emit needs a key and text.

Request: {question}
Prior turn responses:
{responses}
Suggestion/comment: {comment}
Graph state (summary): {summary}

Current behavior registry (title / point key / text):
{registry}
"""


class Patch(BaseModel):
    """A single structured edit over behavior entries or collection nodes.

    Behavior-targeted patches (the default) carry ``title``/``point_key``/
    ``content``. Node-targeted patches instead carry a ``node`` payload — field
    edits (``{name, description, prompt, params, writes}``) for
    ``NodeCollection.update`` or a full type-tagged serialized node (has a
    ``"type"`` key) for ``NodeCollection.add``/``replace`` — and omit the
    behaviors fields.

    :param seq: Global monotonically increasing id preserving apply order.
    :type seq: int
    :param action: What to do with the targeted entry.
    :type action: Literal["add", "update", "remove"]
    :param title: Behavior group the entry lives in.
    :type title: str | None
    :param point_key: Stable key of the behavior point (from ``behaviors.yaml``).
    :type point_key: str | None
    :param content: New text for ``add``/``update``; must be ``None`` for
        ``remove`` (remove only names the key).
    :type content: str | None
    :param node_id: Target id in the ``NodeCollection`` (node patches).
    :type node_id: str | None
    :param node: Node payload — field edits or a type-tagged serialized node.
    :type node: dict[str, object] | None

    :raises ValueError: missing title/key, missing content on add/update,
        content on a remove, or a malformed node payload.
    """

    seq: int
    action: Literal["add", "update", "remove"]
    title: str | None = None
    point_key: str | None = None
    content: str | None = None
    node_id: str | None = None
    node: dict[str, object] | None = None

    @model_validator(mode="after")
    def _validate_shape(self) -> Patch:
        if self.node is not None or self.node_id is not None:
            if self.node is None:
                raise ValueError("node-targeted patch needs a 'node' payload")
            if not isinstance(self.node, dict):
                raise ValueError("node patch payload must be a dict")
            if self.action == "remove":
                raise ValueError("node-targeted 'remove' is unsupported")
            if self.action == "update" and self.node_id is None:
                raise ValueError("update node patch needs a 'node_id'")
            if self.action == "add" and "type" not in self.node:
                raise ValueError(
                    "a node 'add' needs a full serialized node payload"
                )
            return self
        if not self.title:
            raise ValueError("patch needs a non-empty 'title'")
        if not self.point_key:
            raise ValueError("patch needs a non-empty 'point_key'")
        if self.action in ("add", "update") and self.content is None:
            raise ValueError(f"{self.action} patch needs 'content'")
        if self.action == "remove" and self.content is not None:
            raise ValueError("remove patch must not carry 'content'")
        return self


class RegeneratedPoint(BaseModel):
    """One point in the LLM-regenerated registry (keyed)."""

    key: str = Field(min_length=1)
    text: str = ""


class RegeneratedGroup(BaseModel):
    """One titled group of regenerated points."""

    title: str = Field(min_length=1)
    points: list[RegeneratedPoint]


class RegeneratedBehaviors(BaseModel):
    """Full regenerated behavior registry the LLM must return."""

    groups: list[RegeneratedGroup]


def parse_regenerated(content: str) -> list[RegeneratedGroup]:
    """Parse the LLM's regenerated registry JSON into groups.

    :raises ValueError: malformed JSON, missing ``groups``, unkeyed points, or
        empty group titles (all surfaced as the LLM failed the contract).
    """
    if not isinstance(content, str):
        raise TypeError(
            f"regenerated behaviors must be a string, got {type(content).__name__}"
        )
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"regenerated behaviors are not valid JSON: {exc}") from exc
    try:
        registry = RegeneratedBehaviors.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"regenerated behaviors are malformed: {exc}") from exc
    return registry.groups


def diff_behaviors(
    old_groups: Iterable[BehaviorGroup],
    new_groups: Iterable[RegeneratedGroup],
    *,
    seq_prefix: int = 0,
) -> list[Patch]:
    """Programmatically diff the regenerated registry against the snapshot.

    Matching is by ``(title, point_key)``: entries present in both with
    different text become ``update`` patches, entries only in the regenerated
    set become ``add``, entries only in the snapshot become ``remove``, and
    unchanged entries produce no patch. Result is ordered by ``seq`` starting
    at ``seq_prefix`` (adds/updates follow the regenerated order, removes
    follow the snapshot order).

    :raises ValueError: duplicate ``(title, point_key)`` in the regenerated
        set, or a non-negative-integer ``seq_prefix``.
    """
    if (
        not isinstance(seq_prefix, int)
        or isinstance(seq_prefix, bool)
        or seq_prefix < 0
    ):
        raise ValueError("seq_prefix must be a non-negative integer")

    old = {
        (group.title, point.id): point.text
        for group in old_groups
        for point in group.points
    }
    new = [
        (group.title, point.key, point.text)
        for group in new_groups
        for point in group.points
    ]

    seen: set[tuple[str, str]] = set()
    for title, key, _text in new:
        if (title, key) in seen:
            raise ValueError(
                f"regenerated behaviors carry a duplicate point: "
                f"{title!r} / {key!r}"
            )
        seen.add((title, key))

    patches: list[Patch] = []
    seq = seq_prefix
    for title, key, text in new:
        if (title, key) in old:
            if old[(title, key)] != text:
                patches.append(
                    Patch(
                        seq=seq, action="update", title=title, point_key=key, content=text
                    )
                )
                seq += 1
        else:
            patches.append(
                Patch(seq=seq, action="add", title=title, point_key=key, content=text)
            )
            seq += 1
    for title, key in old:
        if (title, key) not in seen:
            patches.append(Patch(seq=seq, action="remove", title=title, point_key=key))
            seq += 1
    return patches


def _render_registry(groups: Iterable[BehaviorGroup]) -> str:
    """Render behaviors as ``## <title>`` / ``- [<key>] <text>`` lines."""
    lines: list[str] = []
    for group in groups:
        lines.append(f"## {group.title}")
        lines.extend(f"- [{point.id}] {point.text}" for point in group.points)
    return "\n".join(lines) or "(none)"


def _render_responses(responses: Iterable[str]) -> str:
    """Render prior turn responses; ``(none)`` when empty."""
    items = [str(response) for response in responses]
    return "\n".join(f"- {item}" for item in items) or "(none)"


def _summarize_state(graph_state: object) -> str:
    """A compact, metadata-only summary of the graph state for the prompt."""
    if graph_state is None:
        return "(none)"
    if isinstance(graph_state, BaseModel):
        data = graph_state.model_dump(exclude_none=True)
    elif isinstance(graph_state, dict):
        data = dict(graph_state)
    else:
        data = {"repr": str(graph_state)}
    text = json.dumps(data, ensure_ascii=False, default=str)
    if len(text) > 800:
        text = f"{text[:800]}... [truncated]"
    return text


def compute_improvement_patches(
    question: str,
    responses: Iterable[str],
    graph_state: object,
    comment: str,
    behaviors_at_time: Iterable[BehaviorGroup],
    llm: Runnable,
    *,
    seq_prefix: int = 0,
) -> list[Patch]:
    """Compute structured behavior patches from a user suggestion.

    Renders the fixed template with the suggestion (question + responses +
    comment), a compact graph-state summary, and the behaviors snapshot, asks
    the LLM for the **full regenerated registry** (keyed), and returns the
    programmatic diff as an ordered ``Patch`` list.

    :raises ValueError: the LLM returned no usable content or malformed
        regenerated entries.
    """
    prompt = _FEEDBACK_TEMPLATE.format(
        question=question if question else "(none)",
        responses=_render_responses(responses),
        comment=comment if comment else "(none)",
        summary=_summarize_state(graph_state),
        registry=_render_registry(behaviors_at_time),
    )
    result = llm.invoke(prompt)
    content = getattr(result, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise ValueError("LLM response carried no regenerated behaviors content")
    groups = parse_regenerated(content)
    return diff_behaviors(behaviors_at_time, groups, seq_prefix=seq_prefix)