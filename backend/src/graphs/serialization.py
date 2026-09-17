"""JSON (de)serialization support for graph artifacts.

The supported annotation set is ``{str, int, float, bool, list, dict}``
(serialized as ``"str"``/``"int"``/...). Generic forms (``list[str]``) and
``Annotated``/optional annotations (``str | None``) normalize onto that set.
Type-tagged node dispatch lives here too: every node type registers a
``from_dict`` builder under its ``"type"`` tag, so ``node_from_dict`` can
rebuild a nested ``to_dict`` payload without the caller knowing the concrete
class. Registration is lazy — the node modules self-register on import, and
``node_from_dict`` imports the owning module for an unknown tag on demand.
"""

from __future__ import annotations

import importlib
import types
from collections.abc import Callable
from typing import Annotated, Union, get_args, get_origin

from graphs.connections import Connection, RoutingConnection

ANNOTATION_NAMES: dict[type, str] = {
    str: "str",
    int: "int",
    float: "float",
    bool: "bool",
    list: "list",
    dict: "dict",
}
_NAME_ANNOTATIONS: dict[str, type] = {name: ann for ann, name in ANNOTATION_NAMES.items()}

_OPTIONAL_PREFIX = "optional:"

# module keys (type tag -> owning module) for lazy node registration.
_NODE_TYPE_MODULES = {
    "text": "graphs.nodes.text_node",
    "graph": "graphs.nodes.graph_node",
    "tool": "graphs.nodes.tool_node",
}


def _unwrap(annotation: object) -> tuple[object, bool]:
    """Return ``(core_annotation, is_optional)`` for a supported annotation.

    ``Annotated`` wrappers are stripped; ``X | None`` (or ``Optional[X]``)
    unions report ``is_optional=True`` with the non-``None`` member as the
    core. Anything else is returned untouched with ``is_optional=False``.
    """
    origin = get_origin(annotation)
    if origin is Annotated:
        return _unwrap(get_args(annotation)[0])
    if origin is Union or origin is types.UnionType:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if not args or len(args) != 1:
            raise ValueError(f"unsupported union annotation: {annotation!r}")
        core, _ = _unwrap(args[0])
        return core, True
    return annotation, False


def serialize_annotation(annotation: object) -> str:
    """Canonical stable string for a supported annotation.

    Generic/optional annotations normalize to their atomic name
    (``list[str]`` -> ``"list"``), optional ones prefix ``"optional:"``
    (``str | None`` -> ``"optional:str"``). Unsupported annotations raise
    ``ValueError``.
    """
    core, optional = _unwrap(annotation)
    if core in ANNOTATION_NAMES:
        name = ANNOTATION_NAMES[core]
    else:
        origin = get_origin(core)
        if origin in (list, dict):
            name = ANNOTATION_NAMES[origin]
        else:
            raise ValueError(
                f"unsupported annotation for serialization: {annotation!r}"
            )
    return f"{_OPTIONAL_PREFIX}{name}" if optional else name


def deserialize_annotation(name: str) -> type:
    """Rebuild the annotation a ``serialize_annotation`` string encodes.

    Optional names restore as ``<type> | None``; element types of generic
    ``list``/``dict`` names are not recovered (bare origins only).
    """
    if not isinstance(name, str):
        raise TypeError(f"serialized annotation must be a string: {name!r}")
    optional = False
    if name.startswith(_OPTIONAL_PREFIX):
        optional = True
        name = name[len(_OPTIONAL_PREFIX):]
    annotation = _NAME_ANNOTATIONS.get(name)
    if annotation is None:
        raise ValueError(f"unknown serialized annotation: {name!r}")
    if optional:
        return annotation | None
    return annotation


_NODE_FROM_DICT: dict[str, Callable[[dict], object]] = {}


def register_node_type(tag: str, from_dict: Callable[[dict], object]) -> None:
    """Register the ``tag`` -> ``from_dict`` builder for a node type."""
    if not isinstance(tag, str) or not tag:
        raise ValueError("node type tag must be a non-empty string")
    if tag in _NODE_FROM_DICT:
        raise ValueError(f"node type {tag!r} already registered")
    _NODE_FROM_DICT[tag] = from_dict


def node_from_dict(data: dict) -> object:
    """Rebuild a node from a ``to_dict`` payload (type-tagged dispatch).

    The owning module is imported on first use so registration is lazy;
    an unknown/unregistered type tag raises ``ValueError``.
    """
    tag = data.get("type")
    builder = _NODE_FROM_DICT.get(tag)
    if builder is None:
        module = _NODE_TYPE_MODULES.get(tag)
        if module is not None:
            try:
                importlib.import_module(module)
            except ImportError:
                pass
            builder = _NODE_FROM_DICT.get(tag)
    if builder is None:
        raise ValueError(f"unknown node type tag: {tag!r}")
    return builder(data)


def connection_from_dict(data: dict) -> Connection | RoutingConnection:
    """Rebuild a connection from its ``as_dict`` payload."""
    connection_type = data.get("type")
    if connection_type == "standard":
        return Connection(source=data["source"], target=data["target"])
    if connection_type == "routing":
        return RoutingConnection(
            source=data["source"],
            targets=tuple(data["targets"]),
            routing_prompt=data["routing_prompt"],
        )
    raise ValueError(f"unknown connection type: {connection_type!r}")