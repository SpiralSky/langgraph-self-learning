"""``ToolCallNode``: run a whitelisted tool from state-provided args.

The node declares a fixed two-param schema (``tool``: which registered tool
to call, ``args``: its JSON-style arguments) and, on ``get_node(llm)``,
produces a LangGraph-ready dual callable that reads both from state by
name, runs the tool through the whitelist registry, and writes the textual
result to the declared ``writes`` keys. The LLM is never invoked — the
``llm`` argument exists only to satisfy the ``GraphNode`` protocol.

The tool name is validated against the registry at construction time
(build-time whitelist), so a generated graph can never reference a tool
outside the whitelist. The ``prompt`` is a fixed template derived from the
tool name, never free-form.
"""

from __future__ import annotations

import json
from typing import Self

from langchain_core.runnables import Runnable
from pydantic import BaseModel

from graphs.nodes.base import AbstractNode, DualCallable
from graphs.nodes.text_node import is_list_annotation
from graphs.serialization import register_node_type
from graphs.tools import ToolRegistry, default_registry

_PROMPT_TEMPLATE = (
    "Execute the tool {tool} with the JSON arguments read from the 'args' "
    "state field and write the plain-text result."
)


class _ToolCallFn:
    """LangGraph-ready dual callable produced by ``ToolCallNode.get_node``.

    Reads ``tool`` and ``args`` from the state (dict or pydantic model) by
    name, dispatches through the node's whitelist registry, and spreads the
    stringified result over the declared ``writes`` keys with the same
    list-wrap rule as ``_TextNodeFn``. The instance carries ``__name__`` so
    LangGraph infers the node name from the callable itself.
    """

    def __init__(
        self,
        *,
        name: str,
        tool: str,
        registry: ToolRegistry,
        writes: dict[str, str],
    ) -> None:
        self.__name__ = name
        self._tool = tool
        self._registry = registry
        self._writes = writes

    def __call__(self, state: dict) -> dict:
        return self.invoke(state)

    @staticmethod
    def _read(state: dict, key: str) -> object:
        """Read ``key`` from a pydantic-model or plain-dict state.

        An absent key yields ``None`` (callers fall back to the node's own
        configured tool / empty args).
        """
        if isinstance(state, BaseModel):
            return getattr(state, key, None)
        return state.get(key)

    @staticmethod
    def _target_is_list(state: dict, state_key: str) -> bool:
        """List-wrap only when the target state field is declared a list.

        Mirrors ``_TextNodeFn._target_is_list``: the field type is read from
        the pydantic state model; a plain-dict state has no schema, so values
        are written unwrapped.
        """
        if not isinstance(state, BaseModel):
            return False
        field = type(state).model_fields.get(state_key)
        return field is not None and is_list_annotation(field.annotation)

    @staticmethod
    def _coerce_args(raw: object) -> dict:
        """Normalize the state ``args`` value to a plain dict.

        Accepts a dict directly or a JSON-encoded object string; anything
        else (list, scalar, malformed JSON) is a call error.
        """
        if raw is None:
            return {}
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"tool args JSON is invalid: {exc}") from exc
            if not isinstance(parsed, dict):
                raise TypeError(
                    "tool args JSON must encode an object, got "
                    f"{type(parsed).__name__}"
                )
            return parsed
        if isinstance(raw, dict):
            return raw
        raise ValueError(f"tool args must be a dict, got {type(raw).__name__}")

    def _update(self, state: dict, content: str) -> dict:
        return {
            state_key: [content] if self._target_is_list(state, state_key) else content
            for state_key in self._writes.values()
        }

    def invoke(self, state: dict) -> dict:
        """Read tool + args from state, run the whitelisted tool, spread writes."""
        tool_name = self._read(state, "tool")
        if tool_name is None:
            tool_name = self._tool
        args = self._coerce_args(self._read(state, "args"))
        run = self._registry.get(str(tool_name))
        result = run(args)
        return self._update(state, str(result))

    async def ainvoke(self, state: dict) -> dict:
        """Async twin of ``invoke`` (tools are run synchronously)."""
        return self.invoke(state)


class ToolCallNode(AbstractNode):
    """Run a registered whitelisted tool; one tool per node.

    :param name: Node name (used as the LangGraph node name).
    :type name: str
    :param description: What this node does, used for retrieval.
    :type description: str
    :param tool: Name of the whitelisted tool this node invokes.
    :type tool: str
    :param registry: Whitelist registry to dispatch through; defaults to the
        shared builtin-tools registry when ``None``.
    :type registry: ToolRegistry | None
    :param writes: Output map ``{field: state_key}``; each key receives the
        tool's string result. Defaults to ``{"result": "response"}``.
    :type writes: dict[str, str] | None

    ``params`` is fixed at ``{"tool": str, "args": dict}`` (both read from
    state by name); the ``prompt`` is a fixed template naming ``tool``.

    :raises ValueError: for an unknown ``tool`` (build-time whitelist check)
        or invalid ``writes``.
    """

    def __init__(
        self,
        name: str,
        description: str,
        tool: str,
        registry: ToolRegistry | None = None,
        *,
        writes: dict[str, str] | None = None,
    ) -> None:
        if not isinstance(tool, str) or not tool:
            raise ValueError("tool must be a non-empty string")
        if registry is None:
            registry = default_registry()
        try:
            registry.get(tool)
        except KeyError as exc:
            raise ValueError(
                f"unknown tool {tool!r}; registered tools: {registry.names()}"
            ) from exc
        self.tool = tool
        self.registry = registry
        prompt = _PROMPT_TEMPLATE.format(tool=tool)
        super().__init__(
            name,
            description,
            prompt,
            params={"tool": str, "args": dict},
            writes=writes,
        )

    def get_node(self, llm: Runnable) -> DualCallable:
        """Bind an LLM (ignored) and return the LangGraph-ready dual callable."""
        return _ToolCallFn(
            name=self.name,
            tool=self.tool,
            registry=self.registry,
            writes=self.writes,
        )

    def updated(self, **changes: object) -> Self:
        """Return a new validated instance with ``changes`` applied.

        ``prompt``/``params`` are fixed for a tool node, so only ``name``,
        ``description``, ``tool`` and ``writes`` may change; a changed
        ``tool`` is re-validated against the whitelist.

        :raises ValueError: if ``prompt`` or ``params`` is changed.
        :raises TypeError: for unknown field names.
        """
        allowed = {"name", "description", "tool", "writes"}
        for key in changes:
            if key in ("prompt", "params"):
                raise ValueError(f"{key} is fixed for a tool node")
            if key not in allowed:
                raise TypeError(
                    f"unsupported fields: {sorted(set(changes))}; supported: "
                    f"{sorted(allowed)}"
                )
        return type(self)(
            name=changes.get("name", self.name),
            description=changes.get("description", self.description),
            tool=changes.get("tool", self.tool),
            registry=self.registry,
            writes=changes.get("writes", self.writes),
        )

    def to_dict(self) -> dict:
        """Type-tagged JSON-ready dict (``"type": "tool"``).

        ``params`` is structural (fixed) and ``prompt`` is derived, so the
        payload carries only metadata, the tool name, and the writes map.
        The registry is a runtime dependency and is not serialized — a
        restored node dispatches through the default registry.
        """
        return {
            "type": "tool",
            "name": self.name,
            "description": self.description,
            "tool": self.tool,
            "writes": dict(self.writes),
        }

    @classmethod
    def from_dict(cls, data: dict) -> ToolCallNode:
        """Rebuild a :class:`ToolCallNode` from a ``to_dict`` payload."""
        return cls(
            name=data["name"],
            description=data["description"],
            tool=data["tool"],
            writes=dict(data["writes"]),
        )


register_node_type("tool", ToolCallNode.from_dict)