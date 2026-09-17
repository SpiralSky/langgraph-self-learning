"""Whitelist tool registry for ``ToolCallNode``.

Every tool a generated graph may invoke lives in a :class:`ToolRegistry`:
a name -> callable map where each callable takes a plain ``dict`` of
arguments and returns a ``str`` result. Only registered tools are callable
— an unknown name is a validation error at node-build time, never a
runtime surprise.

Tool modules stay importable without side effects: no client (DDGS, mem0
``Memory``) is constructed at import time. :func:`default_registry` wires
up the builtin tools lazily on first request.
"""

from __future__ import annotations

from collections.abc import Callable


class ToolError(RuntimeError):
    """A tool failed to run (transport/args problem the caller can surface).

    Runtime failures are distinguishable from programming errors: this is
    raised after a whitelisted tool was dispatched but couldn't complete.
    """


def require_arg(args: dict, key: str) -> object:
    """Return ``args[key]``; raise ``ValueError`` when absent or not a dict."""
    if not isinstance(args, dict):
        raise TypeError(f"tool args must be a dict, got {type(args).__name__}")
    if key not in args:
        raise ValueError(f"missing required tool argument: {key!r}")
    return args[key]


def int_arg(args: dict, key: str, default: int) -> int:
    """Coerce ``args[key]`` to ``int`` (``default`` when unset or null)."""
    value = args.get(key)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"tool argument {key!r} must be an integer") from None


class ToolRegistry:
    """An explicit name -> callable whitelist of runnable tools.

    Tools are plain callables ``dict -> str``. Registration is the only
    way to make a tool callable; the registry never delegates to anything
    not registered here.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Callable[[dict], str]] = {}

    def register(
        self,
        name: str,
        fn: Callable[[dict], str] | None = None,
    ) -> Callable[[dict], str]:
        """Register ``fn`` under ``name`` (or use as ``@register("name")``).

        :raises ValueError: for an empty name, a non-callable ``fn``, or a
            name that is already registered (dup registration is a whitelist
            smell and rejected).
        """
        if fn is None:

            def _decorate(fn: Callable[[dict], str]) -> Callable[[dict], str]:
                self.register(name, fn)
                return fn

            return _decorate
        if not isinstance(name, str) or not name:
            raise ValueError("tool name must be a non-empty string")
        if not callable(fn):
            raise TypeError(f"tool {name!r} must be callable")
        if name in self._tools:
            raise ValueError(f"tool {name!r} is already registered")
        self._tools[name] = fn
        return fn

    def get(self, name: str) -> Callable[[dict], str]:
        """Return the registered callable for ``name``.

        :raises KeyError: if ``name`` is not a registered tool.
        """
        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(f"unknown tool: {name!r}") from None

    def names(self) -> tuple[str, ...]:
        """Registered tool names, sorted for determinism."""
        return tuple(sorted(self._tools))

    def __contains__(self, name: object) -> bool:
        return name in self._tools


_default_registry: ToolRegistry | None = None


def default_registry() -> ToolRegistry:
    """Return the shared registry of builtin tools (built lazily once).

    Only plain functions are wired here — no tool client is constructed.
    """
    global _default_registry
    if _default_registry is None:
        from graphs.tools.ddgs import ddgs_search
        from graphs.tools.mem0 import mem0_remember, mem0_retrieve

        _default_registry = ToolRegistry()
        _default_registry.register("ddgs", ddgs_search)
        _default_registry.register("mem0_remember", mem0_remember)
        _default_registry.register("mem0_retrieve", mem0_retrieve)
    return _default_registry