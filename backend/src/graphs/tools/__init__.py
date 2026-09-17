"""Builtin tool implementations for the whitelist registry.

Only ``graphs.tools`` provides concrete tools; the default registry wires
``ddgs`` (DuckDuckGo web search), ``mem0_remember`` and ``mem0_retrieve``
(local memory store). Each tool is a plain ``dict -> str`` callable so the
registry stays a dumb whitelist.

Network/tool clients are never constructed at import time; importing this
package has no side effects.
"""

from graphs.tools.registry import (
    ToolError,
    ToolRegistry,
    default_registry,
    int_arg,
    require_arg,
)

__all__ = [
    "ToolError",
    "ToolRegistry",
    "default_registry",
    "int_arg",
    "require_arg",
]