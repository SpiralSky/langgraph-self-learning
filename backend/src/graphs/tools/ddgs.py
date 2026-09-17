"""DuckDuckGo web search tool.

Network access happens only when the registered callable actually runs —
importing this module (or the tools package) constructs nothing and opens
no connections. Tests must never invoke this; they inject fakes at the
registry level instead.
"""

from __future__ import annotations

from ddgs import DDGS
from ddgs.exceptions import DDGSException

from graphs.tools.registry import ToolError, int_arg, require_arg

_DEFAULT_MAX_RESULTS = 5


def ddgs_search(args: dict) -> str:
    """Search the web and return a compact text digest of the top results.

    :param dict args: ``{"query": str, "max_results": int (default 5)}``.
        ``query`` is required; ``max_results`` is capped at 10.
    :type args: dict
    :return: Numbered ``title / URL / snippet`` entries, one block per
        result; ``"No results."`` when the search matches nothing.
    :rtype: str
    :raises ToolError: if the underlying search raises ``DDGSException``.
    """
    query = str(require_arg(args, "query"))
    if not query.strip():
        raise ValueError("tool argument 'query' must be a non-empty string")
    max_results = min(max(int_arg(args, "max_results", _DEFAULT_MAX_RESULTS), 1), 10)
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)
    except DDGSException as exc:
        raise ToolError(f"ddgs search failed: {exc}") from exc

    if not results:
        return "No results."
    blocks = [
        f"{index}. {item.get('title', '')}\n   {item.get('href', '')}\n"
        f"   {item.get('body', '')}"
        for index, item in enumerate(results, start=1)
    ]
    return "\n\n".join(block.strip() for block in blocks)