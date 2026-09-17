"""Node-system foundation: the ``GraphNode`` protocol and ``AbstractNode`` ABC.

A graph node bundles retrieval metadata (``name``, ``description``,
``prompt``) with a parameter schema (``params``) and output map (``writes``)
and produces a LangGraph-ready callable on demand via ``get_node(llm)``.

Concrete node types (e.g. ``TextNode``) implement the ``AbstractNode`` ABC;
anything with the right attributes still satisfies the ``GraphNode`` protocol
structurally. Nothing imports langgraph here, keeping the node system loosely
coupled to the graph runtime.
"""

import re
from abc import ABC, abstractmethod
from typing import Protocol, Self

from langchain_core.runnables import Runnable
from pydantic import BaseModel

from graphs.api.stats import OnlineStats

PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_]\w*)\}")


class DualCallable(Protocol):
    """A LangGraph-ready node callable with sync and async entry points.

    Both methods take the full graph state and return a partial-state
    update; ``invoke`` is synchronous while ``ainvoke`` is awaitable.
    """

    def invoke(self, state: dict, *args: object, **kwargs: object) -> dict: ...

    async def ainvoke(self, state: dict, *args: object, **kwargs: object) -> dict: ...


class GraphNode(Protocol):
    """A retrievable graph node: metadata plus a node-callable factory.

    ``get_node(llm)`` returns the LangGraph-ready callable that validates
    the node's input, fills the prompt, and invokes the supplied LLM.
    """

    name: str
    description: str
    prompt: str
    input_shape: type | type[BaseModel] | None

    def get_node(self, llm: Runnable) -> DualCallable: ...


class AbstractNode(ABC):
    """Graph-agnostic base for concrete nodes.

    Declares retrieval metadata (``name``, ``description``, ``prompt``), the
    by-name inputs the node reads from state (``params``) and the explicit
    output map (``writes``: node-local field name -> state key).

    ``params={}`` denotes a *generator* node that reads nothing and produces
    output from the LLM alone. Prompt placeholders ``{param}`` must be a
    subset of ``params`` — an orphan placeholder is a construction error.
    Writes default to ``{"result": "response"}``.
    """

    name: str
    description: str
    prompt: str
    params: dict[str, type]
    writes: dict[str, str]
    run_counts: int
    output_tokens: OnlineStats
    output_time: OnlineStats

    def __init__(
        self,
        name: str,
        description: str,
        prompt: str,
        params: dict[str, type] | None = None,
        writes: dict[str, str] | None = None,
    ) -> None:
        for attr, value in (
            ("name", name),
            ("description", description),
            ("prompt", prompt),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{attr} must be a non-empty string")
        if params is None:
            params = {}
        if writes is None:
            writes = {"result": "response"}
        for field, state_key in writes.items():
            if not isinstance(field, str) or not field:
                raise ValueError("each write field must be a non-empty string")
            if not isinstance(state_key, str) or not state_key:
                raise ValueError("each write state key must be a non-empty string")
        placeholders = set(PLACEHOLDER_RE.findall(prompt))
        missing = placeholders - set(params)
        if missing:
            raise ValueError(
                f"prompt placeholders not declared in params: {sorted(missing)}"
            )
        self.name = name
        self.description = description
        self.prompt = prompt
        self.params = dict(params)
        self.writes = dict(writes)
        self.run_counts = 0
        self.output_tokens = OnlineStats()
        self.output_time = OnlineStats()

    @staticmethod
    def _checked_float(value: object, label: str) -> float:
        """Coerce ``value`` to a finite, non-negative float or raise ValueError."""
        if isinstance(value, bool):
            raise ValueError(f"{label} must be numeric, got bool")
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"{label} must be numeric, got {type(value).__name__}"
            ) from None
        if number != number:  # NaN
            raise ValueError(f"{label} must not be NaN")
        if number < 0:
            raise ValueError(f"{label} must not be negative")
        return number

    def record_run(
        self,
        *,
        tokens: object | None = None,
        elapsed_seconds: object | None = None,
    ) -> None:
        """Record a single run: bump the count and fold time/tokens in.

        ``elapsed_seconds`` is validated and recorded when not ``None``; the
        same holds for ``tokens``. Both reject NaN/negative values with a
        ``ValueError``. Nodes are collection-agnostic — this mutates only this
        instance (shared across a collection via reference semantics).
        """
        seconds = (
            None if elapsed_seconds is None
            else self._checked_float(elapsed_seconds, "elapsed_seconds")
        )
        count = (
            None if tokens is None else self._checked_float(tokens, "output_tokens")
        )
        self.run_counts += 1
        if seconds is not None:
            self.output_time.update(seconds)
        if count is not None:
            self.output_tokens.update(count)

    def record_result(self, result: object, elapsed_seconds: object | None) -> None:
        """Record a run from an LLM result, extracting ``output_tokens`` usage.

        Token usage is read from ``usage_metadata["output_tokens"]``; a missing
        attribute/key or a non-numeric value skips the token update (time is
        still recorded).
        """
        usage = getattr(result, "usage_metadata", None)
        tokens: object | None = None
        if isinstance(usage, dict):
            value = usage.get("output_tokens")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                tokens = value
        self.record_run(tokens=tokens, elapsed_seconds=elapsed_seconds)

    def _carry_stats(self, new: "AbstractNode") -> None:
        """Copy this node's run stats onto a freshly rebuilt instance."""
        new.run_counts = self.run_counts
        new.output_tokens = self.output_tokens
        new.output_time = self.output_time

    def updated(self, **changes: object) -> Self:
        """Return a new, validated instance with ``changes`` applied.

        Rebuilds through the class constructor so every provided value is
        re-validated (non-empty metadata, non-empty writes, prompt
        placeholders a subset of ``params``). The current instance is never
        mutated.

        :raises TypeError: for unknown field names.
        """
        allowed = {"name", "description", "prompt", "params", "writes"}
        unknown = sorted(set(changes) - allowed)
        if unknown:
            raise TypeError(
                f"unsupported fields: {unknown}; supported: {sorted(allowed)}"
            )
        rebuilt = type(self)(
            name=changes.get("name", self.name),
            description=changes.get("description", self.description),
            prompt=changes.get("prompt", self.prompt),
            params=changes.get("params", self.params),
            writes=changes.get("writes", self.writes),
        )
        self._carry_stats(rebuilt)
        return rebuilt

    @abstractmethod
    def get_node(self, llm: Runnable) -> DualCallable:
        """Bind an LLM and return the LangGraph-ready dual callable."""