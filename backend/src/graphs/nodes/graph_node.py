"""``GraphNode``: wrap a full graph as a single reusable node.

A :class:`GraphNode` bundles a complete inner :class:`graphs.graph.Graph`
(retrieval metadata + ``input_map``/``output_map`` bridge maps) and exposes it
through the ``AbstractNode`` interface, so it can be added to a parent graph,
nested arbitrarily deep, and stored/pruned in a ``NodeCollection`` untouched.

Execution: on ``get_node(llm)`` the node yields a LangGraph-ready dual
callable that (1) reads the outer state by ``params`` (the outer keys named by
``input_map`` values), (2) builds the inner initial state mapping each outer
value onto its inner key, (3) runs the inner compiled graph (cached per bound
LLM), and (4) surfaces each ``output_map`` inner key to its outer state key —
multiple entries produce the multi-output partial update.
"""

from typing import TYPE_CHECKING, Self

from pydantic import BaseModel, TypeAdapter

from graphs.nodes.base import AbstractNode, DualCallable
from graphs.serialization import register_node_type

if TYPE_CHECKING:
    from langchain_core.runnables import Runnable

    from graphs.graph import Graph


class _GraphNodeFn:
    """LangGraph-ready dual callable produced by ``GraphNode.get_node``.

    The outer reads are validated once per invocation with one ``TypeAdapter``
    per param (mirroring ``_TextNodeFn``); the inner compiled graph is built
    lazily on first use and cached for the lifetime of this (per-LLM) callable.
    """

    def __init__(
        self,
        *,
        name: str,
        graph: "Graph",
        input_map: dict[str, str],
        output_map: dict[str, str],
        params: dict[str, type],
        llm: "Runnable",
    ) -> None:
        self.__name__ = name
        self._graph = graph
        self._input_map = input_map
        self._output_map = output_map
        self._validators = {p: TypeAdapter(ann) for p, ann in params.items()}
        self._llm = llm
        self._compiled = None

    def __call__(self, state: dict) -> dict:
        return self.invoke(state)

    @staticmethod
    def _read(state: dict, key: str) -> object:
        """Read ``key`` from a pydantic-model or plain-dict state."""
        if isinstance(state, BaseModel):
            return getattr(state, key)
        return state[key]

    def _compiled_graph(self):
        if self._compiled is None:
            self._compiled = self._graph.compile(self._llm)
        return self._compiled

    def _outer_state(self, state: dict) -> dict:
        """Validate the by-name outer reads and map them onto inner keys."""
        outer = {
            param: validator.validate_python(self._read(state, param))
            for param, validator in self._validators.items()
        }
        return {
            inner_key: outer[outer_key]
            for inner_key, outer_key in self._input_map.items()
        }

    def invoke(self, state: dict) -> dict:
        """Run the inner graph on the bridged initial state, surface outputs."""
        inner_state = self._outer_state(state)
        result = self._compiled_graph().invoke(inner_state)
        return {
            outer_key: result[inner_key]
            for inner_key, outer_key in self._output_map.items()
        }

    async def ainvoke(self, state: dict) -> dict:
        """Async twin of ``invoke`` using the inner graph's ``ainvoke``."""
        inner_state = self._outer_state(state)
        result = await self._compiled_graph().ainvoke(inner_state)
        return {
            outer_key: result[inner_key]
            for inner_key, outer_key in self._output_map.items()
        }


class GraphNode(AbstractNode):
    """Wrap a full inner graph as an ``AbstractNode``.

    :param name: Node name (used as the LangGraph node name in parent graphs).
    :type name: str
    :param description: What this node does, used for retrieval.
    :type description: str
    :param prompt: Human/LLM description of what the inner graph does; used for
        retrieval metadata (e.g. in ``NodeCollection``), never executed.
    :type prompt: str
    :param graph: The inner graph being wrapped; it declares its own state
        model. Will be compiled lazily on first execution.
    :type graph: Graph
    :param input_map: ``{inner_key: outer_key}`` — which inner state keys this
        node sources from the OUTER state. Required and non-empty.
    :type input_map: dict[str, str]
    :param output_map: ``{inner_key: outer_key}`` — which inner terminal keys
        to surface to which OUTER keys. Required and non-empty; multiple
        entries produce the multi-output case.
    :type output_map: dict[str, str]

    ``params`` (used by the parent graph's validation) are derived from
    ``input_map``: every param is an outer key referenced by an ``input_map``
    value, annotated with the inner field's type. ``writes`` map each
    ``output_map`` inner key to its outer state key.
    """

    def __init__(
        self,
        name: str,
        description: str,
        prompt: str,
        graph: "Graph",
        input_map: dict[str, str],
        output_map: dict[str, str],
    ) -> None:
        self.graph = graph
        self.input_map = self._validate_map(input_map, "input_map")
        self.output_map = self._validate_map(output_map, "output_map")
        model_fields = graph.state_model.model_fields
        for inner_key in (*self.input_map, *self.output_map):
            if inner_key not in model_fields:
                raise ValueError(
                    f"inner key {inner_key!r} is not a field of the inner "
                    f"state model {graph.state_model.__name__}"
                )
        params = {
            outer_key: model_fields[inner_key].annotation
            for inner_key, outer_key in self.input_map.items()
        }
        missing = set(params) - set(self.input_map.values())
        if missing:
            raise ValueError(
                f"params not backed by an input_map outer key: {sorted(missing)}"
            )
        writes = {
            inner_key: outer_key for inner_key, outer_key in self.output_map.items()
        }
        super().__init__(name, description, prompt, params=params, writes=writes)

    @staticmethod
    def _validate_map(value: dict[str, str], label: str) -> dict[str, str]:
        if not isinstance(value, dict) or not value:
            raise ValueError(f"{label} must be a non-empty dict")
        for key, mapped in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{label} keys must be non-empty strings")
            if not isinstance(mapped, str) or not mapped:
                raise ValueError(f"{label} values must be non-empty strings")
        return dict(value)

    def get_node(self, llm: "Runnable") -> DualCallable:
        """Bind an LLM and return the LangGraph-ready dual callable."""
        return _GraphNodeFn(
            name=self.name,
            graph=self.graph,
            input_map=self.input_map,
            output_map=self.output_map,
            params=self.params,
            llm=llm,
        )

    def updated(self, **changes: object) -> Self:
        """Return a new validated instance; only metadata may change.

        ``params``/``writes`` are derived from the ``input_map``/``output_map``
        bridge, so they cannot be edited independently — swap in a fresh node
        via the collection's ``replace()`` instead.

        :raises ValueError: if ``params`` or ``writes`` is changed.
        :raises TypeError: for unknown field names.
        """
        allowed = {"name", "description", "prompt"}
        for key in changes:
            if key in ("params", "writes"):
                raise ValueError(
                    f"{key} is derived from input_map/output_map; use replace()"
                )
            if key not in allowed:
                raise TypeError(
                    f"unsupported fields: {sorted(set(changes))}; supported: "
                    f"{sorted(allowed)}"
                )
        return type(self)(
            name=changes.get("name", self.name),
            description=changes.get("description", self.description),
            prompt=changes.get("prompt", self.prompt),
            graph=self.graph,
            input_map=self.input_map,
            output_map=self.output_map,
        )

    def to_dict(self) -> dict:
        """Type-tagged JSON-ready dict (``"type": "graph"``).

        The inner graph is serialized in full (its own state model included);
        ``params``/``writes`` are omitted because they are derived from the
        ``input_map``/``output_map`` bridge at construction time.
        """
        return {
            "type": "graph",
            "name": self.name,
            "description": self.description,
            "prompt": self.prompt,
            "graph": self.graph.to_dict(),
            "input_map": dict(self.input_map),
            "output_map": dict(self.output_map),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GraphNode":
        """Rebuild a :class:`GraphNode` from a ``to_dict`` payload.

        The inner ``Graph`` is imported lazily to keep module-level coupling
        loose (mirroring the ``TYPE_CHECKING`` convention above).
        """
        from graphs.graph import Graph

        return cls(
            name=data["name"],
            description=data["description"],
            prompt=data["prompt"],
            graph=Graph.from_dict(data["graph"]),
            input_map=dict(data["input_map"]),
            output_map=dict(data["output_map"]),
        )


register_node_type("graph", GraphNode.from_dict)