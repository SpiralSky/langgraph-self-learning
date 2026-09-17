"""Editable graph structure: ``Graph`` with manual-id nodes and connections.

A :class:`Graph` holds nodes (``AbstractNode`` instances) and connections
(``Connection`` / ``RoutingConnection``) keyed by manual, LLM-referenceable
string ids. It offers full edit ops (add/remove/update), explicit structural
validation, an LLM-parseable ``render()``, name lookup, and ``compile()`` —
turning the structure into a runnable langgraph ``StateGraph``.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

from langchain_core.runnables import Runnable
from langgraph.constants import END as _LG_END
from langgraph.constants import START as _LG_START
from langgraph.graph import StateGraph
from pydantic import BaseModel, create_model

from graphs.structure.connections import Connection, RoutingConnection
from graphs.nodes.base import PLACEHOLDER_RE, AbstractNode
from graphs.persistence.serialization import (
    connection_from_dict,
    deserialize_annotation,
    node_from_dict,
    serialize_annotation,
)
from graphs.structure.state import LearningGraphState

if TYPE_CHECKING:
    from typing import TypeAlias

START = "START"
END = "END"
RESERVED_IDS = frozenset({START, END})

ConnectionSpec: TypeAlias = Connection | RoutingConnection


class GraphValidationError(ValueError):
    """Aggregate structural/schema validation error.

    ``errors`` lists every problem found; the message joins them so no
    single problem hides behind the first failing check.
    """

    def __init__(self, errors: list[str]) -> None:
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


@dataclass(frozen=True)
class NodeView:
    """Read-only LLM-inspection view of a node (not the raw node object)."""

    id: str
    name: str
    description: str
    prompt: str
    params: dict[str, type]
    writes: dict[str, str]


class Graph:
    """An editable, frameless graph structure.

    :param state_model: Pydantic model every node reads from / writes to.
        Defaults to ``LearningGraphState``; custom per-graph schemas are
        intended (inner graphs declare their own).
    """

    def __init__(self, *, state_model: type[BaseModel] = LearningGraphState) -> None:
        self._state_model = state_model
        self._nodes: dict[str, AbstractNode] = {}
        self._node_llm: dict[str, Runnable | None] = {}
        self._edges: dict[str, ConnectionSpec] = {}
        self._edge_keys: dict[tuple, str] = {}
        self._tombstoned_nodes: set[str] = set()
        self._tombstoned_edges: set[str] = set()

    @property
    def state_model(self) -> type[BaseModel]:
        """The pydantic state model every node reads/writes against."""
        return self._state_model

    # -------------------------------------------------------------- edit ops

    def add_node(
        self, node_id: str, node: AbstractNode, llm: Runnable | None = None
    ) -> None:
        """Store ``node`` under ``node_id`` with an optional per-node LLM."""
        self._check_node_id(node_id)
        self._nodes[node_id] = node
        self._node_llm[node_id] = llm

    def add_edge(
        self,
        edge_id: str,
        source: str,
        target: str | Sequence[str],
        *,
        routing_prompt: str | None = None,
        model: Runnable | None = None,
    ) -> None:
        """Add a connection under ``edge_id``.

        A single ``target`` with no ``routing_prompt`` yields a
        :class:`Connection`; multiple targets (or any target) with a
        ``routing_prompt`` yields a :class:`RoutingConnection`.
        """
        if not isinstance(edge_id, str) or not edge_id:
            raise ValueError("edge id must be a non-empty string")
        if edge_id in self._edges:
            raise ValueError(f"edge id already present: {edge_id!r}")
        if edge_id in self._tombstoned_edges:
            raise ValueError(f"edge id was removed and cannot be re-added: {edge_id!r}")
        targets = (target,) if isinstance(target, str) else tuple(target)
        connection = self._build_connection(
            edge_id, source, targets, routing_prompt=routing_prompt, model=model
        )
        self._edges[edge_id] = connection
        self._edge_keys[self._edge_key(connection)] = edge_id

    def remove_node(self, node_id: str) -> None:
        """Remove ``node_id`` and CASCADE every edge incident to it."""
        if node_id not in self._nodes:
            raise KeyError(node_id)
        self._nodes.pop(node_id)
        self._node_llm.pop(node_id, None)
        for edge_id in list(self._edges):
            connection = self._edges[edge_id]
            source, targets = self._edge_endpoints(connection)
            if node_id == source or node_id in targets:
                self._drop_edge(edge_id)
        self._tombstoned_nodes.add(node_id)

    def remove_edge(self, edge_id: str) -> None:
        """Remove edge by id (KeyError if absent)."""
        if edge_id not in self._edges:
            raise KeyError(edge_id)
        self._drop_edge(edge_id)

    def update_node(
        self, node_id: str, new_node: AbstractNode, llm: Runnable | None = None
    ) -> None:
        """Replace the instance under ``node_id``; no graph-wide revalidation."""
        if node_id not in self._nodes:
            raise KeyError(node_id)
        self._nodes[node_id] = new_node
        self._node_llm[node_id] = llm

    def update_edge(self, edge_id: str, **changes) -> None:
        """Replace the stored edge under ``edge_id`` re-running add-edge checks."""
        if edge_id not in self._edges:
            raise KeyError(edge_id)
        old = self._edges[edge_id]
        source = changes.get("source", old.source)
        if "targets" in changes:
            target = changes["targets"]
            targets = (target,) if isinstance(target, str) else tuple(target)
        elif "target" in changes:
            target = changes["target"]
            targets = (target,) if isinstance(target, str) else tuple(target)
        else:
            _, targets = self._edge_endpoints(old)
        if isinstance(old, RoutingConnection):
            routing_prompt = changes.get("routing_prompt", old.routing_prompt)
            model = changes.get("model", old.model)
        else:
            routing_prompt = changes.get("routing_prompt", None)
            model = changes.get("model", None)
        connection = self._build_connection(
            edge_id,
            source,
            targets,
            routing_prompt=routing_prompt,
            model=model,
            exclude_edge_id=edge_id,
        )
        self._edge_keys.pop(self._edge_key(old), None)
        self._edges[edge_id] = connection
        self._edge_keys[self._edge_key(connection)] = edge_id

    # ------------------------------------------- validation helpers (private)

    def _check_node_id(self, node_id: str) -> None:
        if not isinstance(node_id, str) or not node_id:
            raise ValueError("node id must be a non-empty string")
        if node_id in RESERVED_IDS:
            raise ValueError(f"node id is reserved: {node_id!r}")
        if node_id in self._nodes:
            raise ValueError(f"node id already present: {node_id!r}")
        if node_id in self._tombstoned_nodes:
            raise ValueError(f"node id was removed and cannot be re-added: {node_id!r}")

    def _build_connection(
        self,
        edge_id: str,
        source: str,
        targets: tuple[str, ...],
        *,
        routing_prompt: str | None,
        model: Runnable | None = None,
        exclude_edge_id: str | None = None,
    ) -> ConnectionSpec:
        if not isinstance(source, str) or not source:
            raise ValueError(f"edge {edge_id!r}: source must be a non-empty string")
        if not targets:
            raise ValueError(f"edge {edge_id!r}: target must name at least one node")
        if routing_prompt is None:
            if len(targets) != 1:
                raise ValueError(
                    f"edge {edge_id!r}: multiple targets require routing_prompt"
                )
            connection: ConnectionSpec = Connection(source=source, target=targets[0])
        else:
            if not isinstance(routing_prompt, str) or not routing_prompt:
                raise ValueError(
                    f"edge {edge_id!r}: routing_prompt must be a non-empty string"
                )
            connection = RoutingConnection(
                source=source,
                targets=targets,
                routing_prompt=routing_prompt,
                model=model,
            )
        self._check_role(edge_id, source, targets)
        self._check_endpoints(edge_id, source, targets)
        key = self._edge_key(connection)
        existing = self._edge_keys.get(key)
        if existing is not None and existing != exclude_edge_id:
            raise ValueError(
                f"edge {edge_id!r}: equivalent edge {existing!r} already exists: {key!r}"
            )
        return connection

    def _check_role(self, edge_id: str, source: str, targets: tuple[str, ...]) -> None:
        if source == END:
            raise ValueError(f"edge {edge_id!r}: 'END' may only be a target")
        for target in targets:
            if target == START:
                raise ValueError(f"edge {edge_id!r}: 'START' may only be a source")

    def _check_endpoints(
        self, edge_id: str, source: str, targets: tuple[str, ...]
    ) -> None:
        for ref in (source, *targets):
            if ref in RESERVED_IDS:
                continue
            if ref not in self._nodes:
                raise ValueError(
                    f"edge {edge_id!r}: endpoint node {ref!r} does not exist"
                )

    def _drop_edge(self, edge_id: str) -> None:
        connection = self._edges.pop(edge_id)
        self._edge_keys.pop(self._edge_key(connection), None)
        self._tombstoned_edges.add(edge_id)

    @staticmethod
    def _edge_key(connection: ConnectionSpec) -> tuple:
        source, targets = Graph._edge_endpoints(connection)
        return (source, frozenset(targets), connection.type)

    @staticmethod
    def _edge_endpoints(connection: ConnectionSpec) -> tuple[str, tuple[str, ...]]:
        if isinstance(connection, RoutingConnection):
            return connection.source, connection.targets
        return connection.source, (connection.target,)

    @staticmethod
    def _alias(node_id: str) -> str:
        """Map a graph-level ``START``/``END`` id to the langgraph constant."""
        if node_id == START:
            return _LG_START
        if node_id == END:
            return _LG_END
        return node_id

    @staticmethod
    def _routing_path(edge_id: str, routing_prompt: str, llm: Runnable):
        """A StateGraph path callable: run an LLM and return the chosen target."""

        def route(state) -> str:
            result = llm.invoke(routing_prompt)
            return str(result.content).strip()

        route.__name__ = f"route_{edge_id}"
        return route

    # --------------------------------------------------------------- compile

    def compile(self, llm: Runnable | None = None) -> StateGraph:
        """Compile the structure into a runnable langgraph ``StateGraph``.

        Validates the graph first, then maps every node and connection onto
        a langgraph ``StateGraph``: node ids become langgraph node names,
        ``START``/``END`` alias to the framework constants, and each routing
        edge becomes a conditional edge whose path callable runs the edge's
        routing prompt through an LLM and returns the chosen candidate id.

        :param llm: Default LLM for any node/edge without its own override
            (per-node ``add_node`` llm wins; routing edges use their own
            ``add_edge`` model). A node or routing edge with neither raises
            ``ValueError``.
        """
        self.validate()
        builder = StateGraph(self._state_model)
        for node_id, node in self._nodes.items():
            node_llm = self._node_llm.get(node_id) or llm
            if node_llm is None:
                raise ValueError(
                    f"node {node_id!r} needs an LLM; pass one via add_node "
                    "or to compile"
                )
            builder.add_node(node_id, node.get_node(node_llm))
        for edge_id, connection in self._edges.items():
            source, targets = self._edge_endpoints(connection)
            if isinstance(connection, RoutingConnection):
                routing_llm = connection.model or llm
                if routing_llm is None:
                    raise ValueError(
                        f"routing edge {edge_id!r} needs an LLM to pick a "
                        "target; pass one via add_edge or to compile"
                    )
                builder.add_conditional_edges(
                    self._alias(source),
                    self._routing_path(edge_id, connection.routing_prompt, routing_llm),
                    {target: self._alias(target) for target in targets},
                )
            else:
                builder.add_edge(self._alias(source), self._alias(targets[0]))
        return builder.compile()

    # ------------------------------------------------------------ accessors

    def nodes(self) -> dict[str, AbstractNode]:
        """Snapshot of ``{node_id: deep copy of node}``."""
        return {node_id: deepcopy(node) for node_id, node in self._nodes.items()}

    def edges(self) -> dict[str, ConnectionSpec]:
        """Snapshot of ``{edge_id: connection}``."""
        return dict(self._edges)

    def get_node(self, node_id: str) -> AbstractNode:
        """Deep copy of the node stored under ``node_id`` (KeyError if absent).

        The graph's stored instance is read-only through this getter; use
        ``update_node`` to replace the stored prototype.
        """
        return deepcopy(self._nodes[node_id])

    def get_edge(self, edge_id: str) -> ConnectionSpec:
        """Return the connection stored under ``edge_id`` (KeyError if absent)."""
        return self._edges[edge_id]

    def get_node_llm(self, node_id: str) -> Runnable | None:
        """Return the per-node LLM override (None when none was set)."""
        return self._node_llm[node_id]

    def __len__(self) -> int:
        """Number of nodes."""
        return len(self._nodes)

    def __contains__(self, node_id: object) -> bool:
        """True when ``node_id`` is a stored node."""
        return node_id in self._nodes

    # ------------------------------------------------------------ validation

    def validate(self) -> None:
        """Full structural + schema validation; raises :class:`GraphValidationError`."""
        errors: list[str] = []
        self._validate_ids(errors)
        self._validate_connections(errors)
        self._validate_state_schema(errors)
        if errors:
            raise GraphValidationError(errors)

    def _validate_ids(self, errors: list[str]) -> None:
        for node_id in self._nodes:
            if not isinstance(node_id, str) or not node_id:
                errors.append(f"node id must be a non-empty string: {node_id!r}")
            if node_id in RESERVED_IDS:
                errors.append(f"node id is reserved: {node_id!r}")
            if node_id in self._tombstoned_nodes:
                errors.append(f"node id is still tombstoned: {node_id!r}")
        for edge_id in self._edges:
            if not isinstance(edge_id, str) or not edge_id:
                errors.append(f"edge id must be a non-empty string: {edge_id!r}")
            if edge_id in self._tombstoned_edges:
                errors.append(f"edge id is still tombstoned: {edge_id!r}")

    def _validate_connections(self, errors: list[str]) -> None:
        start_sources = 0
        for edge_id, connection in self._edges.items():
            source, targets = self._edge_endpoints(connection)
            if source == START:
                start_sources += 1
            if source == END:
                errors.append(f"edge {edge_id!r}: 'END' may only be a target")
            for target in targets:
                if target == START:
                    errors.append(f"edge {edge_id!r}: 'START' may only be a source")
            for ref in (source, *targets):
                if ref not in self._nodes and ref not in RESERVED_IDS:
                    errors.append(f"edge {edge_id!r}: references missing node {ref!r}")
        if start_sources != 1:
            errors.append(
                f"graph must have exactly one START-source edge (found {start_sources})"
            )

    def _validate_state_schema(self, errors: list[str]) -> None:
        model_fields = set(self._state_model.model_fields)
        for node_id, node in self._nodes.items():
            params = getattr(node, "params", None) or {}
            writes = getattr(node, "writes", None) or {}
            reads = set(params)
            state_keys = set(writes.values())
            if not reads <= model_fields:
                errors.append(
                    f"node {node_id!r} reads state fields not in the state model: "
                    f"{sorted(reads - model_fields)}"
                )
            if not state_keys <= model_fields:
                errors.append(
                    f"node {node_id!r} writes state keys not in the state model: "
                    f"{sorted(state_keys - model_fields)}"
                )
            placeholders = set(PLACEHOLDER_RE.findall(getattr(node, "prompt", "") or ""))
            orphan = placeholders - set(params)
            if orphan:
                errors.append(
                    f"node {node_id!r} prompt placeholders not declared in params: "
                    f"{sorted(orphan)}"
                )

    # -------------------------------------------------- renderer + name lookup

    def to_dict(self) -> dict:
        """Type-tagged JSON-ready dict; ``node_llm`` and edge ``model`` dropped.

        The state model is either ``{"kind": "LearningGraphState"}`` or
        ``{"kind": "dynamic", "fields": {field: serialized-annotation}}``;
        the dynamic form is rebuilt via pydantic ``type(...)`` on restore.
        Tombstones are not persisted — restored graphs are fresh structures.
        """
        if self._state_model is LearningGraphState:
            state_model: dict = {"kind": "LearningGraphState"}
        else:
            state_model = {
                "kind": "dynamic",
                "fields": {
                    name: serialize_annotation(field.annotation)
                    for name, field in self._state_model.model_fields.items()
                },
            }
        return {
            "type": "graph",
            "state_model": state_model,
            "nodes": {
                node_id: node.to_dict() for node_id, node in self._nodes.items()
            },
            "edges": {edge_id: edge.as_dict() for edge_id, edge in self._edges.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> Graph:
        """Rebuild a :class:`Graph` from a ``to_dict`` payload.

        Nodes are added first (dict order), then edges; connections keep their
        routing configuration while per-node LLM overrides and edge models are
        not recoverable — supply them again at ``compile()`` time.
        """
        state = data.get("state_model", {})
        kind = state.get("kind")
        if kind == "LearningGraphState":
            model: type[BaseModel] = LearningGraphState
        elif kind == "dynamic":
            model = create_model(
                "GeneratedState",
                **{
                    name: (deserialize_annotation(annotation), None)
                    for name, annotation in state.get("fields", {}).items()
                },
            )
        else:
            raise ValueError(f"unknown state model kind: {kind!r}")

        graph = cls(state_model=model)
        for node_id, node_data in data.get("nodes", {}).items():
            graph.add_node(node_id, node_from_dict(node_data))
        for edge_id, edge_data in data.get("edges", {}).items():
            connection = connection_from_dict(edge_data)
            if isinstance(connection, RoutingConnection):
                graph.add_edge(
                    edge_id,
                    connection.source,
                    connection.targets,
                    routing_prompt=connection.routing_prompt,
                )
            else:
                graph.add_edge(edge_id, connection.source, connection.target)
        return graph

    def _label(self, node_id: str, *, show_ids: bool) -> str:
        if node_id in RESERVED_IDS:
            return node_id
        name = self._nodes[node_id].name if node_id in self._nodes else node_id
        return f"{name} ({node_id})" if show_ids else name

    def render(
        self,
        *,
        show_ids: bool = False,
        show_descriptions: bool = False,
        show_prompts: bool = False,
    ) -> str:
        """Render the graph as deterministic, LLM-parseable lines.

        Default shows the name chain (``START -> n1 -> n2 -> END``); routing
        edges render their candidates as ``n1 ->? (n2 | n3)``. With the show
        flags, per-node blocks with descriptions/prompts are appended.
        """
        lines: list[str] = []
        for edge in self._edges.values():
            source, targets = self._edge_endpoints(edge)
            source_label = self._label(source, show_ids=show_ids)
            if isinstance(edge, RoutingConnection):
                candidates = " | ".join(
                    self._label(t, show_ids=show_ids) for t in targets
                )
                lines.append(f"{source_label} ->? ({candidates})")
            else:
                target_label = self._label(targets[0], show_ids=show_ids)
                lines.append(f"{source_label} -> {target_label}")
        if show_descriptions or show_prompts:
            for node_id, node in self._nodes.items():
                block = [self._label(node_id, show_ids=show_ids)]
                if show_descriptions:
                    block.append(f"  description: {node.description}")
                if show_prompts:
                    block.append(f"  prompt: {node.prompt}")
                lines.extend(block)
        return "\n".join(lines) if lines else ""

    def nodes_by_name(self, name: str) -> list[NodeView]:
        """Return a read-only :class:`NodeView` for every node named ``name``."""
        return [
            NodeView(
                id=node_id,
                name=node.name,
                description=node.description,
                prompt=node.prompt,
                params=dict(node.params),
                writes=dict(node.writes),
            )
            for node_id, node in self._nodes.items()
            if node.name == name
        ]
