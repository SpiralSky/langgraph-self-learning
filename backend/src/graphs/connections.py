"""Edge flavors stored inside a :class:`graphs.graph.Graph`.

Both classes are pure data: the graph holds them keyed by a manual string id
and enforces structure. ``Connection`` is a plain one-to-one edge;
``RoutingConnection`` fans the source out to an LLM-chosen candidate target
at runtime. ``model`` is carried for a later ``compile`` step and is typed
loosely (``object | None``) so this module never depends on a specific
langchain type.
"""

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class Connection:
    """A plain one-to-one edge from ``source`` node to ``target`` node."""

    source: str
    target: str
    type: ClassVar[str] = "standard"

    def as_dict(self) -> dict:
        """Type-tagged JSON-ready dict (see ``graphs.serialization``)."""
        return {"type": self.type, "source": self.source, "target": self.target}


@dataclass(frozen=True)
class RoutingConnection:
    """An LLM-routed fan-out edge: ``source`` reaches one of ``targets``.

    :param source: The from-node id.
    :param targets: The candidate to-node ids (1+).
    :param routing_prompt: Instruction telling the LLM how to pick a target.
    :param model: langchain ``Runnable`` used to choose the target at runtime
        (carried here, typed loosely to keep this module decoupled).
    """

    source: str
    targets: tuple[str, ...]
    routing_prompt: str
    model: object | None = None
    type: ClassVar[str] = "routing"

    def as_dict(self) -> dict:
        """Type-tagged JSON-ready dict; ``model`` is dropped (never serialized)."""
        return {
            "type": self.type,
            "source": self.source,
            "targets": list(self.targets),
            "routing_prompt": self.routing_prompt,
        }