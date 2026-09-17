"""``GeneratorNode``: build a nested graph at runtime and run it in-turn.

A :class:`GeneratorNode` is the runtime counterpart to a catalog of reusable
steps: given the user's request plus the sectioned behavior points, it asks an
LLM for a single-pass list of builder tool calls (``add_node`` / ``add_edge``),
applies them to a fresh :class:`graphs.graph.Graph` with an automatically
derived inner state model, validates the structure (retrying the whole spec a
bounded number of times with error feedback), and finally executes the result
as a nested subgraph through the existing :class:`GraphNode` bridge machinery.

The node catalog is fixed: ``TextNode`` (type ``"text"``) and ``ToolCallNode``
(type ``"tool"``, whitelisted through a ``ToolRegistry``). Nothing arbitrary
can be instantiated. The core prompt template is a module constant and is
never rewritten; behavior points are data rendered into it per-node.

The generated artifact graph and any ``reuse``-flagged node ids are exposed on
the bound callable (``last_graph`` / ``reuse_ids``). With auto-save enabled
(the default) the reuse artifacts are wrapped into save-ready prototypes
(``wrap_reused``) and persisted into a dedicated :class:`NodeCollection` at the
end of the turn (``save_reused`` -> storage layer); ``auto_save=False`` keeps
the node fully transient.

When that collection holds saved steps, a separate internal **retrieve** step
selects the ones worth reusing for the request and the builder can pull them
into the generated graph via ``add_node(from_collection=...)`` instead of
building them from scratch; pulled nodes are the collection's **live**
instances, so run stats recorded while the nested graph runs fold back into
the collection.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

from langchain_core.runnables import Runnable
from pydantic import BaseModel, create_model

from graphs.api.collection import NodeCollection, NodeCollectionRecord
from graphs.learning.behaviors import BehaviorGroup, load_behaviors, render_behaviors
from graphs.structure.graph import RESERVED_IDS, Graph, GraphValidationError
from graphs.nodes.base import AbstractNode, DualCallable
from graphs.nodes.graph_node import GraphNode
from graphs.nodes.text_node import TextNode
from graphs.nodes.tool_node import ToolCallNode
from graphs.persistence.serialization import deserialize_annotation
from graphs.persistence.storage import COLLECTION_PATH, dump_collection, load_collection
from graphs.tools import ToolRegistry, default_registry

_GENERATOR_TEMPLATE = """Build a single-pass graph that fulfills the user's request.
Return the entire graph in ONE response as builder tool calls: zero or more
add_node calls followed by add_edge calls, and nothing else; do not chat.

add_node(id, type, name, description, prompt?, params?, writes?, tool?, reuse?)
  - type "text": an LLM step. "prompt" is an instruction template that may
    contain placeholders for the fields it reads, and every placeholder must
    be declared in "params": a mapping of field name to a type name; use
    "str" for text, "list" for a list of strings, "int" for an integer, and
    "dict" for an object. "writes" maps a local output name to a state key;
    the default (result -> response) makes that step's output the final
    user-facing answer.
  - type "tool": a whitelisted tool step. "tool" must be a registered name
    (for example ddgs, mem0_remember, or mem0_retrieve). The tool reads its
    arguments from the state key named "args"; to pass arguments add an
    upstream text step that writes "args" as a JSON object string.
  - set "reuse" to true to mark the step's output for saving into the
    reusable set, so it can be reused in later answers.
  - set "from_collection" to a collection node id to reuse an existing step
    from the "Reusable steps" list instead of building one; then omit
    "prompt", "params", "writes", and "tool" (fresh or pulled, never both).

add_edge(source, target)
  - wires step "source" to step "target"; the reserved ids START and END
    open and close the graph.

User request: {user_message}

Behaviors to follow:
"""

#: JSON-schema tool definitions bound to the LLM before the generator call.
#: ``add_node``/``add_edge`` are the ONLY tools a generated graph may emit —
#: the whitelist lives in the fixed instruction template and here.
BUILDER_TOOL_DEFS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "add_node",
            "description": (
                "Add a step to the generated graph. type 'text' adds an LLM "
                "step; type 'tool' adds a whitelisted tool step; "
                "'from_collection' reuses an existing step from the collection."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Unique step id referenced by add_edge."},
                    "type": {"type": "string", "enum": ["text", "tool"]},
                    "name": {"type": "string", "description": "Step name."},
                    "description": {"type": "string", "description": "What this step does."},
                    "prompt": {
                        "type": "string",
                        "description": "Required for text steps: instruction template whose placeholders are declared in params.",
                    },
                    "params": {
                        "type": "object",
                        "description": "Text steps only: {field: type-name} with type names str, int, float, bool, list, dict.",
                    },
                    "writes": {
                        "type": "object",
                        "description": '{output-name: state-key}; default {"result": "response"}.',
                    },
                    "tool": {
                        "type": "string",
                        "description": "Required for tool steps: whitelisted tool, e.g. ddgs, mem0_remember, mem0_retrieve.",
                    },
                    "from_collection": {
                        "type": "string",
                        "description": (
                            "Collection node id to reuse as-is instead of "
                            "building a fresh step; when set, omit prompt, "
                            "params, writes, and tool."
                        ),
                    },
                    "reuse": {
                        "type": "boolean",
                        "description": "Whether this step's output should be saved into the reusable set.",
                    },
                },
                "required": ["id", "type", "name", "description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_edge",
            "description": "Wire step source to step target; START and END are reserved ids.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                },
                "required": ["source", "target"],
            },
        },
    },
]


def decode_builder_calls(response: object) -> list[dict]:
    """Extract the builder call list from an LLM response.

    Prefers langchain ``tool_calls`` on the response (each item is a mapping
    with ``name`` and ``args``); when the runtime yields none, falls back to
    parsing the response content as a JSON object ``{"calls": [...]}``. Each
    call normalizes to ``{"name": str, "args": dict}``.

    :raises ValueError: when neither source yields a usable call list.
    """
    tool_calls = getattr(response, "tool_calls", None)
    if tool_calls:
        return [
            {"name": call.get("name"), "args": dict(call.get("args") or {})}
            for call in tool_calls
        ]
    content = getattr(response, "content", None)
    if isinstance(content, str) and content.strip():
        return _decode_json_calls(content)
    raise ValueError(
        "LLM response carried no builder tool calls (no tool_calls and no "
        "JSON content with a 'calls' list)"
    )


def _decode_json_calls(content: str) -> list[dict]:
    """Parse a JSON ``{"calls": [{name, args}, ...]}`` blob."""
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"builder JSON is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict) or "calls" not in payload:
        raise TypeError('builder JSON must be an object with a "calls" list')
    calls = payload["calls"]
    if not isinstance(calls, list):
        raise TypeError("builder JSON 'calls' must be a list")
    decoded: list[dict] = []
    for call in calls:
        if not isinstance(call, dict):
            raise TypeError(
                f"each builder call must be an object, got {type(call).__name__}"
            )
        args = call.get("args", {})
        if not isinstance(args, dict):
            raise TypeError(
                f"each builder call 'args' must be an object, got {type(args).__name__}"
            )
        decoded.append({"name": call.get("name"), "args": args})
    return decoded


#: Max records shown to the LLM in the retrieve-step catalog. The catalog
#: stays compact because the retrieve call is one extra LLM input per turn.
_CATALOG_LIMIT = 20

#: Fixed prompt for the internal retrieve step: given the user's request and a
#: catalog of reusable steps, it returns the collection node ids worth pulling
#: into the generated graph. Runs only when the reuse collection has nodes.
_RETRIEVE_TEMPLATE = """Select reusable steps for the user's request.

You can reuse existing steps from a node collection instead of building them
from scratch. A step is worth reusing when it already performs (or clearly
covers) part of the request.

Available reusable steps:
{catalog}

User request: {user_message}

Return ONLY a JSON object with the chosen steps' ids, e.g.
{{"ids": ["<id>", "..."]}} -- no other text. An empty list means nothing fits.
"""


def _decode_retrieved_ids(response: object) -> list[str]:
    """Extract a list of collection node ids from an LLM retrieve response.

    Mirrors :func:`decode_builder_calls`' fallback style: tool calls carrying
    an ``id``/``ids`` argument are preferred; otherwise the response content
    is parsed as JSON in the form of an object ``{"ids": [...]}`` or a bare
    list of strings.

    :raises ValueError: when no usable id list can be read from the response.
    """
    tool_calls = getattr(response, "tool_calls", None)
    if tool_calls:
        extracted: list[str] = []
        for call in tool_calls:
            args = dict(call.get("args") or {})
            value = args.get("id") or args.get("ids")
            if isinstance(value, str):
                extracted.append(value)
            elif isinstance(value, list):
                extracted.extend(item for item in value if isinstance(item, str))
        if extracted:
            return extracted
    content = getattr(response, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise ValueError(
            "retrieve response carried no usable ids (no tool_calls and no "
            "JSON content with an 'ids' list)"
        )
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"retrieve JSON is not valid JSON: {exc}") from exc
    ids = payload.get("ids") if isinstance(payload, dict) else payload
    if not isinstance(ids, list):
        raise TypeError('retrieve JSON must be an object with an "ids" list')
    if not all(isinstance(an_id, str) for an_id in ids):
        raise TypeError("retrieve ids must all be strings")
    return ids


def build_state_model(nodes: Mapping[str, AbstractNode]) -> type[BaseModel]:
    """Auto-build the inner state model for a set of generated nodes.

    The model unions the declared ``params`` keys (typed by the node's actual
    annotation) with the ``writes`` target state keys (``str`` unless a param
    already typed the key) plus the always-present ``user_message`` and
    ``response`` fields. Every node reads/writes against the union, so the
    structure always validates.
    """
    fields: dict[str, type] = {}
    for node in nodes.values():
        for param, annotation in node.params.items():
            fields[param] = annotation
        for state_key in node.writes.values():
            fields.setdefault(state_key, str)
    fields.setdefault("user_message", str)
    fields.setdefault("response", str)
    return create_model(
        "GeneratedState",
        **{name: (annotation, None) for name, annotation in fields.items()},
    )


def wrap_reused(
    reuse_ids: Iterable[str],
    graph: Graph,
    *,
    name: str,
    description: str,
) -> list[AbstractNode]:
    """Build save-ready prototypes for the ``reuse``-flagged node ids.

    A single flagged id saves that node directly (a deep copy, so the stored
    prototype is detached from the run); multiple flagged ids indicate the
    whole generated chain is the artifact, saved as a :class:`GraphNode`
    wrapping the full graph with the generator bridge maps (``user_message``
    in, ``response`` out). Empty input yields nothing.

    :raises KeyError: if a reuse id is not present in ``graph``.
    """
    ids = list(dict.fromkeys(reuse_ids))
    if not ids:
        return []
    if len(ids) == 1:
        return [graph.get_node(ids[0])]
    return [
        GraphNode(
            name=name,
            description=description,
            prompt="Reusable graph generated at runtime from the request.",
            graph=graph,
            input_map={"user_message": "user_message"},
            output_map={"response": "response"},
        )
    ]


def save_reused(
    nodes: Iterable[AbstractNode],
    collection: NodeCollection,
    data_path: str | Path = COLLECTION_PATH,
    *,
    pinned: bool = False,
) -> list[str]:
    """Store ``nodes`` into ``collection`` and persist it to ``data_path``.

    Every node must implement ``to_dict`` — a non-serializable prototype
    raises ``ValueError`` before anything is added or written. Entries are
    added with the ``pinned`` flag so callers can protect high-value
    artifacts from auto-pruning; empty input is a no-op.

    :returns: The collection-assigned ids, in input order.
    """
    items = list(nodes)
    if not items:
        return []
    for node in items:
        if getattr(node, "to_dict", None) is None:
            raise ValueError(f"cannot save a node without to_dict(): {node.name!r}")
    ids = [collection.add(node, pinned=pinned) for node in items]
    dump_collection(collection, data_path)
    return ids


class _GeneratorFn:
    """LangGraph-ready dual callable produced by ``GeneratorNode.get_node``.

    Holds the assembled prompt (fixed template + rendered behaviors) and the
    retry budget; each invocation generates, validates, and runs the nested
    graph. The generated :class:`Graph` and any ``reuse``-flagged node ids are
    exposed on ``last_graph`` / ``reuse_ids`` after a successful run.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str,
        prompt: str,
        behaviors_text: str,
        retries: int,
        registry: ToolRegistry,
        llm: Runnable,
        auto_save: bool = True,
        collection: NodeCollection | None = None,
        save_path: str | Path = COLLECTION_PATH,
        pinned: bool = False,
    ) -> None:
        self.__name__ = name
        self._description = description
        self._prompt = prompt
        self._behaviors_text = behaviors_text
        self._retries = retries
        self._registry = registry
        self._llm = llm
        self._auto_save = auto_save
        self._collection = collection
        self._save_path = Path(save_path) if save_path is not None else COLLECTION_PATH
        self._pinned = pinned
        self.last_graph: Graph | None = None
        self.reuse_ids: list[str] = []
        self._retrieved_ids: list[str] = []
        self._retrieved: list[NodeCollectionRecord] = []
        self._retrieve_note: str | None = None

    def __call__(self, state: dict) -> dict:
        return self.invoke(state)

    @staticmethod
    def _read(state: dict, key: str) -> object:
        """Read ``key`` from a pydantic-model or plain-dict state."""
        if isinstance(state, BaseModel):
            return getattr(state, key, None)
        return state.get(key)

    def _bind(self) -> Runnable:
        """Bind the builder tools when the LLM supports it (else return as-is)."""
        if callable(getattr(self._llm, "bind_tools", None)):
            return self._llm.bind_tools(BUILDER_TOOL_DEFS)
        return self._llm

    def _assemble(self, user_message: object, errors: list[str]) -> str:
        prompt = self._prompt.format(user_message=user_message)
        prompt += self._behaviors_text
        prompt += self._reuse_section()
        if errors:
            prompt += (
                "\n\nThe previous attempt was rejected for these reasons:\n"
                + "\n".join(f"- {error}" for error in errors)
                + "\n\nReturn a corrected full graph specification."
            )
        return prompt

    def _reuse_section(self) -> str:
        """Render the retrieved-steps section appended to the build prompt.

        The retrieve note is rendered even when nothing was retrieved, so a
        degraded retrieve step still surfaces its warning to the model.
        """
        if not self._retrieved and not self._retrieve_note:
            return ""
        parts: list[str] = []
        if self._retrieved:
            lines = "\n".join(
                f"- {record.id} ({record.name}): {record.description}"
                for record in self._retrieved
            )
            parts.append(
                "Reusable steps selected for this request. To reuse one, use "
                "add_node(from_collection='<id>') and omit prompt/params/writes "
                f"(and tool):\n{lines}"
            )
        if self._retrieve_note:
            parts.append(self._retrieve_note)
        return "\n\n" + "\n".join(parts)

    def _retrieve_prompt(self, user_message: object) -> str | None:
        """Render the retrieve prompt, or ``None`` when the step is skipped.

        Skipped without a collection or when the collection is empty — the
        retrieve step is an extra LLM call that only pays off when there is
        already something to reuse.
        """
        if self._collection is None or len(self._collection) == 0:
            return None
        records = self._collection.records()[:_CATALOG_LIMIT]
        catalog = "\n".join(
            f"- {record.id}: {record.name} — {record.description} "
            f"(ran {record.run_counts} times)"
            for record in records
        )
        return _RETRIEVE_TEMPLATE.format(
            catalog=catalog, user_message=user_message
        )

    def _apply_retrieve(self, response: object) -> None:
        """Fold a retrieve response into ``_retrieved``/``_retrieve_note``.

        A malformed response, or ids that are not in the collection, never
        crash the generation — they degrade to a short note appended to the
        build prompt so the model can correct itself.
        """
        try:
            ids = _decode_retrieved_ids(response)
        except (TypeError, ValueError) as exc:
            self._retrieved_ids = []
            self._retrieved = []
            self._retrieve_note = f"note: retrieve step was ignored ({exc})"
            return
        unknown = [an_id for an_id in ids if an_id not in self._collection]
        known = [an_id for an_id in ids if an_id in self._collection]
        by_id = {record.id: record for record in self._collection.records()}
        self._retrieved_ids = known
        self._retrieved = [by_id[an_id] for an_id in known]
        self._retrieve_note = (
            f"note: retrieve step returned unknown collection ids "
            f"{sorted(unknown)!r}; they were ignored — choose only from "
            "the catalog."
            if unknown
            else None
        )

    def _retrieve(self, user_message: object) -> None:
        """Run the internal retrieve step (sync); no-op when skipped."""
        prompt = self._retrieve_prompt(user_message)
        if prompt is None:
            self._retrieved_ids = []
            self._retrieved = []
            self._retrieve_note = None
            return
        self._apply_retrieve(self._llm.invoke(prompt))

    async def _aretrieve(self, user_message: object) -> None:
        """Run the internal retrieve step; async twin of :meth:`_retrieve`."""
        prompt = self._retrieve_prompt(user_message)
        if prompt is None:
            self._retrieved_ids = []
            self._retrieved = []
            self._retrieve_note = None
            return
        self._apply_retrieve(await self._llm.ainvoke(prompt))

    def _generate_sync(self, user_message: object) -> Graph:
        self._retrieve(user_message)
        last_errors: list[str] = []
        bound = self._bind()
        for _ in range(self._retries + 1):
            prompt = self._assemble(user_message, last_errors)
            try:
                calls = decode_builder_calls(bound.invoke(prompt))
            except (TypeError, ValueError) as exc:
                last_errors = [str(exc)]
                continue
            graph, errors = self._build_graph(calls)
            self._merge_validation(graph, errors)
            if not errors:
                return graph
            last_errors = errors
        self._raise_generation_failed(last_errors, self._retries + 1)

    async def _generate_async(self, user_message: object) -> Graph:
        await self._aretrieve(user_message)
        last_errors: list[str] = []
        bound = self._bind()
        for _ in range(self._retries + 1):
            prompt = self._assemble(user_message, last_errors)
            try:
                calls = decode_builder_calls(await bound.ainvoke(prompt))
            except (TypeError, ValueError) as exc:
                last_errors = [str(exc)]
                continue
            graph, errors = self._build_graph(calls)
            self._merge_validation(graph, errors)
            if not errors:
                return graph
            last_errors = errors
        self._raise_generation_failed(last_errors, self._retries + 1)

    @staticmethod
    def _merge_validation(graph: Graph, errors: list[str]) -> None:
        try:
            graph.validate()
        except GraphValidationError as exc:
            errors.extend(exc.errors)

    @staticmethod
    def _raise_generation_failed(last_errors: list[str], attempts: int) -> None:
        detail = "; ".join(last_errors) if last_errors else "unknown generation error"
        raise RuntimeError(
            f"could not generate a valid graph after {attempts} attempts: {detail}"
        )

    def _build_graph(self, calls: list[dict]) -> tuple[Graph, list[str]]:
        nodes: dict[str, AbstractNode] = {}
        edges: list[tuple[str, str]] = []
        errors: list[str] = []
        pulled: set[str] = set()
        for call in calls:
            name = call.get("name")
            args = call.get("args") or {}
            try:
                if name == "add_node":
                    node_id, node = self._make_node(args, pulled)
                    self._check_node_id(node_id, nodes)
                    nodes[node_id] = node
                elif name == "add_edge":
                    edges.append(self._edge_endpoints(args))
                else:
                    raise ValueError(
                        f"unknown builder call {name!r}; expected 'add_node' "
                        "or 'add_edge'"
                    )
            except (TypeError, ValueError) as exc:
                errors.append(str(exc))
        self.reuse_ids = [
            call["args"]["id"]
            for call in calls
            if call.get("name") == "add_node"
            and call.get("args", {}).get("reuse")
            and not call.get("args", {}).get("from_collection")
        ]
        graph = Graph(state_model=build_state_model(nodes))
        for node_id, node in nodes.items():
            graph.add_node(node_id, node)
        for index, (source, target) in enumerate(edges, start=1):
            try:
                graph.add_edge(f"e{index}", source, target)
            except (TypeError, ValueError) as exc:
                errors.append(str(exc))
        return graph, errors

    def _make_node(
        self, args: dict, pulled: set[str]
    ) -> tuple[str, AbstractNode]:
        """Build the node (plus its id) for an ``add_node`` call.

        ``pulled`` tracks collection node ids already added under a graph id
        in this build, so a collection node cannot be reused twice.
        """
        node_id = args.get("id")
        node_type = args.get("type")
        name = args.get("name")
        description = args.get("description")
        if not isinstance(node_id, str) or not node_id:
            raise ValueError("add_node 'id' must be a non-empty string")
        if not isinstance(name, str) or not name:
            raise ValueError("add_node 'name' must be a non-empty string")
        if not isinstance(description, str) or not description:
            raise ValueError("add_node 'description' must be a non-empty string")
        from_collection = args.get("from_collection")
        if from_collection is not None:
            return self._pull_node(node_id, from_collection, args, pulled)
        writes = _normalize_writes(args.get("writes"))
        if node_type == "text":
            prompt = args.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                raise ValueError("text node needs a non-empty 'prompt'")
            node: AbstractNode = TextNode(
                name,
                description,
                prompt,
                params=_normalize_params(args.get("params")),
                writes=writes,
            )
        elif node_type == "tool":
            tool = args.get("tool")
            if not isinstance(tool, str) or not tool:
                raise ValueError("tool node needs a non-empty 'tool' name")
            tool_args = args.get("tool_args")
            if tool_args not in (None, {}):
                raise ValueError(
                    "fixed 'tool_args' are not supported; set the 'args' state "
                    "key with an upstream text step"
                )
            node = ToolCallNode(
                name, description, tool, self._registry, writes=writes
            )
        else:
            raise ValueError(
                f"unknown node type {node_type!r}; expected 'text' or 'tool'"
            )
        return node_id, node

    def _pull_node(
        self, node_id: str, from_collection: str, args: dict, pulled: set[str]
    ) -> tuple[str, AbstractNode]:
        """Resolve ``add_node(from_collection=...)`` to the collection's node.

        The collection's **live instance** is pulled into the inner graph
        (shared reference), so run stats recorded while the nested graph runs
        fold back into the collection. Duplicate use of one collection node in
        a single graph is a build error, as is mixing a pull with a fresh spec
        or with ``reuse`` — the error text feeds the retry loop.

        :raises ValueError: for an unknown id, a repeated pull, a mixed spec,
            or ``reuse`` combined with a pull.
        """
        if self._collection is None or from_collection not in self._collection:
            raise ValueError(
                f"add_node 'from_collection' references unknown collection "
                f"node {from_collection!r}"
            )
        if args.get("reuse"):
            raise ValueError(
                "add_node cannot set both 'reuse' and 'from_collection': a "
                "collection node is already in the reusable set"
            )
        if from_collection in pulled:
            raise ValueError(
                f"collection node {from_collection!r} was already added to "
                "this graph under another id; use each collection node at "
                "most once"
            )
        for key in ("prompt", "params", "writes", "tool"):
            if args.get(key) not in (None, {}, ""):
                raise ValueError(
                    f"add_node 'from_collection' cannot be combined with a "
                    f"fresh spec; omit {key!r}"
                )
        pulled.add(from_collection)
        return node_id, self._collection.get(from_collection)

    @staticmethod
    def _check_node_id(node_id: str, nodes: dict[str, AbstractNode]) -> None:
        if node_id in RESERVED_IDS:
            raise ValueError(f"node id is reserved: {node_id!r}")
        if node_id in nodes:
            raise ValueError(f"duplicate node id: {node_id!r}")

    @staticmethod
    def _edge_endpoints(args: dict) -> tuple[str, str]:
        source = args.get("source")
        target = args.get("target")
        if not isinstance(source, str) or not source:
            raise ValueError("add_edge needs a non-empty 'source' id")
        if not isinstance(target, str) or not target:
            raise ValueError("add_edge needs a non-empty 'target' id")
        return source, target

    def _run_nested(self, graph: Graph, user_message: object) -> str:
        node = self._graph_node(graph)
        result = node.get_node(self._llm).invoke({"user_message": user_message})
        return result["response"]

    async def _arun_nested(self, graph: Graph, user_message: object) -> str:
        node = self._graph_node(graph)
        result = await node.get_node(self._llm).ainvoke(
            {"user_message": user_message}
        )
        return result["response"]

    def _graph_node(self, graph: Graph) -> GraphNode:
        return GraphNode(
            name=self.__name__,
            description=self._description,
            prompt="Auto-generated subgraph produced by a generator node.",
            graph=graph,
            input_map={"user_message": "user_message"},
            output_map={"response": "response"},
        )

    def _maybe_save(self, graph: Graph) -> None:
        """End-of-turn auto-save of reuse-flagged artifacts, when enabled."""
        if not self._auto_save or self._collection is None or not self.reuse_ids:
            return
        save_reused(
            wrap_reused(
                self.reuse_ids,
                graph,
                name=self.__name__,
                description=self._description,
            ),
            self._collection,
            self._save_path,
            pinned=self._pinned,
        )

    def invoke(self, state: dict) -> dict:
        """Generate, validate, and run the nested graph; surface ``response``."""
        user_message = self._read(state, "user_message")
        graph = self._generate_sync(user_message)
        self.last_graph = graph
        result = {"response": self._run_nested(graph, user_message)}
        self._maybe_save(graph)
        return result

    async def ainvoke(self, state: dict) -> dict:
        """Async twin of :meth:`invoke`."""
        user_message = self._read(state, "user_message")
        graph = await self._generate_async(user_message)
        self.last_graph = graph
        result = {"response": await self._arun_nested(graph, user_message)}
        self._maybe_save(graph)
        return result


class GeneratorNode(AbstractNode):
    """Receive the user task + behavior points and generate a nested graph.

    :param name: Node name (used as the LangGraph node name).
    :type name: str
    :param description: What this node does, used for retrieval.
    :type description: str
    :param behaviors: Behavior groups injected as objects, a ``str``/``Path``
        pointing at a sectioned YAML file, or ``None`` for the default path.
    :type behaviors: list[BehaviorGroup] | str | Path | None
    :param retries: Whole-spec regeneration attempts after a failed one.
    :type retries: int
    :param registry: Tool registry forwarded to generated ``ToolCallNode``
        instances; defaults to the shared builtin tools.
    :type registry: ToolRegistry | None
    :param auto_save: Whether ``reuse``-flagged artifacts are saved to the
        dedicated collection at the end of each turn (default ``True``).
    :type auto_save: bool
    :param collection: The dedicated :class:`NodeCollection` reuse artifacts
        land in; defaults to an internal one with a lazily resolved default
        embedder. Pass one with an injected embedder when embedding must be
        controlled.
    :type collection: NodeCollection | None
    :param save_path: File the collection is persisted to after each save
        (default ``backend/data/collection.json``).
    :type save_path: str | Path
    :param reuse_pinned: Mark saved reuse artifacts with the collection's
        ``pinned`` flag so auto-pruning never evicts them (default ``False``).
    :type reuse_pinned: bool
    :param writes: Outer output map; defaults to ``{"result": "response"}``.
    :type writes: dict[str, str] | None

    The core prompt template is fixed (never rewritable) and the behaviors are
    rendered into it from data. Reads ``user_message`` from the outer state.
    """

    def __init__(
        self,
        name: str,
        description: str,
        *,
        behaviors: list[BehaviorGroup] | str | Path | None = None,
        retries: int = 3,
        registry: ToolRegistry | None = None,
        auto_save: bool = True,
        collection: NodeCollection | None = None,
        save_path: str | Path = COLLECTION_PATH,
        reuse_pinned: bool = False,
        writes: dict[str, str] | None = None,
    ) -> None:
        if not isinstance(retries, int) or isinstance(retries, bool) or retries < 0:
            raise ValueError("retries must be a non-negative integer")
        if not isinstance(auto_save, bool):
            raise TypeError(f"auto_save must be a bool, got {type(auto_save).__name__}")
        if not isinstance(reuse_pinned, bool):
            raise TypeError(
                f"reuse_pinned must be a bool, got {type(reuse_pinned).__name__}"
            )
        if not isinstance(save_path, (str, Path)):
            raise TypeError(
                f"save_path must be a str or Path, got {type(save_path).__name__}"
            )
        if collection is not None and not isinstance(collection, NodeCollection):
            raise TypeError(
                f"collection must be a NodeCollection, got {type(collection).__name__}"
            )
        self._retries = retries
        self._registry = registry if registry is not None else default_registry()
        self._auto_save = auto_save
        self._pinned = reuse_pinned
        self._save_path = Path(save_path)
        if collection is not None:
            self._reuse_collection = collection
        elif auto_save:
            self._reuse_collection = load_collection(self._save_path)
        else:
            self._reuse_collection = None
        groups = self._load_behaviors(behaviors)
        self.behaviors_text = render_behaviors(groups)
        super().__init__(
            name,
            description,
            _GENERATOR_TEMPLATE,
            params={"user_message": str},
            writes=writes,
        )

    @staticmethod
    def _load_behaviors(
        behaviors: list[BehaviorGroup] | str | Path | None,
    ) -> list[BehaviorGroup]:
        """Resolve a behaviors argument (objects, path, or default) to groups."""
        if behaviors is None:
            return load_behaviors()
        if isinstance(behaviors, (str, Path)):
            return load_behaviors(behaviors)
        if isinstance(behaviors, list):
            if not all(isinstance(group, BehaviorGroup) for group in behaviors):
                raise TypeError(
                    "behaviors list items must be BehaviorGroup instances"
                )
            return list(behaviors)
        raise TypeError(
            "behaviors must be a path, a list of BehaviorGroup, or None, got "
            f"{type(behaviors).__name__}"
        )

    def get_node(self, llm: Runnable) -> DualCallable:
        """Bind an LLM and return the LangGraph-ready dual callable."""
        return _GeneratorFn(
            name=self.name,
            description=self.description,
            prompt=self.prompt,
            behaviors_text=self.behaviors_text,
            retries=self._retries,
            registry=self._registry,
            llm=llm,
            auto_save=self._auto_save,
            collection=self._reuse_collection,
            save_path=self._save_path,
            pinned=self._pinned,
        )

    def updated(self, **changes: object) -> GeneratorNode:
        """Return a new validated instance; only metadata may change.

        The core prompt, params, writes, behaviors, and retries are structural
        for a generator node, so only ``name``/``description`` are editable
        (swap in a fresh node otherwise).

        :raises ValueError: if a structural field is changed.
        :raises TypeError: for unknown field names.
        """
        allowed = {"name", "description"}
        for key in changes:
            if key in ("prompt", "params", "writes"):
                raise ValueError(f"{key} is structural for a generator node")
            if key not in allowed:
                raise TypeError(
                    f"unsupported fields: {sorted(set(changes))}; supported: "
                    f"{sorted(allowed)}"
                )
        rebuilt = type(self)(
            name=changes.get("name", self.name),
            description=changes.get("description", self.description),
            auto_save=self._auto_save,
            collection=self._reuse_collection,
            save_path=self._save_path,
            reuse_pinned=self._pinned,
        )
        self._carry_stats(rebuilt)
        return rebuilt


def _normalize_params(raw: object) -> dict[str, type]:
    """Canonicalize a ``params`` mapping ``{field: type-name}``.

    Type names are the serialized annotation names (``"str"``, ``"list"``,
    ...) resolved through :func:`graphs.serialization.deserialize_annotation`;
    an unknown name is a build error the retry loop feeds back.
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise TypeError(f"params must be an object, got {type(raw).__name__}")
    params: dict[str, type] = {}
    for field, type_name in raw.items():
        if not isinstance(field, str) or not field:
            raise ValueError("params field names must be non-empty strings")
        if not isinstance(type_name, str):
            raise TypeError(
                f"params field {field!r} must map to a type name string, got "
                f"{type(type_name).__name__}"
            )
        params[field] = deserialize_annotation(type_name)
    return params


def _normalize_writes(raw: object) -> dict[str, str] | None:
    """Canonicalize a ``writes`` mapping ``{output-name: state-key}``."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise TypeError(f"writes must be an object, got {type(raw).__name__}")
    writes: dict[str, str] = {}
    for output, state_key in raw.items():
        if not isinstance(output, str) or not output:
            raise ValueError("writes output names must be non-empty strings")
        if not isinstance(state_key, str) or not state_key:
            raise ValueError("writes state keys must be non-empty strings")
        writes[output] = state_key
    return writes