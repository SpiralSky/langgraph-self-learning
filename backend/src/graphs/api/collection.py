"""``NodeCollection``: an in-memory store for node prototypes.

The collection holds instances of the ``GraphNode`` protocol (e.g.
``TextNode`` objects) keyed by a uuid4 id, tracks usage/recency, supports
combined-filter metadata retrieval, and hands its contents out via
``records()`` / ``snapshot()`` for an external system to persist. No graph
wiring and no persistence live here.

``NodeCollection`` also owns an internal **ephemeral** chromadb cosine index
for semantic search: ``add`` embeds ``"{name}\\n{description}"`` per node,
``remove``/pruning delete the vectors, and ``semantic_search`` embeds the
query and ranks by similarity. Persistence (vectors included) stays external.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import uuid4

import chromadb

from graphs.api.stats import OnlineStats
from graphs.nodes.base import GraphNode

_DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
_default_embedding_model: object | None = None


def get_default_embedder() -> Callable[[str], list[float]]:
    """Build (and cache) the default fastembed-backed embedder, lazily.

    The first call downloads ``_DEFAULT_EMBEDDING_MODEL`` (one-time; offline
    afterwards). Tests must never call this — inject a fake embedder instead.
    """
    global _default_embedding_model
    if _default_embedding_model is None:
        from fastembed import TextEmbedding

        _default_embedding_model = TextEmbedding(model_name=_DEFAULT_EMBEDDING_MODEL)

    def _embed_text(text: str) -> list[float]:
        return next(_default_embedding_model.embed([text])).tolist()

    return _embed_text


@dataclass(frozen=True)
class NodeCollectionRecord:
    """Read-only metadata view of a stored node."""

    id: str
    name: str
    description: str
    use_counts: int
    pinned: bool
    created_at: datetime
    last_used_at: datetime | None
    run_counts: int = 0
    tokens_count: int = 0
    tokens_mean: float | None = None
    tokens_std: float | None = None
    time_count: int = 0
    time_mean: float | None = None
    time_std: float | None = None


@dataclass
class _Entry:
    """Internal store entry: the prototype plus per-node bookkeeping."""

    id: str
    node: GraphNode
    use_counts: int = 0
    pinned: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_used_at: datetime | None = None
    embedded: bool = False


class NodeCollection:
    """An insertion-ordered, auto-pruning in-memory store of node prototypes.

    :param max_nodes: Max non-pinned entries before auto-pruning triggers.
    :type max_nodes: int
    :param prune_to: Non-pinned count pruned down to on ``add``.
    :type prune_to: int
    :param embedder: Embedding callable ``str -> list[float]`` used for the
        semantic-search index; defaults to a lazy fastembed-backed helper.
    :type embedder: Callable[[str], list[float]] | None
    """

    def __init__(
        self,
        *,
        max_nodes: int = 100,
        prune_to: int = 80,
        embedder: Callable[[str], list[float]] | None = None,
    ) -> None:
        if not 0 < prune_to <= max_nodes:
            raise ValueError("expected 0 < prune_to <= max_nodes")
        self._max_nodes = max_nodes
        self._prune_to = prune_to
        self._entries: dict[str, _Entry] = {}
        self._embedder = embedder
        self._vector_collection = self._make_vector_collection()

    def _make_vector_collection(self):
        client = chromadb.EphemeralClient()
        # EphemeralClient shares one in-memory sqlite per process, so use a
        # unique collection name to keep each instance's index private.
        return client.get_or_create_collection(
            f"nodes-{uuid4()}", metadata={"hnsw:space": "cosine"}
        )

    def add(self, node: GraphNode, *, pinned: bool = False) -> str:
        """Store a prototype and return its new id; auto-prunes on overflow."""
        if not isinstance(node.name, str) or not node.name:
            raise ValueError("node.name must be a non-empty string")
        node_id = str(uuid4())
        entry = _Entry(id=node_id, node=node, pinned=pinned)
        self._entries[node_id] = entry
        try:
            self._embed(entry)
        except Exception:  # noqa: BLE001 - embedding must never drop the node
            entry.embedded = False
        self._prune()
        return node_id

    def _embed(self, entry: _Entry) -> None:
        """Embed the entry's name+description into the vector index."""
        embedder = self._embedder if self._embedder is not None else get_default_embedder()
        text = f"{entry.node.name}\n{entry.node.description}"
        vector = embedder(text)
        self._vector_collection.add(
            ids=[entry.id],
            embeddings=[vector],
            documents=[text],
            metadatas=[{"node_id": entry.id}],
        )
        entry.embedded = True

    def _unembed(self, node_id: str) -> None:
        self._vector_collection.delete(ids=[node_id])

    def _sync_embedding(self, entry: _Entry, old_node: GraphNode) -> None:
        """Re-embed a mutating entry iff its embedded text changed.

        An unembedded entry re-attempts embedding; a changed entry swaps its
        vector. Any failure degrades to ``embedded=False`` — the node change
        is kept and this never raises.
        """
        old_text = f"{old_node.name}\n{old_node.description}"
        new_text = f"{entry.node.name}\n{entry.node.description}"
        if new_text == old_text and entry.embedded:
            return
        if entry.embedded:
            self._unembed(entry.id)
        try:
            self._embed(entry)
        except Exception:  # noqa: BLE001 - embedding must never drop the node
            entry.embedded = False

    def get(self, node_id: str) -> GraphNode:
        """Return the live stored prototype by reference, bumping use/recency.

        This is the collection's own instance (no deep copy), so edits must
        go through the API: :meth:`update_inplace` (validated rebuild in the
        store), :meth:`edit` (validated copy, store untouched) or
        :meth:`replace` (explicit commit). Mutating the returned object
        outside those methods is unsupported.

        :raises KeyError: if ``node_id`` is unknown.
        """
        entry = self._entries[node_id]
        entry.use_counts += 1
        entry.last_used_at = datetime.now(UTC)
        return entry.node

    def update_inplace(
        self,
        node_id: str,
        *,
        reset_stats: bool = False,
        **changes: object,
    ) -> None:
        """Field-level edit of the stored prototype; no use/recency bump.

        Only non-``None`` kwargs reach the stored node's ``updated()``; an
        empty call is a no-op. The new instance is rebuilt and validated
        BEFORE it replaces the stored one, so a failed edit (e.g. an orphan
        prompt placeholder) leaves the store unchanged.

        Run stats are carried onto the rebuilt node by ``updated()`` and
        therefore preserved — pass ``reset_stats=True`` to zero them instead.

        :raises KeyError: if ``node_id`` is unknown.
        :raises TypeError: if the stored node has no ``updated()``.
        """
        entry = self._entries[node_id]
        updated = getattr(entry.node, "updated", None)
        if updated is None:
            raise TypeError(
                "stored node has no updated(); use replace() to swap in a new prototype"
            )
        changes = {
            key: value for key, value in changes.items() if value is not None
        }
        if not changes:
            return
        new_node = updated(**changes)
        if reset_stats:
            new_node.run_counts = 0
            new_node.output_tokens = OnlineStats()
            new_node.output_time = OnlineStats()
        old_node = entry.node
        entry.node = new_node
        self._sync_embedding(entry, old_node)

    def edit(self, node_id: str, **changes: object) -> GraphNode:
        """Return a validated copy of the stored prototype; store untouched.

        Rebuilds a new instance via the stored node's ``updated()`` (so all
        provided values are re-validated and run stats are carried over) and
        returns it WITHOUT touching the store. Commit the result explicitly
        with :meth:`replace` when ready.

        :raises KeyError: if ``node_id`` is unknown.
        :raises TypeError: if the stored node has no ``updated()``.
        """
        entry = self._entries[node_id]
        updated = getattr(entry.node, "updated", None)
        if updated is None:
            raise TypeError(
                "stored node has no updated(); use replace() to swap in a new prototype"
            )
        return updated(**changes)

    def replace(self, node_id: str, new_node: GraphNode) -> None:
        """Swap a new prototype under the same id; no use/recency bump.

        Bookkeeping (``id``, ``use_counts``, ``pinned``, ``created_at``,
        ``last_used_at``) is preserved. The incoming node's own run stats
        come with it; nothing is carried over from the outgoing prototype.

        :raises KeyError: if ``node_id`` is unknown.
        :raises ValueError: if ``new_node.name`` is empty.
        """
        entry = self._entries[node_id]
        if not isinstance(new_node.name, str) or not new_node.name:
            raise ValueError("node.name must be a non-empty string")
        old_node = entry.node
        entry.node = new_node
        self._sync_embedding(entry, old_node)

    def get_by_name(self, name: str) -> list[NodeCollectionRecord]:
        """All records with a matching name, in insertion order. No bump."""
        return [
            self._record(entry)
            for entry in self._entries.values()
            if entry.node.name == name
        ]

    def search(
        self,
        query: str | None = None,
        *,
        limit: int = 10,
        sort_by: str = "use",
        within_newest: int | None = None,
    ) -> list[NodeCollectionRecord]:
        """Combined-filter metadata retrieval. Never mutates the store.

        Candidates are pages of the store narrowed by ``within_newest``,
        filtered by a case-insensitive substring match of ``query`` against
        name OR description, then sorted and truncated to ``limit``.

        :param query: Keyword filter; ``None`` matches everything.
        :type query: str | None
        :param limit: Max records returned (must be >= 1).
        :type limit: int
        :param sort_by: ``'use'`` (use_counts desc, created_at desc tie-break)
            or ``'newest'`` (created_at desc).
        :type sort_by: str
        :param within_newest: Narrow candidates to the ``z`` most recently
            created entries before filtering and sorting.
        :type within_newest: int | None

        :raises ValueError: for invalid ``limit``/``sort_by``/``within_newest``.
        """
        if limit < 1:
            raise ValueError("limit must be >= 1")
        candidates = list(self._entries.values())
        if within_newest is not None:
            if within_newest < 1:
                raise ValueError("within_newest must be >= 1")
            candidates = sorted(candidates, key=lambda e: e.created_at, reverse=True)
            candidates = candidates[:within_newest]
        if query:
            lowered = query.lower()
            candidates = [
                e
                for e in candidates
                if lowered in e.node.name.lower()
                or lowered in e.node.description.lower()
            ]
        if sort_by == "use":
            candidates = sorted(
                candidates, key=lambda e: (e.use_counts, e.created_at), reverse=True
            )
        elif sort_by == "newest":
            candidates = sorted(candidates, key=lambda e: e.created_at, reverse=True)
        else:
            raise ValueError(f"unknown sort_by: {sort_by!r}")
        return [self._record(e) for e in candidates[:limit]]

    def top_by_use(self, k: int) -> list[NodeCollectionRecord]:
        """Top ``k`` records by usage (see ``search`` with ``sort_by='use'``)."""
        return self.search(sort_by="use", limit=k)

    def newest(self, k: int) -> list[NodeCollectionRecord]:
        """Newest ``k`` records by creation time."""
        return self.search(sort_by="newest", limit=k)

    def semantic_search(
        self,
        query: str,
        *,
        limit: int = 10,
        within_newest: int | None = None,
        sort_by: str = "similarity",
    ) -> list[NodeCollectionRecord]:
        """Semantic ("top-n similar") retrieval. Never mutates the store.

        Embeds ``query``, retrieves the ``limit`` most similar entries from the
        internal cosine index (optionally narrowed to the ``within_newest``
        newest-created candidates), then optionally re-sorts the top-n by
        usage or recency before truncating to ``limit``. Unembedded entries
        (e.g. when an embedder failed at ``add`` time) are skipped because
        they have no vector in the index.

        :param query: Text whose embedding is matched against stored nodes.
        :type query: str
        :param limit: Max records returned (must be >= 1).
        :type limit: int
        :param within_newest: Only consider the ``z`` most recently created
            entries as candidates.
        :type within_newest: int | None
        :param sort_by: ``'similarity'`` (index order) | ``'use'`` | ``'newest'``.
        :type sort_by: str

        :raises ValueError: for invalid ``limit``/``within_newest``/``sort_by``.
        """
        if limit < 1:
            raise ValueError("limit must be >= 1")
        if within_newest is not None and within_newest < 1:
            raise ValueError("within_newest must be >= 1")
        if sort_by not in {"similarity", "use", "newest"}:
            raise ValueError(f"unknown sort_by: {sort_by!r}")
        if not self._entries:
            return []

        candidates = list(self._entries.values())
        where = None
        if within_newest is not None:
            newest = sorted(
                candidates, key=lambda e: e.created_at, reverse=True
            )[:within_newest]
            if not newest:
                return []
            where = {"node_id": {"$in": [e.id for e in newest]}}

        embedder = self._embedder if self._embedder is not None else get_default_embedder()
        vector = embedder(query)
        n_results = min(limit, len(self._entries))
        if n_results < 1:
            return []
        results = self._vector_collection.query(
            query_embeddings=[vector], n_results=n_results, where=where
        )
        ids = results["ids"][0] if results["ids"] else []
        entries = [self._entries[nid] for nid in ids if nid in self._entries]

        if sort_by == "use":
            entries = sorted(
                entries, key=lambda e: (e.use_counts, e.created_at), reverse=True
            )
        elif sort_by == "newest":
            entries = sorted(entries, key=lambda e: e.created_at, reverse=True)
        return [self._record(e) for e in entries[:limit]]

    def remove(self, node_id: str) -> None:
        """Drop an entry (allowed even when pinned).

        :raises KeyError: if ``node_id`` is unknown.
        """
        entry = self._entries.pop(node_id)
        if entry.embedded:
            self._unembed(node_id)

    def pin(self, node_id: str) -> None:
        """Protect an entry from auto-pruning (idempotent).

        :raises KeyError: if ``node_id`` is unknown.
        """
        self._entries[node_id].pinned = True

    def unpin(self, node_id: str) -> None:
        """Release an entry from pruning protection (idempotent).

        :raises KeyError: if ``node_id`` is unknown.
        """
        self._entries[node_id].pinned = False

    def records(self) -> list[NodeCollectionRecord]:
        """All records in insertion order."""
        return [self._record(e) for e in self._entries.values()]

    def snapshot(self) -> list[GraphNode]:
        """Deep copies of the stored prototypes, insertion order, for persist."""
        return [deepcopy(e.node) for e in self._entries.values()]

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._entries

    def _record(self, entry: _Entry) -> NodeCollectionRecord:
        node = entry.node
        return NodeCollectionRecord(
            id=entry.id,
            name=node.name,
            description=node.description,
            use_counts=entry.use_counts,
            pinned=entry.pinned,
            created_at=entry.created_at,
            last_used_at=entry.last_used_at,
            run_counts=node.run_counts,
            tokens_count=node.output_tokens.count,
            tokens_mean=node.output_tokens.mean if node.output_tokens.count else None,
            tokens_std=node.output_tokens.std,
            time_count=node.output_time.count,
            time_mean=node.output_time.mean if node.output_time.count else None,
            time_std=node.output_time.std,
        )

    def _prune(self) -> None:
        """If non-pinned entries exceed ``max_nodes``, trim down to ``prune_to``.

        The lowest-use non-pinned entries are evicted (oldest ``created_at``
        breaks ties); pinned entries are never counted nor evicted.
        """
        non_pinned = [e for e in self._entries.values() if not e.pinned]
        if len(non_pinned) <= self._max_nodes:
            return
        excess = len(non_pinned) - self._prune_to
        for entry in sorted(
            non_pinned, key=lambda e: (e.use_counts, e.created_at)
        )[:excess]:
            self._entries.pop(entry.id)
            if entry.embedded:
                self._unembed(entry.id)