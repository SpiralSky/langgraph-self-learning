"""File layer for persisted graph artifacts under ``backend/data/``.

Storage helpers write atomic JSON (temp file + rename) and expose
``dump_collection`` / ``load_collection`` for round-tripping a whole
``NodeCollection`` through the type-tagged serializers. Persisted collections
store each node's output-token/time summary (count/mean/std) and ``run_counts``
inline per node. ``load_collection`` applies those stats straight onto the
deserialized node (run stats live on nodes, not on collection entries).
Per-entry bookkeeping that cannot be restored (ids, ``pinned``, timestamps,
use counts) is still deliberately not stored — restore rebuilds prototypes
with the external system's default configuration.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from config import BACKEND_ROOT
from graphs.api.collection import NodeCollection
from graphs.api.stats import OnlineStats
from graphs.serialization import node_from_dict

DATA_ROOT = BACKEND_ROOT / "data"
COLLECTION_PATH = DATA_ROOT / "collection.json"


def ensure_data_dir() -> Path:
    """Create (idempotently) and return the ``backend/data/`` directory."""
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    return DATA_ROOT


def read_json(path: str | Path) -> object:
    """Load a JSON file at ``path`` (missing file raises ``FileNotFoundError``)."""
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: str | Path, obj: object) -> None:
    """Write ``obj`` as JSON, atomically (temp file + ``os.replace``).

    The parent directory is created on demand; a JSON document writes to a
    ``*.tmp`` sibling first so a failed write never corrupts an existing file.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.tmp")
    tmp.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(tmp, target)


def dump_collection(collection: NodeCollection, path: str | Path = COLLECTION_PATH) -> None:
    """Persist a collection's prototypes as type-tagged node dicts.

    Only nodes with a ``to_dict`` implementation can be persisted; a
    non-serializable prototype raises ``ValueError`` naming it.

    :param collection: The collection to snapshot.
    :type collection: NodeCollection
    :param path: Destination file (default ``backend/data/collection.json``).
    :type path: str | Path
    """
    nodes: list[dict] = []
    for node in collection.snapshot():
        to_dict = getattr(node, "to_dict", None)
        if to_dict is None:
            raise ValueError(
                f"collection contains a node without to_dict(): {node.name!r}"
            )
        tokens: OnlineStats = getattr(node, "output_tokens", None) or OnlineStats()
        time: OnlineStats = getattr(node, "output_time", None) or OnlineStats()
        nodes.append(
            {
                "node": to_dict(),
                "stats": {
                    "run_counts": getattr(node, "run_counts", 0),
                    "tokens": {
                        "count": tokens.count,
                        "mean": tokens.mean if tokens.count else None,
                        "std": tokens.std,
                    },
                    "time": {
                        "count": time.count,
                        "mean": time.mean if time.count else None,
                        "std": time.std,
                    },
                },
            }
        )
    write_json(path, {"nodes": nodes})


def _apply_node_stats(node, stats: dict) -> None:
    """Fold persisted run telemetry ``stats`` onto a freshly built node.

    ``run_counts`` must be a non-negative integer; token/time summaries are
    restored as ``OnlineStats`` (Welford ``_m2`` included) so recording can
    continue seamlessly after a reload.

    :raises ValueError: for a negative ``run_counts``.
    """
    run_counts = int(stats["run_counts"])
    if run_counts < 0:
        raise ValueError("run_counts must not be negative")
    node.run_counts = run_counts
    node.output_tokens = OnlineStats.from_dict(stats["tokens"])
    node.output_time = OnlineStats.from_dict(stats["time"])


def load_collection(
    path: str | Path = COLLECTION_PATH,
    embedder: object | None = None,
) -> NodeCollection:
    """Restore a collection persisted by :func:`dump_collection`.

    Vectors are re-embedded on ``add`` with the supplied (or default)
    embedder; an embedding failure degrades to ``embedded=False`` exactly as a
    normal ``add`` would. A missing file yields an empty collection.

    Each entry's persisted run stats are applied directly onto the rebuilt
    node before it is added; legacy ``{"nodes": [<node dict>]}`` dumps (no
    ``stats``) load with empty telemetry.

    :param path: Source file (default ``backend/data/collection.json``).
    :type path: str | Path
    :param embedder: Embedding callable to inject (injected in tests; None
        resolves the default lazily, which tests must never trigger).
    :type embedder: Callable[[str], list[float]] | None
    """
    collection = NodeCollection(embedder=embedder)
    try:
        data = read_json(path)
    except FileNotFoundError:
        return collection
    for item in data["nodes"]:
        node_data = item["node"] if isinstance(item, dict) and "node" in item else item
        node = node_from_dict(node_data)
        stats = item.get("stats") if isinstance(item, dict) else None
        if stats:
            _apply_node_stats(node, stats)
        collection.add(node)
    return collection