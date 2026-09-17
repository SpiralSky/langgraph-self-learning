"""FastAPI endpoint exposing the feedback pipeline over HTTP.

Thin wrapper around :mod:`graphs.feedback` (patch computation) and
:mod:`graphs.patches` (JSONL storage + deterministic application). The
``POST /feedback`` handler loads the current behaviors, computes structured
patches from the suggestion via the LLM, appends them to the patch log, and
applies them — returning the applied patch list.

The LLM is bound lazily from ``model_settings.yaml`` (``get_chat_model``)
unless a callable is injected, so importing or building the app never touches
the network, the environment, or model settings.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from graphs.behaviors import load_behaviors
from graphs.feedback import compute_improvement_patches
from graphs.llm import get_chat_model
from graphs.patches import append_patches, apply_patches, load_patches


class FeedbackRequest(BaseModel):
    """Body of a ``POST /feedback`` request."""

    question: str
    responses: list[str]
    comment: str
    graph_state: object | None = None


def _next_seq(patches_path: str | Path | None) -> int:
    """The seq for the next patch, continuing after the log's last entry.

    Keeps ``seq`` monotonic across requests so the most-recent-wins precedence
    (:mod:`graphs.patches`) stays well-defined.
    """
    existing = load_patches(patches_path)
    return max((patch.seq for patch in existing), default=-1) + 1


def create_app(
    *,
    llm: Runnable | None = None,
    behaviors_path: str | Path | None = None,
    patches_path: str | Path | None = None,
    collection: object | None = None,
) -> FastAPI:
    """Build the FastAPI app exposing ``POST /feedback``.

    :param llm: Runnable emitting ``{content: ...}`` for regeneration; defaults
        to ``get_chat_model("fast")`` resolved lazily on first request so tests
        never touch the environment.
    :type llm: Runnable | None
    :param behaviors_path: Behaviors data file (default ``backend/behaviors.yaml``).
    :type behaviors_path: str | Path | None
    :param patches_path: Patch JSONL log (default ``backend/data/patches.jsonl``).
    :type patches_path: str | Path | None
    :param collection: Optional ``NodeCollection`` for node-targeted patches.
    :type collection: object | None
    :return: A configured FastAPI app.
    :rtype: FastAPI
    """
    app = FastAPI(title="graph-feedback")

    @app.post("/feedback")
    def post_feedback(payload: FeedbackRequest) -> dict[str, object]:
        active_llm = llm if llm is not None else get_chat_model("fast")
        try:
            behaviors = load_behaviors(behaviors_path)
            computed = compute_improvement_patches(
                question=payload.question,
                responses=payload.responses,
                graph_state=payload.graph_state,
                comment=payload.comment,
                behaviors_at_time=behaviors,
                llm=active_llm,
                seq_prefix=_next_seq(patches_path),
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail=f"invalid feedback: {exc}"
            ) from exc
        try:
            append_patches(computed, patches_path)
            applied = apply_patches(
                computed, behaviors_path=behaviors_path, collection=collection
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"failed to apply feedback: {exc}"
            ) from exc
        serialized = [
            patch.model_dump(mode="json", exclude_none=True) for patch in applied
        ]
        return {"applied": serialized}

    return app