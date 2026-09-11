from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

FEEDBACK_DIR = Path(os.environ.get("FEEDBACK_DIR", "/shared/workspace/feedback"))

router = APIRouter(prefix="/api/feedback", tags=["feedback"])


class FeedbackEntry(BaseModel):
    scenario_id: str
    learner_profile: str
    domain: str
    initial_query: str
    transcript: str
    ratings: dict[str, float] = Field(default_factory=dict)
    comment: str = ""
    submitted_at: str = ""


class FeedbackSubmit(BaseModel):
    scenario_id: str
    learner_profile: str
    domain: str
    initial_query: str
    transcript: str
    ratings: dict[str, float] = Field(default_factory=dict)
    comment: str = ""


_DIMENSIONS = [
    "socratic_depth",
    "scaffolding_adequacy",
    "error_feedback_quality",
    "knowledge_model_accuracy",
    "adaptive_branching",
    "comprehension_checking",
    "goal_adherence",
    "overall_pedagogical_quality",
]


def _ensure_dir() -> Path:
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    return FEEDBACK_DIR


def _load_all() -> list[dict]:
    d = _ensure_dir()
    entries = []
    for f in sorted(d.glob("*.json")):
        try:
            with open(f) as fh:
                entries.append(json.load(fh))
        except (json.JSONDecodeError, OSError):
            continue
    return entries


@router.post("/submit")
async def submit_feedback(payload: FeedbackSubmit) -> dict:
    entry = FeedbackEntry(
        scenario_id=payload.scenario_id,
        learner_profile=payload.learner_profile,
        domain=payload.domain,
        initial_query=payload.initial_query,
        transcript=payload.transcript,
        ratings=payload.ratings,
        comment=payload.comment,
        submitted_at=datetime.now(timezone.utc).isoformat(),
    )
    d = _ensure_dir()
    fname = f"feedback_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}.json"
    fpath = d / fname
    with open(fpath, "w") as f:
        json.dump(entry.model_dump(), f, indent=2, default=str)
    return {"status": "ok", "file": str(fpath)}


@router.get("/list")
async def list_feedback(limit: int = 50) -> dict:
    entries = _load_all()
    return {"count": len(entries), "entries": entries[-limit:]}


@router.get("/summary")
async def feedback_summary() -> dict:
    entries = _load_all()
    if not entries:
        return {"count": 0, "averages": {}, "dimensions": _DIMENSIONS}
    avgs: dict[str, float] = {}
    for dim in _DIMENSIONS:
        vals = [e["ratings"].get(dim) for e in entries if dim in e.get("ratings", {})]
        avgs[dim] = sum(vals) / len(vals) if vals else 0.0
    return {"count": len(entries), "averages": avgs, "dimensions": _DIMENSIONS}