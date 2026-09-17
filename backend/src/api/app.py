"""Runnable FastAPI application exposing ``POST /feedback``.

Run with ``uvicorn api.app:app`` from ``backend/``.
"""

from api.feedback import create_app

app = create_app()

__all__ = ["app"]