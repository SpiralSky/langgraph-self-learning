"""Central state threaded through the learning graph."""

from pydantic import BaseModel


class LearningGraphState(BaseModel):
    """State threaded through every graph turn.

    All nodes read from and write to this state; ``user_message`` carries the
    turn's input and ``response`` carries the terminal output. Fields hold
    plain values — there is no reducer accumulation.
    """

    user_message: str | None = None
    response: str | None = None