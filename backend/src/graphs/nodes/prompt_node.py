"""Bind a prompt string as the first argument of a node function.

Returns a LangGraph-ready callable ``(state, *rest) -> Rt``. The wrapper
re-exposes ``fn``'s post-prompt parameters via ``__signature__``, so
LangGraph can inject extra runtime kwargs (e.g. ``config: RunnableConfig``).
"""

import functools
import inspect
from collections.abc import Callable
from typing import Concatenate, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def with_prompt(
    prompt: str,
) -> Callable[[Callable[Concatenate[str, P], R]], Callable[P, R]]:
    """Return a decorator that binds ``prompt`` as the first argument of ``fn``.

    The decorated ``fn`` must be ``fn(prompt: str, state, *rest) -> Rt``; the
    returned wrapper is ``wrapper(state, *rest) -> Rt``.
    """

    def decorator(fn: Callable[Concatenate[str, P], R]) -> Callable[P, R]:
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
        if len(positional) < 2:
            raise TypeError("node function must take at least 2 parameters: (prompt: str, state, ...)")
        if params[0].annotation is not str:
            raise TypeError("node function's first parameter must be annotated as str")

        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return fn(prompt, *args, **kwargs)

        wrapper.__signature__ = sig.replace(parameters=params[1:])
        return wrapper

    return decorator