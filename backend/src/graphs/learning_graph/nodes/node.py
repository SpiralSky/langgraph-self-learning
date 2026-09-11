import functools
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


class Node:
    """Wraps a node callable with an optional prompt and intent declaration.

    The ``@node`` decorator turns a plain function into a :class:`Node`
    instance that the graph compiler can inspect for prompt and intent
    metadata.  ``functools.update_wrapper`` preserves the original
    function's ``__name__``, ``__doc__``, etc.
    """


class Node:
    def __init__(
        self,
        fn: Callable[..., Any],
        prompt: str | None = None,
        intent: str | None = None,
    ) -> None:
        self.fn = fn
        self.prompt = prompt
        self.intent = intent
        functools.update_wrapper(self, fn, updated=[])

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)


def node(
    *,
    prompt: str | None = None,
    intent: str | None = None,
) -> Callable[[F], "Node"]:
    def decorator(func: F) -> Node:
        return Node(fn=func, prompt=prompt, intent=intent)
    return decorator