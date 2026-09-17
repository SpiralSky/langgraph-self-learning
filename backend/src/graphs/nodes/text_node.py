"""``TextNode``: a by-name-parameter LLM node producing str results.

On ``get_node(llm)`` the node produces a LangGraph-ready dual callable: for
every declared ``param`` it reads the value from state (dict or pydantic
model) by name and validates it, fills the corresponding ``{param}``
placeholder into the prompt, runs the supplied LLM, and returns a partial
update mapping each ``writes`` field to its state key. The whole state flows
through — there is no stack reading.

Writes are plain values; a state key is list-wrapped only when the target
pydantic field is declared list-typed.

Nothing from langgraph is imported here; the callable is consumed by the
graph runtime loosely, exactly like the ``GraphNode`` protocol intends.
"""

from typing import Annotated, get_args, get_origin

from langchain_core.runnables import Runnable
from pydantic import BaseModel, TypeAdapter

from graphs.nodes.base import AbstractNode, DualCallable
from graphs.serialization import (
    deserialize_annotation,
    register_node_type,
    serialize_annotation,
)


def is_list_annotation(annotation: object) -> bool:
    """True when ``annotation`` is a list type (e.g. ``list``, ``list[str]``)."""
    if annotation is list:
        return True
    origin = get_origin(annotation)
    if origin is Annotated:
        return is_list_annotation(get_args(annotation)[0])
    return origin is list


class _TextNodeFn:
    """LangGraph-ready dual callable produced by ``TextNode.get_node``.

    One ``TypeAdapter`` per param is built once and reused for every
    invocation. The instance carries ``__name__`` so LangGraph infers the
    node name from the callable itself.
    """

    def __init__(
        self,
        *,
        name: str,
        prompt: str,
        params: dict[str, type],
        writes: dict[str, str],
        llm: Runnable,
    ) -> None:
        self.__name__ = name
        self._prompt = prompt
        self._validators = {p: TypeAdapter(ann) for p, ann in params.items()}
        self._writes = writes
        self._llm = llm

    def __call__(self, state: dict) -> dict:
        return self.invoke(state)

    @staticmethod
    def _read(state: dict, key: str) -> object:
        """Read ``key`` from a pydantic-model or plain-dict state."""
        if isinstance(state, BaseModel):
            return getattr(state, key)
        return state[key]

    def _fill_prompt(self, state: dict) -> str:
        filled = self._prompt
        for param, validator in self._validators.items():
            value = self._read(state, param)
            filled = filled.replace(f"{{{param}}}", str(validator.validate_python(value)))
        return filled

    @staticmethod
    def _target_is_list(state: dict, state_key: str) -> bool:
        """List-wrap only when the target state field is declared a list.

        The field type is read from the pydantic state model; a plain-dict
        state has no schema, so values are written unwrapped.
        """
        if not isinstance(state, BaseModel):
            return False
        field = type(state).model_fields.get(state_key)
        return field is not None and is_list_annotation(field.annotation)

    def _update(self, state: dict, content: str) -> dict:
        return {
            state_key: [content] if self._target_is_list(state, state_key) else content
            for state_key in self._writes.values()
        }

    def invoke(self, state: dict) -> dict:
        """Read params by name, run the LLM on the filled prompt, spread writes."""
        result = self._llm.invoke(self._fill_prompt(state))
        return self._update(state, str(result.content))

    async def ainvoke(self, state: dict) -> dict:
        """Async twin of ``invoke`` using ``await llm.ainvoke(...)``."""
        result = await self._llm.ainvoke(self._fill_prompt(state))
        return self._update(state, str(result.content))


class TextNode(AbstractNode):
    """A LLM node: read params by name, fill the prompt, run an LLM.

    :param name: Node name (used as the LangGraph node name).
    :type name: str
    :param description: What this node does, used for retrieval.
    :type description: str
    :param prompt: Prompt template; ``{param}`` placeholders must be declared in
        ``params``.
    :type prompt: str
    :param params: By-name inputs read from state — ``{key: annotation}``.
        Empty (default) makes this a generator node that reads nothing.
    :type params: dict[str, type] | None
    :param writes: Output map ``{field: state_key}``; each key receives the
        single LLM result. Defaults to ``{"result": "response"}``.
    :type writes: dict[str, str] | None
    """

    def get_node(self, llm: Runnable) -> DualCallable:
        """Bind an LLM and return the LangGraph-ready dual callable."""
        return _TextNodeFn(
            name=self.name,
            prompt=self.prompt,
            params=self.params,
            writes=self.writes,
            llm=llm,
        )

    def to_dict(self) -> dict:
        """Type-tagged JSON-ready dict (``"type": "text"``).

        Param annotations are canonicalized via
        :func:`graphs.serialization.serialize_annotation`.
        """
        return {
            "type": "text",
            "name": self.name,
            "description": self.description,
            "prompt": self.prompt,
            "params": {
                param: serialize_annotation(annotation)
                for param, annotation in self.params.items()
            },
            "writes": dict(self.writes),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TextNode":
        """Rebuild a :class:`TextNode` from a ``to_dict`` payload."""
        return cls(
            name=data["name"],
            description=data["description"],
            prompt=data["prompt"],
            params={
                param: deserialize_annotation(annotation)
                for param, annotation in data["params"].items()
            },
            writes=dict(data["writes"]),
        )


register_node_type("text", TextNode.from_dict)