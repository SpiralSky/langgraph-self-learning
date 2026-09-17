import inspect
from typing import TypedDict

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, StateGraph

from graphs.prompt_node import with_prompt


def test_prompt_bound_as_first_arg():
    @with_prompt("fixed prompt")
    def f(prompt: str, state):
        return prompt

    assert f({"x": 1}) == "fixed prompt"


def test_extra_args_forwarded():
    @with_prompt("p")
    def f(prompt: str, state, a, b="default"):
        return (prompt, state, a, b)

    assert f({"x": 1}, 2, b=3) == ("p", {"x": 1}, 2, 3)


def test_return_passthrough():
    expected = {"partial": "update"}

    @with_prompt("p")
    def f(prompt: str, state):
        return expected

    assert f({"x": 1}) is expected


def test_metadata_preserved():
    @with_prompt("p")
    def my_node(prompt: str, state):
        """docstring here"""
        return state

    assert my_node.__name__ == "my_node"
    assert my_node.__doc__ == "docstring here"
    assert my_node.__module__ == __name__


def test_signature_re_exposed():
    @with_prompt("p")
    def f(prompt: str, state, config: int = 1):
        return state

    sig = inspect.signature(f)
    params = sig.parameters
    assert list(params) == ["state", "config"]
    assert "prompt" not in params
    assert params["state"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert params["config"].annotation is int
    assert params["config"].default == 1


def test_validation():
    with pytest.raises(TypeError):
        with_prompt("p")(lambda state: state)

    with pytest.raises(TypeError):
        with_prompt("p")(lambda prompt: prompt)

    with pytest.raises(TypeError):

        def bad(prompt: int, state):
            return state

        with_prompt("p")(bad)

    with pytest.raises(TypeError):

        def unannotated(prompt, state):
            return state

        with_prompt("p")(unannotated)


class State(TypedDict):
    x: int


def test_langgraph_integration_regression():
    @with_prompt("grade it")
    def grade_node(prompt: str, state: State, config: RunnableConfig | None) -> State:
        return {"x": state["x"] + 1}

    builder = StateGraph(State)
    builder.add_node(grade_node)
    builder.add_edge(START, "grade_node")
    graph = builder.compile()

    assert graph.invoke({"x": 41}) == {"x": 42}