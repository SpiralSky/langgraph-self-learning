"""Unit tests for ``TextNode`` — no live LLM; a fake Runnable echoes prompts.

FakeLLM.invoke returns ``{content}:{prompt}`` and FakeLLM.ainvoke returns
``a:{prompt}``, so assertions are easy to read. The callable's raw return is
``{state_key: result}`` written as a plain value; list-wrapping happens only
when the target state field is declared list-typed.
"""

import pytest
from langgraph.graph import START, StateGraph
from pydantic import BaseModel, Field, ValidationError

from graphs.nodes.text_node import TextNode
from graphs.state import LearningGraphState


class FakeLLM:
    def __init__(self, content="ok"):
        self.content = content

    def invoke(self, prompt):
        return type("R", (), {"content": f"{self.content}:{prompt}"})()

    async def ainvoke(self, prompt):
        return type("R", (), {"content": f"a:{prompt}"})()


@pytest.mark.parametrize(
    ("name", "description", "prompt"),
    [
        ("", "desc", "prompt"),
        (None, "desc", "prompt"),
        (0, "desc", "prompt"),
        ("name", "", "prompt"),
        ("name", None, "prompt"),
        ("name", "desc", ""),
        ("name", "desc", None),
    ],
)
def test_construction_validates(name, description, prompt):
    with pytest.raises(ValueError):
        TextNode(name, description, prompt)


def test_orphan_placeholder_raises_at_construction():
    with pytest.raises(ValueError, match="not declared in params"):
        TextNode("n", "d", "Summarize {topic}")


def test_param_read_by_name_from_dict_state():
    node = TextNode(
        "summarize", "d", "Summarize: {topic}", params={"topic": str}
    )
    fn = node.get_node(FakeLLM())
    assert fn.invoke({"topic": "cats"}) == {"response": "ok:Summarize: cats"}


def test_param_read_by_name_from_pydantic_model_state():
    node = TextNode(
        "summarize", "d", "Summarize: {user_message}", params={"user_message": str}
    )
    fn = node.get_node(FakeLLM())
    state = LearningGraphState(user_message="cats")
    assert fn.invoke(state) == {"response": "ok:Summarize: cats"}


def test_param_typing_mismatch_raises_validation_error():
    node = TextNode(
        "levels", "d", "Level {level}", params={"level": int}
    )
    with pytest.raises(ValidationError):
        node.get_node(FakeLLM()).invoke({"level": "high"})


def test_generator_node_reads_nothing_runs_with_empty_fill():
    fn = TextNode("gen", "d", "Generate a haiku").get_node(FakeLLM())
    assert fn.invoke({}) == {"response": "ok:Generate a haiku"}
    assert fn.invoke(LearningGraphState()) == {"response": "ok:Generate a haiku"}


def test_writes_map_honored():
    node = TextNode(
        "analyst", "d", "Analyze {user_message}",
        params={"user_message": str},
        writes={"result": "analysis"},
    )
    fn = node.get_node(FakeLLM())
    assert fn.invoke({"user_message": "hi"}) == {"analysis": "ok:Analyze hi"}


def test_multi_key_writes_spread_across_fields():
    node = TextNode(
        "twin", "d", "Go {topic}",
        params={"topic": str},
        writes={"first": "a_out", "second": "b_out"},
    )
    fn = node.get_node(FakeLLM())
    assert fn.invoke({"topic": "x"}) == {
        "a_out": "ok:Go x",
        "b_out": "ok:Go x",
    }


def test_default_writes_target_response():
    assert TextNode("n", "d", "p").writes == {"result": "response"}


def test_prompt_without_placeholder_passes_through():
    fn = TextNode("plain", "d", "Do it:").get_node(FakeLLM())
    assert fn.invoke({}) == {"response": "ok:Do it:"}


async def test_async_ainvoke():
    node = TextNode(
        "summarize", "d", "Summarize {user_message}", params={"user_message": str}
    )
    fn = node.get_node(FakeLLM())
    assert await fn.ainvoke({"user_message": "hi"}) == {
        "response": "a:Summarize hi"
    }


def test_node_name_attribute():
    node = TextNode("my_awesome_node", "d", "p")
    assert node.get_node(FakeLLM()).__name__ == "my_awesome_node"


def test_langgraph_integration_plain_value_write():
    node = TextNode(
        "summarize", "d", "Summarize {user_message}", params={"user_message": str}
    )
    builder = StateGraph(LearningGraphState)
    builder.add_node(node.get_node(FakeLLM()))
    builder.add_edge(START, "summarize")
    graph = builder.compile()

    result = graph.invoke({"user_message": "seed"})
    assert result["response"] == "ok:Summarize seed"


def test_langgraph_integration_model_initial_state():
    node = TextNode(
        "shout", "d", "Shout {user_message}", params={"user_message": str}
    )
    builder = StateGraph(LearningGraphState)
    builder.add_node(node.get_node(FakeLLM()))
    builder.add_edge(START, "shout")
    graph = builder.compile()

    result = graph.invoke(LearningGraphState(user_message="hey"))
    assert result["response"] == "ok:Shout hey"


class ListResponseState(BaseModel):
    """A state model whose ``response`` field is declared list-typed."""

    response: list[str] = Field(default_factory=list)


def test_list_typed_field_gets_wrapped_value():
    fn = TextNode("n", "d", "P").get_node(FakeLLM())
    assert fn.invoke(ListResponseState()) == {"response": ["ok:P"]}


def test_plain_dict_state_never_wraps_even_colliding_field_name():
    fn = TextNode("n", "d", "P").get_node(FakeLLM())
    assert fn.invoke({"response": "seed"}) == {"response": "ok:P"}