"""Structural validation of the ``AbstractNode`` ABC and the ``GraphNode``
protocol plus the dual-callable contract produced by ``get_node(llm)``.

``GraphNode`` is a plain ``typing.Protocol`` (not ``runtime_checkable``), so
conformance is asserted structurally — by checking required members exist —
rather than with ``isinstance``.
"""

import pytest

from graphs.nodes.base import AbstractNode, GraphNode
from graphs.nodes.text_node import TextNode

EXPECTED_NODE_ATTRS = ("name", "description", "prompt", "input_shape")


class _FakeLLM:
    def __init__(self, tokens=None):
        self.tokens = tokens

    def _result(self, prompt):
        if self.tokens is None:
            return type("R", (), {"content": prompt})()
        return type("R", (), {
            "content": prompt,
            "usage_metadata": {"output_tokens": self.tokens},
        })()

    def invoke(self, prompt):
        return self._result(prompt)

    async def ainvoke(self, prompt):
        return self._result(prompt)


class _StubFn:
    def invoke(self, state: dict) -> dict:
        return {"response": "ok"}

    async def ainvoke(self, state: dict) -> dict:
        return {"response": "ok"}


class _ConcreteNode(AbstractNode):
    """Minimal concrete node used to exercise the ABC directly."""

    def get_node(self, llm):
        return _StubFn()


class StubGraphNode:
    """Minimal structural stand-in for ``GraphNode``."""

    name: str = "stub"
    description: str = "a stub node"
    prompt: str = "do {input}"
    input_shape: type | None = str

    def get_node(self, llm):
        return _StubFn()


def test_abstractnode_constructs_and_records_defaults():
    node = _ConcreteNode("n", "d", "p")
    assert node.name == "n"
    assert node.description == "d"
    assert node.prompt == "p"
    assert node.params == {}
    assert node.writes == {"result": "response"}


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
def test_abstractnode_validates_metadata(name, description, prompt):
    with pytest.raises(ValueError):
        _ConcreteNode(name, description, prompt)


def test_abstractnode_rejects_orphan_placeholders():
    with pytest.raises(ValueError, match="not declared in params"):
        _ConcreteNode("n", "d", "Hello {topic}")


def test_abstractnode_accepts_declared_placeholders():
    node = _ConcreteNode("n", "d", "Hello {topic}", params={"topic": str})
    assert node.params == {"topic": str}
    assert node.prompt == "Hello {topic}"


def test_abstractnode_stores_params_and_writes():
    node = _ConcreteNode(
        "n", "d", "p", params={"a": int, "b": str}, writes={"x": "out_a", "y": "out_b"}
    )
    assert node.params == {"a": int, "b": str}
    assert node.writes == {"x": "out_a", "y": "out_b"}


def test_abstractnode_validates_write_keys_and_values():
    with pytest.raises(ValueError):
        _ConcreteNode("n", "d", "p", writes={"": "out"})
    with pytest.raises(ValueError):
        _ConcreteNode("n", "d", "p", writes={"x": ""})


def test_textnode_is_an_abstractnode():
    node = TextNode("n", "d", "p")
    assert isinstance(node, AbstractNode)
    assert node.params == {}
    assert node.writes == {"result": "response"}


def test_stub_satisfies_graphnode_structurally():
    for attr in EXPECTED_NODE_ATTRS:
        assert hasattr(StubGraphNode, attr)
    assert callable(StubGraphNode.get_node)


def test_plain_protocol_not_runtime_checkable():
    with pytest.raises(TypeError):
        isinstance(StubGraphNode(), GraphNode)


def test_get_node_returns_dualcable():
    fn = TextNode("n", "d", "p").get_node(_FakeLLM())
    assert callable(fn.invoke)
    assert callable(fn.ainvoke)


def test_dualcable_contract_shape():
    fn = StubGraphNode().get_node(_FakeLLM())
    assert callable(fn.invoke)
    assert callable(fn.ainvoke)
    assert fn.invoke({}) == {"response": "ok"}


# ---------------------------------------------------------------- updated()


def test_updated_returns_validated_fresh_instance():
    node = _ConcreteNode("n", "d", "Hello {topic}", params={"topic": str})
    updated = node.updated(prompt="Hello {topic} there")
    assert updated is not node
    assert updated.name == "n"
    assert updated.description == "d"
    assert updated.prompt == "Hello {topic} there"
    assert node.prompt == "Hello {topic}"


def test_updated_placeholders_must_stay_subset_of_params():
    node = _ConcreteNode("n", "d", "p")
    ok = node.updated(params={"topic": str}, prompt="Do {topic}")
    assert ok.params == {"topic": str}
    assert ok.prompt == "Do {topic}"
    with pytest.raises(ValueError, match="not declared in params"):
        node.updated(params={"topic": str}, prompt="Do {missing}")


def test_updated_preserves_unrelated_braces():
    node = _ConcreteNode("n", "d", "Score {score}", params={"score": str})
    updated = node.updated(prompt="Score {score} out of 100")
    assert updated.prompt == "Score {score} out of 100"
    assert updated.params == {"score": str}


def test_updated_unknown_kwarg_raises_type_error():
    node = _ConcreteNode("n", "d", "p")
    with pytest.raises(TypeError, match="unsupported"):
        node.updated(bogus=1)


def test_updated_no_changes_returns_fresh_instance():
    node = _ConcreteNode("n", "d", "p")
    fresh = node.updated()
    assert fresh is not node
    assert (fresh.name, fresh.description, fresh.prompt) == ("n", "d", "p")
    assert fresh.params == {}
    assert fresh.writes == {"result": "response"}


# --------------------------------------------------------------- run stats


def test_default_stats_are_zeroed():
    node = _ConcreteNode("n", "d", "p")
    assert node.run_counts == 0
    assert (node.output_tokens.count, node.output_tokens.mean) == (0, 0.0)
    assert (node.output_time.count, node.output_time.mean) == (0, 0.0)


def test_record_run_bumps_count_and_folds_time_and_tokens():
    node = _ConcreteNode("n", "d", "p")
    node.record_run(tokens=10, elapsed_seconds=0.5)
    node.record_run(tokens=10, elapsed_seconds=0.5)
    assert node.run_counts == 2
    assert node.output_tokens.count == 2
    assert node.output_tokens.mean == 10.0
    assert node.output_time.count == 2
    assert node.output_time.mean == 0.5
    assert node.output_tokens.std == 0.0
    assert node.output_time.std == 0.0


def test_record_run_time_only_and_stats_detail():
    node = _ConcreteNode("n", "d", "p")
    node.record_run(elapsed_seconds=0.2)
    node.record_run(elapsed_seconds=1.0)
    assert node.run_counts == 2
    assert node.output_tokens.count == 0
    assert node.output_time.count == 2
    assert node.output_time.mean == pytest.approx(0.6)
    assert node.output_time.std is not None


def test_record_run_skips_none_fields_but_bumps_count():
    node = _ConcreteNode("n", "d", "p")
    node.record_run()
    assert node.run_counts == 1
    assert node.output_tokens.count == 0
    assert node.output_time.count == 0


def test_record_run_rejects_negative_and_nan():
    node = _ConcreteNode("n", "d", "p")
    with pytest.raises(ValueError, match="output_tokens must not be negative"):
        node.record_run(tokens=-1, elapsed_seconds=0.1)
    with pytest.raises(ValueError, match="must not be NaN"):
        node.record_run(tokens=float("nan"), elapsed_seconds=0.1)
    with pytest.raises(ValueError, match="elapsed_seconds must not be negative"):
        node.record_run(tokens=1, elapsed_seconds=-0.1)
    with pytest.raises(ValueError, match="must not be NaN"):
        node.record_run(tokens=1, elapsed_seconds=float("nan"))
    with pytest.raises(ValueError, match="output_tokens must be numeric"):
        node.record_run(tokens="ten", elapsed_seconds=0.1)
    assert node.run_counts == 0


def test_record_result_extracts_output_tokens_and_skips_when_absent():
    node = _ConcreteNode("n", "d", "p")

    class Result:
        usage_metadata = {"output_tokens": 7}

    node.record_result(Result(), 0.25)
    assert node.run_counts == 1
    assert node.output_tokens.count == 1
    assert node.output_tokens.mean == 7.0
    assert node.output_time.count == 1

    class NoUsage:
        pass

    node.record_result(NoUsage(), 0.25)
    assert node.run_counts == 2
    assert node.output_tokens.count == 1
    assert node.output_time.count == 2

    class TextTokens:
        usage_metadata = {"output_tokens": "twenty"}

    node.record_result(TextTokens(), 0.25)
    assert node.run_counts == 3
    assert node.output_tokens.count == 1
    assert node.output_time.count == 3


def test_updated_preserves_run_stats():
    node = _ConcreteNode("n", "d", "p")
    node.record_run(tokens=5, elapsed_seconds=0.3)
    updated = node.updated(name="renamed")
    assert updated.run_counts == 1
    assert updated.output_tokens.count == 1
    assert updated.output_tokens.mean == 5.0
    assert updated.output_time.count == 1
    assert node.run_counts == 1