"""Unit tests for ``Graph.compile`` — compiling structure to langgraph.

Pure-unit: every LLM is a fake whose result content is deterministic and
derived from the prompt it was given; nothing touches the network. Scenario
coverage mirrors ``docs/graphs.md`` intent: linear chains, read/write wiring,
per-node LLM overrides, routing edges, and validation/LLM error paths.
"""

import pytest

from graphs.graph import END, START, Graph, GraphValidationError
from graphs.nodes.text_node import TextNode


class FakeLLM:
    """Returns ``{content}:{prompt}`` so tests can tell which LLM ran."""

    def __init__(self, content="ok"):
        self.content = content

    def invoke(self, prompt):
        return type("R", (), {"content": f"{self.content}:{prompt}"})()

    async def ainvoke(self, prompt):
        return type("R", (), {"content": f"a:{prompt}"})()


class RoutingLLM:
    """Picks an edge candidate: ``invoke`` returns ``choice`` verbatim."""

    def __init__(self, choice):
        self.choice = choice
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return type("R", (), {"content": self.choice})()

    async def ainvoke(self, prompt):
        self.prompts.append(prompt)
        return type("R", (), {"content": self.choice})()


def _text_node(node_id, name, prompt, *, params=None, writes=None):
    return TextNode(name, f"description of {node_id}", prompt, params=params, writes=writes)


def test_compile_linear_chain_runs_both_nodes():
    graph = Graph()
    graph.add_node(
        "a", _text_node("a", "alpha", "Summarize {user_message}",
                        params={"user_message": str}), llm=FakeLLM("A"))
    graph.add_node(
        "b", _text_node("b", "beta", "Work on {user_message}",
                        params={"user_message": str}), llm=FakeLLM("B"))
    graph.add_edge("e_start_a", START, "a")
    graph.add_edge("e_a_b", "a", "b")
    graph.add_edge("e_b_end", "b", END)

    graph_runtime = graph.compile()
    result = graph_runtime.invoke({"user_message": "hey"})

    assert result["response"] == "B:Work on hey"


def test_compile_read_writes_across_nodes():
    graph = Graph()
    graph.add_node(
        "a", _text_node("a", "alpha", "Honk {user_message}",
                        params={"user_message": str}), llm=FakeLLM("A"))
    graph.add_node(
        "b", _text_node("b", "beta", "Merge {response} into the answer",
                        params={"response": str}), llm=FakeLLM("B"))
    graph.add_edge("e_start_a", START, "a")
    graph.add_edge("e_a_b", "a", "b")
    graph.add_edge("e_b_end", "b", END)

    result = graph.compile().invoke({"user_message": "hey"})

    assert result["response"] == "B:Merge A:Honk hey into the answer"


def test_compile_per_node_llm_beats_default():
    graph = Graph()
    graph.add_node(
        "a", _text_node("a", "alpha", "Summarize {user_message}",
                        params={"user_message": str}), llm=FakeLLM("A"))
    graph.add_node(
        "b", _text_node("b", "beta", "Work on {user_message}",
                        params={"user_message": str}))
    graph.add_edge("e_start_a", START, "a")
    graph.add_edge("e_a_b", "a", "b")
    graph.add_edge("e_b_end", "b", END)

    result = graph.compile(llm=FakeLLM("default")).invoke(
        {"user_message": "hey"}
    )

    assert result["response"] == "default:Work on hey"


def test_compile_routing_edge_follows_chosen_candidate():
    graph = Graph()
    for node_id, name, content in (("a", "alpha", "A"), ("b", "beta", "B"), ("c", "gamma", "C")):
        graph.add_node(
            node_id,
            _text_node(node_id, name, f"{name.title()} {name.title()}",
                       params={"user_message": str}),
            llm=FakeLLM(content))
    router = RoutingLLM("b")
    graph.add_edge("e_start_a", START, "a")
    graph.add_edge("e_a_bc", "a", ["b", "c"], routing_prompt="pick one", model=router)
    graph.add_edge("e_b_end", "b", END)
    graph.add_edge("e_c_end", "c", END)

    result = graph.compile().invoke({"user_message": "hey"})

    assert router.prompts == ["pick one"]
    assert result["response"] == "B:Beta Beta"


def test_compile_routing_edge_to_end_candidate():
    graph = Graph()
    graph.add_node("a", _text_node("a", "alpha", "Alpha {user_message}",
                                   params={"user_message": str}), llm=FakeLLM("A"))
    router = RoutingLLM(END)
    graph.add_edge("e_start_a", START, "a")
    graph.add_edge("e_a_end", "a", [END], routing_prompt="stop?", model=router)

    result = graph.compile().invoke({"user_message": "hey"})

    assert result["response"] == "A:Alpha hey"


def test_compile_validates_first():
    graph = Graph()
    graph.add_node("a", _text_node("a", "alpha", "p"), llm=FakeLLM("A"))
    with pytest.raises(GraphValidationError, match="exactly one START-source edge"):
        graph.compile()

    empty = Graph()
    with pytest.raises(GraphValidationError):
        empty.compile()


def test_compile_missing_node_llm_errors():
    graph = Graph()
    graph.add_node("a", _text_node("a", "alpha", "Do {user_message}",
                                   params={"user_message": str}))
    graph.add_edge("e_start_a", START, "a")
    graph.add_edge("e_a_end", "a", END)
    with pytest.raises(ValueError, match="needs an LLM"):
        graph.compile()


def test_compile_routing_edge_missing_llm_errors():
    graph = Graph()
    graph.add_node("a", _text_node("a", "alpha", "p"), llm=FakeLLM("A"))
    graph.add_node("b", _text_node("b", "beta", "p"), llm=FakeLLM("B"))
    graph.add_edge("e_start_a", START, "a")
    graph.add_edge("e_a_b", "a", ["b"], routing_prompt="pick one")
    graph.add_edge("e_b_end", "b", END)
    with pytest.raises(ValueError, match="routing edge.*needs an LLM"):
        graph.compile()