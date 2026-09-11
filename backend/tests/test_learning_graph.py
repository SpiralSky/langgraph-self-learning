import pytest
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from graphs.learning_graph.compiler import compile_prompts
from graphs.learning_graph.config import config
from graphs.learning_graph.learning_graph import _NODES, builder, export, graph
from graphs.learning_graph.nodes.node import Node


class TestNodesDict:
    EXPECTED_KEYS = [
        "user_input",
        "input_analyzer",
        "retrieve_memory",
        "decision_maker",
        "response_builder",
        "save_memory",
        "format_output",
        "model_output",
        "execute_code",
    ]

    def test_contains_all_expected_keys(self):
        assert set(_NODES.keys()) == set(self.EXPECTED_KEYS)

    def test_each_value_is_node_instance(self):
        for name, node_obj in _NODES.items():
            assert isinstance(node_obj, Node), f"{name} is not a Node"

    def test_no_extra_keys(self):
        assert len(_NODES) == len(self.EXPECTED_KEYS)


class TestGraphCompilation:
    def test_builder_is_state_graph(self):
        assert isinstance(builder, StateGraph)

    def test_compile_succeeds(self):
        assert isinstance(graph, CompiledStateGraph)


class TestGraphEdges:
    @pytest.fixture(autouse=True)
    def _graph_info(self):
        g = graph.get_graph()
        self.nodes = {n for n in g.nodes}
        self.edges = {(e.source, e.target) for e in g.edges}

    def test_start_to_user_input(self):
        assert ("__start__", "user_input") in self.edges

    def test_user_input_to_retrieve_memory(self):
        assert ("user_input", "retrieve_memory") in self.edges

    def test_user_input_to_input_analyzer(self):
        assert ("user_input", "input_analyzer") in self.edges

    def test_retrieve_memory_to_decision_maker(self):
        assert ("retrieve_memory", "decision_maker") in self.edges

    def test_input_analyzer_to_decision_maker(self):
        assert ("input_analyzer", "decision_maker") in self.edges

    def test_decision_maker_to_response_builder(self):
        assert ("decision_maker", "response_builder") in self.edges

    def test_response_builder_to_save_memory(self):
        assert ("response_builder", "save_memory") in self.edges

    def test_save_memory_to_format_output(self):
        assert ("save_memory", "format_output") in self.edges

    def test_format_output_to_model_output(self):
        assert ("format_output", "model_output") in self.edges

    def test_model_output_to_end(self):
        assert ("model_output", "__end__") in self.edges


class TestExport:
    def test_export_has_required_keys(self):
        assert "graph" in export
        assert "nodes" in export
        assert "prompts" in export

    def test_graph_is_compiled(self):
        assert isinstance(export["graph"], CompiledStateGraph)

    def test_nodes_is_same_dict_as_nodes(self):
        assert export["nodes"] is _NODES

    def test_prompts_maps_each_name_to_compiled_prompt(self):
        compiled = compile_prompts(_NODES, config.customizations)
        for name, node_obj in _NODES.items():
            assert export["prompts"][name] == compiled[name]

    def test_prompts_extend_base_prompt_when_present(self):
        for name, node_obj in _NODES.items():
            if node_obj.prompt is None:
                assert export["prompts"][name] is None
            else:
                assert export["prompts"][name].startswith(node_obj.prompt)


class TestTimedDecoration:
    def test_all_nodes_wrapped_with_timed(self):
        for name in _NODES:
            node_fn = graph.nodes[name].bound.func
            assert hasattr(node_fn, "__wrapped__"), f"{name} missing __wrapped__"