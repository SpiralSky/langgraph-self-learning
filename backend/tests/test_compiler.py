import pytest

from graphs.learning_graph.compiler import (
    _COMPILED,
    compile_prompts,
    get_node_prompt,
)
from graphs.learning_graph.config import (
    CustomizationsConfig,
    GraphCustomization,
    NodeCustomization,
)
from graphs.learning_graph.nodes.node import Node


@pytest.fixture(autouse=True)
def _isolate_registry():
    saved = dict(_COMPILED)
    yield
    _COMPILED.clear()
    _COMPILED.update(saved)


def _node(prompt: str | None = "BASE", intent: str | None = "decide") -> Node:
    return Node(fn=lambda state: {}, prompt=prompt, intent=intent)


class TestCompilePrompts:
    def test_node_extra_instructions_appended(self):
        cc = CustomizationsConfig(
            nodes={"decision_maker": NodeCustomization(extra_instructions="EXTRA")}
        )
        out = compile_prompts({"decision_maker": _node()}, cc)
        assert out["decision_maker"] == "BASE\n\nEXTRA"

    def test_graph_block_broadcast_by_intent(self):
        cc = CustomizationsConfig(graph=GraphCustomization(pacing_and_pivot="PIVOT"))
        nodes = {
            "decider": _node(intent="decide"),
            "writer": _node(intent="generate"),
        }
        out = compile_prompts(nodes, cc)
        assert out["decider"] == "BASE\n\nPIVOT"
        assert out["writer"] == "BASE"

    def test_prompt_overrides_substitute_in_base(self):
        cc = CustomizationsConfig(
            nodes={"decision_maker": NodeCustomization(prompt_overrides={"tone": "warm"})}
        )
        out = compile_prompts({"decision_maker": _node(prompt="Write with {tone} tone.")}, cc)
        assert out["decision_maker"] == "Write with warm tone."

    def test_prompt_overrides_leave_other_placeholders(self):
        cc = CustomizationsConfig(
            nodes={"input_analyzer": NodeCustomization(prompt_overrides={"style": "X"})}
        )
        out = compile_prompts(
            {"input_analyzer": _node(prompt="A {style} and {session_catalog}.")}, cc
        )
        assert out["input_analyzer"] == "A X and {session_catalog}."

    def test_precedence_base_then_graph_then_node(self):
        cc = CustomizationsConfig(
            graph=GraphCustomization(tone_calibration="GRAPH_BLOCK"),
            nodes={"decision_maker": NodeCustomization(constraints="NODE_CONSTRAINT", tone="friendly")},
        )
        out = compile_prompts({"decision_maker": _node()}, cc)
        assert out["decision_maker"] == "BASE\n\nGRAPH_BLOCK\n\nTone: friendly.\n\nNODE_CONSTRAINT"

    def test_structured_style_only_for_teaching_intents(self):
        cc = CustomizationsConfig(
            graph=GraphCustomization(pedagogical_style="socratic", verbosity="concise")
        )
        nodes = {
            "decider": _node(intent="decide"),
            "formatter": _node(intent="format"),
        }
        out = compile_prompts(nodes, cc)
        assert "Pedagogical style: socratic." in out["decider"]
        assert "Verbosity: concise." in out["decider"]
        assert out["formatter"] == "BASE"

    def test_unknown_graph_block_raises(self):
        cc = CustomizationsConfig(graph=GraphCustomization(bogus_block="X"))
        with pytest.raises(ValueError, match="Unknown graph customization block"):
            compile_prompts({"decision_maker": _node()}, cc)

    def test_pedagogy_blocks_broadcast_by_intent(self):
        cc = CustomizationsConfig(
            graph=GraphCustomization(
                verify_before_inform="VERIFY",
                comprehension_check="CHECK",
                adaptive_branching="BRANCH",
            )
        )
        nodes = {
            "decider": _node(intent="decide"),
            "writer": _node(intent="generate"),
            "formatter": _node(intent="format"),
        }
        out = compile_prompts(nodes, cc)
        assert out["decider"] == "BASE\n\nVERIFY\n\nCHECK\n\nBRANCH"
        assert out["writer"] == "BASE\n\nCHECK"
        assert out["formatter"] == "BASE"

    def test_no_prompt_node_compiles_to_none(self):
        out = compile_prompts({"user_input": _node(prompt=None)}, CustomizationsConfig())
        assert out["user_input"] is None

    def test_formatting_rules_appended(self):
        cc = CustomizationsConfig(
            nodes={"format_output": NodeCustomization(formatting_rules="RULES")}
        )
        out = compile_prompts({"format_output": _node(intent="format")}, cc)
        assert out["format_output"] == "BASE\n\nRULES"


class TestGetNodePrompt:
    def test_returns_compiled_prompt(self):
        cc = CustomizationsConfig(
            nodes={"decision_maker": NodeCustomization(extra_instructions="EXTRA")}
        )
        compile_prompts({"decision_maker": _node()}, cc)
        assert get_node_prompt("decision_maker") == "BASE\n\nEXTRA"

    def test_falls_back_to_default_when_not_compiled(self):
        _COMPILED.pop("ghost", None)
        assert get_node_prompt("ghost") is None
        assert get_node_prompt("ghost", default="D") == "D"