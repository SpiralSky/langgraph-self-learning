"""Prompt compiler: applies ``graph_config.yaml`` customizations to node prompts.

Each node declares a :attr:`Node.intent` (via the ``@node`` decorator).
``compile_prompts`` merges, per node, in precedence order:

1. the node's base prompt (``Node.prompt``), after ``prompt_overrides``
   key-value substitution;
2. graph-level free-form blocks whose intent targets match the node's intent;
3. structured graph style directives (``pedagogical_style``/``verbosity``/
   ``tone``) for teaching intents;
4. per-node ``tone``, ``constraints``, ``formatting_rules`` and
   ``extra_instructions``.

Nodes call :func:`get_node_prompt` at runtime to read their compiled prompt.
"""

from collections.abc import Mapping
from typing import Any

from graphs.learning_graph.config import CustomizationsConfig
from graphs.learning_graph.nodes.node import Node

# Free-form graph customization blocks -> node intents they broadcast to.
# Every key present under ``customizations.graph`` must appear here; unknown
# keys fail loudly in :func:`compile_prompts`.
INTENT_TARGETS: dict[str, tuple[str, ...]] = {
    "tone_calibration": ("analyze", "decide", "generate", "format"),
    "analogy_policy": ("generate",),
    "pacing_and_pivot": ("decide",),
    "verify_before_inform": ("decide",),
    "comprehension_check": ("decide", "generate"),
    "adaptive_branching": ("decide",),
}

# Intents that receive the structured graph-level style directives.
STYLE_INTENTS: tuple[str, ...] = ("analyze", "decide", "generate")

# Compiled prompt per node name, populated by :func:`compile_prompts`.
_COMPILED: dict[str, str | None] = {}


def get_node_prompt(name: str, *, default: str | None = None) -> str | None:
    """
    Return the compiled prompt for a node.

    Falls back to ``default`` when the registry has not been populated yet
    (e.g. when a node module is imported without building the full graph).

    :param name: Node name (``_NODES`` key).
    :type name: str
    :param default: Prompt to return when no compiled entry exists.
    :type default: str | None
    :return: The compiled prompt for the node, else ``default``.
    :rtype: str | None
    """
    return _COMPILED.get(name, default)


def _apply_overrides(prompt: str, overrides: Mapping[str, Any]) -> str:
    """
    Substitute ``{key}`` placeholders in ``prompt`` with override values.

    Only placeholders present in ``overrides`` are replaced, leaving any other
    ``{...}`` format placeholders (e.g. ``{session_catalog}``) untouched.

    :param prompt: Base prompt text.
    :type prompt: str
    :param overrides: Mapping of placeholder key to replacement value.
    :type overrides: Mapping[str, Any]
    :return: Prompt with the matching placeholders substituted.
    :rtype: str
    """
    for key, value in overrides.items():
        prompt = prompt.replace(f"{{{key}}}", str(value))
    return prompt


def _graph_blocks(customizations: CustomizationsConfig) -> dict[str, str]:
    """Free-form blocks declared under ``customizations.graph``."""
    blocks = customizations.graph.model_extra or {}
    return {key: str(value) for key, value in blocks.items() if value is not None}


def compile_prompts(
    nodes: Mapping[str, Node],
    customizations: CustomizationsConfig,
) -> dict[str, str | None]:
    """
    Build the compiled prompt for every node in ``nodes``.

    The result is stored in the module registry (see :func:`get_node_prompt`)
    and returned for the graph ``export``.

    :param nodes: Mapping of node name to :class:`Node`.
    :type nodes: Mapping[str, Node]
    :param customizations: Loaded customization config.
    :type customizations: CustomizationsConfig
    :return: Node name -> compiled prompt (``None`` for nodes without a base
        prompt).
    :rtype: dict[str, str | None]
    :raises ValueError: If a graph customization block has no intent target.
    """
    graph_cfg = customizations.graph
    blocks = _graph_blocks(customizations)

    unknown = sorted(set(blocks) - set(INTENT_TARGETS))
    if unknown:
        raise ValueError(
            "Unknown graph customization block(s): " + ", ".join(unknown)
            + ". Add an INTENT_TARGETS entry to declare their target intents."
        )

    compiled: dict[str, str | None] = {}
    for name, node_obj in nodes.items():
        base = node_obj.prompt
        if base is None:
            compiled[name] = None
            continue

        node_cfg = customizations.nodes.get(name)
        if node_cfg is not None and node_cfg.prompt_overrides:
            base = _apply_overrides(base, node_cfg.prompt_overrides)

        parts = [base]
        intent = node_obj.intent
        if intent:
            for block, targets in INTENT_TARGETS.items():
                block_text = blocks.get(block)
                if block_text and intent in targets:
                    parts.append(block_text)
            if intent in STYLE_INTENTS:
                directives = []
                if graph_cfg.pedagogical_style:
                    directives.append(f"Pedagogical style: {graph_cfg.pedagogical_style}.")
                if graph_cfg.verbosity:
                    directives.append(f"Verbosity: {graph_cfg.verbosity}.")
                if graph_cfg.tone:
                    directives.append(f"Default tone: {graph_cfg.tone}.")
                if directives:
                    parts.append(" ".join(directives))
        if node_cfg is not None:
            if node_cfg.tone:
                parts.append(f"Tone: {node_cfg.tone}.")
            if node_cfg.constraints:
                parts.append(node_cfg.constraints)
            if node_cfg.formatting_rules:
                parts.append(node_cfg.formatting_rules)
            if node_cfg.extra_instructions:
                parts.append(node_cfg.extra_instructions)

        compiled[name] = "\n\n".join(parts)

    _COMPILED.update(compiled)
    return compiled