import os
from pathlib import Path
from typing import Any, Dict

import pytest
from pydantic import ValidationError

from graphs.learning_graph.config import (
    Config,
    CustomizationsConfig,
    GraphCustomization,
    NodeCustomization,
    ProviderData,
    _resolve_dict,
    _resolve_env_vars,
    read_config,
)


# ---------------------------------------------------------------------------
# _resolve_env_vars
# ---------------------------------------------------------------------------

class TestResolveEnvVars:
    def test_placeholder_replaced(self, monkeypatch):
        monkeypatch.setenv("MY_KEY", "secret123")
        assert _resolve_env_vars("${MY_KEY}") == "secret123"

    def test_no_placeholder_returns_original(self):
        assert _resolve_env_vars("hello world") == "hello world"

    def test_multiple_placeholders(self, monkeypatch):
        monkeypatch.setenv("A", "foo")
        monkeypatch.setenv("B", "bar")
        assert _resolve_env_vars("${A}_${B}") == "foo_bar"

    def test_missing_env_var_raises(self):
        with pytest.raises(ValueError, match="Environment variable 'DOES_NOT_EXIST' is not set"):
            _resolve_env_vars("${DOES_NOT_EXIST}")

    def test_partial_missing_raises(self, monkeypatch):
        monkeypatch.setenv("A", "ok")
        with pytest.raises(ValueError, match="Environment variable 'B' is not set"):
            _resolve_env_vars("${A}_${B}")


# ---------------------------------------------------------------------------
# _resolve_dict
# ---------------------------------------------------------------------------

class TestResolveDict:
    def test_plain_string_bypassed(self):
        assert _resolve_dict("hello") == "hello"

    def test_string_no_dollar_bypassed(self):
        assert _resolve_dict("no placeholder") == "no placeholder"

    def test_nested_dict_keys_resolved(self, monkeypatch):
        monkeypatch.setenv("API_KEY", "sk-abc")
        data = {"api_key": "${API_KEY}", "endpoint": "https://example.com"}
        assert _resolve_dict(data) == {"api_key": "sk-abc", "endpoint": "https://example.com"}

    def test_list_members_resolved(self, monkeypatch):
        monkeypatch.setenv("MODEL", "gpt-4")
        data = ["${MODEL}", "fixed"]
        assert _resolve_dict(data) == ["gpt-4", "fixed"]

    def test_int_float_bool_bypassed(self):
        assert _resolve_dict(42) == 42
        assert _resolve_dict(3.14) == 3.14
        assert _resolve_dict(True) is True

    def test_mixed_structure(self, monkeypatch):
        monkeypatch.setenv("KEY", "val")
        data: Dict[str, Any] = {
            "outer": [
                {"inner": "${KEY}"},
                "plain",
            ],
            "count": 5,
        }
        assert _resolve_dict(data) == {
            "outer": [
                {"inner": "val"},
                "plain",
            ],
            "count": 5,
        }


# ---------------------------------------------------------------------------
# ProviderData
# ---------------------------------------------------------------------------

class TestProviderData:
    def test_reasoning_capabilities_prefers_reasoning_levels(self):
        pd = ProviderData(
            api_endpoint="https://example.com",
            api_key="key",
            model_id="m1",
            reasoning_levels=["low", "medium", "high"],
            thinking_level="medium",
        )
        assert pd.reasoning_capabilities == ("low", "medium", "high")

    def test_reasoning_capabilities_falls_back_to_thinking(self):
        pd = ProviderData(
            api_endpoint="https://example.com",
            api_key="key",
            model_id="m1",
            thinking_level="high",
        )
        assert pd.reasoning_capabilities == ("high",)

    def test_reasoning_capabilities_empty_when_none(self):
        pd = ProviderData(
            api_endpoint="https://example.com",
            api_key="key",
            model_id="m1",
        )
        assert pd.reasoning_capabilities == ()

    def test_default_reasoning_returns_explicit_default(self):
        pd = ProviderData(
            api_endpoint="https://example.com",
            api_key="key",
            model_id="m1",
            reasoning_levels=["low", "medium", "high"],
            reasoning_default="low",
        )
        assert pd.default_reasoning == "low"

    def test_default_reasoning_middle_of_levels(self):
        pd = ProviderData(
            api_endpoint="https://example.com",
            api_key="key",
            model_id="m1",
            reasoning_levels=["low", "medium", "high"],
        )
        assert pd.default_reasoning == "medium"

    def test_default_reasoning_middle_for_even_count(self):
        pd = ProviderData(
            api_endpoint="https://example.com",
            api_key="key",
            model_id="m1",
            reasoning_levels=["a", "b", "c", "d"],
        )
        assert pd.default_reasoning == "c"

    def test_default_reasoning_none_when_no_reasoning(self):
        pd = ProviderData(
            api_endpoint="https://example.com",
            api_key="key",
            model_id="m1",
        )
        assert pd.default_reasoning is None


# ---------------------------------------------------------------------------
# Config construction
# ---------------------------------------------------------------------------

class TestConfig:
    def test_from_dict_with_required_fields(self):
        cfg = Config(
            provider_data={
                "p1": ProviderData(
                    api_endpoint="https://ep.com",
                    api_key="key1",
                    model_id="m1",
                ),
            },
            model_providers={"node_a": "p1"},
        )
        assert cfg.provider_data["p1"].model_id == "m1"
        assert cfg.model_providers["node_a"] == "p1"
        assert cfg.memory_top_k == 5  # default

    def test_with_customizations_populated(self):
        cfg = Config(
            provider_data={
                "p1": ProviderData(
                    api_endpoint="https://ep.com",
                    api_key="key1",
                    model_id="m1",
                ),
            },
            model_providers={"node_a": "p1"},
            customizations=CustomizationsConfig(
                graph=GraphCustomization(pedagogical_style="socratic"),
                nodes={
                    "response_builder": NodeCustomization(
                        tone="friendly",
                    ),
                },
            ),
        )
        assert cfg.customizations.graph.pedagogical_style == "socratic"
        assert cfg.customizations.nodes["response_builder"].tone == "friendly"

    def test_get_model_data_returns_correct_provider(self):
        cfg = Config(
            provider_data={
                "p1": ProviderData(
                    api_endpoint="https://ep.com",
                    api_key="key1",
                    model_id="m1",
                ),
                "p2": ProviderData(
                    api_endpoint="https://ep2.com",
                    api_key="key2",
                    model_id="m2",
                ),
            },
            model_providers={"node_a": "p1", "node_b": "p2"},
        )
        assert cfg.get_model_data("node_b").model_id == "m2"

    def test_reasoning_effort_for_override_wins(self):
        cfg = Config(
            provider_data={
                "p1": ProviderData(
                    api_endpoint="https://ep.com",
                    api_key="key1",
                    model_id="m1",
                    reasoning_default="medium",
                ),
            },
            model_providers={"node_a": "p1"},
            reasoning_defaults={"node_a": "low"},
        )
        assert cfg.reasoning_effort_for("node_a") == "low"

    def test_reasoning_effort_for_falls_back_to_provider_default(self):
        cfg = Config(
            provider_data={
                "p1": ProviderData(
                    api_endpoint="https://ep.com",
                    api_key="key1",
                    model_id="m1",
                    reasoning_default="medium",
                ),
            },
            model_providers={"node_a": "p1"},
        )
        assert cfg.reasoning_effort_for("node_a") == "medium"

    def test_reasoning_effort_for_none_for_non_reasoning(self):
        cfg = Config(
            provider_data={
                "p1": ProviderData(
                    api_endpoint="https://ep.com",
                    api_key="key1",
                    model_id="m1",
                ),
            },
            model_providers={"node_a": "p1"},
        )
        assert cfg.reasoning_effort_for("node_a") is None

    def test_reasoning_effort_for_empty_map_falls_back(self):
        cfg = Config(
            provider_data={
                "p1": ProviderData(
                    api_endpoint="https://ep.com",
                    api_key="key1",
                    model_id="m1",
                    reasoning_default="high",
                ),
            },
            model_providers={"node_a": "p1"},
        )
        assert cfg.reasoning_effort_for("node_a") == "high"


# ---------------------------------------------------------------------------
# NodeCustomization / GraphCustomization / CustomizationsConfig
# ---------------------------------------------------------------------------

class TestNodeCustomization:
    def test_all_fields_default_to_none(self):
        nc = NodeCustomization()
        assert nc.tone is None
        assert nc.constraints is None
        assert nc.extra_instructions is None
        assert nc.prompt_overrides is None

    def test_partial_population(self):
        nc = NodeCustomization(tone="friendly")
        assert nc.tone == "friendly"
        assert nc.constraints is None


class TestGraphCustomization:
    def test_all_fields_default_to_none(self):
        gc = GraphCustomization()
        assert gc.pedagogical_style is None
        assert gc.verbosity is None
        assert gc.tone is None


class TestCustomizationsConfig:
    def test_empty_graph_uses_defaults(self):
        cc = CustomizationsConfig()
        assert isinstance(cc.graph, GraphCustomization)
        assert cc.graph.pedagogical_style is None
        assert cc.nodes == {}

    def test_empty_nodes_dict(self):
        cc = CustomizationsConfig(graph=GraphCustomization(pedagogical_style="direct"))
        assert cc.graph.pedagogical_style == "direct"
        assert cc.nodes == {}
