"""Config loading, provider data, and customization models."""

import os
import re
from pathlib import Path
from typing import Any, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field
from yaml import safe_load


PROJECT_ROOT = Path(__file__).resolve().parents[3]


_ENV_VAR_PATTERN = re.compile(r"\$\{(\w+)\}")


def _resolve_env_vars(value: str) -> str:
    """Replace ${VAR_NAME} placeholders with values from the environment."""

    def _replace(match: re.Match) -> str:
        var_name = match.group(1)
        env_value = os.getenv(var_name)
        if env_value is None:
            raise ValueError(
                f"Environment variable '{var_name}' is not set "
                f"(referenced in config.yaml as ${{{var_name}}})"
            )
        return env_value

    return _ENV_VAR_PATTERN.sub(_replace, value)


def _resolve_dict(obj: object) -> object:
    """Recursively walk the YAML tree and resolve env-var placeholders in strings."""
    if isinstance(obj, str):
        return _resolve_env_vars(obj)
    if isinstance(obj, dict):
        return {k: _resolve_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_dict(item) for item in obj]
    return obj


class ProviderData(BaseModel):
    """
    Model provider entry: endpoint, key, model id, and its reasoning capability.

    Reasoning capability is declared on the model itself, not per node:
    ``reasoning_levels`` lists the accepted ``reasoning_effort`` values native
    to the model (e.g. ``["low", "medium", "high"]``) and ``reasoning_default``
    picks the effort applied for every request from this provider. Non-reasoning
    models omit both. The legacy single ``thinking_level`` is still accepted and
    treated as a fixed capability (its only level) when the new fields are
    absent, so existing configs keep working unchanged.
    """
    api_endpoint: str
    api_key: str
    model_id: str
    thinking_level: Optional[str] = None
    reasoning_levels: Optional[List[str]] = None
    reasoning_default: Optional[str] = None

    @property
    def reasoning_capabilities(self) -> tuple[str, ...]:
        """
        The reasoning effort levels this model accepts, native scale order.

        :return: ``reasoning_levels`` when declared, else the legacy
            ``thinking_level`` as a single-level capability. Empty when the
            model does not reason.
        :rtype: tuple[str, ...]
        """
        if self.reasoning_levels:
            return tuple(self.reasoning_levels)
        if self.thinking_level is not None:
            return (self.thinking_level,)
        return ()

    @property
    def default_reasoning(self) -> Optional[str]:
        """
        The reasoning effort applied to every request.

        :return: ``reasoning_default`` when declared, else the middle declared
            level, else ``None`` for non-reasoning models.
        :rtype: Optional[str]
        """
        if self.reasoning_default is not None:
            return self.reasoning_default
        levels = self.reasoning_capabilities
        if not levels:
            return None
        return levels[len(levels) // 2]

class NodeCustomization(BaseModel):
    """Per-node customization settings from YAML config."""

    model_config = ConfigDict(extra="allow")

    tone: Optional[str] = Field(default=None, description="Override the node's tone / style.")
    constraints: Optional[str] = Field(default=None, description="Additional hard rules for this node.")
    extra_instructions: Optional[str] = Field(default=None, description="Extra instructions appended to the node's prompt.")
    prompt_overrides: Optional[dict[str, Any]] = Field(default=None, description="Key-value overrides for the node's prompts.")
    formatting_rules: Optional[str] = Field(default=None, description="Markdown/formatting rules appended to the node's prompt.")


class GraphCustomization(BaseModel):
    """Graph-level customization settings broadcast to matching nodes.

    Free-form keys (e.g. ``tone_calibration``) are retained via
    ``extra="allow"``; the compiler broadcasts them by ``NodeIntent``.
    """

    model_config = ConfigDict(extra="allow")

    pedagogical_style: Optional[str] = Field(default=None, description="'socratic', 'direct', 'scaffolding', etc.")
    verbosity: Optional[str] = Field(default=None, description="'concise', 'detailed', 'balanced'.")
    tone: Optional[str] = Field(default=None, description="Default tone applied to all nodes without their own.")


class CustomizationsConfig(BaseModel):
    """Top-level customization config loaded from ``config.yaml[customizations]``."""

    graph: GraphCustomization = Field(default_factory=GraphCustomization, description="Graph-level settings broadcast by intent.")
    nodes: dict[str, NodeCustomization] = Field(default_factory=dict, description="Per-node settings by node name.")


class Config(BaseModel):
    """Top-level config loaded from config.yaml (+ graph_config.yaml customizations)."""
    model_providers: dict[str, str]
    workspace_host_path: str = "/workspace/shared/workspace"
    memory_top_k: int = 5
    memory_max_chars: int = 300
    session_min_overlap: int = 1
    customizations: CustomizationsConfig = Field(
        default_factory=CustomizationsConfig,
        description="Node and graph customization settings.",
    )
    reasoning_defaults: dict[str, str] = Field(default_factory=dict)
    model_override: Optional[str] = Field(
        default=None,
        description="Optional model override for all LLM calls. When set, uses this model ID instead of the configured default.",
    )
    provider_data: dict[str, ProviderData]

    def get_model_data(self, name: str) -> ProviderData:
        """Return the ProviderData for a named model usage.

        :param name: Config key naming the model usage (e.g. ``"response_builder"``).
        :return: ProviderData for the resolved model.
        :rtype: ProviderData
        """
        model_name = self.model_providers[name]
        return self.provider_data[model_name]

    def reasoning_effort_for(self, name: str) -> Optional[str]:
        """
        Reasoning effort for a node usage: per-node override when declared,
        else the provider's static ``default_reasoning``. ``None`` for
        non-reasoning models.
        """
        if name in self.reasoning_defaults:
            return self.reasoning_defaults[name]
        return self.get_model_data(name).default_reasoning

def read_config(path: Path) -> Config:
    """
    Read config.yaml, resolve ${VAR_NAME} env-var placeholders, merge
    graph_config.yaml customizations, and validate with the Config model.

    :param path: Path to config.yaml.
    :type path: Path
    :return: Validated Config instance.
    :rtype: Config
    """
    load_dotenv()

    with open(path) as stream:
        yaml = safe_load(stream)

    resolved = _resolve_dict(yaml)

    graph_cfg_path = PROJECT_ROOT / "src" / "config" / "graph_config.yaml"
    try:
        with open(graph_cfg_path) as f:
            graph_yaml = safe_load(f) or {}
        resolved["customizations"] = graph_yaml.get("customizations", {})
    except FileNotFoundError:
        resolved["customizations"] = {}

    try:
        parsed_config = Config(**resolved)
        return parsed_config
    except Exception as e:
        raise ValueError(f"Failed to parse config: {e}")


config: Config = read_config(PROJECT_ROOT / "config.yaml")

