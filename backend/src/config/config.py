"""Model and provider settings, loaded from ``model_settings.yaml``.

This module owns the `providers` / `models` YAML file the backend reads at
startup. Values of the form ``${ENV_VAR}`` are resolved from the environment
(``.env`` is loaded first) so secrets never need to live in the YAML file.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_SETTINGS_PATH = BACKEND_ROOT / "model_settings.yaml"

_ENVVAR_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ProviderSettings(BaseModel):
    """A single named provider entry under the ``providers`` key."""

    api_type: str
    api_key: str | None = None
    base_url: str | None = None
    default_model: str


class ModelSettings(BaseModel):
    """A single named model entry under the ``models`` key.

    ``model`` and ``thinking_level`` optionally override what the referenced
    provider would otherwise contribute (its ``default_model``, and any
    provider-specific reasoning setting).
    """

    provider: str
    model: str | None = None
    thinking_level: str | None = None


class ModelSettingsFile(BaseModel):
    """Top-level schema of ``model_settings.yaml``."""

    providers: dict[str, ProviderSettings]
    models: dict[str, ModelSettings] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_provider_references(self) -> ModelSettingsFile:
        for name, model in self.models.items():
            if model.provider not in self.providers:
                raise ValueError(
                    f"model {name!r} references unknown provider {model.provider!r}"
                )
        return self


def _resolve_env_ref(match: re.Match[str]) -> str:
    name = match.group(1)
    if name not in os.environ:
        raise ValueError(
            f"environment variable {name!r} referenced in model settings is not set"
        )
    return os.environ[name]


def _interpolate_env(value):
    """Recursively replace ``${ENV_VAR}`` references inside string values."""
    if isinstance(value, dict):
        return {key: _interpolate_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_interpolate_env(item) for item in value]
    if isinstance(value, str):
        return _ENVVAR_REF.sub(_resolve_env_ref, value)
    return value


def get_model_settings(
    config_path: Path = DEFAULT_MODEL_SETTINGS_PATH,
) -> ModelSettingsFile:
    """Load and validate ``model_settings.yaml``.

    Loads ``.env`` from the backend root first (existing environment variables
    take precedence), then reads the YAML file, resolves ``${ENV_VAR}``
    references, and validates the result against ``ModelSettingsFile``.

    Uncached: every call re-reads the file. Use ``get_model_settings_cached``
    when repeated callers should share one parsed result.
    """
    load_dotenv(BACKEND_ROOT / ".env")

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"model settings file not found: {path} (copy model_settings.template.yaml)"
        )

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"{path} must contain a top-level mapping")

    interpolated = _interpolate_env(raw)
    return ModelSettingsFile.model_validate(interpolated)


@lru_cache
def get_model_settings_cached(
    config_path: Path = DEFAULT_MODEL_SETTINGS_PATH,
) -> ModelSettingsFile:
    """Cached wrapper around :func:`get_model_settings`.

    The cache is keyed per ``config_path``, so callers using different paths
    never share (or clobber) each other's results.
    """
    return get_model_settings(config_path)
