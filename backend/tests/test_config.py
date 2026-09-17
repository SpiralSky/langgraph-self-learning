from pathlib import Path

import pytest
import yaml

from config import (
    DEFAULT_MODEL_SETTINGS_PATH,
    ModelSettingsFile,
    get_model_settings,
    get_model_settings_cached,
)


def write_settings(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "model_settings.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _base_settings() -> dict:
    return {
        "providers": {
            "openai": {
                "api_type": "openai",
                "api_key": "${TEST_API_KEY}",
                "default_model": "gpt-4o-mini",
            }
        },
        "models": {},
    }


def test_env_reference_resolved(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "secret-123")
    path = write_settings(tmp_path, _base_settings())

    settings = get_model_settings(path)

    assert settings.providers["openai"].api_key == "secret-123"
    assert settings.providers["openai"].api_type == "openai"
    assert settings.providers["openai"].default_model == "gpt-4o-mini"


def test_undefined_env_var_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("MISSING_TEST_KEY", raising=False)
    data = _base_settings()
    data["providers"]["openai"]["api_key"] = "${MISSING_TEST_KEY}"
    path = write_settings(tmp_path, data)

    with pytest.raises(ValueError, match="MISSING_TEST_KEY"):
        get_model_settings(path)


def test_unknown_provider_reference_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "secret-123")
    data = _base_settings()
    data["models"] = {"fast": {"provider": "no_such_provider"}}
    path = write_settings(tmp_path, data)

    with pytest.raises(ValueError, match="no_such_provider"):
        get_model_settings(path)


def test_model_overrides_provider_default(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "secret-123")
    data = _base_settings()
    data["models"] = {
        "fast": {
            "provider": "openai",
            "model": "gpt-4o-turbo",
            "thinking_level": "high",
        }
    }
    path = write_settings(tmp_path, data)

    settings = get_model_settings(path)

    assert settings.models["fast"].provider == "openai"
    assert settings.models["fast"].model == "gpt-4o-turbo"
    assert settings.models["fast"].thinking_level == "high"
    assert settings.providers["openai"].default_model == "gpt-4o-mini"


def test_model_without_override_falls_back_to_provider_default(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "secret-123")
    data = _base_settings()
    data["models"] = {"fast": {"provider": "openai"}}
    path = write_settings(tmp_path, data)

    settings = get_model_settings(path)

    assert settings.models["fast"].model is None
    assert settings.models["fast"].thinking_level is None


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="model_settings.yaml"):
        get_model_settings(tmp_path / "model_settings.yaml")


def test_bad_root_type_raises(tmp_path):
    path = tmp_path / "model_settings.yaml"
    path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(TypeError):
        get_model_settings(path)


def test_uncached_reloads_on_every_call(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "secret-123")
    path = write_settings(tmp_path, _base_settings())

    first = get_model_settings(path)
    second = get_model_settings(path)

    assert isinstance(first, ModelSettingsFile)
    assert first is not second


def test_cached_returns_same_object(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "secret-123")
    path = write_settings(tmp_path, _base_settings())
    get_model_settings_cached.cache_clear()

    first = get_model_settings_cached(path)
    second = get_model_settings_cached(path)

    assert first is second
    get_model_settings_cached.cache_clear()


def test_cached_is_keyed_per_path(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "secret-123")
    path_a = write_settings(tmp_path / "a", _base_settings())
    data = _base_settings()
    data["providers"]["openai"]["default_model"] = "other-model"
    path_b = write_settings(tmp_path / "b", data)
    get_model_settings_cached.cache_clear()

    cached_a = get_model_settings_cached(path_a)
    cached_b = get_model_settings_cached(path_b)

    assert cached_a is not cached_b
    assert cached_a.providers["openai"].default_model == "gpt-4o-mini"
    assert cached_b.providers["openai"].default_model == "other-model"
    assert get_model_settings_cached(path_a) is cached_a
    get_model_settings_cached.cache_clear()


def test_default_path_points_to_backend_model_settings(tmp_path):
    assert DEFAULT_MODEL_SETTINGS_PATH.name == "model_settings.yaml"
    assert DEFAULT_MODEL_SETTINGS_PATH.exists()


def test_no_arg_loads_real_default_file():
    settings = get_model_settings()

    assert isinstance(settings, ModelSettingsFile)
    assert "openai" in settings.providers
