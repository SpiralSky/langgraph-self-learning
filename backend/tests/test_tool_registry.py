"""Unit tests for the tool whitelist registry.

Pure-unit: no tool is ever constructed or invoked over the network — the
default registry's builtin tools are plain functions and are never called
here.
"""

import pytest

from graphs.tools import (
    ToolError,
    ToolRegistry,
    default_registry,
    int_arg,
    require_arg,
)


def _echo(args):
    return f"echo:{args['text']}"


def test_register_and_get_round_trip():
    registry = ToolRegistry()
    registry.register("echo", _echo)
    assert registry.get("echo") is _echo
    assert "echo" in registry


def test_get_unknown_tool_raises_key_error_naming_it():
    registry = ToolRegistry()
    with pytest.raises(KeyError, match="unknown tool: 'nope'"):
        registry.get("nope")


def test_names_sorted_and_deterministic():
    registry = ToolRegistry()
    registry.register("beta", _echo)
    registry.register("alpha", _echo)
    assert registry.names() == ("alpha", "beta")


def test_register_validates_name():
    registry = ToolRegistry()
    for bad in ("", None, 0):
        with pytest.raises(ValueError, match="non-empty string"):
            registry.register(bad, _echo)


def test_register_rejects_non_callable():
    registry = ToolRegistry()
    with pytest.raises(TypeError, match="must be callable"):
        registry.register("echo", "not a function")


def test_duplicate_registration_rejected():
    registry = ToolRegistry()
    registry.register("echo", _echo)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("echo", _echo)


def test_register_as_decorator():
    registry = ToolRegistry()

    @registry.register("shout")
    def shout(args):
        return str(args["text"]).upper()

    assert registry.get("shout")({"text": "hi"}) == "HI"


def test_default_registry_has_builtin_whitelist():
    names = default_registry().names()
    assert "ddgs" in names
    assert "mem0_remember" in names
    assert "mem0_retrieve" in names


def test_default_registry_is_a_shared_singleton():
    assert default_registry() is default_registry()


def test_require_arg_returns_value_and_validates():
    assert require_arg({"text": "hi"}, "text") == "hi"
    with pytest.raises(ValueError, match="missing required tool argument: 'text'"):
        require_arg({}, "text")
    with pytest.raises(TypeError, match="tool args must be a dict"):
        require_arg(["not", "a", "dict"], "text")


def test_int_arg_coerces_and_defaults():
    assert int_arg({}, "k", 5) == 5
    assert int_arg({"k": None}, "k", 5) == 5
    assert int_arg({"k": "7"}, "k", 5) == 7
    with pytest.raises(ValueError, match="must be an integer"):
        int_arg({"k": "lots"}, "k", 5)


def test_tool_error_is_a_runtime_error():
    assert issubclass(ToolError, RuntimeError)