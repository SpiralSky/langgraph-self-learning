"""Unit tests for the concrete ddgs + mem0 tool wrappers.

No tool client is ever constructed and no network is touched: network /
mem0 construction are patched with fakes so the argument handling and
result formatting are exercised, not the remote systems.
"""

import pytest

import graphs.tools.ddgs as ddgs_mod
import graphs.tools.mem0 as mem0_mod
from graphs.tools import ToolError


class FakeDDGS:
    """Stand-in for the real ``DDGS`` class: ``__enter__`` yields self with a
    canned ``text()`` and ``__exit__`` is a no-op."""

    def __init__(self, results, exc=None):
        self.results = results
        self.exc = exc

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def text(self, query, max_results=5):
        if self.exc is not None:
            raise self.exc
        self.last_query = query
        self.last_max_results = max_results
        return self.results[:max_results]


_RESPONSES = [
    {"title": "Cats", "href": "https://example.com/cats", "body": "All about cats."},
    {"title": "Dogs", "href": "https://example.com/dogs", "body": "All about dogs."},
]


@pytest.fixture
def fake_ddgs(monkeypatch):
    def _install(results=_RESPONSES, exc=None):
        fake = FakeDDGS(results, exc)
        monkeypatch.setattr(ddgs_mod, "DDGS", lambda *a, **k: fake)
        return fake

    return _install


def test_ddgs_search_formats_compact_digest(fake_ddgs):
    fake = fake_ddgs()
    out = ddgs_mod.ddgs_search({"query": "cats", "max_results": 1})
    assert "1. Cats" in out
    assert "https://example.com/cats" in out
    assert "All about cats." in out
    assert "Dogs" not in out
    assert fake.last_query == "cats"
    assert fake.last_max_results == 1


def test_ddgs_search_requires_query(fake_ddgs):
    with pytest.raises(ValueError, match="missing required tool argument"):
        ddgs_mod.ddgs_search({})
    with pytest.raises(ValueError, match="non-empty string"):
        ddgs_mod.ddgs_search({"query": "   "})


def test_ddgs_search_caps_max_results(fake_ddgs):
    fake = fake_ddgs()
    ddgs_mod.ddgs_search({"query": "q", "max_results": 999})
    assert fake.last_max_results == 10


def test_ddgs_search_rejects_non_integer_max_results(fake_ddgs):
    with pytest.raises(ValueError, match="must be an integer"):
        ddgs_mod.ddgs_search({"query": "q", "max_results": "many"})


def test_ddgs_search_empty_results(fake_ddgs):
    fake_ddgs(results=[])
    assert ddgs_mod.ddgs_search({"query": "q"}) == "No results."


def test_ddgs_search_surfaces_engine_error_as_tool_error(fake_ddgs):
    from ddgs.exceptions import DDGSException

    fake_ddgs(exc=DDGSException("timeout"))
    with pytest.raises(ToolError, match="ddgs search failed"):
        ddgs_mod.ddgs_search({"query": "q"})


class FakeMemory:
    """Fake mem0 ``Memory``: records adds, answers searches from a dict."""

    def __init__(self, hits=()):
        self.hits = hits
        self.add_calls = []
        self.search_calls = []

    def add(self, text, *, user_id=None, infer=True):
        self.add_calls.append((text, user_id, infer))

    def search(self, query, *, top_k=5, filters=None):
        self.search_calls.append((query, top_k, filters))
        return {"results": [{"memory": hit} for hit in self.hits[:top_k]]}


@pytest.fixture
def fake_memory(monkeypatch):
    def _install(hits=()):
        fake = FakeMemory(hits)
        monkeypatch.setattr(mem0_mod, "get_memory", lambda: fake)
        return fake

    return _install


def test_mem0_remember_stores_raw_text_no_inference(fake_memory):
    fake = fake_memory()
    out = mem0_mod.mem0_remember({"text": "prefers cats", "user_id": "u1"})
    assert out == "Remembered: prefers cats"
    assert fake.add_calls == [("prefers cats", "u1", False)]


def test_mem0_remember_defaults_user_id(fake_memory):
    fake = fake_memory()
    mem0_mod.mem0_remember({"text": "x"})
    assert fake.add_calls == [("x", "default", False)]


def test_mem0_remember_validates_text(fake_memory):
    with pytest.raises(ValueError, match="missing required tool argument"):
        mem0_mod.mem0_remember({})
    with pytest.raises(ValueError, match="non-empty string"):
        mem0_mod.mem0_remember({"text": ""})


def test_mem0_retrieve_formats_top_k(fake_memory):
    fake = fake_memory(hits=("cats are great", "dogs are great"))
    out = mem0_mod.mem0_retrieve({"query": "pets", "top_k": 1, "user_id": "u1"})
    assert out == "- cats are great"
    query, top_k, filters = fake.search_calls[0]
    assert query == "pets"
    assert top_k == 1
    assert filters == {"user_id": "u1"}


def test_mem0_retrieve_defaults_and_empty(fake_memory):
    fake = fake_memory()
    assert mem0_mod.mem0_retrieve({"query": "nothing here"}) == "No memories found."
    _query, top_k, filters = fake.search_calls[0]
    assert top_k == 5
    assert filters == {"user_id": "default"}


def test_mem0_retrieve_validates_query(fake_memory):
    with pytest.raises(ValueError, match="missing required tool argument"):
        mem0_mod.mem0_retrieve({})