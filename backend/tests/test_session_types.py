from pathlib import Path

import pytest
from yaml import safe_load

from graphs.learning_graph.session_types import (
    SESSION_TYPES_PATH,
    SessionType,
    _load_session_types,
    analyzer_catalog_text,
    session_types,
)


class TestLoadSessionTypes:
    def test_successful_parse(self, tmp_path):
        yaml_file = tmp_path / "types.yaml"
        yaml_file.write_text("""
guided_learning:
  label: "Guided Step-by-Step"
  main: true
  summary: "Guided tutoring."
  triggers: "teach me, walk me through"

quiz_game:
  label: "Quiz Game"
  main: true
  summary: "A points-based quiz."
  state_keys: ["score", "question_index"]
""")
        result = _load_session_types(yaml_file)
        assert len(result) == 2
        gl = result["guided_learning"]
        assert gl.label == "Guided Step-by-Step"
        assert gl.is_main is True
        assert gl.summary == "Guided tutoring."
        assert gl.triggers == "teach me, walk me through"
        assert gl.state_keys == []
        qg = result["quiz_game"]
        assert qg.label == "Quiz Game"
        assert qg.state_keys == ["score", "question_index"]

    def test_file_not_found_returns_empty(self, tmp_path):
        missing = tmp_path / "nonexistent.yaml"
        result = _load_session_types(missing)
        assert result == {}

    def test_empty_yaml_returns_empty(self, tmp_path):
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")
        assert _load_session_types(yaml_file) == {}

    def test_null_yaml_returns_empty(self, tmp_path):
        yaml_file = tmp_path / "null.yaml"
        yaml_file.write_text("null")
        result = _load_session_types(yaml_file)
        assert result == {}

    def test_malformed_entry_raises_value_error(self, tmp_path):
        yaml_file = tmp_path / "bad.yaml"
        yaml_file.write_text("""
bad_type:
  summary: "missing label field"
""")
        with pytest.raises(ValueError, match="Failed to parse session type 'bad_type'"):
            _load_session_types(yaml_file)

    def test_non_dict_entry_skipped_silently(self, tmp_path):
        yaml_file = tmp_path / "bad2.yaml"
        yaml_file.write_text("""
bad_type: "just a string"
good_type:
  label: "Good"
  summary: "A good type"
""")
        result = _load_session_types(yaml_file)
        assert "bad_type" not in result
        assert "good_type" in result

    def test_skips_customizations_key(self, tmp_path):
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("""
customizations:
  graph:
    verbosity: concise

guided_learning:
  label: "Guided Step-by-Step"
  main: true
  summary: "Guided tutoring."
""")
        result = _load_session_types(yaml_file)
        assert "customizations" not in result
        assert "guided_learning" in result
        assert len(result) == 1


class TestSessionType:
    def test_minimal_fields(self):
        st = SessionType(label="Test", summary="A test type")
        assert st.label == "Test"
        assert st.summary == "A test type"
        assert st.triggers == ""
        assert st.state_keys == []
        assert st.system_block == ""
        assert st.is_main is True

    def test_all_fields_populated(self):
        st = SessionType(
            label="Full",
            summary="Full type",
            triggers="test, example",
            state_keys=["key1", "key2"],
            system_block="Some block",
            is_main=False,
        )
        assert st.label == "Full"
        assert st.triggers == "test, example"
        assert st.state_keys == ["key1", "key2"]
        assert st.system_block == "Some block"
        assert st.is_main is False

    def test_main_alias_to_is_main(self):
        st = SessionType(label="Alias", summary="Alias test", main=True)
        assert st.is_main is True
        st2 = SessionType(label="Alias2", summary="Alias test 2", main=False)
        assert st2.is_main is False

    def test_main_true_default(self):
        st = SessionType(label="Default", summary="Default main")
        assert st.is_main is True


class TestAnalyzerCatalogText:
    def test_renders_only_main_types(self):
        catalog = {
            "type_a": SessionType(label="A", summary="Type A", is_main=True, triggers="foo"),
            "type_b": SessionType(label="B", summary="Type B", is_main=False),
            "type_c": SessionType(label="C", summary="Type C", is_main=True),
        }
        with pytest.MonkeyPatch().context() as mp:
            import graphs.learning_graph.session_types as st_mod
            mp.setattr(st_mod, "session_types", catalog)
            text = analyzer_catalog_text()
        lines = text.split("\n")
        assert len(lines) == 2
        assert "- type_a: Type A; cues: foo" in lines
        assert "- type_c: Type C" in lines
        assert "type_b" not in text

    def test_triggers_appended_when_present(self):
        catalog = {
            "t1": SessionType(label="T1", summary="First", is_main=True, triggers="alpha, beta"),
        }
        with pytest.MonkeyPatch().context() as mp:
            import graphs.learning_graph.session_types as st_mod
            mp.setattr(st_mod, "session_types", catalog)
            text = analyzer_catalog_text()
        assert "- t1: First; cues: alpha, beta" in text

    def test_empty_catalog_returns_none(self):
        with pytest.MonkeyPatch().context() as mp:
            import graphs.learning_graph.session_types as st_mod
            mp.setattr(st_mod, "session_types", {})
            text = analyzer_catalog_text()
        assert text == "(none)"

    def test_main_types_without_triggers(self):
        catalog = {
            "plain": SessionType(label="Plain", summary="No triggers", is_main=True),
        }
        with pytest.MonkeyPatch().context() as mp:
            import graphs.learning_graph.session_types as st_mod
            mp.setattr(st_mod, "session_types", catalog)
            text = analyzer_catalog_text()
        assert "- plain: No triggers" == text.strip()


class TestSessionTypesModuleVariable:
    def test_is_dict_of_session_types(self):
        assert isinstance(session_types, dict)
        for key, val in session_types.items():
            assert isinstance(key, str)
            assert isinstance(val, SessionType)

    def test_does_not_contain_customizations(self):
        assert "customizations" not in session_types

    def test_contains_known_types(self):
        known = {"guided_learning", "quiz_game", "debate", "mastery_check", "flashcard_review"}
        assert known.issubset(session_types.keys())