from typing import Any, Dict, List, Optional

import pytest
from pydantic import ValidationError

from graphs.learning_graph.pydantic_models import (
    InputAnalysisResult,
    JsonValue,
    KnowledgeComponent,
    PrerequisiteGap,
    QueryResult,
    ResponseBuilderOutput,
    SessionContext,
    SessionControl,
    SessionRecord,
    TeachingStrategy,
)


# ---------------------------------------------------------------------------
# JsonValue – recursive JSON root model
# ---------------------------------------------------------------------------

class TestJsonValue:
    def test_str(self):
        assert JsonValue("hello").root == "hello"

    def test_int(self):
        assert JsonValue(42).root == 42

    def test_float(self):
        assert JsonValue(3.14).root == 3.14

    def test_bool(self):
        assert JsonValue(True).root is True

    def test_none(self):
        assert JsonValue(None).root is None

    def test_list(self):
        v = JsonValue([1, "two", None])
        assert v.model_dump() == [1, "two", None]

    def test_dict(self):
        v = JsonValue({"a": 1, "b": "two"})
        assert v.model_dump() == {"a": 1, "b": "two"}

    def test_nested_recursion(self):
        v = JsonValue({"outer": [{"inner": "deep"}, None, [1, 2, [3]]]})
        dumped = v.model_dump()
        assert dumped["outer"][0]["inner"] == "deep"  # type: ignore[index]
        assert dumped["outer"][2][2] == [3]  # type: ignore[index]

    def test_round_trip(self):
        payload: Dict[str, Any] = {"name": "test", "count": 5, "tags": ["a", "b"]}
        v = JsonValue(payload)
        assert v.model_dump() == payload

    def test_invalid_type_raises(self):
        with pytest.raises(ValidationError):
            JsonValue(object())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# SessionControl – defaults and round-trip
# ---------------------------------------------------------------------------

class TestSessionControl:
    def test_defaults(self):
        c = SessionControl()
        assert c.awaiting_user_input is False
        assert c.pending_item_id is None
        assert c.expected_input_kind is None
        assert c.allow_reveal_answer is False
        assert c.one_move_per_turn is False

    def test_all_fields_populated(self):
        c = SessionControl(
            awaiting_user_input=True,
            pending_item_id="task_3",
            expected_input_kind="answer",
            allow_reveal_answer=True,
            one_move_per_turn=True,
        )
        assert c.awaiting_user_input is True
        assert c.pending_item_id == "task_3"
        assert c.expected_input_kind == "answer"
        assert c.allow_reveal_answer is True
        assert c.one_move_per_turn is True

    def test_json_round_trip(self):
        original = SessionControl(
            awaiting_user_input=True,
            pending_item_id="q_42",
            expected_input_kind="answer",
        )
        data = original.model_dump()
        restored = SessionControl.model_validate(data)
        assert restored == original


# ---------------------------------------------------------------------------
# KnowledgeComponent – valid and edge cases
# ---------------------------------------------------------------------------

class TestKnowledgeComponent:
    def test_minimal_fields(self):
        kc = KnowledgeComponent(concept="Bayes theorem", status="missing")
        assert kc.concept == "Bayes theorem"
        assert kc.status == "missing"
        assert kc.evidence == ""
        assert kc.turn == 0
        assert kc.series is None

    def test_all_fields_populated(self):
        kc = KnowledgeComponent(
            concept="conditional probability",
            status="mastered",
            evidence="learner solved it correctly",
            turn=3,
            series="probability_101",
        )
        assert kc.evidence == "learner solved it correctly"
        assert kc.turn == 3
        assert kc.series == "probability_101"

    def test_evidence_defaults_to_empty_string(self):
        kc = KnowledgeComponent(concept="KL divergence", status="rusty")
        assert kc.evidence == ""

    def test_status_accepts_valid_values(self):
        for status in ("missing", "rusty", "inferred", "mastered"):
            kc = KnowledgeComponent(concept="x", status=status)
            assert kc.status == status

    def test_empty_concept_still_accepted(self):
        kc = KnowledgeComponent(concept="", status="missing")
        assert kc.concept == ""


# ---------------------------------------------------------------------------
# PrerequisiteGap – valid entries
# ---------------------------------------------------------------------------

class TestPrerequisiteGap:
    def test_full_construction(self):
        gap = PrerequisiteGap(
            concept="Bayes theorem",
            status="rusty",
            evidence="learner said 'I forget Bayes'",
        )
        assert gap.concept == "Bayes theorem"
        assert gap.status == "rusty"
        assert gap.evidence == "learner said 'I forget Bayes'"

    def test_empty_evidence(self):
        gap = PrerequisiteGap(concept="integration", status="missing", evidence="")
        assert gap.evidence == ""

    def test_missing_fields_raises(self):
        with pytest.raises(ValidationError):
            PrerequisiteGap()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# InputAnalysisResult – all fields
# ---------------------------------------------------------------------------

class TestInputAnalysisResult:
    def test_clear_true_with_all_fields(self):
        r = InputAnalysisResult(
            is_clear=True,
            intent="conceptual",
            key_points=["conditional probability"],
            prerequisite_gaps=[
                PrerequisiteGap(concept="Bayes", status="rusty", evidence="...")
            ],
            comments="Clear question.",
            session_hint="guided_learning",
            suggest_session=False,
        )
        assert r.is_clear is True
        assert r.intent == "conceptual"
        assert r.key_points == ["conditional probability"]
        assert len(r.prerequisite_gaps) == 1
        assert r.comments == "Clear question."
        assert r.session_hint == "guided_learning"
        assert r.suggest_session is False
        assert r.suggested_clarifications is None

    def test_clear_false_with_suggested_clarifications(self):
        r = InputAnalysisResult(
            is_clear=False,
            intent="unclear",
            key_points=[],
            comments="Not enough context.",
            suggested_clarifications=["What topic are you studying?"],
        )
        assert r.is_clear is False
        assert r.suggested_clarifications == ["What topic are you studying?"]

    def test_session_hint_null(self):
        r = InputAnalysisResult(
            is_clear=True,
            intent="factual",
            key_points=["Python"],
            comments="Fine.",
        )
        assert r.session_hint is None

    def test_session_hint_populated(self):
        r = InputAnalysisResult(
            is_clear=True,
            intent="conceptual",
            key_points=["Python"],
            comments="Session start.",
            session_hint="quiz_game",
        )
        assert r.session_hint == "quiz_game"

    def test_suggest_session_when_hint_is_none(self):
        r = InputAnalysisResult(
            is_clear=True,
            intent="conceptual",
            key_points=["probability"],
            comments="User wants guided learning.",
            suggest_session=True,
        )
        assert r.suggest_session is True
        assert r.session_hint is None

    def test_prerequisite_gaps_defaults_to_empty(self):
        r = InputAnalysisResult(
            is_clear=True,
            intent="factual",
            key_points=[],
            comments="Ok.",
        )
        assert r.prerequisite_gaps == []


# ---------------------------------------------------------------------------
# SessionContext – lifecycle states
# ---------------------------------------------------------------------------

class TestSessionContext:
    def test_default_status_none(self):
        sc = SessionContext()
        assert sc.status == "none"
        assert sc.goal is None
        assert sc.current_focus is None
        assert sc.resume_anchor is None
        assert sc.next_move_hint is None

    def test_active_session(self):
        sc = SessionContext(
            goal="Learn Bayes theorem",
            current_focus="conditional probability",
            status="active",
        )
        assert sc.goal == "Learn Bayes theorem"
        assert sc.current_focus == "conditional probability"
        assert sc.status == "active"

    def test_paused_with_resume_anchor(self):
        sc = SessionContext(
            status="paused",
            resume_anchor="explained_posterior",
            next_move_hint="ask_guiding_question",
        )
        assert sc.status == "paused"
        assert sc.resume_anchor == "explained_posterior"
        assert sc.next_move_hint == "ask_guiding_question"

    def test_completed(self):
        sc = SessionContext(status="completed")
        assert sc.status == "completed"

    def test_invalid_status_raises(self):
        with pytest.raises(ValidationError):
            SessionContext(status="unknown")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# TeachingStrategy – all pedagogical moves
# ---------------------------------------------------------------------------

class TestTeachingStrategy:
    def test_default_execution_plan_empty_list(self):
        ts = TeachingStrategy()
        assert ts.execution_plan == []

    def test_default_updated_session_context(self):
        ts = TeachingStrategy()
        assert isinstance(ts.updated_session_context, SessionContext)
        assert ts.updated_session_context.status == "none"

    def test_default_pedagogical_move(self):
        ts = TeachingStrategy()
        assert ts.pedagogical_move == "explain_concept"

    @pytest.mark.parametrize("move", [
        "explain_concept",
        "ask_guiding_question",
        "answer_side_quest",
        "pivot_to_goal",
        "propose_session",
    ])
    def test_all_pedagogical_moves(self, move):
        ts = TeachingStrategy(
            pedagogical_move=move,  # type: ignore[arg-type]
            focus_concept="Bayes",
            tone="socratic",
            execution_plan=["Step 1", "Step 2"],
            updated_session_context=SessionContext(status="active"),
        )
        assert ts.pedagogical_move == move
        assert ts.focus_concept == "Bayes"
        assert ts.tone == "socratic"
        assert ts.execution_plan == ["Step 1", "Step 2"]
        assert ts.updated_session_context.status == "active"

    def test_invalid_pedagogical_move_raises(self):
        with pytest.raises(ValidationError):
            TeachingStrategy(pedagogical_move="invalid_move")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ResponseBuilderOutput – standard fields
# ---------------------------------------------------------------------------

class TestResponseBuilderOutput:
    def test_full_construction(self):
        r = ResponseBuilderOutput(
            draft_response="# Conditional Probability\n\nP(A|B) = P(A∩B) / P(B)",
            tone="socratic",
            sources_used=["Textbook: Probability Theory"],
        )
        assert "Conditional Probability" in r.draft_response
        assert r.tone == "socratic"
        assert r.sources_used == ["Textbook: Probability Theory"]

    def test_empty_sources_used(self):
        r = ResponseBuilderOutput(
            draft_response="Based on my knowledge...",
            tone="encouraging",
            sources_used=[],
        )
        assert r.sources_used == []

    def test_missing_fields_raises(self):
        with pytest.raises(ValidationError):
            ResponseBuilderOutput(draft_response="hi")  # missing tone, sources_used


# ---------------------------------------------------------------------------
# SessionRecord – comprehensive
# ---------------------------------------------------------------------------

class TestSessionRecord:
    def test_minimal_defaults(self):
        sr = SessionRecord()
        assert sr.session_id == ""
        assert sr.thread_id == ""
        assert sr.description == ""
        assert sr.goal is None
        assert sr.state == {}
        assert sr.state_meta == {}
        assert sr.instructions == []
        assert sr.control is None
        assert sr.template_id is None
        assert sr.status == "active"
        assert sr.created_at == 0.0
        assert sr.updated_at == 0.0

    def test_full_record(self):
        sr = SessionRecord(
            session_id="sess_1",
            thread_id="thread_abc",
            description="Probability quiz",
            goal="Master Bayes theorem",
            state={"score": JsonValue(10), "level": JsonValue("intermediate")},
            state_meta={"score": "points earned", "level": "difficulty"},
            instructions=["Ask 3 questions", "Check mastery"],
            control=SessionControl(awaiting_user_input=True),
            template_id="quiz_template_v1",
            status="active",
            created_at=1000.0,
            updated_at=2000.0,
        )
        assert sr.session_id == "sess_1"
        assert sr.state["score"].root == 10
        assert sr.control is not None
        assert sr.control.awaiting_user_input is True
        assert sr.template_id == "quiz_template_v1"
        assert sr.created_at == 1000.0

    def test_state_with_nested_json_value(self):
        sr = SessionRecord(
            state={"nested": JsonValue({"a": [1, {"b": None}]})},
        )
        dumped = sr.model_dump()
        assert dumped["state"]["nested"] == {"a": [1, {"b": None}]}

    def test_control_as_session_control(self):
        ctrl = SessionControl(awaiting_user_input=True, pending_item_id="task_1")
        sr = SessionRecord(control=ctrl)
        assert sr.control is not None
        assert sr.control.awaiting_user_input is True
        assert sr.control.pending_item_id == "task_1"

    def test_json_round_trip(self):
        original = SessionRecord(
            session_id="sess_2",
            thread_id="thread_xyz",
            description="Linear algebra review",
            status="active",
        )
        data = original.model_dump()
        restored = SessionRecord.model_validate(data)
        assert restored == original


# ---------------------------------------------------------------------------
# QueryResult – simple construction
# ---------------------------------------------------------------------------

class TestQueryResult:
    def test_all_fields_populated(self):
        qr = QueryResult(
            confidence=0.95,
            information="Conditional probability definition...",
            queries=["P(A|B) definition"],
            reasoning="Direct match from textbook.",
        )
        assert qr.confidence == 0.95
        assert qr.information.startswith("Conditional")
        assert qr.queries == ["P(A|B) definition"]
        assert qr.reasoning == "Direct match from textbook."

    def test_missing_fields_raises(self):
        with pytest.raises(ValidationError):
            QueryResult(confidence=0.5, information="")  # missing queries, reasoning