"""Pydantic models for the Learning Graph.

All data structures used across graph nodes are defined here:
session control, knowledge components, input analysis, teaching
strategy, response output, session records, and code execution.
"""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, RootModel


class JsonValue(RootModel[Union[str, int, float, bool, None, List["JsonValue"], Dict[str, "JsonValue"]]]):
    """
    A recursively-structured JSON value.

    Models the shape of any deserialized JSON document: a scalar (``str``,
    ``int``, ``float``, ``bool``), ``None``, a JSON ``list``, or a JSON
    ``dict`` mapping ``str`` to another ``JsonValue``. Declared as a model
    (rather than a module-level type alias) so Pydantic resolves the
    self-reference by name instead of inlining it, avoiding
    ``maximum recursion depth exceeded`` on schema build.
    """


class SessionControl(BaseModel):
    """
    Machine-readable session rules the graph keys routing and gating off.

    Unlike ``phase`` (a free, human-friendly label the model may author inside
    ``state`` for display), this object is the authoritative signal: when
    ``awaiting_user_input`` is set, the turn is a learner reply to a pending item
    (``pending_item_id``) and must be routed to the assessor rather than treated
    as a fresh discovery turn. ``allow_reveal_answer`` gates whether the tutor may
    show the answer this turn, and ``one_move_per_turn`` keeps a session moving at
    most one rung per exchange.

    :param awaiting_user_input: True when the session is waiting on a specific
    leaner reply (an answer, a setup confirmation, or a slot the learner must
    fill); False once that input is consumed or the session is idle.
    :vartype awaiting_user_input: bool
    :param pending_item_id: Identifier of the pending item the awaited reply
    resolves (e.g. ``"task_3"`` or ``"setup_confirmation"``), or ``None``.
    :vartype pending_item_id: str | None
    :param expected_input_kind: Kind of input awaited, e.g. ``"answer"``,
    ``"setup_confirmation"``, ``"setup_revision"``, or ``"slot"``.
    :vartype expected_input_kind: str | None
    :param allow_reveal_answer: True to permit showing the correct answer this
    turn; False (default) to keep the tutor from revealing it early.
    :vartype allow_reveal_answer: bool
    :param one_move_per_turn: True when the session advances at most one rung per
    learner turn, regardless of how much the answer demonstrates.
    :vartype one_move_per_turn: bool
    """

    awaiting_user_input: bool = Field(
    default=False,
    description="True when the session is waiting on a specific learner reply; the turn should be graded as an answer, not a fresh question.",
    )
    pending_item_id: Optional[str] = Field(
    default=None,
    description="Identifier of the pending item the awaited reply belongs to, e.g. 'task_3' or 'setup_confirmation'.",
    )
    expected_input_kind: Optional[str] = Field(
    default=None,
    description="What the session is awaiting: 'answer', 'setup_confirmation', 'setup_revision', 'slot', etc.",
    )
    allow_reveal_answer: bool = Field(
    default=False,
    description="True to let the tutor reveal the answer this turn; False to keep the pending answer hidden until the learner supplies it.",
    )
    one_move_per_turn: bool = Field(
    default=False,
    description="True when the session advances at most one rung per learner turn.",
    )


class KnowledgeComponent(BaseModel):
    """
    A single concept in the learner's persistent knowledge model.

    Unlike ``PrerequisiteGap`` (per-turn diagnosis), this persists across turns
    and sessions so the tutor can gate depth inside the learner's zone of
    proximal development. Status lives on a ``missing -> rusty -> mastered``
    rung ladder; ``inferred`` marks a deduction without an explicit admission.

    :param concept: The concept, named at the rung level.
    :vartype concept: str
    :param status: ``missing``, ``rusty``, ``inferred``, or ``mastered``.
    :vartype status: str
    :param evidence: Supporting quote or signal.
    :vartype evidence: str
    :param turn: Turn number this component was last observed on.
    :vartype turn: int
    :param series: Optional label of the learning session/series this belongs to.
    :vartype series: str | None
    """

    concept: str = Field(description="The concept, named at the rung level (e.g. 'conditional probability notation q(z|x)').")
    status: str = Field(description="Rung ladder: 'missing', 'rusty', 'inferred', or 'mastered'.")
    evidence: str = Field(default="", description="The supporting quote or signal.")
    turn: int = Field(default=0, description="Turn count when this component was last observed.")
    series: Optional[str] = Field(default=None, description="Optional label of the learning session/series this belongs to.")


class PrerequisiteGap(BaseModel):
    """Per-turn diagnosis of a missing prerequisite knowledge item.

    Unlike :class:`KnowledgeComponent`, this does not persist across turns.
    Populated by :func:`input_analyzer` when the learner demonstrates a gap.

    :param concept: The concrete prerequisite at the rung level.
    :vartype concept: str
    :param status: ``missing``, ``rusty``, or ``inferred``.
    :vartype status: str
    :param evidence: Supporting quote or signal.
    :vartype evidence: str
    """

    concept: str = Field(
        description="The concrete prerequisite the learner lacks, named at the rung level (e.g. 'conditional probability notation q(z|x)', 'what the posterior means', 'the vertical bar means conditioning', 'expectation', 'KL divergence')."
    )
    status: str = Field(
        description="'missing' when the learner explicitly admits not knowing it, 'rusty' when they say they forgot or are uncertain, 'inferred' when deduced from conversation history/memory without an explicit admission."
    )
    evidence: str = Field(
        description="The supporting quote or signal ('learner said \"...\"', 'inferred from earlier turns', 'inferred from memory')."
    )


class InputAnalysisResult(BaseModel):
    """Output of the :func:`input_analyzer` node.

    :param is_clear: True if the question is specific enough to answer directly.
    :vartype is_clear: bool
    :param intent: ``'factual'``, ``'conceptual'``, ``'problem_solving'``, ``'debugging'``, or ``'unclear'``.
    :vartype intent: str
    :param key_points: Extracted core concepts or entities. Empty if unclear.
    :vartype key_points: list[str]
    :param prerequisite_gaps: Prerequisite knowledge the learner lacks or is unsure of. Empty when no credible gap signal exists.
    :vartype prerequisite_gaps: list[PrerequisiteGap]
    :param comments: Natural language feedback for the user.
    :vartype comments: str
    :param suggested_clarifications: 1-2 questions to resolve ambiguity if unclear.
    :vartype suggested_clarifications: list[str] | None
    :param session_hint: Short hint for interactive session format; null when no session is intended.
    :vartype session_hint: str | None
    :param suggest_session: True when the user wants guided/tracked learning without naming a format.
    :vartype suggest_session: bool
    """

    is_clear: bool = Field(
        description="True if the question is specific enough to answer without guessing intent."
    )
    intent: str = Field(
        description="The primary intent: 'factual', 'conceptual', 'problem_solving', 'debugging', or 'unclear'."
    )
    key_points: List[str] = Field(
        description="Extracted core concepts or entities from the user's input. Empty if unclear."
    )
    prerequisite_gaps: List[PrerequisiteGap] = Field(
        default_factory=list,
        description="Prerequisite knowledge the learner lacks or is unsure of, derived from explicit admissions or history. Empty when no credible gap signal exists — never invent gaps."
    )
    comments: str = Field(
        description="Natural language feedback for the user. If clear, note specifically and factually what was asked well, without unearned generic praise. If unclear, gently explain exactly what is missing."
    )
    suggested_clarifications: Optional[List[str]] = Field(
        default=None,
        description="If is_clear is False, provide 1-2 specific questions to ask the user to resolve ambiguity."
    )
    session_hint: Optional[str] = Field(
        default=None,
        description="A short free-form hint when the user clearly initiates an interactive session (e.g. 'guided_learning', 'quiz_game', or a model-invented id like 'word_game'). Guidance only, never binding. Null when no session is intended."
    )
    suggest_session: bool = Field(
        default=False,
        description="True when the user clearly wants guided/tracked learning without naming a format, so the tutor should hint at the available session types. False otherwise."
    )


class SessionContext(BaseModel):
    """
    A lightweight, mutable description of where a learning session stands.

    Unlike the heavier :class:`SessionRecord` (full state, ``state_meta``,
    ``instructions``, gating ``control``), this is the decision-first summary the
    graph threads between turns: the learner's goal, what we are currently
    focusing on, the lifecycle ``status``, a ``resume_anchor`` for when the turn
    detours, and a hint of the next expected move. It is deliberately lean and
    LLM-authored so the decision_maker can rewrite it in place as intent shifts.

    :param goal: The learner's overall objective for the session.
    :vartype goal: str | None
    :param current_focus: What the session is currently focused on.
    :vartype current_focus: str | None
    :param status: ``none``, ``active``, ``paused``, or ``completed``.
    :vartype status: Literal["none", "active", "paused", "completed"]
    :param control: Machine-readable session rules (awaiting_user_input /
    pending_item_id / expected_input_kind / one_move_per_turn); persisted
    so downstream nodes (e.g. assessor) can gate routing.
    :vartype control: Optional[SessionControl]
    :param state: Model-authored session values (e.g. mastery signal
    ``mastered_concepts``) persisted across turns.
    :vartype state: Dict[str, JsonValue]
    :param resume_anchor: Marker of where to resume after a detour/pause.
    :vartype resume_anchor: str | None
    :param next_move_hint: Proposal for the next pedagogical move, if decided.
    :vartype next_move_hint: str | None
    """

    goal: Optional[str] = Field(default=None, description="The learner's overall objective for the session.")
    current_focus: Optional[str] = Field(default=None, description="What the session is currently focused on.")
    status: Literal["none", "active", "paused", "completed"] = Field(
    default="none",
    description="Session lifecycle: 'none', 'active', 'paused', or 'completed'.",
    )
    control: Optional[SessionControl] = Field(
    default=None,
    description="Machine-readable session rules (awaiting_user_input / pending_item_id / expected_input_kind / one_move_per_turn); persisted so downstream nodes can gate routing.",
    )
    state: Dict[str, JsonValue] = Field(
    default_factory=dict,
    description="Model-authored session values (e.g. mastery signal mastered_concepts) persisted across turns.",
    )
    resume_anchor: Optional[str] = Field(default=None, description="Where to resume after a detour or pause.")
    next_move_hint: Optional[str] = Field(default=None, description="Hint for the next pedagogical move, if decided.")


class TeachingStrategy(BaseModel):
    """
    The decision of a pedagogical-director node for this turn.

    Encodes what move the tutor should make (explain, ask a guiding question,
    answer a side quest, pivot back to the goal, or propose a new session),
    which concept to focus on, the response's tone, a short executable plan,
    and the updated :class:`SessionContext` to persist. ``response_builder``
    consumes this as instruction rather than re-deciding pedagogically.

    :param pedagogical_move: ``explain_concept``, ``ask_guiding_question``,
    ``answer_side_quest``, ``pivot_to_goal``, or ``propose_session``.
    :vartype pedagogical_move: Literal[...]
    :param focus_concept: The concept to centre the turn on.
    :vartype focus_concept: str
    :param tone: Intended response tone (e.g. ``"encouraging"``, ``"socratic"``).
    :vartype tone: str
    :param execution_plan: 2-3 bullet steps for the tutor to execute.
    :vartype execution_plan: list[str]
    :param updated_session_context: The authoritatively-updated session context.
    :vartype updated_session_context: SessionContext
    """

    pedagogical_move: Literal[
    "explain_concept",
    "ask_guiding_question",
    "answer_side_quest",
    "pivot_to_goal",
    "propose_session",
    "execute_code",
    ] = Field(
    default="explain_concept",
    description="'explain_concept', 'ask_guiding_question', 'answer_side_quest', 'pivot_to_goal', or 'propose_session'.",
    )
    focus_concept: Optional[str] = Field(default=None, description="The concept to focus this turn on.")
    tone: str = Field(default="encouraging", description="Intended response tone (e.g. 'encouraging', 'socratic').")
    execution_plan: List[str] = Field(
    default_factory=list,
    description="2-3 bullet steps the tutor should execute this turn.",
    )
    updated_session_context: SessionContext = Field(
    default_factory=SessionContext,
    description="The authoritatively new session context after this decision.",
    )


class ResponseBuilderOutput(BaseModel):
    """Output of the :func:`response_builder` node.

    :param draft_response: Main educational response in Markdown.
    :vartype draft_response: str
    :param tone: Tone applied (e.g. ``'encouraging'``, ``'socratic'``).
    :vartype tone: str
    :param sources_used: Key facts/URLs from search results. Empty if based on internal knowledge.
    :vartype sources_used: list[str]
    """

    draft_response: str = Field(
        description="The main educational response. Integrate all pedagogical strategies here, including direct answers, socratic questions, scaffolding, or reflection prompts. Use Markdown for formatting."
    )
    tone: str = Field(
        description="The specific tone applied in the response (e.g., 'encouraging', 'formal', 'direct', 'socratic', 'harsh'). This helps the final formatter verify consistency."
    )
    sources_used: List[str] = Field(
        description="A list of key facts, URLs, or source titles from the search results that were used to construct the answer. Empty if based solely on internal knowledge/memory."
    )


class SessionRecord(BaseModel):
    """
    A model-authored learning session.

    The model freely authors and manages the session: ``description`` states
    what it is about, ``goal`` is an optional free-form objective, and ``state``
    holds any tracked values the session needs. ``state_meta`` annotates each
    key with a plain-language meaning so later turns (and the input analyzer)
    know what a value means without a hardcoded schema. ``session_id``,
    ``thread_id``, ``status``, and the timestamps are system-stamped on persist.

    :param session_id: Short unique id, system-stamped on first persist.
    :vartype session_id: str
    :param thread_id: Thread the session belongs to, system-stamped.
    :vartype thread_id: str
    :param description: What the session is about.
    :vartype description: str
    :param goal: Optional free-form objective for the session.
    :vartype goal: str | None
    :param state: Session state values, e.g. ``{"score": 12, "current_step": 3}``.
    :vartype state: dict[str, JsonValue]
    :param state_meta: Mapping of state key to plain-language meaning.
    :vartype state_meta: dict[str, str]
    :param instructions: Optional model-authored 'session program': ordered
    tasks, mastery criteria, and gating rules. Guidance for the session
    assessor, never a fixed schema.
    :vartype instructions: list[str]
    :param control: Machine-readable session rules (awaiting input, pending item,
    reveal-answer gate); ``None`` when the session has no active gating.
    :vartype control: SessionControl | None
    :param template_id: Optional originating template id, kept as provenance;
    custom/improvised sessions leave it unset.
    :vartype template_id: str | None
    :param status: Session lifecycle status.
    :vartype status: str
    :param created_at: Unix timestamp of creation, system-stamped.
    :vartype created_at: float
    :param updated_at: Unix timestamp of last update, system-stamped.
    :vartype updated_at: float
    """

    session_id: str = Field(default="", description="Short unique id, system-stamped on first persist.")
    thread_id: str = Field(default="", description="Thread the session belongs to, system-stamped.")
    description: str = Field(default="", description="What the session is about.")
    goal: Optional[str] = Field(default=None, description="Optional free-form objective for the session.")
    state: Dict[str, JsonValue] = Field(
    default_factory=dict,
    description="Session state, e.g. {'score': 12, 'current_step': 3}.",
    )
    state_meta: Dict[str, str] = Field(
    default_factory=dict,
    description="Maps each state key to a plain-language meaning, e.g. {'score': 'points earned so far'}.",
    )
    instructions: List[str] = Field(
    default_factory=list,
    description="Optional model-authored 'session program': ordered tasks, mastery criteria, and gating rules. Guidance for the session assessor, never a fixed schema.",
    )
    control: Optional[SessionControl] = Field(
    default=None,
    description="Machine-readable session rules (awaiting input, pending item, reveal-answer gate); None when the session has no active gating.",
    )
    template_id: Optional[str] = Field(
    default=None,
    description="Optional originating template id, kept as provenance; custom/improvised sessions leave it unset.",
    )
    status: str = Field(default="active", description="Session lifecycle status.")
    created_at: float = Field(default=0.0, description="Unix timestamp of creation, system-stamped.")
    updated_at: float = Field(default=0.0, description="Unix timestamp of last update, system-stamped.")


class ExecuteCodeInput(BaseModel):
    code: str = Field(description="Python code to execute via SSH executor.")
    timeout: Optional[int] = Field(
    default=60,
    description="Execution timeout in seconds. Default 60s per task-02 policy.",
    )


class PackageProposal(BaseModel):
    package_name: str = Field(description="Name of the package to install/propose.")
    version: Optional[str] = Field(
    default=None,
    description="Optional version constraint; null means latest.",
    )
    reason: str = Field(description="Why this package is needed for the learning session.")


class ExecuteCodeOutput(BaseModel):
    final_output: str = Field(description="Captured stdout+stderr, capped at 100 lines / 10000 bytes per task-02 policy.")
    success: bool = Field(description="Whether execution completed without error or timed out.")
    execution_time: Optional[float] = Field(default=None, description="Wall-clock time in seconds, if available.")
    files: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of files created/modified in the workspace after execution. Each entry has path, type, host_path, size.",
    )


class QueryResult(BaseModel):
    """Web search result from the :func:`input_analyzer` node.

    :param confidence: Search confidence score.
    :vartype confidence: float
    :param information: Search result text.
    :vartype information: str
    :param queries: Queries that produced this result.
    :vartype queries: list[str]
    :param reasoning: Reasoning for the query selection.
    :vartype reasoning: str
    """

    confidence: float = Field(description="Search confidence score.")
    information: str = Field(description="Search result text.")
    queries: List[str] = Field(description="Queries that produced this result.")
    reasoning: str = Field(description="Reasoning for the query selection.")