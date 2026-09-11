from typing import Any
import json

from langchain_core.messages import SystemMessage

from graphs.learning_graph.compiler import get_node_prompt
from graphs.learning_graph.llm import get_chat_model
from graphs.learning_graph.nodes.node import node
from graphs.learning_graph.pydantic_models import TeachingStrategy
from graphs.learning_graph.state import LearningGraphState



PEDAGOGICAL_DIRECTOR_SYSTEM_PROMPT = """You are the Pedagogical Director. Your job is to evaluate the user's input against their learning goals and output a TeachingStrategy.

Input JSON keys: `user_message` (learner's message), `memory_results` (past interactions, known gaps/strengths), `analysis_results` (intent/clarity/key concepts, plus `prerequisite_gaps` and the session signals `session_hint` and `suggest_session`), `active_session` (the current SessionContext: goal, current_focus, status, resume_anchor, next_move_hint, control, state).

Rules:
- If the user asks a foundational side-question, set `pedagogical_move` to `'answer_side_quest'`. You MAY update `updated_session_context.status` to `'paused'` and set a `resume_anchor` if it is a major detour, OR keep `status` `'active'` if it is a quick clarification.
- If the user says 'continue' (or otherwise signals resuming), set `pedagogical_move` to `'pivot_to_goal'` or `'explain_concept'` using the existing `resume_anchor`.
- If `suggest_session` is true from the analyzer, set `pedagogical_move` to `'propose_session'` and draft the `updated_session_context` (goal and current_focus based on the request).
- Otherwise (a normal on-goal question), choose `'explain_concept'` or `'ask_guiding_question'` as the situation warrants.
- Persist the current session state into `updated_session_context` whenever a session exists; only invent fields when none exists yet.
- Return ONLY valid JSON matching the TeachingStrategy schema. No markdown fences, no commentary.
"""


@node(prompt=PEDAGOGICAL_DIRECTOR_SYSTEM_PROMPT, intent="decide")
def decision_maker(state: LearningGraphState) -> dict[str, Any]:
    """
    Decide the pedagogical move for this turn.

    Acts as the pedagogical director, evaluating the user's message against the
    learner's goals and memory, and emits a :class:`TeachingStrategy` under
    ``"teaching_strategy"``. Also threads the (possibly updated)
    :class:`SessionContext` forward via the strategy so downstream nodes can
    persist it without re-deciding pedagogy.

    Move selection logic:
    - ``answer_side_quest`` — foundational side-question; may pause the session
      for a major detour.
    - ``pivot_to_goal`` / ``explain_concept`` — user says "continue" or signals
      resumption, using the ``resume_anchor``.
    - ``propose_session`` — analyzer set ``suggest_session``; drafts a new
      :class:`SessionContext`.
    - ``explain_concept`` / ``ask_guiding_question`` — normal on-goal questions.

    :param state: Current graph state carrying ``user_message``,
        ``memory_results``, ``analysis_results``, and ``active_session``.
    :type state: LearningGraphState
    :return: Mapping the state key ``"teaching_strategy"`` to a
        :class:`TeachingStrategy`.
    :rtype: dict[str, Any]
    """
    model = get_chat_model("decision_maker").with_structured_output(
        TeachingStrategy, method="function_calling"
    )

    active_session = None
    if state.active_session is not None:
        active_session = state.active_session.model_dump()

    memory_results = state.memory_results or []
    knowledge_model = state.knowledge_model or []
    context = {
        "user_message": state.user_message,
        "memory_results": [
            m.get("text") for m in memory_results if isinstance(m, dict)
        ],
        "analysis_results": state.analysis_results.model_dump() if state.analysis_results is not None else None,
        "active_session": active_session,
        "knowledge_model": [
            f"{kc.concept} ({kc.status}, {kc.evidence[:60]})" for kc in knowledge_model
        ],
    }

    context_json = json.dumps(context, indent=2, default=str)

    system_prompt = get_node_prompt("decision_maker", default=PEDAGOGICAL_DIRECTOR_SYSTEM_PROMPT)

    strategy = model.invoke(
        [
            SystemMessage(system_prompt),
            SystemMessage(f"Current Learning Context:\n{context_json}"),
        ]
    )

    return {
        "teaching_strategy": strategy
    }