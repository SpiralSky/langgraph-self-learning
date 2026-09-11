from typing import Any

from langchain_core.messages import SystemMessage

from graphs.learning_graph.compiler import get_node_prompt
from graphs.learning_graph.llm import get_chat_model
from graphs.learning_graph.nodes.node import node
from graphs.learning_graph.pydantic_models import InputAnalysisResult
from graphs.learning_graph.session_types import analyzer_catalog_text
from graphs.learning_graph.state import LearningGraphState

RECENT_HISTORY_WINDOW = 3

INPUT_ANALYSER_PROMPT = """You are a Learning Input Analyst. Analyze the user's question for clarity, intent, and key learning concepts.

Goals:
1. Decide if the question is clear enough to answer directly.
2. Identify learning intent: 'factual' (definition/date), 'conceptual' (mechanism/theory), 'problem_solving' (task or code), 'unclear'.
3. Extract the most relevant nouns/concepts; ignore filler.
4. Diagnose prerequisite gaps: fill `prerequisite_gaps` when the learner demonstrably lacks or is unsure of knowledge the answer DEPENDS on. Signals:
   - Explicit admissions: "I don't know how X...", "i forgot what X is", "I'm not sure about X".
   - Expressing a required concept as a question to the tutor ("what does X mean?", "is X the same as Y?") instead of using it.
   - Uncertainty implied across the recent 2-3 turns (e.g. the learner fumbled a step the last time it appeared).
   Name each gap at the concrete rung level (e.g. "conditional probability notation `q(z|x)`", "what a posterior is", "the `|` means conditioning", "expectation", "KL divergence") — NOT the whole topic (never list "variational autoencoders" as the gap). `status`: 'missing' for explicit admissions, 'rusty' for forgot/uncertain, 'inferred' when you deduce it from history. Always fill `evidence` with the supporting quote or signal. Leave the list EMPTY when there is no credible signal — do NOT invent gaps to be safe.
5. Comment encouragingly but honestly; push vague questions to be more specific.

Session hint detection (optional — never forced, guidance only):
{session_catalog}
EXPLICIT triggers — ALWAYS classify these as an interactive learning session, never as plain Q&A:
- If the user uses phrases like "step-by-step", "teach me", "walk me through", "guide me", "show me how" — set `session_hint` to `"guided_learning"` and `suggest_session` to `true`.
- Do NOT downgrade such interactive requests to plain Q&A. A request to be taught/guided in steps is a session-initiation signal even when it lacks an explicitly named format.
- (If the user ALSO names a different concrete format, e.g. "quiz me", prefer the named template id over `"guided_learning"`.)
- When the user explicitly initiates a tracked, turn-based session — clearly asking to be guided step by step, quizzed, debated, or otherwise learn in a recurring format that matches one of the templates (e.g. "walk me through", "teach me step by step", "quiz me") — set `session_hint` to the matching template id (e.g. "guided_learning", "quiz_game"). A normal direct question (definition, "what is X") is NOT a session request: set `session_hint` to null.
- The hint is guidance only, never binding: you may also set a short descriptive id for an improvised tracked format not covered by a template (e.g. "word_game" for "let's play a word-association game to learn latin roots", "story_learning" for "teach me through a story"). The tutor authors the actual session freely.
- `suggest_session` is the fallback when the user clearly WANTS guided/tracked learning but did not name a concrete format and no session is active: set `session_hint` to null and `suggest_session` to true so the tutor can invite them to pick a format. When `session_hint` is set, leave `suggest_session` false — the tutor seeds the session directly.
- Otherwise set both `session_hint` (null) and `suggest_session` (false). Only set either when the user genuinely asks for a tracked learning experience, never for a plain Q&A turn.

The message list may contain up to 3 preceding conversation turns (earlier user questions and AI tutor replies) before the current user question. Use this history to resolve pronouns, ellipsis, and references to earlier topics when judging clarity and extracting key concepts. The final message is always the current question to analyze. If a session is already active, the current message often continues it — do not re-detect a new format.

Vague questions (e.g. "Hey, can you show me again?") => is_clear=false.

Return a valid JSON object matching the provided schema. No markdown fences.
"""

@node(prompt=INPUT_ANALYSER_PROMPT, intent="analyze")
def input_analyzer(state: LearningGraphState) -> dict[str, Any]:
    """
    Analyze the user's question for clarity, intent, and key learning concepts.

    Calls the configured LLM with a structured-output schema and stores the
    resulting :class:`InputAnalysisResult` under ``"analysis_results"``. Prior
    conversation turns (up to :data:`RECENT_HISTORY_WINDOW`) are passed to the
    model before the current message so pronouns and implicit references in the
    follow-up are resolved. Emits no widget markup.

    Prerequisite-gap detection: only fills ``prerequisite_gaps`` when there is
    a credible signal (explicit admission, expressed-as-question, or uncertainty
    across recent turns); never invents gaps. Each gap is named at the concrete
    rung level (e.g. ``"conditional probability notation q(z|x)"``), not the
    whole topic.

    Session-hint detection: interactive phrases (``step-by-step``, ``teach me``,
    ``walk me through``, ``quiz me``) set ``session_hint`` to the matching
    template id and ``suggest_session`` to ``False``; when the user wants
    guided learning but didn't name a format, sets ``suggest_session`` to
    ``True`` with ``session_hint`` to ``None``.

    :param state: Current graph state carrying ``user_message`` and the
        ``messages`` channel (used for the preceding turns).
    :type state: LearningGraphState
    :return: Mapping the state key ``"analysis_results"`` to an
        :class:`InputAnalysisResult`.
    :rtype: dict[str, Any]
    """
    user_message = state.user_message

    model = get_chat_model("input_analyzer").with_structured_output(
        InputAnalysisResult, method="function_calling"
    )

    recent_history = state.messages[-RECENT_HISTORY_WINDOW - 1:-1]

    base_prompt = get_node_prompt("input_analyzer", default=INPUT_ANALYSER_PROMPT)
    system_prompt = base_prompt.format(session_catalog=analyzer_catalog_text())

    analysis = model.invoke(
        [
            SystemMessage(system_prompt),
            *recent_history,
            user_message
        ]
    )

    return {
        "analysis_results": analysis
    }