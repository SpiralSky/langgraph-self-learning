import json

from langchain_core.messages import SystemMessage, HumanMessage

from graphs.learning_graph.llm import get_structured_model
from graphs.learning_graph.pydantic_models import ResponseImproverOutput
from graphs.learning_graph.state import LearningGraphState

RESPONSE_IMPROVER_SYSTEM_PROMPT = """
You are a Pedagogical Editor. Refine the draft response so it aligns with the user's learning needs and input quality.

Input JSON keys: `draft_response`, `current_tone`, `sources_used`, `user_intent`, `input_clarity` (clear/ambiguous), `input_comments` (e.g. "lazy", "unclear", "well-structured").

Rules:
1. Strategy by intent:
   - Factual: direct and concise; correct misconceptions immediately.
   - Conceptual: analogies; switch to a Socratic style (guiding questions) if understanding seems incomplete.
   - Problem solving: scaffolding; break down steps clearly.
   - Open-ended: reflection; connect topics to broader contexts.
2. Tone (critical):
   - High-quality input: keep tone encouraging and professional.
   - Lazy/gibberish input (per `input_comments`): shift to harsh-but-formal, e.g. "Your question lacks the necessary context for a meaningful answer. Please specify [X] before proceeding." Be stern, not rude.
   - Ambiguous input: open with a clarification statement, then give a best-guess answer.
3. Content integrity:
   - Cite every `sources_used` item in the text, or list them at the bottom if confidence was low.
   - Keep Markdown clean (headers, bolding, lists).
   - Produce plain Markdown only: never use HTML tags or widget markup (e.g. `<div data-widget="...">` or `:::` fences) — widget formatting is applied downstream.
   - Weave Socratic/Reflection questions naturally into the flow, not just appended at the end.
7. Session tracking:
   - Fill `session_delta` with the consolidated learning `topic` and a short list of `ledger_updates` capturing what the learner demonstrably now knows or still struggles with after this exchange (e.g. "knows list slicing", "struggles with nested loops").
   - Set `continue_session` to false only when the topic is exhausted and the learner should move on to a new topic.
   - Leave `session_delta` null when this exchange does not advance a learning session.

Output JSON:
- `final_response`: str, refined Markdown
- `strategy_used`: str, primary strategy applied
- `tone_applied`: str, final tone descriptor
- `session_delta`: object|null, session update as described above
"""


def response_improver(state: LearningGraphState) -> dict[str, ResponseImproverOutput]:
    """
    Refine the draft response to align with the user's intent and input quality.

    Passes the draft, tone, sources, and analysis feedback to the configured LLM
    with structured output, producing a polished markdown response. Produces
    plain markdown only — widget formatting is deferred to ``format_output``.

    :param state: Current graph state carrying ``draft_response`` and
        ``analysis_results``.
    :type state: LearningGraphState
    :return: Mapping the state key ``"improved_response"`` to a
        :class:`ResponseImproverOutput`.
    :rtype: dict[str, ResponseImproverOutput]
    :raises ValueError: If ``draft_response`` or ``analysis_results`` is
        missing from state.
    """
    if state.draft_response is None:
        raise ValueError("Missing required field(s): draft_response")
    if state.analysis_results is None:
        raise ValueError("Missing required field(s): analysis_results")
    builder_output = state.draft_response
    analysis = state.analysis_results

    context = {
        "draft_response": builder_output.draft_response,
        "current_tone": builder_output.tone,
        "sources_used": builder_output.sources_used,
        "user_intent": analysis.intent,
        "input_clarity": analysis.is_clear,
        "input_comments": analysis.comments
    }

    messages = [
        SystemMessage(content=RESPONSE_IMPROVER_SYSTEM_PROMPT),
        HumanMessage(content=json.dumps(context, indent=2, default=str))
    ]

    model = get_structured_model("response_improver", ResponseImproverOutput)
    result = model.invoke(messages)

    # noinspection bad-return
    return {
        "improved_response": result
    }
