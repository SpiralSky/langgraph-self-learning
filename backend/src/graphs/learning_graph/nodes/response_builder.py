import json

from langchain_core.messages import SystemMessage, HumanMessage

from graphs.learning_graph.compiler import get_node_prompt
from graphs.learning_graph.llm import get_chat_model
from graphs.learning_graph.nodes.node import node
from graphs.learning_graph.pydantic_models import ResponseBuilderOutput
from graphs.learning_graph.state import LearningGraphState
from graphs.learning_graph.text_utils import message_to_text


RESPONSE_BUILDER_SYSTEM_PROMPT = """
You are a skilled tutor's writer. You do NOT make pedagogical decisions — those are decided upstream. Your single job is to turn a given teaching plan into a beautifully formatted, empathetic Markdown reply to the learner.

Input JSON keys:
- `user_message`: the learner's message you are replying to.
- `focus_concept`: the concept this turn centres on.
- `tone`: the tone to write in (e.g. encouraging, socratic, direct).
- `execution_plan`: 2-3 bullet steps decided for this turn. Follow them in order.
- `session_context`: the current SessionContext (goal, current_focus, status, resume_anchor, next_move_hint). Use it to stay on-goal and reference where the learner is.

Guidelines:
- Write ONLY about what the `execution_plan` says. Do not introduce new topics, new tasks, or new planning.
- Make it beautiful and easy to read: use short paragraphs, bullet lists for steps, and `**bold**` for key terms. Keep plain Markdown only — no HTML, no `:::` widget fences (formatting is applied downstream).
- Be empathetic: acknowledge the learner's effort and their words; mirror their level; never condescend.
- Address the learner naturally as "you". Do not refer to "the execution plan", "the tutor", or "the strategy".
- Follow the `tone` you are given.

Output JSON:
- `draft_response`: str, the final Markdown reply following the execution_plan and tone.
- `tone`: str, the exact tone you applied.
- `sources_used`: list[str], always empty — no web search input here.
"""

RESPONSE_BUILDER_OUTPUT_SCHEMA_HINT = """
The `execution_plan` may contain terse bullet notes written for a tutor; expand
each into a helpful, human Markdown passage, but do not re-order or drop any.
"""

@node(prompt=RESPONSE_BUILDER_SYSTEM_PROMPT, intent="generate")
def response_builder(state: LearningGraphState) -> dict[str, ResponseBuilderOutput]:
    """
    Translate the decided teaching plan into a polished empathetic reply.

    This node is the pedagogical executor (not planner). It consumes the
    :class:`TeachingStrategy` produced upstream and simply renders its
    ``execution_plan``, ``tone``, and ``focus_concept`` into a beautifully
    formatted :class:`ResponseBuilderOutput` draft. It makes no pedagogical
    decisions of its own. Structured output is requested on a single attempt;
    no LangGraph ``RetryPolicy`` is configured for this node.

    Output is plain Markdown only — no ``:::`` widget fences and no HTML;
    widget concerns are handled downstream by :func:`format_output`.

    :param state: Current graph state carrying ``user_message`` and the
        ``teaching_strategy`` produced by ``decision_maker``.
    :type state: LearningGraphState
    :return: Mapping the state key ``"draft_response"`` to a
        :class:`ResponseBuilderOutput`.
    :rtype: dict[str, ResponseBuilderOutput]
    :raises ValueError: If ``user_message`` or ``teaching_strategy`` are missing.
    """
    if state.user_message is None:
        raise ValueError("Missing required field(s): user_message")
    if state.teaching_strategy is None:
        raise ValueError(
            "Missing required field(s): teaching_strategy (decision_maker must run first)"
        )

    strategy = state.teaching_strategy

    context = {
        "user_message": message_to_text(state.user_message),
        "focus_concept": strategy.focus_concept,
        "tone": strategy.tone,
        "execution_plan": strategy.execution_plan,
        "session_context": strategy.updated_session_context.model_dump(),
    }

    context_json = json.dumps(context, indent=2, default=str)

    model = get_chat_model("response_builder").with_structured_output(
        ResponseBuilderOutput, method="function_calling"
    )
    system_prompt = get_node_prompt(
        "response_builder", default=RESPONSE_BUILDER_SYSTEM_PROMPT
    )
    res = model.invoke(
        [
            SystemMessage(system_prompt),
            SystemMessage(RESPONSE_BUILDER_OUTPUT_SCHEMA_HINT),
            HumanMessage(f"Current Teaching Context:\n{context_json}"),
        ]
    )

    return {
        "draft_response": res
    }