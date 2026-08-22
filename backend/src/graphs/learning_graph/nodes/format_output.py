import json

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage

from graphs.learning_graph.config import config
from graphs.learning_graph.state import LearningGraphState

FORMAT_OUTPUT_SYSTEM_PROMPT = """You are a formatting expert. Improve the readability and visual appeal of the given response with markdown.

You are the SOLE producer of widget fences (rendered by the chat UI):
- `:::code language="<lang>" title="<label>"` ... `:::` renders a syntax-highlighted code block.
- `:::text title="<label>" color="<color>"` ... `:::` renders a Discord-style info box.

Rules:
1. Emphasis: selectively bold key concepts/important terms or info; use italics sparingly on specific words. Don't overuse.
2. Structure: add ##/### headings if the content lacks structure; use bullets/numbered lists where appropriate; keep logical flow.
3. Code: wrap code snippets in :::code blocks; set only `language`/`title` attributes (renderer handles the rest).
   :::code language="python" title="Example"
   def f(): pass
   :::
4. Notes: use :::text for important notes, warnings, tips, or supplementary info; set only `title`/`color` attributes. Colors: blue (info), amber (warning), red (error/critical), green (success/tip), purple, gray.
   :::text title="Important Note" color="blue"
   Your note content here
   :::
5. Preservation: keep all facts and meaning; add no new information and change no substance; only enhance presentation so the output is clean, professional, and easy to scan."""


def format_output(state: LearningGraphState) -> dict[str, str]:
    """
    Format the improved response into a presentable markdown string.

    This is the ONLY node in the graph that may emit widget-fence markup
    (``:::code``, ``:::text``). Earlier nodes keep their output as plain
    markdown and leave widget concerns to here.

    :param state: Current graph state, must have ``improved_response``
        populated with a ``final_response`` to format.
    :type state: LearningGraphState
    :return: Mapping the state key ``"final_output"`` to the formatted markdown
        response string ready for the chat UI.
    :rtype: dict[str, str]
    :raises ValueError: If ``state.improved_response`` is empty or missing.
    """
    improved_response = state.improved_response

    if not improved_response:
        raise ValueError("Improved response is required but not present in state")

    context = {
        "response_to_format": improved_response.final_response,
        "original_tone": improved_response.tone_applied,
        "strategy_used": improved_response.strategy_used
    }

    messages = [
        SystemMessage(content=FORMAT_OUTPUT_SYSTEM_PROMPT),
        HumanMessage(content=json.dumps(context, indent=2, default=str))
    ]

    model_config = config.get_model_data("format_output")
    model = init_chat_model(
        model_config.model_id,
        api_key=model_config.api_key,
        base_url=model_config.api_endpoint,
        temperature=0.2,
        model_provider="openai"
    )

    result = model.invoke(messages)

    return {
        "final_output": result.content
    }