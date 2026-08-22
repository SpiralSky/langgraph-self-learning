import json
import re

from langchain_core.messages import SystemMessage, HumanMessage

from graphs.learning_graph.llm import get_chat_model
from graphs.learning_graph.state import LearningGraphState

FORMAT_OUTPUT_SYSTEM_PROMPT = """You are a formatting expert. Improve the readability and visual appeal of the given response with markdown.

You are the SOLE producer of widgets (rendered by the chat UI). Widgets use the line-based `:::` fence syntax ONLY — never HTML. Each widget is an opening line, the raw body lines, and a closing line containing only `:::`:
- :::text title="<label>" color="<color>"
  body lines...
  :::
  Renders a Discord-style info box. Set ONLY `title` and `color` attributes.
- :::code language="<lang>" title="<label>"
  code lines...
  :::
  Renders a syntax-highlighted code block. Set ONLY `language` and `title` attributes.

FORBIDDEN:
- NEVER emit HTML tags. In particular, never emit `<div data-widget="text" data-widget-title="..." data-widget-color="...">...</div>` style markup — the UI renders raw HTML as escaped plain text, breaking the widget.
- NEVER use any other wrapper syntax for widgets; the `:::` fences above are the only valid form.
- Every widget fence MUST end with a line containing only `:::`.

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
5. Math: render mathematical expressions with LaTeX in KaTeX delimiters — inline math uses `$...$` (e.g. `$E = mc^2$`), display math uses `$$...$$` on its own line (e.g. `$$\int_0^1 x^2\,dx$$`). Write the LaTeX source directly; never use ASCII-art equations.
6. Preservation: keep all facts and meaning; add no new information and change no substance; only enhance presentation so the output is clean, professional, and easy to scan."""

_WIDGET_DIV_RE = re.compile(
    r'<div\s+data-widget="(?P<widget>text|code)"(?P<attrs>[^>]*)>(?P<body>.*?)</div>',
    re.DOTALL
)
_WIDGET_ATTR_RE = re.compile(r'data-widget-(title|color|language)="([^"]*)"')

# html data-* attribute -> ::: fence attribute, per widget type.
_WIDGET_FENCE_ATTRS: dict[str, dict[str, str]] = {
    "text": {"title": "title", "color": "color"},
    "code": {"title": "title", "language": "language"}
}


def _convert_widget_div(match: re.Match) -> str:
    """
    Convert one ``<div data-widget="...">...</div>`` match to a ``:::`` fence.

    Renders only the attributes the frontend widget supports (e.g. ``text``
    accepts ``title``/``color``, ``code`` accepts ``language``/``title``).

    :param match: Regular-expression match for a single widget ``<div>``.
    :type match: re.Match
    :return: Equivalent ``:::`` fence markup.
    :rtype: str
    """
    widget = match.group("widget")
    attrs = dict(_WIDGET_ATTR_RE.findall(match.group("attrs")))
    fence_attrs = _WIDGET_FENCE_ATTRS[widget]
    rendered = [
        f'{fence_attr}="{attrs[attr]}"'
        for attr, fence_attr in fence_attrs.items()
        if attr in attrs
    ]
    opening = f":::{widget} {' '.join(rendered)}".rstrip()
    body = match.group("body").strip()

    return f"{opening}\n{body}\n:::"


def normalize_widget_fences(markdown: str) -> str:
    """
    Convert any HTML ``<div data-widget="text|code">`` markup to ::: fences.

    Frontend widgets are declared with ::: fences (see ``remark-widget-fence``
    in the chat UI); raw HTML is escaped and rendered as plain text. This is a
    safety net for the unlikely event the model emits the HTML ``<div>`` form
    instead of the fence form.

    :param markdown: Model output to normalize.
    :type markdown: str
    :return: The same string with HTML-style widget markup replaced by ::: fences.
    :rtype: str
    """
    return _WIDGET_DIV_RE.sub(_convert_widget_div, markdown)


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
    :raises ValueError: If ``state.improved_response`` is missing.
    """
    if state.improved_response is None:
        raise ValueError("Missing required field(s): improved_response")
    improved_response = state.improved_response

    context = {
        "response_to_format": improved_response.final_response,
        "original_tone": improved_response.tone_applied,
        "strategy_used": improved_response.strategy_used
    }

    messages = [
        SystemMessage(content=FORMAT_OUTPUT_SYSTEM_PROMPT),
        HumanMessage(content=json.dumps(context, indent=2, default=str))
    ]

    model = get_chat_model("format_output", temperature=0.2)
    result = model.invoke(messages)

    formatted = result.content
    if isinstance(formatted, str):
        formatted = normalize_widget_fences(formatted)

    return {
        "final_output": formatted
    }