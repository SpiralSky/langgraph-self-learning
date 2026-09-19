"""Messages-looping chat graph wrapping the single-turn agent entrypoint.

The ``agent`` graph (``graphs.entrypoint``) is single-turn:
``user_message`` in, ``response`` out. This module exposes a thin ``chat``
graph with a proper ``messages`` channel so SDK ``messages``-aware runtimes
(the assistant-ui ``react-langgraph`` runtime) can drive multi-turn
conversations: each run forwards the last human message as ``user_message``
and appends the resulting ``response`` to the running ``messages`` channel.

Strictly additive — the ``agent`` contract, its state, and backend tests are
untouched. ``langgraph.json`` registers both graphs; the frontend defaults to
the ``chat`` graph id.
"""

from typing import Annotated

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from typing_extensions import TypedDict
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from graphs.entrypoint import graph as agent_graph


class ChatState(TypedDict):
    """Running conversation: langchain message tuples accumulated in-order.

    ``ui`` is an optional pass-through channel for generative UI messages
    emitted by the nested ``agent`` graph. The assistant-ui runtime reads it
    from ``values`` events / ``getState()`` (``state.values[uiStateKey]``),
    so forwarding it here is what lets the frontend render artifacts without
    a dedicated ``custom``-event stream from the nested graph.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    ui: list[dict]


def _last_human_message(messages: list[AnyMessage]) -> str:
    """Return the content of the most recent human message, or ``""``."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            content = message.content
            return content if isinstance(content, str) else str(content)
    return ""


def _run_turn(state: ChatState) -> dict:
    """Run the nested single-turn agent and append its reply to ``messages``."""
    result = agent_graph.invoke({"user_message": _last_human_message(state["messages"])})
    response = result.get("response") or ""
    update: dict = {"messages": AIMessage(content=response)}
    # Forward any generative UI the nested agent surfaced in its final state.
    # The nested agent's ``custom`` events do NOT propagate through
    # ``.invoke()``, so the values-state ``ui`` channel is the transport.
    ui = result.get("ui")
    if ui:
        update["ui"] = ui
    return update


_builder = StateGraph(ChatState)
_builder.add_node("agent", _run_turn)
_builder.add_edge(START, "agent")
_builder.add_edge("agent", END)

graph = _builder.compile()