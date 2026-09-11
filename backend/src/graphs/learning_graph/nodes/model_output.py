from langchain_core.messages import AIMessage

from graphs.learning_graph.nodes.node import node
from graphs.learning_graph.state import LearningGraphState


@node(prompt=None, intent="emit")
def model_output(state: LearningGraphState) -> dict:
    """
    Convert the final response into a chat-ready AI message.

    Wraps the final text (``state.final_output`` when present, otherwise
    ``state.draft_response`` as a fallback) into an :class:`AIMessage`
    appended to the ``messages`` channel.

    :param state: Current graph state carrying ``final_output`` or
        ``draft_response``.
    :type state: LearningGraphState
    :return: Mapping the state key ``"messages"`` to a list containing a single
        :class:`AIMessage` with the final text.
    :rtype: dict
    """
    final_text = state.final_output

    if final_text is None and state.draft_response is not None:
        final_text = state.draft_response.draft_response

    return {
        "messages": [AIMessage(content=final_text)]
    }