from langchain_core.messages import AIMessage

from graphs.learning_graph.state import LearningGraphState


def model_output(state: LearningGraphState) -> dict:
    """
    Convert the formatted final output into a chat-ready AI message.

    Wraps the final text (``state.final_output`` when present, otherwise the
    plain-markdown ``improved_response`` on the skip-format path) into an
    :class:`AIMessage` appended to the messages channel.

    :param state: Current graph state carrying ``final_output`` and/or
        ``improved_response``.
    :type state: LearningGraphState
    :return: Mapping the state key ``"messages"`` to a list containing a single
        :class:`AIMessage` with the final text.
    :rtype: dict
    """
    final_text = state.final_output

    if final_text is None and state.improved_response is not None:
        final_text = state.improved_response.final_response

    return {
        "messages": [AIMessage(content=final_text)]
    }