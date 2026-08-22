from langchain_core.messages import AIMessage

from graphs.learning_graph.state import LearningGraphState


def model_output(state: LearningGraphState) -> dict:
    """
    Convert the formatted final output into a chat-ready AI message.

    Wraps ``state.final_output`` (already formatted, possibly containing
    ``:::`` widget fences) into an :class:`AIMessage` appended to the messages
    channel.

    :param state: Current graph state carrying ``final_output``.
    :type state: LearningGraphState
    :return: Mapping the state key ``"messages"`` to a list containing a single
        :class:`AIMessage` with the final formatted text.
    :rtype: dict
    """
    final_text = state.final_output

    return {
        "messages": [AIMessage(content=final_text)]
    }