from graphs.learning_graph.state import LearningGraphState


def user_input(state: LearningGraphState) -> dict:
    """
    Read the latest message from the message channel.

    :param state: Current graph state carrying a populated ``messages`` channel.
    :type state: LearningGraphState
    :return: Mapping the state key ``"user_message"`` to the most recent
        message.
    :rtype: dict
    """
    latest_msg = state.messages[-1]

    return {
        "user_message": latest_msg
    }