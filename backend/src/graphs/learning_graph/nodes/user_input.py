from graphs.learning_graph.nodes.node import node
from graphs.learning_graph.state import LearningGraphState


@node(prompt=None, intent="receive")
def user_input(state: LearningGraphState) -> dict:
    """
    Read the latest message from the message channel.

    Extracts ``state.messages[-1]`` and stores it under the
    ``"user_message"`` state key so downstream nodes can access the
    current learner input without re-reading the channel.

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