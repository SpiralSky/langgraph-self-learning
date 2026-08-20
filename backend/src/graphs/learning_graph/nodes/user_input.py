from graphs.learning_graph.state import LearningGraphState


def user_input(state: LearningGraphState) -> dict:
    latest_msg = state.messages[-1]

    return {
        "user_message": latest_msg
    }