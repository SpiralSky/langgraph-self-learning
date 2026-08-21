from langchain_core.messages import AIMessage

from graphs.learning_graph.state import LearningGraphState


def model_output(state: LearningGraphState) -> dict:
    final_text = state.final_output

    return {
        "messages": [AIMessage(content=final_text)]
    }