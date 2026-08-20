from langchain_core.messages import AIMessage

from graphs.learning_graph.state import LearningGraphState


def model_output(state: LearningGraphState) -> dict:
    final_text = state.improved_response.final_response

    return {
        "messages": [AIMessage(content=final_text)]
    }