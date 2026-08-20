from typing import TypedDict

from langchain_core.messages import HumanMessage


class LearningGraphState(TypedDict):
    user_message: HumanMessage

