from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage

from graphs.learning_graph.config import config
from graphs.learning_graph.state import LearningGraphState

from pydantic import BaseModel, Field
from typing import List, Optional, Any


class InputAnalysisResult(BaseModel):
    is_clear: bool = Field(
        description="True if the question is specific enough to answer without guessing intent."
    )
    intent: str = Field(
        description="The primary intent: 'factual', 'conceptual', 'problem_solving', 'debugging', or 'unclear'."
    )
    key_points: List[str] = Field(
        description="Extracted core concepts or entities from the user's input. Empty if unclear."
    )
    comments: str = Field(
        description="Natural language feedback for the user. If clear, praise specificity. If unclear, gently explain what is missing."
    )
    suggested_clarifications: Optional[List[str]] = Field(
        default=None,
        description="If is_clear is False, provide 1-2 specific questions to ask the user to resolve ambiguity."
    )

INPUT_ANALYSER_PROMPT = """
You are an expert Pedagogical Input Analyst. Analyze the user's question for clarity, intent, and key concepts.

# GOALS
1. Determine if the question is clear enough to answer directly.
2. Identify the user's learning intent (e.g., wanting a fact, understanding a concept, solving a problem).
3. Extract key technical terms or concepts mentioned.
4. Provide constructive feedback on the question's quality.

# GUIDELINES
- **Clarity**: If the question is vague (e.g., "How does it work?"), mark `is_clear` as false.
- **Intent**: 
  - 'factual': Seeking a specific definition or date.
  - 'conceptual': Seeking understanding of a mechanism or theory.
  - 'problem_solving': Needs help with a specific task or code.
  - 'unclear': Cannot determine intent.
- **Key Points**: Extract only the most relevant nouns/concepts. Ignore filler words.
- **Comments**: Be encouraging but honest. If the question is lazy, guide them to be more specific.

# OUTPUT FORMAT
Return a valid JSON object matching the schema provided. Do not include markdown formatting like ```json.
"""

def input_analyzer(state: LearningGraphState) -> dict[str, Any]:
    user_message = state["user_message"]

    model_config = config.get_model_data("input_analyzer")

    model = init_chat_model(
        model_config.model_id,
        api_key=model_config.api_key,
        base_url=model_config.api_endpoint,
        temperature=0
    ).with_structured_output(InputAnalysisResult)

    analysis = model.invoke(
        [
            SystemMessage(INPUT_ANALYSER_PROMPT),
            user_message
        ]
    )

    return {
        "analysis_results": analysis
    }