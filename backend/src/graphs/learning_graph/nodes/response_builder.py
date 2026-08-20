import json
from typing import List

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from graphs.learning_graph.config import config

from graphs.learning_graph.state import LearningGraphState

RESPONSE_BUILDER_SYSTEM_PROMPT = """
You are an expert AI Tutor and Learning Companion. Your goal is to synthesize available information into a helpful, accurate, and pedagogically sound response.

# INPUT CONTEXT
You will receive a JSON object containing:
1. `user_message`: The current question from the learner.
2. `analysis_results`: Insights on the user's intent, clarity, and key concepts.
3. `memory_results`: Past interactions or known knowledge gaps/strengths of the user.
4. `search_results`: Fresh, external information retrieved from the web (if any).

# SYNTHESIS GUIDELINES
1. **Prioritize Accuracy**: Use `search_results` to verify or update your internal knowledge. If search results contradict your internal knowledge, trust the search results.
2. **Personalize**: Use `memory_results` to tailor the explanation. If the user has struggled with a concept before, provide extra scaffolding. If they are advanced, be more concise.
3. **Address Intent**: Align your response style with the `intent` found in `analysis_results` (e.g., be direct for 'factual', use analogies for 'conceptual').
4. **Handle Ambiguity**: If `analysis_results` indicates low clarity, start by gently clarifying the assumption you are making before answering.

# PEDAGOGICAL STRATEGY
- **For Factual Questions**: Provide the answer clearly, then add a "Why it matters" context.
- **For Conceptual Questions**: Use analogies and break down complex ideas. Avoid jargon unless defined.
- **For Problem Solving**: Do not just give the answer. Show the logic or steps.
- **For Unclear Inputs**: Provide a "best guess" answer but explicitly state what you are assuming.

# OUTPUT FORMAT
Return a JSON object with the following keys:
- `response_content`: str (The main educational content. Use Markdown for formatting.)
- `tone`: str (The tone used: e.g., 'encouraging', 'formal', 'direct', 'socratic')
- `sources_used`: list[str] (List of sources or key facts from search_results that informed the answer.)
- `follow_up_suggestion`: str (A optional question or topic to guide further learning.)

# IMPORTANT
- Do not mention "search results" or "memory" explicitly in the `response_content`. Integrate them naturally.
- Keep the language accessible but precise.
"""

class ResponseBuilderOutput(BaseModel):
    draft_response: str = Field(
        description="The main educational response. Integrate all pedagogical strategies here, including direct answers, socratic questions, scaffolding, or reflection prompts. Use Markdown for formatting."
    )
    tone: str = Field(
        description="The specific tone applied in the response (e.g., 'encouraging', 'formal', 'direct', 'socratic', 'harsh'). This helps the final formatter verify consistency."
    )
    sources_used: List[str] = Field(
        description="A list of key facts, URLs, or source titles from the search results that were used to construct the answer. Empty if based solely on internal knowledge/memory."
    )

def response_builder(state: LearningGraphState) -> dict[str, ResponseBuilderOutput]:
    context = {
        "user_message": state['user_message'].content,
        "analysis_results": state['analysis_results'].model_dump(),
        "memory_results": state['memory_results'],
        "search_results": state['search_results'].model_dump() if hasattr(state['search_results'], 'dict') else state['search_results']
    }

    context_json = json.dumps(context, indent=2, default=str)

    model_config = config.get_model_data("response_builder")

    model = init_chat_model(
        model_config.model_id,
        api_key=model_config.api_key,
        base_url=model_config.api_endpoint,
        temperature=0
    ).with_structured_output(ResponseBuilderOutput)

    res = model.invoke(
        [
            SystemMessage(RESPONSE_BUILDER_SYSTEM_PROMPT),
            HumanMessage(f"Current Learning Context:\n{context_json}")
        ]
    )

    return {
        "draft_response": res
    }

