import json

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage

from graphs.learning_graph.config import config
from graphs.learning_graph.state import LearningGraphState

FORMAT_OUTPUT_SYSTEM_PROMPT = """You are a formatting expert tasked with making responses more readable and visually appealing using markdown.

Your task is to format the improved response by:

1. **Selective Emphasis**: 
   - Bold key concepts, important terms, or critical information
   - Use italics for emphasis on specific words or phrases where appropriate
   - Don't overuse bold/italics - be selective and strategic

2. **Structure Enhancement**:
   - Add proper headings (##, ###) if the content lacks clear structure
   - Use bullet points or numbered lists where appropriate
   - Ensure logical flow and readability

3. **Code Blocks**:
   - Wrap any code snippets in :::code blocks with appropriate language specification
   - Example: :::code language="python" title="Example"
     your code here
     :::

4. **Info/Note Boxes**:
   - Use :::text blocks for important notes, warnings, tips, or supplementary information
   - Choose appropriate colors: blue (info), yellow (warning), red (error/critical), green (success/tip)
   - Example: :::text title="Important Note" color="blue"
     Your note content here
     :::

5. **Preservation**:
   - Maintain all factual content and meaning from the original response
   - Don't add new information or change the substance
   - Only enhance the presentation and readability

Format the response to be clean, professional, and easy to scan while maintaining the original message's intent and accuracy."""


def format_output(state: LearningGraphState) -> dict[str, str]:
    improved_response = state.improved_response

    if not improved_response:
        raise ValueError("Improved response is required but not present in state")

    context = {
        "response_to_format": improved_response.final_response,
        "original_tone": improved_response.tone_applied,
        "strategy_used": improved_response.strategy_used
    }

    messages = [
        SystemMessage(content=FORMAT_OUTPUT_SYSTEM_PROMPT),
        HumanMessage(content=json.dumps(context, indent=2, default=str))
    ]

    model_config = config.get_model_data("format_output")
    model = init_chat_model(
        model_config.model_id,
        api_key=model_config.api_key,
        base_url=model_config.api_endpoint,
        temperature=0.2,
        model_provider="openai"
    )

    result = model.invoke(messages)

    return {
        "final_output": result.content
    }