from typing import List, Optional

from pydantic import BaseModel, Field


class FetcherOutput(BaseModel):
    knowledge: str = Field(
        description="A concise summary of what the LLM already knows. If unknown, leave empty."
    )
    confidence: float = Field(
        description="0.0-1.0. High (>0.8) for basic facts. Medium (0.5-0.8) for niche topics. Low (<0.5) for recent/ambiguous data."
    )
    search_queries: List[str] = Field(
        description="List of keyword-rich search queries. ONLY populate if confidence < 0.8. Use boolean operators if needed."
    )
    reasoning: str = Field(
        description="Brief justification for the confidence score and query choice."
    )


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


class ResponseImproverOutput(BaseModel):
    final_response: str = Field(
        description="The final, polished response ready for the user. Must be in Markdown."
    )
    strategy_used: str = Field(
        description="The pedagogical strategy applied: 'direct', 'socratic', 'scaffolding', or 'reflection'."
    )
    tone_applied: str = Field(
        description="The final tone used: 'encouraging', 'formal', 'harsh_formal', or 'neutral'."
    )


class QueryResult(BaseModel):
    confidence: float
    information: str
    queries: list[str]
    reasoning: str
