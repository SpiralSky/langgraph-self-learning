# Architecture Overview

## System Overview

The backend is a **LangGraph-based Socratic tutoring system**. It orchestrates a multi-step teaching loop where each turn:

1. Receives the learner's message
2. Analyzes intent and clarity
3. Retrieves relevant memories
4. Decides the next pedagogical move
5. Builds a response
6. Persists the exchange to memory
7. Formats the output for the chat UI

## Graph Flow

```
START
  │
  ▼
user_input ──┬──► retrieve_memory ──┐
  │          │                     │
  ▼          ▼                     │
input_analyzer                           │
  │                                     │
  └──────────► decision_maker ◄─────────┘
                   │
        ┌──────────┼──────────┐
        │          │          │
        ▼          │          ▼
execute_code      │   response_builder
        │         │          │
        └─────────┴──► save_memory
                          │
                       format_output
                          │
                       model_output
                          │
                          ▼
                         END
```

- **user_input** reads the latest message from the message channel
- **retrieve_memory** + **input_analyzer** run in parallel after user_input
- **decision_maker** receives both and decides the pedagogical move
- **response_builder** renders the teaching plan into Markdown
- **save_memory** persists the exchange to mem0
- **format_output** applies widget fences, LaTeX, and Markdown structure
- **model_output** wraps the result as an AIMessage
- **execute_code** branches off decision_maker when `pedagogical_move == "execute_code"` and flows back to response_builder

## State Data Flow

`LearningGraphState` carries the following keys through the graph:

| Key | Type | Purpose |
|-----|------|---------|
| `messages` | `list[BaseMessage]` | Conversation history (LangGraph channel) |
| `user_message` | `BaseMessage \| None` | Latest user message, extracted by user_input |
| `memory_results` | `list[dict] \| None` | Compacted memory search hits |
| `analysis_results` | `InputAnalysisResult \| None` | Intent, clarity, key concepts, prerequisite gaps |
| `search_results` | `QueryResult \| None` | Web search results (if used) |
| `active_session` | `SessionRecord \| None` | Current learning session state |
| `knowledge_model` | `list[KnowledgeComponent]` | Persistent mastery model |
| `use_web_search` | `bool` | Whether to use web search |
| `skip_format` | `bool` | Skip formatting for this turn |
| `assess_session` | `bool` | Whether to assess the session |
| `teaching_strategy` | `TeachingStrategy \| None` | Pedagogical decision from decision_maker |
| `draft_response` | `ResponseBuilderOutput \| None` | Draft response from response_builder |
| `final_output` | `str \| None` | Formatted output ready for the UI |

## Config Files

| File | Role |
|------|------|
| `config.yaml` | Runtime config: model providers, API keys (via env vars), reasoning defaults |
| `config.template.yaml` | Template with example provider configurations |
| `graph_config.yaml` | Pedagogical customizations (graph-level + per-node) and session type templates |

### Config resolution order
1. `config.yaml` is loaded and env-var placeholders (`${VAR}`) are resolved
2. `graph_config.yaml` is merged in under the `customizations` key
3. The `Config` model validates the result

## Session Types

Session templates loaded from `graph_config.yaml`. Each defines a `system_block` (starter instructions), `state_keys` (suggested state fields), and `triggers` (phrases that indicate the format).

| Type | Label | Triggers |
|------|-------|----------|
| `guided_learning` | Guided Step-by-Step | teach me, walk me through, step by step |
| `quiz_game` | Quiz Game | quiz, quiz me, test me, trivia |
| `mastery_check` | Mastery Check Loop | check my understanding, test my mastery |
| `debate` | Back-and-Forth Debate | debate, argue, convince me |
| `flashcard_review` | Flashcard Review | flashcards, drill me, recall practice |

The `main` flag controls whether a type is offered to the input analyzer for auto-trigger.

## Node Registry

Nodes are registered in `_NODES` (a `dict[str, Node]` mapping name to `Node` instance). Each `Node` wraps a callable with an optional `prompt` and `intent`:

```python
_NODES = {
    "user_input": user_input,         # intent: "receive"
    "input_analyzer": input_analyzer, # intent: "analyze"
    "retrieve_memory": retrieve_memory, # intent: "memory_read"
    "decision_maker": decision_maker, # intent: "decide"
    "response_builder": response_builder, # intent: "generate"
    "save_memory": save_memory,       # intent: "memory_write"
    "format_output": format_output,   # intent: "format"
    "model_output": model_output,     # intent: "emit"
    "execute_code": execute_code,     # intent: "execute_code"
}
```

The `@node` decorator declares a function's `prompt` and `intent`. Intents determine which graph-level customization blocks are broadcast to the node's prompt by the compiler.

## Export Structure

The module exports a `LearningGraphExport` TypedDict:

```python
class LearningGraphExport(TypedDict):
    graph: StateGraph          # Compiled LangGraph graph
    nodes: dict[str, Node]     # Node registry
    prompts: dict[str, str | None]  # Compiled prompts per node
```

This is consumed by `langgraph.json` to launch the graph via LangGraph serve.