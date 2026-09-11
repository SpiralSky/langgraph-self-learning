# State & Models

## LearningGraphState

The central state object threaded through every graph turn. All nodes read from and write to this state.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `messages` | `Annotated[list[BaseMessage], add_messages]` | `[]` | Conversation history; LangGraph-managed channel |
| `user_message` | `BaseMessage \| None` | `None` | Latest user message, extracted by `user_input` |
| `memory_results` | `list[dict] \| None` | `[]` | Compacted memory search hits from `retrieve_memory` |
| `analysis_results` | `InputAnalysisResult \| None` | `None` | Intent/clarity/key-concepts analysis from `input_analyzer` |
| `search_results` | `QueryResult \| None` | `None` | Web search results (if `use_web_search` is set) |
| `active_session` | `SessionRecord \| None` | `None` | Current learning session state |
| `knowledge_model` | `list[KnowledgeComponent]` | `[]` | Persistent per-thread mastery model |
| `use_web_search` | `bool` | `True` | Whether to query web search |
| `skip_format` | `bool` | `False` | Skip formatting for this turn |
| `assess_session` | `bool` | `False` | Whether to assess the session this turn |
| `teaching_strategy` | `TeachingStrategy \| None` | `None` | Pedagogical decision from `decision_maker` |
| `draft_response` | `ResponseBuilderOutput \| None` | `None` | Draft response from `response_builder` |
| `final_output` | `str \| None` | `None` | Formatted output ready for the chat UI |

## Pydantic Models

### JsonValue
Recursive JSON value type: `str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]`. Declared as a `RootModel` (not a type alias) so Pydantic resolves the self-reference by name, avoiding `maximum recursion depth exceeded` on schema build.

### SessionControl
Machine-readable session rules the graph keys routing and gating off.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `awaiting_user_input` | `bool` | `False` | True when the session is waiting on a specific learner reply |
| `pending_item_id` | `str \| None` | `None` | Identifier of the pending item (e.g. `"task_3"`) |
| `expected_input_kind` | `str \| None` | `None` | What input is awaited: `"answer"`, `"setup_confirmation"`, etc. |
| `allow_reveal_answer` | `bool` | `False` | True to permit showing the answer this turn |
| `one_move_per_turn` | `bool` | `False` | True when the session advances at most one rung per turn |

### KnowledgeComponent
A single concept in the learner's persistent knowledge model. Status lives on a `missing → rusty → inferred → mastered` rung ladder.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `concept` | `str` | — | Concept name at the rung level |
| `status` | `str` | — | `'missing'`, `'rusty'`, `'inferred'`, or `'mastered'` |
| `evidence` | `str` | `""` | Supporting quote or signal |
| `turn` | `int` | `0` | Turn number last observed |
| `series` | `str \| None` | `None` | Optional session/series label |

### PrerequisiteGap
Per-turn diagnosis of a missing prerequisite. Unlike `KnowledgeComponent`, this does not persist across turns.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `concept` | `str` | — | Concrete prerequisite at the rung level |
| `status` | `str` | — | `'missing'`, `'rusty'`, or `'inferred'` |
| `evidence` | `str` | — | Supporting quote or signal |

### InputAnalysisResult
Output of the `input_analyzer` node.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `is_clear` | `bool` | — | True if the question is specific enough to answer directly |
| `intent` | `str` | — | `'factual'`, `'conceptual'`, `'problem_solving'`, `'debugging'`, or `'unclear'` |
| `key_points` | `list[str]` | `[]` | Extracted core concepts or entities |
| `prerequisite_gaps` | `list[PrerequisiteGap]` | `[]` | Prerequisite knowledge the learner lacks |
| `comments` | `str` | — | Natural language feedback for the user |
| `suggested_clarifications` | `list[str] \| None` | `None` | 1-2 questions to resolve ambiguity if unclear |
| `session_hint` | `str \| None` | `None` | Short hint for interactive session format |
| `suggest_session` | `bool` | `False` | True when user wants guided/tracked learning without naming a format |

### SessionContext
Lightweight, mutable description of where a learning session stands. Threads between turns; the `decision_maker` can rewrite it in place.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `goal` | `str \| None` | `None` | Learner's overall objective |
| `current_focus` | `str \| None` | `None` | What the session is currently focused on |
| `status` | `Literal["none","active","paused","completed"]` | `"none"` | Session lifecycle |
| `control` | `SessionControl \| None` | `None` | Machine-readable session rules |
| `state` | `dict[str, JsonValue]` | `{}` | Model-authored session values |
| `resume_anchor` | `str \| None` | `None` | Where to resume after a detour |
| `next_move_hint` | `str \| None` | `None` | Hint for the next pedagogical move |

### TeachingStrategy
The pedagogical-director node's decision for this turn.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `pedagogical_move` | `Literal[...]` | `"explain_concept"` | Move type |
| `focus_concept` | `str \| None` | `None` | Concept to centre the turn on |
| `tone` | `str` | `"encouraging"` | Intended response tone |
| `execution_plan` | `list[str]` | `[]` | 2-3 bullet steps for the tutor |
| `updated_session_context` | `SessionContext` | `SessionContext()` | Updated session context |

### ResponseBuilderOutput
Output of the `response_builder` node.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `draft_response` | `str` | — | Main educational response in Markdown |
| `tone` | `str` | — | Tone applied (e.g. `'encouraging'`, `'socratic'`) |
| `sources_used` | `list[str]` | `[]` | Key facts/URLs from search results |

### SessionRecord
A model-authored learning session. System-stamps `session_id`, `thread_id`, `created_at`, `updated_at` on persist.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `session_id` | `str` | `""` | Short unique id, system-stamped |
| `thread_id` | `str` | `""` | Thread the session belongs to |
| `description` | `str` | `""` | What the session is about |
| `goal` | `str \| None` | `None` | Optional free-form objective |
| `state` | `dict[str, JsonValue]` | `{}` | Session state values |
| `state_meta` | `dict[str, str]` | `{}` | Maps state keys to plain-language meanings |
| `instructions` | `list[str]` | `[]` | Model-authored session program |
| `control` | `SessionControl \| None` | `None` | Machine-readable session rules |
| `template_id` | `str \| None` | `None` | Originating template id |
| `status` | `str` | `"active"` | Session lifecycle status |
| `created_at` | `float` | `0.0` | Unix timestamp of creation |
| `updated_at` | `float` | `0.0` | Unix timestamp of last update |

### ExecuteCodeInput / ExecuteCodeOutput
Code execution request and result.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `code` | `str` | — | Python code to execute |
| `timeout` | `int \| None` | `60` | Execution timeout in seconds |
| `final_output` | `str` | — | Captured stdout+stderr, capped |
| `success` | `bool` | — | Whether execution completed without error |
| `execution_time` | `float \| None` | `None` | Wall-clock time in seconds |

### PackageProposal
A package to install/propose for a learning session.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `package_name` | `str` | — | Package name |
| `version` | `str \| None` | `None` | Optional version constraint |
| `reason` | `str` | — | Why this package is needed |

### QueryResult
Web search result.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `confidence` | `float` | — | Search confidence score |
| `information` | `str` | — | Search result text |
| `queries` | `list[str]` | — | Queries that produced this result |
| `reasoning` | `str` | — | Reasoning for the query selection |