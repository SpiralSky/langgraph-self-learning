# Per-Node Documentation

All 9 nodes in the learning graph. Each is registered with `@node(prompt=..., intent=...)` and declared in `_NODES` — see [Architecture](architecture.md#node-registry).

## Node Reference

### user_input

- **Intent:** `receive`
- **Prompt:** none
- **Purpose:** Reads the latest message from the `messages` channel and stores it in `state.user_message`.
- **Inputs:** `state.messages` (non-empty list)
- **Outputs:** `{"user_message": <latest message>}`

### input_analyzer

- **Intent:** `analyze`
- **Prompt:** `INPUT_ANALYSER_PROMPT` (formatted with the session catalog)
- **Purpose:** Analyzes the user's question for clarity, intent, key concepts, prerequisite gaps, and session hints. Uses the `InputAnalysisResult` structured-output schema.
- **Key logic:**
  - Passes the last `RECENT_HISTORY_WINDOW = 3` turns before the current message so pronouns/references resolve.
  - Fills `prerequisite_gaps` only when there is a credible signal (explicit admission, expressed-as-question, or uncertainty across recent turns); never invents gaps.
  - Detects interactive session triggers (`step-by-step`, `teach me`, `walk me through`, `quiz me`, …) and sets `session_hint` + `suggest_session` accordingly.
- **Inputs:** `state.user_message`, `state.messages`
- **Outputs:** `{"analysis_results": InputAnalysisResult}`

### retrieve_memory

- **Intent:** `memory_read`
- **Prompt:** none
- **Purpose:** Searches mem0 for relevant past memories scoped to the thread, and loads the thread's active session record + knowledge model.
- **Key logic:**
  - Searches mem0 with `user_id = thread_id`, then compacts hits (drops bookkeeping entries, truncates to `memory_max_chars`).
  - Calls `memory.get_all` to find the active `SessionRecord` and `KnowledgeComponent` entries, normalizing across mem0 response shapes.
- **Inputs:** `state.user_message`, `config` (for `thread_id`)
- **Outputs:** `{"memory_results": list[dict], "active_session": SessionRecord | None, "knowledge_model": list[KnowledgeComponent]}`

### decision_maker

- **Intent:** `decide`
- **Prompt:** `PEDAGOGICAL_DIRECTOR_SYSTEM_PROMPT`
- **Purpose:** Acts as the pedagogical director, choosing the next teaching move and threading the (possibly updated) session context forward.
- **Key logic:**
  - `answer_side_quest` for foundational side-questions (may pause the session).
  - `pivot_to_goal` / `explain_concept` when the user says "continue".
  - `propose_session` when the analyzer set `suggest_session`.
  - Otherwise `explain_concept` or `ask_guiding_question`.
- **Inputs:** `state.user_message`, `state.memory_results`, `state.analysis_results`, `state.active_session`, `state.knowledge_model`
- **Outputs:** `{"teaching_strategy": TeachingStrategy}`

### response_builder

- **Intent:** `generate`
- **Prompt:** `RESPONSE_BUILDER_SYSTEM_PROMPT`
- **Purpose:** Renders the `TeachingStrategy.execution_plan` into a polished empathetic Markdown draft. Makes **no** pedagogical decisions — only formats what upstream decided.
- **Key logic:** Validates that `user_message` and `teaching_strategy` are present; builds context JSON and invokes the LLM with `ResponseBuilderOutput` structured output.
- **Inputs:** `state.user_message`, `state.teaching_strategy`
- **Outputs:** `{"draft_response": ResponseBuilderOutput}`

### save_memory

- **Intent:** `memory_write`
- **Prompt:** none
- **Purpose:** Persists the conversation exchange to mem0 and deterministically promotes knowledge-component statuses based on the session's `mastered_concepts`.
- **Key logic:**
  - Writes `[user_msg, assistant_msg]` pair to mem0 scoped by `thread_id`.
  - Walks `knowledge_model`; if a concept appears in `active_session.state.mastered_concepts`, promotes its status one rung up the ladder (`missing → rusty → inferred → mastered`).
  - The knowledge-model update is persisted via the graph's LangGraph checkpointer (available next turn); no extra `memory.add` call is made to avoid double-writing.
- **Inputs:** `state.user_message`, `state.final_output`, `state.knowledge_model`, `state.active_session`, `config` (for `thread_id`)
- **Outputs:** `{}` (write-only node)

### format_output

- **Intent:** `format`
- **Prompt:** `FORMAT_OUTPUT_SYSTEM_PROMPT`
- **Purpose:** The **only** node that may emit widget-fence markup (`:::code`, `:::text`). Converts the draft into presentable Markdown with LaTeX, bold/headings, and widget fences. Also normalizes any stray HTML widget markup to fences.
- **Key logic:** Invokes the LLM with `temperature=0.2`, then runs `normalize_widget_fences()` to convert any `<div data-widget="...">` into `:::` fences as a safety net.
- **Inputs:** `state.draft_response`
- **Outputs:** `{"final_output": str}`

### model_output

- **Intent:** `emit`
- **Prompt:** none
- **Purpose:** Wraps the final formatted text into an `AIMessage` appended to the `messages` channel.
- **Key logic:** Uses `state.final_output` when present; falls back to `state.draft_response.draft_response`.
- **Inputs:** `state.final_output`, `state.draft_response`
- **Outputs:** `{"messages": [AIMessage]}`

### execute_code

- **Intent:** `execute_code`
- **Prompt:** none
- **Purpose:** Executes a Python command in a code-runner container via SSH, returning the capped output to the graph.
- **Key logic:**
  - Resolves the SSH target from `sessions.json` registry (active session container) or falls back to `code-runner` pool.
  - Extracts the `python3 -c "..."` command from the execution plan step; falls back to the raw step text.
  - Runs via SSH with a 60s timeout, then caps stdout/stderr at `MAX_OUTPUT_LINES = 100` lines and `MAX_OUTPUT_BYTES = 10000` bytes.
  - Returns `{"final_output": <capped output>}` on success, or `{}` when no code step is found in the plan.
- **Inputs:** `state.teaching_strategy.execution_plan`, `config` (for thread_id)
- **Outputs:** `{"final_output": str}` (when a code step is found); `{}` otherwise
