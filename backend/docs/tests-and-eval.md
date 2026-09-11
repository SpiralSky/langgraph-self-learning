# Tests & Evaluation

## Test Structure

18 test files live in ``tests/``, one per module plus a graph-level file:

| File | Covers |
|------|--------|
| ``test_learning_graph.py`` | Graph edges, export structure, ``@timed`` wrapping |
| ``test_config.py`` | ``read_config``, ``ProviderData``, ``Config`` resolution |
| ``test_pydantic_models.py`` | All 10 pydantic models |
| ``test_learning_graph.py`` | Graph edges, export, timed decoration |
| ``test_user_input.py`` | ``user_input`` node |
| ``test_input_analyzer.py`` | ``input_analyzer`` node |
| ``test_retrieve_memory.py`` | ``retrieve_memory`` node |
| ``test_decision_maker.py`` | ``decision_maker`` node |
| ``test_response_builder.py`` | ``response_builder`` node |
| ``test_save_memory.py`` | ``save_memory`` node |
| ``test_format_output.py`` | ``format_output`` node |
| ``test_model_output.py`` | ``model_output`` node |
| ``test_execute_code.py`` | ``execute_code`` node |
| ``test_node.py`` | ``Node`` class, ``@node`` decorator |
| ``test_observability.py`` | ``timed`` decorator |
| ``test_text_utils.py`` | ``message_to_text`` |
| ``test_session_types.py`` | ``SessionType``, ``_load_session_types``, ``analyzer_catalog_text`` |
| ``test_compiler.py`` | Prompt compiler |
| ``test_integration.py`` | End-to-end graph invocation: invocation, multi-turn, full turn, scenarios, execute_code branch, error handling |

## Integration Tests

``tests/test_integration.py`` exercises the graph end-to-end via `graph.invoke()` (13 tests, class `TestGraphInvocation`). They cover:

- **Graph invocation** — returns a state dict, `thread_id` routing, multi-turn state/session persistence
- **Full turn** — populates all state fields; memory search called; state threaded between nodes
- **Scenarios** — explanation, question, propose_session
- **execute_code branch** — routes when `pedagogical_move="execute_code"`; `_ssh_exec` is mocked
- **Error handling** — missing `user_message` and empty `messages` raise

**Run:**
```bash
pytest tests/test_integration.py -v
```

Uses `mock_llm` (patches `get_chat_model`) and `mock_mem0` fixtures; no external services required.

## Conftest Fixtures

All fixtures are in ``conftest.py`` (session-scoped where possible):

| Fixture | Scope | Provides |
|---------|-------|----------|
| ``config`` | session | Full ``Config`` from ``config.yaml`` |
| ``state`` | function | Empty ``LearningGraphState`` |
| ``state_with_message`` | function | State with a user message |
| ``state_with_analysis`` | function | Adds ``InputAnalysisResult`` |
| ``state_with_strategy`` | function | Adds ``TeachingStrategy`` |
| ``state_full_turn`` | function | Adds draft, final output, search results, session |
| ``runnable_config`` | function | ``RunnableConfig`` with ``thread_id`` |
| ``mock_mem0`` | function | Patches ``graphs.learning_graph.memory.memory`` |
| ``mock_llm`` | function | Patches ``graphs.learning_graph.llm.get_chat_model`` |

## Test Conventions

- Framework: ``pytest`` + ``pytest-asyncio`` + ``pytest-mock``
- Snapshot testing: ``syrupy``
- HTTP: ``httpx``
- Each test class groups related assertions; fixtures build up state incrementally (``state`` → ``state_with_message`` → ``state_with_analysis`` → …).

## Evaluation Setup

The ``automatic-evaluation/`` folder contains the LLM-as-judge harness:

| File | Purpose |
|------|---------|
| ``run_evaluation.py`` | CLI entry point (argparse) |
| ``scenario_generator.py`` | Generates learner scenarios |
| ``conversation_runner.py`` | Runs tutor↔learner conversations |
| ``evaluator.py`` | ``EnsembleEvaluator`` — scores transcripts |
| ``report.py`` | Markdown report generation |
| ``pydantic_models.py`` | Eval-specific models (``Scenario``, ``EvaluationResult``, …) |
| ``config.yaml`` | Eval config (models, rubric, counts) |

### Ensemble

Three free models from different families for bias reduction:

| Model | Family |
|---|---|
| ``nvidia/nemotron-3-ultra-550b-a55b:free`` | NVIDIA |
| ``minimax/minimax-m3:free`` | MiniMax |
| ``thinkingmachines/inkling:free`` | Thinking Machines |

### Rubric Dimensions

8 pedagogical dimensions scored 1–5 per evaluator. Reports include ensemble scores (mean ± std), per-evaluator breakdown, inter-evaluator agreement, best/worst scenarios, and qualitative commentary.

### Running Evaluation

```bash
uv run run-evaluation --config config.yaml --scenarios 5 --turns 3
```

## Running Tests

```bash
pytest tests/                    # all tests
pytest tests/test_integration.py -v  # integration tests only
```

Requires a working Python environment with the project dependencies installed (``uv sync`` or ``pip install -e .``). The ``config.yaml`` must be present with valid provider keys for non-mocked tests; integration tests use ``mock_mem0`` / ``mock_llm`` fixtures and need no external services.
