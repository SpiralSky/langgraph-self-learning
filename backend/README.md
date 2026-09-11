#### Warning: Mainly Vibe-Coded

The base graph was written completely by me, but everything after revision `ef63dd40ebc4a8aa1b5d78fbdc67199cfdcdb675`
is vibe-coded.

Feel free to use this project as you wish, fork or make a better one. I honestly really hope someone makes a *actually good implementation* of this.

Please.

#### Details
Basically tries to reduce shortcoming of AI learning:
AI does not feed you information directly, instead asking questions to lead the learner.

Planned to add support with code so the learner can run code for learning. After all, to properly learn something,
you need to either: explain it in your own words (possibly given a scenario) or build it yourself.

#### Documentation

See ``docs/`` for developer/operator documentation:

| Doc | Covers |
|-----|--------|
| ``docs/architecture.md`` | System overview, graph flow, state keys, config files, node registry |
| ``docs/state-and-models.md`` | LearningGraphState fields, all 10 pydantic models |
| ``docs/config-and-compiler.md`` | Config system, prompt compiler, LLM init, memory setup |
| ``docs/nodes.md`` | All 9 nodes: purpose, inputs, outputs, key logic |
| ``docs/infrastructure.md`` | Observability, text_utils, session_types, execute_code infra |
| ``docs/tests-and-eval.md`` | Test structure, fixtures, conventions, eval setup |u