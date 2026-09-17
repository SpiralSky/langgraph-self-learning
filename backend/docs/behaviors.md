# Behaviors

Source: `src/graphs/behaviors.py` (`BehaviorGroup`, `BehaviorPoint`,
`load_behaviors`, `render_behaviors`), config: `backend/behaviors.yaml`

## What it is

Behaviors are the **guidance** the `GeneratorNode` should follow when it builds
a graph for a user request — stored as **data**, never as prompt text.

- Behaviors live in a sectioned YAML file: a list of
  `{title, points[]}` groups. Each point is either a plain string or a
  `{id, text}` mapping; `text` is what gets rendered.
- Every point carries a **stable key** — an explicit `id` from the file, or an
  auto-assigned positional key `<title-slug>-p<index>` (1-based) for plain
  strings. Keys exist so later tooling can target individual entries
  (e.g. patch-based improvement).
- `render_behaviors` is the **only** surface that turns the data into prompt
  lines (`## <title>` / `- <point>`). The core generator prompt template lives
  in code and is never rewritten from data.
- This is a deliberate split: changing how the model is guided is a **data
  edit**, while the prompt machinery itself stays fixed and testable.

## File format

`load_behaviors` reads a top-level YAML **list of groups** from
`backend/behaviors.yaml` by default:

```yaml
- title: Explain concepts
  points:
    - id: explain-why
      text: When asked "why" or "explain", give a short reasoning before answering.
    - text: Prefer concrete examples over abstract definitions.
- title: Answer directly
  points:
    - text: Answer the user's question directly without extra chatter.
    - id: cite-uncertainty
      text: When unsure, say so and note what is missing.
```

- Points without an explicit `id` get an auto-assigned stable key
  (`explain-concepts-p2` above). Duplicate keys anywhere in the file raise
  `ValueError`.
- A missing/empty file loads as `[]`; any other shape (non-list root, missing
  titles, malformed points) raises `TypeError`/`ValueError` naming the file.

## API

```python
from graphs.behaviors import BehaviorGroup, load_behaviors, render_behaviors

groups: list[BehaviorGroup] = load_behaviors()          # backend/behaviors.yaml
groups = load_behaviors("/some/other/file.yaml")        # explicit path
groups = [
    BehaviorGroup(title="Explain concepts", points=[
        BehaviorPoint(id="explain-why", text="Give a short reasoning first."),
    ]),
]

text: str = render_behaviors(groups)
# "## Explain concepts\n- Give a short reasoning first."
```

- `load_behaviors(path=None) -> list[BehaviorGroup]` — never `None`; an empty
  or absent file yields `[]`.
- `render_behaviors(groups) -> str` — renders in order; groups with no points
  are skipped; empty input renders to the empty string.

## Behavior

- **Data not prompt** — the YAML holds *what* to do, not model instructions.
  The `GeneratorNode` concatenates the fixed template with the rendered
  behaviors at call time; editing `behaviors.yaml` never touches that
  template.
- **Stable keys** — effective `id` values are explicit when present, otherwise
  the deterministic `<title-slug>-p<idx>` form, with `"group"` as the slug
  fallback for non-alphanumerics-only titles.

## Tests

- `tests/test_behaviors.py` (16 tests) — loading shapes/errors, auto-keys,
  duplicates, rendering, empty behavior.

The suite is pure-unit — run from `backend/`:
`.venv/bin/python -m pytest tests/ -q`.