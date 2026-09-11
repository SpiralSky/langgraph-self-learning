from pathlib import Path

from pydantic import BaseModel, Field
from yaml import safe_load

from graphs.learning_graph.pydantic_models import SessionRecord

SESSION_TYPES_PATH = Path(__file__).resolve().parents[2] / "config" / "graph_config.yaml"


class SessionType(BaseModel):
    """
    One declared session template loaded from ``graph_config.yaml``.

    The YAML is *guidance only*: a type listed here provides a starter
    ``system_block`` and suggested ``state_keys``, but the model may always
    author its own session freely (its description, goal, state, and
    ``state_meta``). The catalog only anchors the input analyzer's trigger
    vocabulary and the optional hint block.

    :ivar label: Human-readable display name.
    :vartype label: str
    :ivar summary: One-line description used in prompts and session hints.
    :vartype summary: str
    :ivar triggers: Comma-separated phrases that commonly indicate this type.
    :vartype triggers: str
    :ivar state_keys: State keys the type suggests (not enforced).
    :vartype state_keys: list[str]
    :ivar system_block: Template instruction block spliced when no authored
        plan exists.
    :vartype system_block: str
    :ivar is_main: Whether the type is offered to the input analyzer.
    :vartype is_main: bool
    """

    label: str
    summary: str
    triggers: str = ""
    state_keys: list[str] = Field(default_factory=list)
    system_block: str = ""
    is_main: bool = Field(default=True, alias="main")

    model_config = {"populate_by_name": True}


def _load_session_types(path: Path) -> dict[str, SessionType]:
    """
    Load and parse the session-type catalog from YAML.

    Skips the ``customizations`` top-level key and malformed entries.
    Returns ``{}`` if the file is missing; raises ``ValueError`` on
    parse failures for known types.

    :param path: Absolute path to the session-types YAML file
        (typically ``graph_config.yaml``).
    :type path: Path
    :return: Mapping of type id to its parsed :class:`SessionType`.
    :rtype: dict[str, SessionType]
    :raises ValueError: If a known type cannot be parsed.
    """
    try:
        with open(path) as stream:
            data = safe_load(stream) or {}
    except FileNotFoundError:
        return {}

    types: dict[str, SessionType] = {}
    for type_id, raw in data.items():
        if type_id == "customizations":
            continue
        if not isinstance(raw, dict):
            continue
        try:
            types[str(type_id)] = SessionType(**raw)
        except Exception:
            continue
    return types


session_types: dict[str, SessionType] = _load_session_types(SESSION_TYPES_PATH)


def analyzer_catalog_text() -> str:
    """
    Render the compact main-type catalog for the input analyzer prompt.

    Only ``is_main`` types are listed, each as ``id: summary`` (plus trigger
    phrases when declared) so the analyzer can pick a matching type without
    paying for every niche type's detail. The analyzer is also told it may
    invent an id for an improvised format.

    :return: Newline-joined catalog lines, or ``"(none)"`` when empty.
    :rtype: str
    """
    lines = []
    for type_id, session_type in session_types.items():
        if not session_type.is_main:
            continue
        detail = session_type.summary
        if session_type.triggers:
            detail += f"; cues: {session_type.triggers}"
        lines.append(f"- {type_id}: {detail}")
    return "\n".join(lines) if lines else "(none)"