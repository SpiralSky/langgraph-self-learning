from typing import Any, Optional


def requires_present(
    template: Optional[str] = None,
    **fields: Any,
) -> None:
    """
    Validate that all named fields carry a non-``None`` value.

    Collects the names of every ``field=value`` keyword argument whose value is
    ``None`` and, when any are missing, raises a :class:`ValueError`.
    With no ``template``, the default message lists the missing field names;
    pass ``template`` to override it, replacing each ``{}`` with the
    comma-separated list of missing names.

    :param template: Optional message template; ``{}`` is replaced by the
        comma-joined missing names. ``None`` selects a default message.
    :type template: Optional[str]
    :param fields: Field names mapped to their current values; only ``None``
        counts as missing.
    :type fields: Any
    :return: ``None`` when every field is present.
    :rtype: None
    :raises ValueError: When any field value is ``None``.
    """
    missing = [name for name, value in fields.items() if value is None]
    if not missing:
        return

    names = ", ".join(missing)
    message = template.format(names) if template else f"Missing required field(s): {names}"
    raise ValueError(message)