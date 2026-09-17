"""LLM model initialization for graph nodes."""

from langchain_openai import ChatOpenAI

from config import get_model_settings_cached


def get_chat_model(name: str, temperature: float = 0.0) -> ChatOpenAI:
    """Build a configured chat model for a named logical model.

    Reads the ``models`` entry named ``name`` from ``model_settings.yaml``
    (cached) and resolves it to the concrete provider + model id: the model's
    ``model`` override when set, else the provider's ``default_model``. Any
    provider ``base_url`` is forwarded, so OpenAI-compatible gateways work.

    :param name: Config key naming the logical model (e.g. ``"fast"``).
    :type name: str
    :param temperature: Sampling temperature to pass to the model.
    :type temperature: float
    :return: An initialized ``ChatOpenAI`` runnable.
    :rtype: ChatOpenAI
    """
    settings = get_model_settings_cached()
    model_cfg = settings.models.get(name)
    if model_cfg is None:
        raise KeyError(f"no model named {name!r} in model_settings.yaml")

    provider = settings.providers[model_cfg.provider]
    model_id = model_cfg.model or provider.default_model

    kwargs: dict = {
        "model": model_id,
        "temperature": temperature,
        "api_key": provider.api_key,
    }
    if provider.base_url:
        kwargs["base_url"] = provider.base_url

    return ChatOpenAI(**kwargs)