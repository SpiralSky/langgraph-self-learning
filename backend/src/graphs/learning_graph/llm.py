import logging

from langchain.chat_models import init_chat_model
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from graphs.learning_graph.config import config

logger = logging.getLogger(__name__)


def get_chat_model(name: str, temperature: float = 0.0):
    """
    Build a configured chat model for a node/phase.

    Reads the provider data for ``name`` from the shared config and forwards an
    optional ``thinking_level`` (if set) to the provider as ``reasoning_effort``.
    Missing ``thinking_level`` keeps the provider's default thinking behavior.

    :param name: Config key naming the model usage (e.g. ``"response_builder"``).
    :type name: str
    :param temperature: Sampling temperature to pass to the model.
    :type temperature: float
    :return: An initialized chat model runnable.
    """
    provider_data = config.get_model_data(name)

    kwargs: dict = {
        "api_key": provider_data.api_key,
        "base_url": provider_data.api_endpoint,
        "temperature": temperature,
        "model_provider": "openai",
    }

    if provider_data.thinking_level:
        kwargs["model_kwargs"] = {"reasoning_effort": provider_data.thinking_level}
        logger.info(
            "Applying thinking_level=%s to model %s",
            provider_data.thinking_level,
            name,
        )

    return init_chat_model(provider_data.model_id, **kwargs)


def get_structured_model(
    name: str,
    schema: type[BaseModel],
    temperature: float = 0.0,
) -> Runnable:
    """
    Build a chat model constrained to emit a given structured output schema.

    :param name: Config key naming the model usage.
    :type name: str
    :param schema: Pydantic model describing the expected structured output.
    :type schema: type[BaseModel]
    :param temperature: Sampling temperature to pass to the model.
    :type temperature: float
    :return: The chat model wrapped with structured-output parsing.
    :rtype: Runnable
    """
    return get_chat_model(name, temperature=temperature).with_structured_output(schema)