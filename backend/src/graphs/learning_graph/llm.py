"""LLM model initialization for graph nodes."""

import logging
from typing import Optional

from langchain.chat_models import init_chat_model

from graphs.learning_graph.config import config

logger = logging.getLogger(__name__)


def get_chat_model(
    name: str,
    temperature: float = 0.0,
    model_override: Optional[str] = None,
):
    """
    Build a configured chat model for a node/phase.

    Reads the provider data for ``name`` from the shared config. Reasoning
    effort is the per-node override from config (``Config.reasoning_defaults``),
    falling back to the provider's static ``default_reasoning``. Models
    without declared reasoning never receive the parameter, keeping the
    provider's default thinking behavior.

    If ``model_override`` is provided, it takes precedence over the configured
    model_id, allowing evaluation runs to use a different model (e.g. a free
    model) without changing node logic.

    :param name: Config key naming the model usage (e.g. ``"response_builder"``).
    :type name: str
    :param temperature: Sampling temperature to pass to the model.
    :type temperature: float
    :param model_override: Optional model ID to override the configured default.
    :type model_override: Optional[str]
    :return: An initialized chat model runnable.
    :rtype: Any
    """
    provider_data = config.get_model_data(name)
    reasoning_effort = config.reasoning_effort_for(name)

    # Use model_override if set, otherwise use the configured model_id
    model_id = model_override or provider_data.model_id

    kwargs: dict = {
        "base_url": provider_data.api_endpoint,
        "temperature": temperature,
        "model_provider": "openai",
    }

    if provider_data.api_key and not provider_data.api_key.startswith("${"):
        kwargs["api_key"] = provider_data.api_key

    if reasoning_effort:
        kwargs["model_kwargs"] = {"reasoning_effort": reasoning_effort}
        logger.info(
            "Applying reasoning_effort=%s to model %s",
            reasoning_effort,
            name,
        )

    return init_chat_model(model_id, **kwargs)