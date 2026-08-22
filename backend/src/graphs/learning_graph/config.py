from pathlib import Path
from typing import Optional

from pydantic import BaseModel
from yaml import safe_load


PROJECT_ROOT = Path(__file__).resolve().parents[3]

# TODO Possibly change api key to reference ENV variable instead
class ProviderData(BaseModel):
    """
    Provider Data class containing api endpoint, key, model id and an optional
    reasoning/thinking level.

    ``thinking_level`` is forwarded to the provider as ``reasoning_effort`` for
    reasoning-capable models (e.g. ``low``, ``medium``, ``high``). When omitted
    the provider's default thinking behavior applies.
    """
    api_endpoint: str
    api_key: str
    model_id: str
    thinking_level: Optional[str] = None

class Config(BaseModel):
    provider_data: dict[str, ProviderData]
    model_providers: dict[str, str]
    memory_top_k: int = 5
    memory_max_chars: int = 300
    session_min_overlap: int = 1
    session_max_ledger_entries: int = 50

    def get_model_data(self, name: str) -> ProviderData:
        """
        Used to get model provider data, such as API key, endpoint and model id.
        :param name: Name of node/configuration using the model.
        :return: ProviderData class.
        """
        model_name = self.model_providers[name]
        return self.provider_data[model_name]

def read_config(path: Path) -> Config:
    """
    Reads config.yaml to get model data such as: API key, endpoint and model id.
    :return: Dictionary mapping model names to model data.
    """
    with open(path) as stream:
        yaml = safe_load(stream)

        try:
            parsed_config = Config(**yaml)
            return parsed_config
        except Exception as e:
            raise ValueError(f"Failed to parse config: {e}")


config: Config = read_config(PROJECT_ROOT / "config.yaml")

