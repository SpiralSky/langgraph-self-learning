from pathlib import Path

from pydantic import BaseModel
from sqlalchemy.testing.plugin.plugin_base import read_config
from yaml import safe_load


PROJECT_ROOT = Path(__file__).resolve().parents[3]

# TODO Possibly change api key to reference ENV variable instead
class ProviderData(BaseModel):
    """
    Provider Data class containing api endpoint, key and model id.
    """
    api_endpoint: str
    api_key: str
    model_id: str

class Config(BaseModel):
    provider_data: dict[str, ProviderData]
    model_providers: dict[str, str]

    def get_model_data(self, name: str) -> ProviderData:
        """
        Used to get model provider data, such as API key, endpoint and model id.
        :param name: Name of node/configuration using the model.
        :return: ProviderData class.
        """
        model_name = self.model_providers[name]
        return self.provider_data[model_name]


config: Config = read_config(PROJECT_ROOT / "config.yaml")

def read_config(path: Path) -> Config | None:
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
            print(f"Failed to parse config: {e}")