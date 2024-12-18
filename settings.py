from pydantic_settings import BaseSettings
import yaml
from typing import Optional

class Settings(BaseSettings):
    GRAPHRAG_LLM_TYPE: str = "openai_chat" # openai_chat or azure_openai_chat
    OPENAI_API_KEY: Optional[str] = None
    GRAPHRAG_LLM_MODEL: str = "gpt-4o"
    GRAPHRAG_LLM_API_BASE: Optional[str] = None
    GRAPHRAG_EMBEDDING_MODEL: str = "text-embedding-3-small"
    GRAPHRAG_EMBEDDING_API_BASE: Optional[str] = None

    AZURE_OPENAI_API_KEY: Optional[str] = None
    AZURE_ENDPOINT: Optional[str] = None
    AZURE_LLM_DEPLOYMENT: Optional[str] = None
    AZURE_EMBEDDINGS_DEPLOYMENT: Optional[str] = None
    AZURE_API_VERSION: Optional[str] = None

    GRAPHRAG_CLAIM_EXTRACTION_ENABLED: bool = False
    INPUT_DIR: str = "./artifacts"
    COMMUNITY_LEVEL: int = 2
    RESPONSE_TYPE: str = "single paragraph"

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        
def load_settings_from_yaml(yaml_file: str) -> Settings:
    """
    Load settings from a YAML file and override with environment variables.

    Args:
        yaml_file (str): Path to the YAML configuration file.        

    Returns:
        Settings: Pydantic Settings object with merged configurations.
    """
    with open(yaml_file, 'r', encoding='utf-8') as file:
        config_dict = yaml.safe_load(file)
    return Settings(**config_dict)
