from pydantic_settings import BaseSettings
import yaml
from pathlib import Path

class Settings(BaseSettings):
    GRAPHRAG_LLM_TYPE: str # openai_chat or azure_openai_chat
    OPENAI_API_KEY: str
    AZURE_OPENAI_API_KEY: str = None
    AZURE_ENDPOINT: str = None
    AZURE_LLM_DEPLOYMENT: str = None
    AZURE_EMBEDDINGS_DEPLOYMENT: str = None
    AZURE_API_VERSION: str = None
    GRAPHRAG_LLM_MODEL: str
    GRAPHRAG_EMBEDDING_MODEL: str
    GRAPHRAG_CLAIM_EXTRACTION_ENABLED: bool
    INPUT_DIR: str
    COMMUNITY_LEVEL: int

    RESPONSE_TYPE: str    

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
