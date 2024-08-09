from pydantic_settings import BaseSettings
import yaml

class Settings(BaseSettings):
    GRAPHRAG_LLM_MODEL: str
    GRAPHRAG_EMBEDDING_MODEL: str
    GRAPHRAG_CLAIM_EXTRACTION_ENABLED: bool
    INPUT_DIR: str
    COMMUNITY_LEVEL: int
    GRAPHRAG_API_KEY: str

    class Config:
        env_file = ".env"
        

def load_settings_from_yaml(yaml_file: str) -> Settings:
    with open(yaml_file, 'r', encoding='utf-8') as file:
        config_dict = yaml.safe_load(file)
    return Settings(**config_dict)
