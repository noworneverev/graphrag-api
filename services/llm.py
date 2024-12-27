from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.llm.oai.typing import OpenaiApiType

from settings import Settings


def setup_llm_and_embedder(settings: Settings):
    """
    Initialize the ChatOpenAI and OpenAIEmbedding instances based on the settings.

    Args:
        settings: The settings object containing configuration details.

    Returns:
        A tuple of (llm, text_embedder).
    """
    common_params = {
        "max_retries": 20,
    }
        
    if settings.GRAPHRAG_LLM_TYPE == "openai_chat":
        llm = ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            api_base=settings.GRAPHRAG_LLM_API_BASE,
            model=settings.GRAPHRAG_LLM_MODEL,
            api_type=OpenaiApiType.OpenAI,
            **common_params,
        )
        text_embedder = OpenAIEmbedding(
            api_key=settings.OPENAI_API_KEY,
            api_base=settings.GRAPHRAG_EMBEDDING_API_BASE,
            api_type=OpenaiApiType.OpenAI,
            model=settings.GRAPHRAG_EMBEDDING_MODEL,
            deployment_name=settings.GRAPHRAG_EMBEDDING_MODEL,
            **common_params,
        )
    else:        
        llm = ChatOpenAI(
            api_base=settings.AZURE_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_API_VERSION,
            deployment_name=settings.AZURE_LLM_DEPLOYMENT,
            model=settings.GRAPHRAG_LLM_MODEL,
            api_type=OpenaiApiType.AzureOpenAI,
            **common_params,
        )
        text_embedder = OpenAIEmbedding(
            api_base=settings.AZURE_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_type=OpenaiApiType.AzureOpenAI,
            api_version=settings.AZURE_API_VERSION,
            model=settings.GRAPHRAG_EMBEDDING_MODEL,
            deployment_name=settings.AZURE_EMBEDDINGS_DEPLOYMENT,
            **common_params,
        )

    return llm, text_embedder
