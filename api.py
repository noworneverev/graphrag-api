from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tiktoken
import uvicorn

from graphrag.query.structured_search.global_search.search import GlobalSearch
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.query.structured_search.drift_search.search import DRIFTSearch
from graphrag.query.structured_search.global_search.community_context import GlobalCommunityContext
from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext
from graphrag.query.structured_search.drift_search.drift_context import DRIFTSearchContextBuilder
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.vector_stores.lancedb import LanceDBVectorStore
from graphrag.config.models.drift_search_config import DRIFTSearchConfig
from graphrag.query.indexer_adapters import (
    read_indexer_covariates,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_report_embeddings,
    read_indexer_reports,
    read_indexer_text_units,
    read_indexer_communities
)

from utils import load_parquet_files, convert_response_to_string, process_context_data, serialize_search_result
from settings import load_settings_from_yaml
from services.llm import setup_llm_and_embedder

_ = load_dotenv()

settings = load_settings_from_yaml("settings.yml")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://noworneverev.github.io"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm, text_embedder = setup_llm_and_embedder(settings)
token_encoder = tiktoken.get_encoding("cl100k_base")

INPUT_DIR = settings.INPUT_DIR
LANCEDB_URI = f"{INPUT_DIR}/lancedb"
COMMUNITY_LEVEL = settings.COMMUNITY_LEVEL
CLAIM_EXTRACTION_ENABLED = settings.GRAPHRAG_CLAIM_EXTRACTION_ENABLED
RESPONSE_TYPE = settings.RESPONSE_TYPE

entity_df, entity_embedding_df, report_df, relationship_df, covariate_df, text_unit_df, community_df = load_parquet_files(INPUT_DIR, CLAIM_EXTRACTION_ENABLED)
entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)
relationships = read_indexer_relationships(relationship_df)
claims = read_indexer_covariates(covariate_df) if CLAIM_EXTRACTION_ENABLED else []
reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)
text_units = read_indexer_text_units(text_unit_df)
communities = read_indexer_communities(community_df, entity_df, report_df)

description_embedding_store = LanceDBVectorStore(
    collection_name="default-entity-description",
)
description_embedding_store.connect(db_uri=LANCEDB_URI)

full_content_embedding_store = LanceDBVectorStore(
    collection_name="default-community-full_content",
)
full_content_embedding_store.connect(db_uri=LANCEDB_URI)

def setup_drift_search() -> DRIFTSearch:
    drift_reports = read_indexer_reports(
        report_df,
        entity_df,
        COMMUNITY_LEVEL,
        content_embedding_col="full_content_embeddings",
    )
    read_indexer_report_embeddings(drift_reports, full_content_embedding_store)

    drift_params = DRIFTSearchConfig(
        temperature=0,
        max_tokens=12_000,
        primer_folds=1,
        drift_k_followups=3,
        n_depth=3,
        n=1,
    )

    context_builder = DRIFTSearchContextBuilder(
        chat_llm=llm,
        text_embedder=text_embedder,
        entities=entities,
        relationships=relationships,
        reports=drift_reports,
        entity_text_embeddings=description_embedding_store,
        text_units=text_units,
        token_encoder=token_encoder,
        config=drift_params
    )

    return DRIFTSearch(
        llm=llm, context_builder=context_builder, token_encoder=token_encoder
    )    

def setup_global_search() -> GlobalSearch:
    try:
        token_encoder = tiktoken.encoding_for_model(settings.GRAPHRAG_LLM_MODEL)
    except KeyError:        
        token_encoder = tiktoken.get_encoding("cl100k_base")
    
    context_builder = GlobalCommunityContext(
        community_reports=reports,
        communities=communities,
        entities=entities,
        token_encoder=token_encoder,
    )

    return GlobalSearch(
        llm=llm,
        context_builder=context_builder,
        token_encoder=token_encoder,
        max_data_tokens=12_000,
        map_llm_params={
            "max_tokens": 1000,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        },
        reduce_llm_params={
            "max_tokens": 2000,
            "temperature": 0.0,
        },
        allow_general_knowledge=False,
        json_mode=True,
        context_builder_params={
            "use_community_summary": False,
            "shuffle_data": True,
            "include_community_rank": True,
            "min_community_rank": 0,
            "community_rank_name": "rank",
            "include_community_weight": True,
            "community_weight_name": "occurrence weight",
            "normalize_community_weight": True,
            "max_tokens": 12_000,
            "context_name": "Reports",
        },
        concurrent_coroutines=32,
        response_type=RESPONSE_TYPE,
    )    

def setup_local_search() -> LocalSearch:
    context_builder = LocalSearchMixedContext(
        community_reports=reports,
        text_units=text_units,
        entities=entities,
        relationships=relationships,
        covariates={"claims": claims} if CLAIM_EXTRACTION_ENABLED else None,
        entity_text_embeddings=description_embedding_store,
        embedding_vectorstore_key=EntityVectorStoreKey.ID,
        text_embedder=text_embedder,
        token_encoder=token_encoder,
    )

    return LocalSearch(
        llm=llm,
        context_builder=context_builder,
        token_encoder=token_encoder,
        llm_params={
            "max_tokens": 2_000,
            "temperature": 0.0,
        },
        context_builder_params={
            "text_unit_prop": 0.5,
            "community_prop": 0.1,
            "conversation_history_max_turns": 5,
            "conversation_history_user_turns_only": True,
            "top_k_mapped_entities": 10,
            "top_k_relationships": 10,
            "include_entity_rank": True,
            "include_relationship_weight": True,
            "include_community_rank": False,
            "return_candidate_context": False,
            "embedding_vectorstore_key": EntityVectorStoreKey.ID,
            "max_tokens": 12_000,
        },
        response_type=RESPONSE_TYPE,
    )    

global_search_engine = setup_global_search()
local_search_engine = setup_local_search()
drift_search_engine = setup_drift_search()

@app.get("/search/drift")
async def drift_search(query: str = Query(..., description="DRIFT search query")):
    try:
        result = await drift_search_engine.asearch(query)        
        response_dict = {
            "response": convert_response_to_string(result.response),
            "context_data": process_context_data(result.context_data),
            "context_text": result.context_text,
            "completion_time": result.completion_time,
            "llm_calls": result.llm_calls,
            "prompt_tokens": result.prompt_tokens,            
        }
        return JSONResponse(content=response_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/global")
async def global_search(query: str = Query(..., description="Search query for global context")):
    try:
        result = await global_search_engine.asearch(query)        
        response_dict = {
            "response": convert_response_to_string(result.response),
            "context_data": process_context_data(result.context_data),
            "context_text": result.context_text,
            "completion_time": result.completion_time,
            "llm_calls": result.llm_calls,
            "prompt_tokens": result.prompt_tokens,
            "reduce_context_data": process_context_data(result.reduce_context_data),
            "reduce_context_text": result.reduce_context_text,
            "map_responses": [serialize_search_result(result) for result in result.map_responses],
        }
        return JSONResponse(content=response_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/local")
async def local_search(query: str = Query(..., description="Search query for local context")):
    try:
        result = await local_search_engine.asearch(query)        
        response_dict = {
            "response": convert_response_to_string(result.response),
            "context_data": process_context_data(result.context_data),
            "context_text": result.context_text,
            "completion_time": result.completion_time,
            "llm_calls": result.llm_calls,
            "prompt_tokens": result.prompt_tokens,            
        }
        return JSONResponse(content=response_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def status():
    return JSONResponse(content={"status": "Server is up and running"})


if __name__ == "__main__":    
    uvicorn.run(app, host="0.0.0.0", port=8000)
