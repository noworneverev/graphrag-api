from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from utils import process_context_data

from pathlib import Path
import graphrag.api as api
from graphrag.config.load_config import load_config
import pandas as pd
from config import PROJECT_DIRECTORY, COMMUNITY_LEVEL, CLAIM_EXTRACTION_ENABLED, RESPONSE_TYPE

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.config = load_config(Path(PROJECT_DIRECTORY))
    app.state.entities = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/entities.parquet")
    app.state.communities = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/communities.parquet")
    app.state.community_reports = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/community_reports.parquet")
    app.state.text_units = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/text_units.parquet")
    app.state.relationships = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/relationships.parquet")
    app.state.covariates = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/covariates.parquet") if CLAIM_EXTRACTION_ENABLED else None
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://noworneverev.github.io"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/search/global")
async def global_search(query: str = Query(..., description="Global Search")):
    try:
        response, context = await api.global_search(
                                config=app.state.config,
                                entities=app.state.entities,
                                communities=app.state.communities,
                                community_reports=app.state.community_reports,                                
                                community_level=COMMUNITY_LEVEL,
                                dynamic_community_selection=False,
                                response_type=RESPONSE_TYPE,
                                query=query,
                            )
        response_dict = {
            "response": response,
            "context_data": process_context_data(context),
        }
        return JSONResponse(content=response_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/local")
async def local_search(query: str = Query(..., description="Local Search")):
    try:
        response, context = await api.local_search(
                                config=app.state.config,
                                entities=app.state.entities,
                                communities=app.state.communities,
                                community_reports=app.state.community_reports,
                                text_units=app.state.text_units,
                                relationships=app.state.relationships,
                                covariates=app.state.covariates,
                                community_level=COMMUNITY_LEVEL,                                
                                response_type=RESPONSE_TYPE,
                                query=query,
                            )
        response_dict = {
            "response": response,
            "context_data": process_context_data(context),
        }        
        return JSONResponse(content=response_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/drift")
async def drift_search(query: str = Query(..., description="DRIFT Search")):
    try:
        response, context = await api.drift_search(
                                config=app.state.config,
                                entities=app.state.entities,
                                communities=app.state.communities,
                                community_reports=app.state.community_reports,
                                text_units=app.state.text_units,
                                relationships=app.state.relationships,
                                community_level=COMMUNITY_LEVEL,                                
                                response_type=RESPONSE_TYPE,
                                query=query,
                            )
        response_dict = {
            "response": response,
            "context_data": process_context_data(context),
        }
        return JSONResponse(content=response_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/basic")
async def basic_search(query: str = Query(..., description="Basic Search")):
    try:
        response, context = await api.basic_search(
                                config=app.state.config,
                                text_units=app.state.text_units,                                
                                query=query,
                            )
        response_dict = {
            "response": response,
            "context_data": process_context_data(context),
        }
        return JSONResponse(content=response_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def status():
    return JSONResponse(content={"status": "Server is up and running"})

if __name__ == "__main__":    
    uvicorn.run(app, host="0.0.0.0", port=8000)
