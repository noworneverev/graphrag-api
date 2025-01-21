# GraphRAG API

This project provides a FastAPI-based server implementation for handling both global and local structured searches using GraphRAG.



## Installation

Clone the repository and install the required dependencies using the following commands:

```bash
git clone git@github.com:noworneverev/graphrag-api.git
cd graphrag-api
```

```bash
python -m venv venv
```

```bash
source venv/bin/activate # for Linux
venv\Scripts\activate # for Windows
```

```bash
pip install -r requirements.txt
```

## Usage

1. Copy the following files and the `lancedb` folder into the `artifacts` folder:
    ```
    lancedb
    create_final_communities.parquet
    create_final_community_reports.parquet
    create_final_nodes.parquet
    create_final_entities.parquet
    create_final_relationships.parquet
    create_final_covariates.parquet
    create_final_text_units.parquet
    ```
2. Run the API
    ```
    python api.py
    ```

Open http://127.0.0.1:8000/docs/ to see the API documentation.

You can also use the interface at [GraphRAG Visualizer](https://noworneverev.github.io/graphrag-visualizer/) to run queries against the server.

![search](static/image.png)

## API Endpoints
- `/search/global`: Perform a global search using GraphRAG.
- `/search/local`: Perform a local search using GraphRAG.
- `/search/drift`: Perform a DRIFT search using GraphRAG.
- `/status`: Check if the server is up and running.