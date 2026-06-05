# Local Hybrid Neural Search Engine

A production-ready, fully-local hybrid neural search engine that combines dense embeddings (BGE), sparse lexical embeddings (SPLADE), and LLM-powered retrieval-augmented generation (RAG). Queries are automatically routed to a private Qdrant knowledge base or to the web via SearXNG using a LangGraph state machine.

## Architecture

```
                    +-------------+
                    |   User      |
                    +------+------+
                           |
                           v
              +------------------------+
              |  FastAPI /search       |
              +-----------+------------+
                          |
                          v
              +------------------------+
              |  LangGraph StateGraph  |
              |  - router_node         |
              |  - local_search_node   |
              |  - web_search_node     |
              |  - generator_node      |
              +-----------+------------+
                          |
            +-------------+-------------+
            |                           |
            v                           v
   +----------------+        +------------------+
   | Qdrant (local) |        | SearXNG (web)    |
   | Hybrid RRF     |        | Google + Bing    |
   | Dense + Sparse |        | JSON API         |
   +--------+-------+        +--------+---------+
            |                           |
            +-------------+-------------+
                          |
                          v
               +---------------------+
               | llama.cpp Local LLM |
               | ChatML RAG Prompt   |
               +---------------------+
```

## Prerequisites

- Docker & Docker Compose
- Python 3.11+
- A local GGUF model file (e.g., any llama.cpp-compatible model)
  - Recommended quantization: **Q4_K_M** (good balance of speed and quality)
  - **Note:** `MODEL_PATH` must be an absolute path to the `.gguf` file.

## Setup

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd neural-search-engine
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the supporting services:
   ```bash
   docker compose up -d
   ```

4. Set the required environment variable:
   ```bash
   # Linux / macOS
   export MODEL_PATH=/absolute/path/to/your-model-Q4_K_M.gguf

   # Windows (PowerShell)
   $env:MODEL_PATH = "C:\\absolute\\path\\to\\your-model-Q4_K_M.gguf"
   ```

## Ingest Documents

Place `.txt` files into a directory and run the ingestion script:

```bash
python scripts/ingest.py --docs-dir ./your-docs
```

Each file is embedded with dense (BGE) and sparse (SPLADE) vectors and stored in Qdrant.

## Run the API

```bash
uvicorn src.main:app --reload
```

The API will be available at `http://localhost:8000`.

## Example Usage

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I configure the search engine?"}'
```

Response:

```json
{
  "answer": "Based on the provided context, ...",
  "routing": "local",
  "sources": [
    "[1] Configuration guide ...",
    "[2] README section ..."
  ]
}
```

## Health Check

```bash
curl http://localhost:8000/health
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | *(required)* | Absolute path to the local GGUF model file |
| `QDRANT_HOST` | `http://localhost:6333` | Qdrant REST API endpoint |
| `COLLECTION_NAME` | `technical_docs` | Qdrant collection name |
| `SEARXNG_HOST` | `http://localhost:8080` | SearXNG instance URL |
| `DENSE_DIM` | `1024` | Dimensionality of dense vectors |
| `N_CTX` | `8192` | LLM context window size |
| `N_THREADS` | `6` | CPU threads for llama.cpp |
| `N_GPU_LAYERS` | `0` | GPU layers to offload (set to `-1` for all layers) |
