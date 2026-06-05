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
              |  Dark Mode UI          |
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
               | Gemma 4 12B Q4_K_M  |
               | Gemma Chat Template |
               +---------------------+
```

## Prerequisites

- Docker & Docker Compose
- Python 3.11+
- ~8GB disk space for the model

## Setup

### 1. Clone the Repository

```bash
git clone <repo-url>
cd neural-search-engine
```

### 2. Create and Activate Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate it
# Linux/macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Important:** The requirements include:
- `numpy<2` (required for sentence-transformers compatibility)
- `llama-cpp-python>=0.3.25` (for Gemma 4 architecture support)
- `qdrant-client>=1.18.0` (for query_points API)
- `hf_xet` (for faster HuggingFace downloads)

### 4. Download the Gemma 4 Model

The model is automatically downloaded on first run, or you can download it manually:

```bash
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='ggml-org/gemma-4-12B-it-GGUF',
    filename='gemma-4-12B-it-Q4_K_M.gguf',
    local_dir='./models'
)
"
```

This downloads the **Gemma 4 12B Instruct** model (Q4_K_M quantization, ~7GB).

### 5. Configure Environment Variables

The `.env` file is automatically loaded by the application. Edit `.env` to set your model path:

```env
MODEL_PATH=C:\Users\your_username\path\to\neural-search-engine\models\gemma-4-12B-it-Q4_K_M.gguf
```

**Note:** `MODEL_PATH` must be an **absolute path** to the `.gguf` file.

### 6. Start Docker Services

```bash
docker compose up -d
```

This starts:
- **Qdrant v1.12.0** on `http://localhost:6333` (vector database)
- **SearXNG** on `http://localhost:8080` (web search fallback)

### 7. Ingest Documents

Place `.txt` files into a directory and run the ingestion script:

```bash
python scripts/ingest.py --docs-dir ./my-docs
```

The script automatically:
- Detects file encoding (UTF-8, UTF-8-sig, Latin-1)
- Generates dense embeddings (BGE) and sparse embeddings (SPLADE)
- Stores vectors in Qdrant with hybrid RRF support

### 8. Run the API Server

```bash
python -m uvicorn src.main:app --host 127.0.0.1 --port 8000
```

The server will:
- Load the Gemma 4 model (~30 seconds)
- Initialize the LangGraph workflow
- Start serving on `http://localhost:8000`

## Web UI

Open your browser to `http://localhost:8000/` to access the dark-mode search interface.

**Features:**
- Dark mode design (background `#0f0f0f`, accent `#6366f1`)
- Real-time search with loading spinner
- Routing badge showing LOCAL (indigo) or WEB (amber)
- Collapsible source cards with expandable text
- Smooth fade-in animations
- JetBrains Mono font for results

## API Usage

### Search Endpoint

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Qdrant?"}'
```

**Response:**

```json
{
  "answer": "Qdrant is an open-source vector database that supports hybrid search with dense and sparse vectors using reciprocal rank fusion...",
  "routing": "local",
  "sources": [
    "Qdrant is a vector database that supports hybrid search...",
    "Reciprocal rank fusion is a method for combining..."
  ]
}
```

### Health Check

```bash
curl http://localhost:8000/health
# Returns: {"status": "ok"}
```

## Environment Variables

All variables can be set in `.env` or exported directly. The `.env` file is auto-loaded.

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | *(required)* | Absolute path to the Gemma 4 GGUF model |
| `QDRANT_HOST` | `http://localhost:6333` | Qdrant REST API endpoint |
| `COLLECTION_NAME` | `technical_docs` | Qdrant collection name |
| `SEARXNG_HOST` | `http://localhost:8080` | SearXNG instance URL |
| `DENSE_DIM` | `1024` | Dimensionality of dense vectors (BGE-large) |
| `N_CTX` | `8192` | LLM context window size |
| `N_THREADS` | `6` | CPU threads for llama.cpp |
| `N_GPU_LAYERS` | `0` | GPU layers to offload (set to `-1` for all) |

## Troubleshooting

### Model Loading Issues

If you see `Failed to load model from file`:
- Ensure `MODEL_PATH` in `.env` is an **absolute path**
- Verify the model file exists and is not corrupted
- Check that `llama-cpp-python>=0.3.25` is installed (required for Gemma 4)

### NumPy Version Conflict

If you see `NumPy 2.x cannot be run in NumPy 1.x`:
```bash
pip install "numpy<2" --force-reinstall
```

### Qdrant Client/Server Version Mismatch

If you see version mismatch warnings:
- Ensure Qdrant Docker is v1.12.0 or higher
- Ensure `qdrant-client>=1.18.0` is installed

### SearXNG Web Search Fails

If web search returns empty results:
- Check that `searxng/settings.yml` has `use_default_settings: true`
- Ensure `formats: [html, json]` is set in the search section
- Restart SearXNG: `docker compose restart searxng`

### Ingestion Script Import Errors

If you see `ModuleNotFoundError: No module named 'src'`:
- Ensure you're running from the project root directory
- The script automatically adds the project root to `sys.path`

## Project Structure

```
neural-search-engine/
├── docker-compose.yml          # Qdrant + SearXNG services
├── requirements.txt            # Pinned dependencies
├── .env                        # Environment variables (auto-loaded)
├── .gitignore                  # Git ignore rules
├── README.md                   # This file
├── models/                     # GGUF model files
│   └── gemma-4-12B-it-Q4_K_M.gguf
├── searxng/                    # SearXNG configuration
│   └── settings.yml
├── scripts/
│   └── ingest.py               # Document ingestion script
└── src/
    ├── __init__.py
    ├── config.py               # Pydantic settings with .env support
    ├── embedder.py             # BGE dense + SPLADE sparse embeddings
    ├── database.py             # Qdrant client and operations
    ├── inference.py            # llama.cpp wrapper with Gemma 4 support
    ├── graph.py                # LangGraph workflow (router, search, generator)
    ├── main.py                 # FastAPI app with lifespan
    └── static/
        └── index.html          # Dark mode web UI
```

## Key Features

- **Hybrid Search**: Combines dense (BGE) and sparse (SPLADE) vectors with RRF fusion
- **Smart Routing**: LLM-based router classifies queries as local or web
- **Local Knowledge**: Search your private documents in Qdrant
- **Web Fallback**: Automatic fallback to SearXNG for general knowledge
- **Gemma 4 12B**: State-of-the-art open-source LLM with ChatML-style prompting
- **Dark Mode UI**: Clean, minimal interface with smooth animations
- **Auto-Config**: `.env` file auto-loading, no manual exports needed
- **Robust Ingestion**: Multi-encoding support, automatic path handling

## License

MIT
