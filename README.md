# neural-search

Hybrid retrieval for technical documentation, fully local. Documents are chunked and indexed twice in Qdrant, as dense BGE-large embeddings and as SPLADE learned sparse vectors, and every query fuses both rankings with weighted Reciprocal Rank Fusion. On top of that retrieval core sits an answering layer: a LangGraph router sends each question to the local index or to a SearXNG web search, and a local Gemma model writes a cited answer from what was retrieved.

Retrieval quality is measured on a public benchmark (BEIR CQADupStack) and latency on the machine it ran on. Every number below is generated from `results/*.json` by a script; see [Reproduce](#reproduce).

## How retrieval works

```
                 .txt / .md documents
                          |
              token-aware chunking (320 tokens, 48 overlap)
              stable ids: uuid5(source#chunk_index)
                          |
          +---------------+----------------+
          |                                |
  BGE-large-en-v1.5                 SPLADE (cocondenser-
  1024-d dense, cosine              ensembledistil), sparse
          |                                |
          +------> Qdrant collection <-----+
                   dense HNSW index + sparse inverted index
                          |
   query --> encode both --> one batched Qdrant request
                          |
              dense ranking     sparse ranking
                    \              /
             weighted RRF: sum_i w_i / (k + rank_i), k = 10
                          |
                    fused ranking
```

- **Dense path.** `BAAI/bge-large-en-v1.5` via sentence-transformers, L2-normalised 1024-d vectors, cosine distance in Qdrant's HNSW index.
- **Sparse path.** `naver/splade-cocondenser-ensembledistil`, a learned sparse model that expands queries and documents over the BERT vocabulary. It is not BM25; BM25 appears only as a baseline in the evaluation.
- **Chunking.** Each document is split into windows of at most 320 tokens with 48 tokens (15%) overlap, measured with the same WordPiece vocabulary both models use, so neither model truncates. Chunks keep their original text and store `source`, `title`, `chunk_index` and character offsets.
- **Idempotent ingestion.** Point ids are `uuid5(source#chunk_index)` and every point carries a fingerprint of its document text plus chunking and model config. Re-running ingestion skips unchanged documents, re-embeds changed ones, deletes chunks a shrunken document no longer produces, and prunes documents that were removed from the folder.
- **Fusion.** Weighted RRF with 1-based ranks: `score(d) = sum_i w_i / (rrf_k + rank_i(d))`. `rrf_k` (default 10, selected on the dev split) is the smoothing constant and is independent of `top_k`, the number of results returned. A document missing from one ranking gets nothing from it, and ties are broken by best single-list rank, then id, so results are deterministic.
- **Why fusion is client-side.** Qdrant 1.20 has a parameterised RRF query, but probing it against a live server showed 0-based ranks (rank 1 in both lists scores 2/60, not 2/61) and weights that do not follow `w_i / (k + rank_i)`. Both per-retriever rankings come back from one batched request anyway, which is also what `/retrieve` returns.
- **Dynamic weights.** `src/weighting.py` leans the fusion towards SPLADE for queries with exact-match signals (identifiers, code, paths, file names, flags, error codes, versions, quoted text) and towards the dense model for natural-language questions, and records the reasons. It is on by default at strength 0.25 because it passed the evaluation's dev-split rule (beat equal weights on dev nDCG@10), but the margin is small and not significant on the test split; see [Results](#results). `DYNAMIC_WEIGHTS=false` gives plain static weights.
- **Empty sparse vectors.** If SPLADE maps a query to an empty vector, hybrid retrieval falls back to dense-only for that query and logs it.

## Answering layer

```
 POST /search --> LangGraph router (Gemma) --+--> local: hybrid retrieval (top 5 chunks)
                                             |
                                             +--> web: SearXNG (Google + Bing), top 3
                                                          |
                                Gemma 4 12B (llama.cpp) answers ONLY from the
                                numbered passages and cites them as [n]
```

The router asks Gemma for one word, `local` or `web`; anything else (including Gemma's thinking-channel markers or garbage) falls back to `local`. The answer prompt tells the model to use only the numbered passages, cite them as `[n]`, and say when the passages are not enough. `/search` returns the answer, the route, and the sources with title, location and chunk index, numbered to match the citations.

## Results

<!-- eval:start -->
Not measured yet. Run `python scripts/eval.py` and then this script.
<!-- eval:end -->

## Latency

<!-- bench:start -->
Not measured yet. Run `python scripts/bench.py` and then this script.
<!-- bench:end -->

`/search` latency depends on the route. The router sends general technical questions, which is most of CQADupStack, to the web route (see the Routing column), so those rows time a SearXNG search plus generation and move with the search engines' response time and the answer length. When SearXNG's engines are blocked the model answers that the passages are not enough, which is quick; see [Troubleshooting](#troubleshooting).

## Reproduce

```bash
pip install -r requirements-dev.txt
docker compose up -d qdrant

# Download BEIR CQADupStack (pinned revision), index it through the normal ingestion path,
# verify the index, tune on the dev split and report on the test split.
python scripts/eval.py --subforums webmasters        # writes results/eval.json

# Verify the indexed corpus, then time retrieval in process, /retrieve under load,
# and /search end to end. Needs the GGUF model for the last part (or pass --skip-e2e).
python scripts/bench.py --model-path models/gemma-4-12B-it-Q4_0.gguf   # writes results/bench.json

python scripts/make_readme_tables.py                 # regenerates the tables above
```

### On an NVIDIA GPU

Embedding uses CUDA automatically (`DEVICE=auto`) and the LLM offloads with `N_GPU_LAYERS=-1`, but both need CUDA builds of their libraries; the default Windows wheels are CPU-only.

```bash
# torch: the Linux default wheel already includes CUDA 12.1; on Windows install it explicitly
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-dev.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# llama-cpp-python with CUDA (needs the CUDA toolkit; on Windows also VS Build Tools)
CMAKE_ARGS="-DGGML_CUDA=on" pip install --force-reinstall --no-cache-dir "llama-cpp-python>=0.3.25"
# PowerShell: $env:CMAKE_ARGS="-DGGML_CUDA=on"; pip install --force-reinstall --no-cache-dir "llama-cpp-python>=0.3.25"
```

Then run eval and bench as above with `N_GPU_LAYERS=-1` in `.env`. On a 12 GB card, Gemma 4 12B Q4_0 (7.2 GB), its KV cache at `--e2e-n-ctx 4096`, and BGE-large plus SPLADE (about 2 GB) fit together. The bench records the GPU name, so latency from different machines is never mixed up.

`--subforums` accepts any CQADupStack subforum (for example `unix programmers webmasters`); results are averaged across them and every selection uses the mean dev score. Indexing is resumable: an interrupted run picks up where it stopped. On a CPU-only laptop, embedding the 17k-document webmasters corpus with BGE-large takes several hours.

## Setup

Requirements: Docker, Python 3.11 or 3.12, about 8 GB of disk for the LLM.

```bash
git clone <repo-url>
cd neural-search
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt                        # requirements-dev.txt adds eval and test tools
cp .env.example .env                                   # then set MODEL_PATH
docker compose up -d                                   # Qdrant v1.20.0 on :6333, SearXNG on :8080
```

Download the LLM (only needed for `/search`; `/retrieve`, eval and bench work without it):

```bash
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('ggml-org/gemma-4-12B-it-GGUF', 'gemma-4-12B-it-Q4_0.gguf', local_dir='models')
"
```

Set `MODEL_PATH` in `.env` to the absolute path of `models/gemma-4-12B-it-Q4_0.gguf`.

Ingest a folder of `.txt` and `.md` files (recursively) and start the API:

```bash
python scripts/ingest.py --docs-dir ./my-docs          # re-run any time; it is idempotent
python -m uvicorn src.main:app --host 127.0.0.1 --port 8000
```

Open `http://localhost:8000/` for the web UI.

## API

### `POST /retrieve`

Retrieval only, no router and no LLM. Returns the SPLADE ranking, the dense ranking and the fused ranking side by side, with raw scores, for a side-by-side view of the three. The bundled web UI at `/` uses `/search`.

```bash
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "nginx returns 502 behind a proxy", "top_k": 5}'
```

Request fields: `query` (required), `top_k` (1 to 50, default 10), `rrf_k` (optional override). Response shape:

```json
{
  "query": "...", "top_k": 5, "rrf_k": 10,
  "weights": {"dense": 0.75, "sparse": 1.25, "reasons": ["lexical:error_code"]},
  "sparse_fallback": false,
  "sparse": [{"rank": 1, "score": 15.4, "id": "...", "title": "...", "source": "...",
              "chunk_index": 0, "snippet": "...", "dense_rank": null, "sparse_rank": null}],
  "dense":  [...],
  "fused":  [{"rank": 1, "score": 0.1818, "...": "...", "dense_rank": 1, "sparse_rank": 1}],
  "timing_ms": {"dense_encode": 0.0, "sparse_encode": 0.0, "qdrant": 0.0, "fusion": 0.0, "total": 0.0}
}
```

Scores are raw: cosine similarity for dense, SPLADE dot product for sparse, RRF score for fused.

### `POST /search`

Router, retrieval or web search, and a cited answer. Returns `503` when no `MODEL_PATH` is configured.

```json
{
  "answer": "Enable gzip in the server config [1] ...",
  "routing": "local",
  "sources": [{"n": 1, "title": "...", "source": "speed.md", "chunk_index": 0, "text": "...", "score": 0.1761}]
}
```

### `GET /health`

Returns `{"status": "ok"}`.

## Configuration

All settings are environment variables, also read from `.env` (see `.env.example`).

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | empty | GGUF file for the router and answers; empty disables `/search` |
| `QDRANT_HOST` | `http://localhost:6333` | Qdrant URL |
| `QDRANT_TIMEOUT` | `60` | Seconds per Qdrant request; bulk upserts can outlast the client default of 5 s |
| `COLLECTION_NAME` | `technical_docs` | Collection the API reads and `ingest.py` writes |
| `SEARXNG_HOST` | `http://localhost:8080` | SearXNG URL for the web route |
| `DENSE_MODEL` | `BAAI/bge-large-en-v1.5` | Dense embedding model |
| `SPARSE_MODEL` | `naver/splade-cocondenser-ensembledistil` | Sparse embedding model |
| `DENSE_DIM` | `1024` | Dense vector size used when creating a collection |
| `CHUNK_SIZE` | `320` | Tokens per chunk |
| `CHUNK_OVERLAP` | `48` | Tokens shared by consecutive chunks |
| `EMBED_BATCH_SIZE` | `16` | Texts per model forward pass |
| `RRF_K` | `10` | RRF smoothing constant, selected on the dev split |
| `TOP_K` | `5` | Chunks returned to the LLM by `/search` |
| `CANDIDATE_DEPTH` | `50` | Results fetched from each retriever before fusion |
| `DENSE_WEIGHT`, `SPARSE_WEIGHT` | `1.0`, `1.0` | Static RRF weights |
| `DYNAMIC_WEIGHTS` | `true` | Use query-dependent weights instead of the static ones |
| `DYNAMIC_STRENGTH` | `0.25` | How far dynamic weights lean, 0 to 1, selected on the dev split |
| `CORS_ORIGINS` | `https://ulucan.dev` and localhost ports | Comma-separated browser origins |
| `N_CTX`, `N_THREADS`, `N_GPU_LAYERS` | `8192`, `6`, `0` | llama.cpp context, threads, GPU offload |

## Development

```bash
pip install -r requirements-dev.txt
pytest                          # no models, LLM or Qdrant server needed: fakes and in-memory Qdrant
ruff check . && ruff format --check .
```

## Project structure

```
neural-search/
├── docker-compose.yml          # Qdrant v1.20.0 + SearXNG
├── requirements*.txt           # runtime, eval/bench, dev
├── .env.example                # every setting, with defaults
├── results/                    # eval.json and bench.json, generated
├── scripts/
│   ├── ingest.py               # idempotent chunked ingestion of .txt/.md
│   ├── eval.py                 # BEIR CQADupStack evaluation
│   ├── bench.py                # latency benchmark
│   └── make_readme_tables.py   # README tables from results/*.json
├── src/
│   ├── config.py               # settings
│   ├── chunking.py             # token-aware chunker
│   ├── embedder.py             # BGE dense + SPLADE sparse, batched
│   ├── database.py             # Qdrant collection setup
│   ├── ingestion.py            # stable ids, fingerprints, stale-chunk cleanup
│   ├── retrieval.py            # dense / sparse / hybrid retrieval and weighted RRF
│   ├── weighting.py            # query-dependent weights (on by default)
│   ├── inference.py            # llama.cpp wrapper and RAG prompt
│   ├── graph.py                # LangGraph router, local and web search, generator
│   ├── evaluation.py           # dataset, split, metrics helpers
│   ├── benchmark.py            # percentiles, workloads, memory sampling
│   ├── main.py                 # FastAPI app: /retrieve, /search, /health
│   └── static/index.html       # web UI
└── tests/                      # pytest suite
```

## Troubleshooting

- **`/search` returns 503.** `MODEL_PATH` is empty. `/retrieve` works without it.
- **The server fails to start with a model error.** `MODEL_PATH` must be an absolute path to an existing `.gguf`, and `llama-cpp-python>=0.3.25` is needed for Gemma 4.
- **Qdrant version warnings or errors.** The server image and `qdrant-client` are both pinned to 1.20. A volume created by an older server (for example v1.12) is several minor versions behind; start from a fresh volume and re-run ingestion.
- **Web route returns nothing.** `searxng/settings.yml` must keep `use_default_settings: true` and the `json` format; restart with `docker compose restart searxng`.
- **Web answers say the passages are not enough.** The web route depends on SearXNG's Google and Bing engines, which search providers may block or degrade for your IP. Check what comes back with `curl "http://localhost:8080/search?q=htaccess+redirect&format=json&engines=google,bing"`; `unresponsive_engines` lists blocked engines, and off-topic titles mean the engine is not passing the full query through.
- **`NumPy 2.x cannot be run in NumPy 1.x`.** `pip install "numpy<2" --force-reinstall`.

## License

MIT
