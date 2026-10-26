from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    QDRANT_HOST: str = "http://localhost:6333"
    QDRANT_TIMEOUT: int = 60  # seconds; the client default of 5 s times out bulk upserts
    COLLECTION_NAME: str = "technical_docs"
    MODEL_PATH: str = ""  # GGUF path; required for /search, not for /retrieve, eval or bench
    SEARXNG_HOST: str = "http://localhost:8080"
    DENSE_MODEL: str = "BAAI/bge-large-en-v1.5"
    SPARSE_MODEL: str = "naver/splade-cocondenser-ensembledistil"
    DENSE_DIM: int = 1024
    DEVICE: str = "auto"  # embedding device: auto (CUDA if available), cpu, cuda, cuda:1, ...
    CHUNK_SIZE: int = 320  # tokens, below the 512-token limit of BGE and SPLADE
    CHUNK_OVERLAP: int = 48  # tokens, 15% of CHUNK_SIZE
    EMBED_BATCH_SIZE: int = 16
    RRF_K: int = 10  # RRF smoothing constant (dev-selected), independent of TOP_K
    TOP_K: int = 5  # results returned (and passed to the LLM as context)
    CANDIDATE_DEPTH: int = 50  # results fetched from each retriever before fusion
    DENSE_WEIGHT: float = 1.0
    SPARSE_WEIGHT: float = 1.0
    DYNAMIC_WEIGHTS: bool = True  # query-dependent weights, see src/weighting.py
    DYNAMIC_STRENGTH: float = 0.25  # dev-selected with RRF_K, see results/eval.json
    # Comma-separated origins allowed to call the API from a browser (e.g. the live demo).
    CORS_ORIGINS: str = (
        "https://ulucan.dev,http://localhost:3000,http://localhost:5173,"
        "http://localhost:8000,http://127.0.0.1:8000"
    )
    N_CTX: int = 8192
    N_THREADS: int = 6
    N_GPU_LAYERS: int = 0

    @property
    def cors_origins(self) -> list[str]:
        """CORS_ORIGINS as a list, ignoring blanks."""
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


settings = Settings()
