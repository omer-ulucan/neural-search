from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    QDRANT_HOST: str = "http://localhost:6333"
    COLLECTION_NAME: str = "technical_docs"
    MODEL_PATH: str = ""  # GGUF path; required for /search, not for /retrieve, eval or bench
    SEARXNG_HOST: str = "http://localhost:8080"
    DENSE_MODEL: str = "BAAI/bge-large-en-v1.5"
    SPARSE_MODEL: str = "naver/splade-cocondenser-ensembledistil"
    DENSE_DIM: int = 1024
    CHUNK_SIZE: int = 320  # tokens, below the 512-token limit of BGE and SPLADE
    CHUNK_OVERLAP: int = 48  # tokens, 15% of CHUNK_SIZE
    EMBED_BATCH_SIZE: int = 16
    RRF_K: int = 60  # RRF smoothing constant, unrelated to how many results are returned
    TOP_K: int = 5  # results returned (and passed to the LLM as context)
    CANDIDATE_DEPTH: int = 50  # results fetched from each retriever before fusion
    DENSE_WEIGHT: float = 1.0
    SPARSE_WEIGHT: float = 1.0
    DYNAMIC_WEIGHTS: bool = False  # query-dependent weights, see src/weighting.py
    DYNAMIC_STRENGTH: float = 0.5
    N_CTX: int = 8192
    N_THREADS: int = 6
    N_GPU_LAYERS: int = 0


settings = Settings()
