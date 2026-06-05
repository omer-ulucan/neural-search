from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    QDRANT_HOST: str = "http://localhost:6333"
    COLLECTION_NAME: str = "technical_docs"
    MODEL_PATH: str  # must be set via env var MODEL_PATH
    SEARXNG_HOST: str = "http://localhost:8080"
    DENSE_DIM: int = 1024
    N_CTX: int = 8192
    N_THREADS: int = 6
    N_GPU_LAYERS: int = 0


settings = Settings()
