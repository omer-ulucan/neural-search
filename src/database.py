import logging

from qdrant_client import QdrantClient, models

from src.config import settings

logger = logging.getLogger(__name__)


def init_qdrant() -> QdrantClient:
    """Initialize Qdrant client and create collection if it does not exist."""
    client = QdrantClient(settings.QDRANT_HOST)
    if not client.collection_exists(settings.COLLECTION_NAME):
        client.create_collection(
            collection_name=settings.COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=settings.DENSE_DIM,
                distance=models.Distance.COSINE,
            ),
            sparse_vectors_config={
                "lexical-sparse": models.SparseVectorParams(),
            },
        )
        logger.info("Created Qdrant collection: %s", settings.COLLECTION_NAME)
    else:
        logger.info("Qdrant collection already exists: %s", settings.COLLECTION_NAME)
    return client
