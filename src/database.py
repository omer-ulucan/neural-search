import logging

from qdrant_client import QdrantClient, models

from src.config import settings

logger = logging.getLogger(__name__)


def init_qdrant() -> QdrantClient:
    """Initialize Qdrant client and create collection if it does not exist."""
    client = QdrantClient(settings.QDRANT_HOST, check_compatibility=False)
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


def insert_document(
    client: QdrantClient,
    doc_id: int,
    text: str,
    metadata: dict,
    dense_vec: list[float],
    sparse_vec: dict[int, float],
) -> None:
    """Upsert a single document with dense and sparse vectors into Qdrant."""
    point = models.PointStruct(
        id=doc_id,
        vector={
            "": dense_vec,
            "lexical-sparse": models.SparseVector(
                indices=list(sparse_vec.keys()),
                values=list(sparse_vec.values()),
            ),
        },
        payload={**metadata, "text": text},
    )
    client.upsert(
        collection_name=settings.COLLECTION_NAME,
        points=[point],
    )
