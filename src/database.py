import logging

from qdrant_client import QdrantClient, models

from src.config import settings

logger = logging.getLogger(__name__)

DENSE_VECTOR = ""
SPARSE_VECTOR = "lexical-sparse"


def init_qdrant(
    client: QdrantClient | None = None,
    collection: str | None = None,
    dense_dim: int | None = None,
) -> QdrantClient:
    """Initialize Qdrant client and create collection if it does not exist."""
    client = client or QdrantClient(settings.QDRANT_HOST)
    collection = collection or settings.COLLECTION_NAME
    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(
                size=dense_dim or settings.DENSE_DIM,
                distance=models.Distance.COSINE,
            ),
            sparse_vectors_config={
                SPARSE_VECTOR: models.SparseVectorParams(),
            },
        )
        client.create_payload_index(
            collection, field_name="source", field_schema=models.PayloadSchemaType.KEYWORD
        )
        logger.info("Created Qdrant collection: %s", collection)
    else:
        logger.info("Qdrant collection already exists: %s", collection)
    return client


def to_sparse_vector(sparse_vec: dict[int, float]) -> models.SparseVector:
    """Convert an {index: weight} mapping into a Qdrant sparse vector."""
    return models.SparseVector(indices=list(sparse_vec.keys()), values=list(sparse_vec.values()))
