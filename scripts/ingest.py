import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import settings  # noqa: E402
from src.database import init_qdrant  # noqa: E402
from src.embedder import LocalHybridEmbedder  # noqa: E402
from src.ingestion import ingest_documents, read_documents  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> None:
    """Ingest .txt and .md documents into the hybrid neural search engine."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Ingest .txt/.md documents into Qdrant")
    parser.add_argument("--docs-dir", required=True, help="Directory of .txt/.md files")
    parser.add_argument("--collection", default=settings.COLLECTION_NAME)
    parser.add_argument("--chunk-size", type=int, default=settings.CHUNK_SIZE, help="Tokens")
    parser.add_argument("--chunk-overlap", type=int, default=settings.CHUNK_OVERLAP)
    parser.add_argument("--batch-size", type=int, default=64, help="Chunks per upsert")
    parser.add_argument(
        "--no-prune",
        action="store_true",
        help="Keep chunks of documents that are no longer in --docs-dir",
    )
    args = parser.parse_args()

    docs = read_documents(args.docs_dir)
    embedder = LocalHybridEmbedder()
    client = init_qdrant(collection=args.collection, dense_dim=embedder.dense_dim)
    stats = ingest_documents(
        client,
        args.collection,
        docs,
        embedder,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
        prune=not args.no_prune,
    )
    logger.info(
        "Ingestion complete: %d docs (%d embedded, %d unchanged, %d pruned), "
        "%d chunks upserted, %d stale chunks deleted",
        len(docs),
        stats.embedded,
        stats.unchanged,
        stats.pruned,
        stats.chunks_upserted,
        stats.stale_chunks_deleted,
    )
    for source in stats.empty:
        logger.warning("Skipped empty document: %s", source)


if __name__ == "__main__":
    main()
