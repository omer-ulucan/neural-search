import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import init_qdrant, insert_document
from src.embedder import LocalHybridEmbedder

logger = logging.getLogger(__name__)


def _read_text_file(fpath: str) -> str:
    """Read a text file trying utf-8, utf-8-sig, and latin-1 encodings."""
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with open(fpath, "r", encoding=enc) as fh:
                return fh.read()
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"Could not decode {fpath} with any supported encoding")


def main() -> None:
    """Ingest .txt documents into the hybrid neural search engine."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Ingest text documents into Qdrant")
    parser.add_argument("--docs-dir", required=True, help="Directory containing .txt files to ingest")
    args = parser.parse_args()

    embedder = LocalHybridEmbedder()
    client = init_qdrant()

    doc_id = 0
    for fname in sorted(os.listdir(args.docs_dir)):
        if not fname.endswith(".txt"):
            continue
        fpath = os.path.join(args.docs_dir, fname)
        text = _read_text_file(fpath)
        logger.info("Processing %s (%d chars)", fname, len(text))
        dense_vec = embedder.get_dense(text)
        sparse_vec = embedder.get_sparse(text)
        insert_document(
            client=client,
            doc_id=doc_id,
            text=text,
            metadata={"filename": fname},
            dense_vec=dense_vec,
            sparse_vec=sparse_vec,
        )
        logger.info("Inserted %s as doc_id=%d", fname, doc_id)
        doc_id += 1

    logger.info("Ingestion complete. Total documents: %d", doc_id)


if __name__ == "__main__":
    main()
