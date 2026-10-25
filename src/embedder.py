import logging

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.config import settings

logger = logging.getLogger(__name__)

MAX_SEQ_LEN = 512


def resolve_device(device: str | None = None) -> str:
    """Map "auto" to "cuda" when a CUDA GPU is visible, otherwise "cpu"."""
    device = (device or settings.DEVICE).strip().lower()
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"DEVICE={device} but torch sees no CUDA GPU (CPU-only torch build?)")
    return device


class LocalHybridEmbedder:
    """Hybrid embedder combining BGE dense and SPLADE sparse vectors."""

    def __init__(
        self,
        dense_model: str | None = None,
        sparse_model: str | None = None,
        device: str | None = None,
    ) -> None:
        """Initialize dense and sparse embedding models."""
        self.dense_model_name = dense_model or settings.DENSE_MODEL
        self.sparse_model_name = sparse_model or settings.SPARSE_MODEL
        self.device = resolve_device(device)
        logger.info("Loading dense embedding model: %s on %s", self.dense_model_name, self.device)
        self.dense_model = SentenceTransformer(self.dense_model_name, device=self.device)
        self.dense_dim = self.dense_model.get_sentence_embedding_dimension()
        logger.info("Loading sparse embedding model: %s", self.sparse_model_name)
        self.sparse_tokenizer = AutoTokenizer.from_pretrained(self.sparse_model_name)
        self.sparse_model = AutoModelForMaskedLM.from_pretrained(
            self.sparse_model_name,
            torch_dtype=torch.float32,
        ).to(self.device)
        self.sparse_model.eval()

    def token_offsets(self, text: str) -> list[tuple[int, int]]:
        """Return character offsets of the dense tokenizer's tokens, used for chunking."""
        enc = self.dense_model.tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True, verbose=False
        )
        return [tuple(o) for o in enc["offset_mapping"]]

    def get_dense(self, text: str) -> list[float]:
        """Encode text into a dense vector using BGE with L2 normalization."""
        return self.encode_dense([text])[0]

    def encode_dense(self, texts: list[str], batch_size: int | None = None) -> list[list[float]]:
        """Encode a batch of texts into L2-normalized BGE vectors."""
        embeddings = self.dense_model.encode(
            texts,
            batch_size=batch_size or settings.EMBED_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def get_sparse(self, text: str) -> dict[int, float]:
        """Extract a SPLADE log-relu sparse vector with nonzero filtering. May be empty."""
        return self.encode_sparse([text])[0]

    def encode_sparse(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[dict[int, float]]:
        """Encode a batch of texts into SPLADE vectors. A vector may be empty."""
        batch_size = batch_size or settings.EMBED_BATCH_SIZE
        # Sorting by length keeps padding, and therefore wasted compute, low.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out: list[dict[int, float]] = [{} for _ in texts]
        for b in range(0, len(order), batch_size):
            idx = order[b : b + batch_size]
            inputs = self.sparse_tokenizer(
                [texts[i] for i in idx],
                padding=True,
                truncation=True,
                max_length=MAX_SEQ_LEN,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = self.sparse_model(**inputs).logits
            activations = torch.log1p(torch.relu(logits))
            masked = activations * inputs["attention_mask"].unsqueeze(-1)
            vecs = masked.max(dim=1).values.cpu()
            for row, i in enumerate(idx):
                nz = torch.nonzero(vecs[row], as_tuple=True)[0]
                out[i] = dict(zip(nz.tolist(), vecs[row, nz].tolist(), strict=True))
        return out
