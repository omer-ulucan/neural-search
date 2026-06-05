import logging

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForMaskedLM, AutoTokenizer

logger = logging.getLogger(__name__)


class LocalHybridEmbedder:
    """Hybrid embedder combining BGE dense and SPLADE sparse vectors."""

    def __init__(self) -> None:
        """Initialize dense and sparse embedding models."""
        logger.info("Loading dense embedding model: BAAI/bge-large-en-v1.5")
        self.dense_model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu")
        logger.info("Loading sparse embedding model: naver/splade-cocondenser-ensembledistil")
        self.sparse_tokenizer = AutoTokenizer.from_pretrained("naver/splade-cocondenser-ensembledistil")
        self.sparse_model = AutoModelForMaskedLM.from_pretrained(
            "naver/splade-cocondenser-ensembledistil",
            torch_dtype=torch.float32,
        )
        self.sparse_model.eval()

    def get_dense(self, text: str) -> list[float]:
        """Encode text into a dense vector using BGE with L2 normalization."""
        embedding = self.dense_model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def get_sparse(self, text: str) -> dict[int, float]:
        """Extract a SPLADE log-relu sparse vector with nonzero filtering."""
        inputs = self.sparse_tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            output = self.sparse_model(**inputs)
        activations = torch.log(1 + torch.relu(output.logits))
        attention_mask = inputs["attention_mask"]
        masked = activations * attention_mask.unsqueeze(-1)
        vec = masked.max(dim=1).values.squeeze()
        nonzero = torch.nonzero(vec, as_tuple=False)
        result = {int(idx): float(vec[idx]) for idx in nonzero.flatten()}
        if not result:
            raise ValueError("SPLADE produced empty sparse vector")
        return result

