import logging

from llama_cpp import Llama

from src.config import settings

logger = logging.getLogger(__name__)


class LocalLLM:
    """Local LLM inference wrapper using llama.cpp."""

    def __init__(self) -> None:
        """Initialize the Llama model with configured parameters."""
        logger.info("Loading LLM from: %s", settings.MODEL_PATH)
        self.llm = Llama(
            model_path=settings.MODEL_PATH,
            n_ctx=settings.N_CTX,
            n_threads=settings.N_THREADS,
            n_gpu_layers=settings.N_GPU_LAYERS,
            verbose=False,
        )
