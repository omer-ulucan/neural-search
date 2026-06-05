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

    def build_rag_prompt(self, query: str, contexts: list[str]) -> str:
        """Build a ChatML prompt with numbered context injection."""
        system_msg = (
            "You are a precise technical assistant. Answer based strictly on the provided context. "
            "If the context is insufficient, say so explicitly."
        )
        context_block = "\n\n".join(
            f"[{i + 1}] {ctx}" for i, ctx in enumerate(contexts)
        )
        prompt = (
            f"<|im_start|>system\n{system_msg}<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {query}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        return prompt

    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 1024) -> str:
        """Generate text from the LLM with stop sequences and error handling."""
        try:
            output = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["</s>", "User:", "\n\nUser"],
            )
            text = output["choices"][0]["text"]
            return text.strip()
        except Exception as exc:
            logger.exception("LLM generation failed")
            raise

