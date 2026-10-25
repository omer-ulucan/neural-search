import logging
import re

from src.config import settings

logger = logging.getLogger(__name__)

_CHANNEL_MARKERS = re.compile(r"<\|channel\>.*?<channel\|>", flags=re.DOTALL)


def strip_channel_markers(text: str) -> str:
    """Remove Gemma 4 thinking-channel blocks, e.g. <|channel>thought ...<channel|>."""
    return _CHANNEL_MARKERS.sub("", text).strip()


def build_rag_prompt(query: str, contexts: list[str]) -> str:
    """Build a Gemma 4 chat prompt with numbered context injection."""
    context_block = "\n\n".join(f"[{i + 1}] {ctx}" for i, ctx in enumerate(contexts))
    return (
        f"<start_of_turn>user\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {query}\n"
        f"<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )


class LocalLLM:
    """Local LLM inference wrapper using llama.cpp."""

    def __init__(self, model_path: str | None = None) -> None:
        """Initialize the Llama model with configured parameters."""
        from llama_cpp import Llama

        model_path = model_path or settings.MODEL_PATH
        if not model_path:
            raise ValueError("MODEL_PATH is not set; point it at a GGUF file")
        logger.info("Loading LLM from: %s", model_path)
        self.model_path = model_path
        self.llm = Llama(
            model_path=model_path,
            n_ctx=settings.N_CTX,
            n_threads=settings.N_THREADS,
            n_gpu_layers=settings.N_GPU_LAYERS,
            verbose=False,
        )

    def build_rag_prompt(self, query: str, contexts: list[str]) -> str:
        """Build a Gemma 4 chat prompt with numbered context injection."""
        return build_rag_prompt(query, contexts)

    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 1024) -> str:
        """Generate text from the LLM with stop sequences and error handling."""
        try:
            output = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["<end_of_turn>", "<start_of_turn>"],
            )
        except Exception:
            logger.exception("LLM generation failed")
            raise
        return strip_channel_markers(output["choices"][0]["text"])
