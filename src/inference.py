import logging
import re
import threading

from src.config import settings

logger = logging.getLogger(__name__)

_CHANNEL_MARKERS = re.compile(r"<\|channel\>.*?<channel\|>", flags=re.DOTALL)

# Gemma 4 control tokens (ids 105 and 106 in the GGUF vocab). Gemma 3's <start_of_turn> and
# <end_of_turn> are not special tokens in Gemma 4 and would reach the model as plain text.
TURN_START = "<|turn>"
TURN_END = "<turn|>"
# An empty thought block after the model header disables thinking, as the GGUF's chat
# template does for enable_thinking=false. Otherwise the model opens a thought channel
# first and a short max_tokens budget (the router uses 10) runs out before the answer.
NO_THINKING = "<|channel>thought\n<channel|>"
STOP = [TURN_END, TURN_START]


def gemma_prompt(user_content: str) -> str:
    """Wrap one user message in the Gemma 4 chat format, with thinking disabled.

    No <bos>: llama-cpp-python prepends it itself (the GGUF sets add_bos_token), and a
    second one degrades output.
    """
    return f"{TURN_START}user\n{user_content.strip()}{TURN_END}\n{TURN_START}model\n{NO_THINKING}"


def strip_channel_markers(text: str) -> str:
    """Remove Gemma 4 thinking-channel blocks, e.g. <|channel>thought ...<channel|>."""
    return _CHANNEL_MARKERS.sub("", text).strip()


INSUFFICIENT_CONTEXT = "The provided context does not contain enough information to answer this."

RAG_INSTRUCTIONS = (
    "Answer the question using ONLY the numbered context passages below.\n"
    "- Cite every claim with the passage number in square brackets, e.g. [1] or [2][3].\n"
    "- Do not use outside knowledge and do not invent passage numbers.\n"
    f'- If the passages do not contain the answer, reply exactly: "{INSUFFICIENT_CONTEXT}" '
    "You may add what the passages do cover, with citations.\n"
)


def format_context(n: int, doc: dict) -> str:
    """Render one numbered passage with its title and location."""
    where = doc.get("source") or ""
    if doc.get("chunk_index") is not None:
        where = f"{where}, chunk {doc['chunk_index']}"
    header = f"[{n}] {doc.get('title') or 'Untitled'}" + (f" ({where})" if where else "")
    return f"{header}\n{doc.get('text', '')}"


def build_rag_prompt(query: str, contexts: list[dict]) -> str:
    """Build a Gemma 4 chat prompt with grounding rules and numbered, attributed passages."""
    context_block = "\n\n".join(format_context(i + 1, d) for i, d in enumerate(contexts))
    return gemma_prompt(f"{RAG_INSTRUCTIONS}\nContext:\n{context_block}\n\nQuestion: {query}")


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
        # A llama.cpp context is not thread-safe; concurrent requests must take turns.
        self._lock = threading.Lock()
        self.llm = Llama(
            model_path=model_path,
            n_ctx=settings.N_CTX,
            n_threads=settings.N_THREADS,
            n_gpu_layers=settings.N_GPU_LAYERS,
            verbose=False,
        )

    def build_rag_prompt(self, query: str, contexts: list[dict]) -> str:
        """Build a Gemma 4 chat prompt with grounding rules and numbered passages."""
        return build_rag_prompt(query, contexts)

    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 1024) -> str:
        """Generate text from the LLM with stop sequences and error handling."""
        try:
            with self._lock:
                output = self.llm(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=STOP,
                )
        except Exception:
            logger.exception("LLM generation failed")
            raise
        return strip_channel_markers(output["choices"][0]["text"])
