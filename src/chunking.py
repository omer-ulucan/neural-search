"""Token-aware chunking that maps token windows back onto the original text."""

from collections.abc import Callable
from dataclasses import dataclass

# Returns (start_char, end_char) for every token of `text`, without special tokens.
OffsetsFn = Callable[[str], list[tuple[int, int]]]


@dataclass(frozen=True)
class Chunk:
    """A contiguous span of a document."""

    index: int
    text: str
    start_char: int
    end_char: int
    n_tokens: int


def chunk_text(text: str, token_offsets: OffsetsFn, chunk_size: int, overlap: int) -> list[Chunk]:
    """Split text into windows of at most `chunk_size` tokens sharing `overlap` tokens.

    Windows are cut on token boundaries and sliced from the original string, so chunk
    text is never re-detokenized. Empty or whitespace-only text yields no chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")
    offsets = [o for o in token_offsets(text) if o[1] > o[0]]
    if not offsets:
        return []

    step = chunk_size - overlap
    chunks: list[Chunk] = []
    start = 0
    while True:
        end = min(start + chunk_size, len(offsets))
        start_char, end_char = offsets[start][0], offsets[end - 1][1]
        chunks.append(
            Chunk(
                index=len(chunks),
                text=text[start_char:end_char],
                start_char=start_char,
                end_char=end_char,
                n_tokens=end - start,
            )
        )
        if end == len(offsets):
            return chunks
        start += step
