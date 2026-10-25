"""Query-dependent RRF weights.

Exact-match signals (identifiers, code, paths, error strings, quoted text) are what
a lexical model like SPLADE handles best, while paraphrased natural-language
questions favour the dense model. The rule is deliberately simple so every decision
can be explained by the reasons it returns.
"""

import re

from src.retrieval import Weights

LEXICAL_SIGNALS: dict[str, re.Pattern] = {
    "quoted": re.compile(r"\"[^\"]{2,}\"|`[^`]+`"),
    "path": re.compile(r"(?:^|\s)(?:~|\.{1,2})?/[\w.-]+|\b[a-zA-Z]:\\|\w+/\w+[\w./-]*/"),
    "filename": re.compile(
        r"(?:^|\s)\.[a-z]{2,}\b|\b[\w-]+\.(?:py|js|ts|php|conf|ini|ya?ml|json|xml|html?|css|"
        r"sh|log|cfg|toml|c|h|cpp|java|go|rs|rb|sql|txt|md|lock)\b"
    ),
    "code": re.compile(r"\w+\(\)|::|->|=>|\b\w+\.\w+\(|[{};<>$]|\b\w+=\S"),
    "identifier": re.compile(r"\b[a-z0-9]+_[a-z0-9_]+\b|\b[a-z]+[A-Z][a-zA-Z0-9]*\b"),
    "constant": re.compile(r"\b[A-Z][A-Z0-9]*_[A-Z0-9_]+\b"),
    "flag": re.compile(r"(?:^|\s)--?[a-zA-Z][\w-]*"),
    "error_code": re.compile(r"\b(?:[A-Z]{1,6}-?\d{3,}|0x[0-9a-fA-F]+|[45]\d\d)\b"),
    "version": re.compile(r"\bv?\d+\.\d+(?:\.\d+)+\b"),
}
QUESTION_START = re.compile(
    r"^(how|what|why|when|where|which|who|can|could|should|is|are|does|do|will|would)\b", re.I
)
MIN_NATURAL_WORDS = 6


def lexical_signals(query: str) -> list[str]:
    """Names of the exact-match signals present in the query."""
    return [name for name, pattern in LEXICAL_SIGNALS.items() if pattern.search(query)]


class QueryWeighting:
    """Lean RRF towards sparse for exact-match queries and towards dense for prose.

    strength s moves weight between retrievers: lexical queries get (1 - s, 1 + s),
    natural-language questions (1 + s, 1 - s), and anything else stays (1, 1).
    """

    def __init__(self, strength: float) -> None:
        if not 0.0 <= strength <= 1.0:
            raise ValueError("strength must be in [0, 1]")
        self.strength = strength

    def __call__(self, query: str) -> Weights:
        s = self.strength
        signals = lexical_signals(query)
        if signals:
            return Weights(1.0 - s, 1.0 + s, tuple(f"lexical:{name}" for name in signals))
        words = query.split()
        if QUESTION_START.match(query.strip()) or len(words) >= MIN_NATURAL_WORDS:
            return Weights(1.0 + s, 1.0 - s, ("natural_language",))
        return Weights(1.0, 1.0, ("neutral",))
