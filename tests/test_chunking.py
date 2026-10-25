import re

import pytest

from src.chunking import chunk_text


def words(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in re.finditer(r"\S+", text)]


def make(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


@pytest.mark.parametrize("text", ["", "   \n\t  "])
def test_empty_text_has_no_chunks(text):
    assert chunk_text(text, words, chunk_size=4, overlap=1) == []


def test_short_text_is_one_chunk_without_surrounding_whitespace():
    chunks = chunk_text("  hello  world \n", words, chunk_size=8, overlap=2)
    assert len(chunks) == 1
    assert chunks[0].text == "hello  world"
    assert chunks[0].n_tokens == 2


def test_windows_overlap_by_exactly_overlap_tokens():
    chunks = chunk_text(make(10), words, chunk_size=4, overlap=1)
    assert [c.text.split() for c in chunks] == [
        ["w0", "w1", "w2", "w3"],
        ["w3", "w4", "w5", "w6"],
        ["w6", "w7", "w8", "w9"],
    ]


def test_no_tiny_trailing_chunk_when_last_window_reaches_end():
    chunks = chunk_text(make(7), words, chunk_size=4, overlap=1)
    assert [c.text.split() for c in chunks] == [["w0", "w1", "w2", "w3"], ["w3", "w4", "w5", "w6"]]


def test_boundaries_follow_tokens_and_cover_the_document():
    text = "alpha beta\n\ngamma   delta epsilon\nzeta eta theta iota kappa"
    tokens = words(text)
    chunks = chunk_text(text, words, chunk_size=3, overlap=1)
    starts = {s for s, _ in tokens}
    ends = {e for _, e in tokens}
    for i, c in enumerate(chunks):
        assert c.index == i
        assert c.n_tokens <= 3
        assert c.start_char in starts and c.end_char in ends
        assert c.text == text[c.start_char : c.end_char]
    assert chunks[0].start_char == tokens[0][0]
    assert chunks[-1].end_char == tokens[-1][1]
    covered = {t for c in chunks for t in tokens if c.start_char <= t[0] and t[1] <= c.end_char}
    assert covered == set(tokens)


def test_zero_overlap_partitions_tokens():
    chunks = chunk_text(make(6), words, chunk_size=2, overlap=0)
    assert [c.text for c in chunks] == ["w0 w1", "w2 w3", "w4 w5"]


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (4, 4), (4, -1), (4, 5)])
def test_invalid_parameters_raise(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("a b c", words, chunk_size=size, overlap=overlap)
