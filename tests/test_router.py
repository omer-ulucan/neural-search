import pytest

from src.graph import parse_route
from src.inference import strip_channel_markers


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("local", "local"),
        ("web", "web"),
        ("  Web.\n", "web"),
        ("LOCAL", "local"),
        ("<|channel>thought\nThe user asks about news.<channel|>web", "web"),
        ("<|channel>thought\nmaybe web?<channel|>local", "local"),
        ("web search would be best", "web"),
    ],
)
def test_parse_route_extracts_decision(raw, expected):
    assert parse_route(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "banana", "I think", "<|channel>thought\n", "42"])
def test_garbage_router_output_falls_back_to_local(raw):
    assert parse_route(raw) == "local"


def test_strip_channel_markers():
    assert strip_channel_markers("<|channel>thought\nhmm<channel|>The answer [1].") == (
        "The answer [1]."
    )
    assert strip_channel_markers("  plain  ") == "plain"
