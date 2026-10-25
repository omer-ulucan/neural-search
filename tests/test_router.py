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


def test_prompts_use_gemma4_turn_tokens_with_thinking_disabled():
    from src.graph import router_prompt
    from src.inference import STOP, build_rag_prompt

    for prompt in (router_prompt("q"), build_rag_prompt("q", [{"title": "t", "text": "x"}])):
        assert prompt.startswith("<|turn>user\n")
        assert prompt.endswith("<turn|>\n<|turn>model\n<|channel>thought\n<channel|>")
        assert "<start_of_turn>" not in prompt and "<end_of_turn>" not in prompt
        assert "<bos>" not in prompt  # llama-cpp-python prepends BOS itself
    assert "<turn|>" in STOP
